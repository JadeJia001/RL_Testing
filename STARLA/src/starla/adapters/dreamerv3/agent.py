"""DreamerV3 agent adapter for STARLA AgentProtocol."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch


class DreamerV3Agent:
    """Adapter that exposes DI-engine DreamerV3 through STARLA AgentProtocol."""

    def __init__(self, policy: Any, world_model: Any):
        self.policy = policy
        self.world_model = world_model
        self.actor = policy._model.actor
        self.critic = policy._model.critic
        self.value_head = world_model.heads.get("value", self.critic)
        self.device = torch.device(getattr(policy, "_device", "cpu"))

        self._collect_dyn_sample = bool(
            getattr(getattr(getattr(policy, "_cfg", None), "collect", None), "collect_dyn_sample", False)
        )
        self._latent_state: dict[str, torch.Tensor] | None = None
        self._prev_action: torch.Tensor | None = None
        self._continuous_prob_warning_emitted = False

    def reset_state(self) -> None:
        """Reset recurrent latent state used by sequential predict calls."""
        self._latent_state = None
        self._prev_action = None

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, dict[str, Any]]:
        obs_tensor = self._to_obs_tensor(obs)
        latent, actor_dist = self._next_latent_and_dist(obs_tensor, use_internal_state=True)

        with torch.no_grad():
            action_tensor = actor_dist.mode() if deterministic else actor_dist.sample()
            logprob = actor_dist.log_prob(action_tensor)

        self._latent_state = {k: v.detach() for k, v in latent.items()}
        self._prev_action = action_tensor.detach()

        action = self._format_action(action_tensor)
        info: dict[str, Any] = {"logprob": self._to_numpy(logprob)}
        return action, info

    def get_action_probabilities(self, obs: Any) -> np.ndarray:
        obs_tensor = self._to_obs_tensor(obs)
        _, actor_dist = self._next_latent_and_dist(obs_tensor, use_internal_state=False)

        with torch.no_grad():
            if self.world_model.action_type == "discrete":
                if hasattr(actor_dist, "probs") and actor_dist.probs is not None:
                    probs = actor_dist.probs
                elif hasattr(actor_dist, "logits"):
                    probs = torch.softmax(actor_dist.logits, dim=-1)
                else:
                    raise NotImplementedError("Actor distribution does not expose logits/probabilities.")
                return np.asarray(probs[0].detach().cpu().numpy(), dtype=float)

            action_size = int(self.world_model.action_size)
            if not self._continuous_prob_warning_emitted:
                logging.warning(
                    "DreamerV3 continuous actor has no categorical probabilities; "
                    "returning a uniform placeholder distribution."
                )
                print(
                    "[DreamerV3Agent] continuous action space detected; "
                    "get_action_probabilities returns a uniform placeholder."
                )
                self._continuous_prob_warning_emitted = True
            return np.full((action_size, ), 1.0 / max(action_size, 1), dtype=float)

    def get_q_values(self, obs: Any) -> np.ndarray:
        obs_tensor = self._to_obs_tensor(obs)
        latent, _ = self._next_latent_and_dist(obs_tensor, use_internal_state=False)
        dynamics = self.world_model.dynamics

        with torch.no_grad():
            if self.world_model.action_type == "discrete":
                q_values: list[float] = []
                action_size = int(self.world_model.action_size)
                for idx in range(action_size):
                    action = torch.zeros((1, action_size), device=self.device)
                    action[:, idx] = 1.0
                    imag_state = dynamics.img_step(latent, action, sample=False)
                    feat = dynamics.get_feat(imag_state)
                    value = self.value_head(feat).mode().reshape(-1)[0]
                    q_values.append(float(value.detach().cpu().item()))
                return np.asarray(q_values, dtype=float)

            feat = dynamics.get_feat(latent)
            value = self.value_head(feat).mode().reshape(-1)[0]
            action_size = int(self.world_model.action_size)
            # Limitation: for continuous actions, DreamerV3 does not provide per-action Q estimates.
            return np.full((action_size, ), float(value.detach().cpu().item()), dtype=float)

    def _next_latent_and_dist(self, obs_tensor: torch.Tensor, use_internal_state: bool) -> tuple[dict[str, torch.Tensor], Any]:
        dynamics = self.world_model.dynamics
        with torch.no_grad():
            if use_internal_state and self._latent_state is not None and self._prev_action is not None:
                latent = {k: v.to(self.device) for k, v in self._latent_state.items()}
                prev_action = self._prev_action.to(self.device)
                if prev_action.dim() == 1:
                    prev_action = prev_action.unsqueeze(0)
                if latent["deter"].shape[0] != obs_tensor.shape[0]:
                    latent = dynamics.initial(obs_tensor.shape[0])
                    prev_action = torch.zeros((obs_tensor.shape[0], int(self.world_model.action_size)), device=self.device)
            else:
                latent = dynamics.initial(obs_tensor.shape[0])
                prev_action = torch.zeros((obs_tensor.shape[0], int(self.world_model.action_size)), device=self.device)

            model_obs = obs_tensor - 0.5 if self.world_model.obs_type == "RGB" else obs_tensor
            embed = self.world_model.encoder(model_obs)
            latent, _ = dynamics.obs_step(latent, prev_action, embed, self._collect_dyn_sample)
            feat = dynamics.get_feat(latent)
            actor_dist = self.actor(feat)
            return latent, actor_dist

    def _to_obs_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs.to(self.device, dtype=torch.float32)
        else:
            obs_tensor = torch.as_tensor(np.asarray(obs), device=self.device, dtype=torch.float32)

        if self.world_model.obs_type == "vector" and obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        if self.world_model.obs_type == "RGB" and obs_tensor.dim() == 3:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor

    def _format_action(self, action_tensor: torch.Tensor) -> Any:
        if self.world_model.action_type == "discrete":
            action_idx = torch.argmax(action_tensor, dim=-1)
            action_np = action_idx.detach().cpu().numpy()
            if action_np.shape[0] == 1:
                return int(action_np[0])
            return action_np

        action_np = action_tensor.detach().cpu().numpy()
        if action_np.shape[0] == 1:
            return action_np[0]
        return action_np

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray | float:
        array = tensor.detach().cpu().numpy()
        if array.size == 1:
            return float(array.reshape(-1)[0])
        return np.asarray(array)
