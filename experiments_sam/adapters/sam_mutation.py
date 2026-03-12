"""SAM-guided mutation operator for DreamerV3 state perturbation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

# Ensure local project modules are importable when running this script directly.
_ROOT = Path(__file__).resolve().parents[2]
_STARLA_SRC = _ROOT / "STARLA" / "src"
_DI_ENGINE = _ROOT / "DI-engine"
for _path in (_STARLA_SRC, _DI_ENGINE):
    _path_str = str(_path)
    if _path.exists() and _path_str not in sys.path:
        sys.path.insert(0, _path_str)


class SAMGuidedMutator:
    """Input-space analogue of SAM first_step, guided by critic value descent."""

    def __init__(self, world_model: Any, actor: Any, critic: Any, *, rho: float = 0.1, device: str = "cpu"):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.rho = float(rho)
        self.device = torch.device(device)
        self._fallback_count = 0

    def _compute_sharpness_gradient(self, state: Any) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32)
        obs = torch.as_tensor(state_np, device=self.device, dtype=torch.float32)
        added_batch_dim = False

        obs_type = getattr(self.world_model, "obs_type", "vector")
        if obs_type == "vector" and obs.dim() == 1:
            obs = obs.unsqueeze(0)
            added_batch_dim = True
        elif obs_type == "RGB" and obs.dim() == 3:
            obs = obs.unsqueeze(0)
            added_batch_dim = True

        obs = obs.clone().detach().requires_grad_(True)
        model_obs = obs - 0.5 if obs_type == "RGB" else obs

        embed = self.world_model.encoder(model_obs)
        batch_size = int(obs.shape[0])
        latent = self.world_model.dynamics.initial(batch_size)
        action_size = int(getattr(self.world_model, "action_size"))
        prev_action = torch.zeros((batch_size, action_size), device=self.device, dtype=torch.float32)

        latent, _ = self.world_model.dynamics.obs_step(latent, prev_action, embed, sample=False)
        feat = self.world_model.dynamics.get_feat(latent)
        value_dist = self.critic(feat)
        value = value_dist.mode() if hasattr(value_dist, "mode") else value_dist
        loss = -value.sum()
        loss.backward()

        if obs.grad is None:
            raise RuntimeError("SAM-guided mutation failed: obs.grad is None.")

        grad = obs.grad.detach().cpu().numpy()
        if added_batch_dim:
            grad = grad[0]
        return np.asarray(grad, dtype=np.float32)

    def transform(self, state: Any) -> Any:
        try:
            for module in (self.world_model, self.actor, self.critic):
                if hasattr(module, "zero_grad"):
                    try:
                        module.zero_grad(set_to_none=True)
                    except TypeError:
                        module.zero_grad()

            with torch.enable_grad():
                grad = self._compute_sharpness_gradient(state)

            grad_norm = float(np.linalg.norm(grad.reshape(-1), ord=2))
            if (not np.isfinite(grad_norm)) or grad_norm < 1e-12:
                self._fallback_count += 1
                return self._random_fallback(state)

            perturbation = self.rho * grad / grad_norm
            return self._apply_perturbation(state, perturbation)
        except Exception:
            self._fallback_count += 1
            return self._random_fallback(state)

    @staticmethod
    def _apply_perturbation(state: Any, perturbation: np.ndarray) -> Any:
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.shape != perturbation.shape:
            raise ValueError(f"State shape {state_np.shape} and perturbation shape {perturbation.shape} mismatch.")
        new_state_np = state_np + perturbation.astype(np.float32, copy=False)

        if isinstance(state, np.ndarray):
            return new_state_np.astype(state.dtype, copy=False)
        if isinstance(state, tuple):
            return tuple(new_state_np.tolist())
        if isinstance(state, list):
            return new_state_np.tolist()
        return new_state_np

    @staticmethod
    def _random_fallback(state: Any) -> Any:
        position = state[0]
        noise = np.random.uniform(low=0.95, high=1.05)
        new_position = position * noise
        if isinstance(state, tuple):
            new_state = list(state)
            new_state[0] = new_position
            return tuple(new_state)
        new_state = deepcopy(state)
        new_state[0] = new_position
        return new_state

    @property
    def fallback_count(self) -> int:
        return self._fallback_count
