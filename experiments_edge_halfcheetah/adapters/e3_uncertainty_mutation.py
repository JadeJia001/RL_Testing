"""E3: Uncertainty-Guided Directional Perturbation (prediction channel) — MuJoCo HalfCheetah (vector).

Uses the DreamerV3 world model's stochastic predictions to estimate
prediction uncertainty, then perturbs the observation in the direction
that maximises model uncertainty.  Adapted for vector observations ``(17,)``.

Reference
---------
Yu et al., "MOPO: Model-based Offline Policy Optimization" (NeurIPS 2020).
arXiv:2005.13239
Adapted from https://github.com/tianheyu927/mopo
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class UncertaintyGuidedMutator:
    """Perturb the vector observation in the direction of highest model uncertainty.

    Parameters
    ----------
    world_model:
        DreamerV3 world model (must expose ``.encoder``, ``.dynamics``, ``.heads``).
    epsilon:
        Magnitude of the normalised perturbation.
    n_samples:
        Number of stochastic RSSM forward passes for variance estimation.
    device:
        Torch device string; must be ``"cpu"`` for the CPU dev container.
    """

    def __init__(
        self,
        world_model: Any,
        *,
        epsilon: float = 0.05,
        n_samples: int = 5,
        device: str = "cpu",
    ) -> None:
        self.world_model = world_model
        self.epsilon = float(epsilon)
        self.n_samples = int(n_samples)
        self.device = torch.device(device)
        self._fallback_count: int = 0
        self._uncertainty_magnitude_accum: list[float] = []

    # ------------------------------------------------------------------
    # Uncertainty gradient computation
    # ------------------------------------------------------------------

    def _estimate_uncertainty_gradient(self, obs: Any) -> np.ndarray:
        """Gradient of reward-prediction variance w.r.t. input observation.

        Returns an ndarray with the same shape as *obs*.
        """
        state_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_tensor = torch.as_tensor(state_np, device=self.device, dtype=torch.float32)
        obs_tensor = obs_tensor.clone().detach().requires_grad_(True)

        embed = self.world_model.encoder(obs_tensor)
        latent = self.world_model.dynamics.initial(1)
        action_size = int(getattr(self.world_model, "action_size"))
        prev_action = torch.zeros((1, action_size), device=self.device, dtype=torch.float32)

        reward_preds: list[torch.Tensor] = []
        for _ in range(self.n_samples):
            lat, _ = self.world_model.dynamics.obs_step(
                latent, prev_action, embed, sample=True,
            )
            feat = self.world_model.dynamics.get_feat(lat)
            reward_head = self.world_model.heads["reward"]
            reward_dist = reward_head(feat)
            reward_val = reward_dist.mode() if hasattr(reward_dist, "mode") else reward_dist
            reward_preds.append(reward_val.squeeze())

        stacked = torch.stack(reward_preds, dim=0)
        variance = torch.var(stacked, dim=0)
        max_var = variance.max() if variance.dim() > 0 else variance

        max_var.backward()

        if obs_tensor.grad is None:
            raise RuntimeError("Uncertainty gradient failed: obs.grad is None.")

        grad = obs_tensor.grad.detach().cpu().numpy().reshape(-1)
        return grad.astype(np.float32)

    # ------------------------------------------------------------------
    # Public transform
    # ------------------------------------------------------------------

    def transform(self, state: Any) -> Any:
        """Directional perturbation toward high model uncertainty."""
        try:
            for module in (self.world_model,):
                if hasattr(module, "zero_grad"):
                    try:
                        module.zero_grad(set_to_none=True)
                    except TypeError:
                        module.zero_grad()

            with torch.enable_grad():
                grad = self._estimate_uncertainty_gradient(state)

            grad_norm = float(np.linalg.norm(grad, ord=2))
            if (not np.isfinite(grad_norm)) or grad_norm < 1e-12:
                self._fallback_count += 1
                return self._random_fallback(state)

            self._uncertainty_magnitude_accum.append(grad_norm)
            perturbation = self.epsilon * grad / grad_norm
            return self._apply_perturbation(state, perturbation)

        except Exception:
            self._fallback_count += 1
            return self._random_fallback(state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_perturbation(state: Any, perturbation: np.ndarray) -> Any:
        state_np = np.asarray(state, dtype=np.float32)
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
        """Perturb ALL dimensions with multiplicative uniform noise [0.95, 1.05]."""
        state_np = np.asarray(state, dtype=np.float32)
        noise = np.random.uniform(low=0.95, high=1.05, size=state_np.shape).astype(np.float32)
        new_state = state_np * noise
        if isinstance(state, np.ndarray):
            return new_state.astype(state.dtype, copy=False)
        if isinstance(state, tuple):
            return tuple(new_state.tolist())
        if isinstance(state, list):
            return new_state.tolist()
        return new_state

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    @property
    def avg_uncertainty_magnitude(self) -> float:
        if not self._uncertainty_magnitude_accum:
            return 0.0
        return float(np.mean(self._uncertainty_magnitude_accum))
