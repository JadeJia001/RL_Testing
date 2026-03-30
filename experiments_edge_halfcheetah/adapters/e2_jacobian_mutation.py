"""E2: Jacobian-Guided Sparse Perturbation (encoding channel) — MuJoCo HalfCheetah (vector).

Uses the Jacobian of the world-model encoder to identify which observation
dimensions most influence the latent representation, then perturbs ONLY those
dimensions.  Adapted for vector observations of shape ``(17,)``.

Reference
---------
Jakubovitz & Giryes, "Improving DNN Robustness to Adversarial Attacks using
Jacobian Regularization" (ECCV 2018).  arXiv:1803.08680
Adapted from https://github.com/facebookresearch/jacobian_regularizer
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class JacobianGuidedMutator:
    """Perturb only the observation dimensions that the encoder is most sensitive to.

    For vector obs ``(17,)`` the Jacobian has shape ``(embed_dim, 17)``.
    Column norms identify the most influential proprioceptive dimensions.

    Parameters
    ----------
    world_model:
        DreamerV3 world model (must expose ``.encoder``).
    epsilon:
        Magnitude of the additive perturbation applied to selected dimensions.
    top_k:
        Number of observation dimensions to perturb (most sensitive first).
    device:
        Torch device string; must be ``"cpu"`` for the CPU dev container.
    """

    def __init__(
        self,
        world_model: Any,
        *,
        epsilon: float = 0.05,
        top_k: int = 5,
        device: str = "cpu",
    ) -> None:
        self.world_model = world_model
        self.epsilon = float(epsilon)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self._fallback_count: int = 0
        self._jacobian_sparsity_accum: list[float] = []

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def _compute_jacobian(self, obs: Any) -> np.ndarray:
        """Compute the Jacobian *J* of ``encoder(obs)`` w.r.t. *obs*.

        Returns an ndarray of shape ``(embed_dim, 17)``.
        """
        state_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_tensor = torch.as_tensor(state_np, device=self.device, dtype=torch.float32)
        obs_tensor = obs_tensor.clone().detach().requires_grad_(True)

        embed = self.world_model.encoder(obs_tensor)  # (1, embed_dim)
        embed_dim = embed.shape[-1]
        obs_dim = obs_tensor.shape[-1]

        jacobian = torch.zeros(embed_dim, obs_dim, device=self.device)
        for i in range(embed_dim):
            if obs_tensor.grad is not None:
                obs_tensor.grad.zero_()
            embed[0, i].backward(retain_graph=True)
            if obs_tensor.grad is not None:
                jacobian[i] = obs_tensor.grad[0].clone()

        return jacobian.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Dimension selection
    # ------------------------------------------------------------------

    def _select_sensitive_obs_dims(self, jacobian: np.ndarray, k: int) -> np.ndarray:
        """Return indices of the *k* observation dims with largest column norms."""
        obs_dim = jacobian.shape[1]
        k = min(k, obs_dim)
        influence = np.linalg.norm(jacobian, axis=0, ord=2)
        return np.argsort(influence)[-k:][::-1]

    # ------------------------------------------------------------------
    # Public transform
    # ------------------------------------------------------------------

    def transform(self, state: Any) -> Any:
        """Sparse perturbation of the most encoder-sensitive observation dims."""
        try:
            for module in (self.world_model,):
                if hasattr(module, "zero_grad"):
                    try:
                        module.zero_grad(set_to_none=True)
                    except TypeError:
                        module.zero_grad()

            with torch.enable_grad():
                jacobian = self._compute_jacobian(state)

            if not np.isfinite(jacobian).all():
                self._fallback_count += 1
                return self._random_fallback(state)

            top_dims = self._select_sensitive_obs_dims(jacobian, self.top_k)
            state_np = np.asarray(state, dtype=np.float32)
            obs_dim = state_np.size

            self._jacobian_sparsity_accum.append(len(top_dims) / max(obs_dim, 1))

            perturbation = np.zeros_like(state_np)
            flat_perturb = perturbation.reshape(-1)
            for j in top_dims:
                col_sum = float(np.sum(jacobian[:, j]))
                flat_perturb[j] = self.epsilon * np.sign(col_sum)

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
    def avg_jacobian_sparsity(self) -> float:
        if not self._jacobian_sparsity_accum:
            return 0.0
        return float(np.mean(self._jacobian_sparsity_accum))
