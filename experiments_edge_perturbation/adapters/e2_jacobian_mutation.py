"""E2: Jacobian-Guided Sparse Perturbation (encoding channel).

Uses the Jacobian of the world-model encoder to identify which observation
dimensions most influence the latent representation, then perturbs ONLY those
dimensions.  This produces a *sparse*, encoder-sensitivity-guided perturbation.

Reference
---------
Jakubovitz & Giryes, "Improving DNN Robustness to Adversarial Attacks using
Jacobian Regularization" (ECCV 2018).  arXiv:1803.08680
Adapted from https://github.com/facebookresearch/jacobian_regularizer
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch


class JacobianGuidedMutator:
    """Perturb only the observation dimensions that the encoder is most sensitive to.

    The Jacobian J of the encoder w.r.t. the input observation is computed via
    ``torch.autograd``.  For each observation dimension *j* the column norm
    ``||J[:, j]||_2`` measures total influence on the latent embedding.  Only
    the ``top_k`` most influential dimensions are perturbed, yielding a sparse
    adversarial mutation.

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
        top_k: int = 3,
        device: str = "cpu",
    ) -> None:
        self.world_model = world_model
        self.epsilon = float(epsilon)
        self.top_k = int(top_k)
        self.device = torch.device(device)
        self._fallback_count: int = 0
        self._jacobian_sparsity_accum: list[float] = []

    # ------------------------------------------------------------------
    # Jacobian computation (adapted from facebook/jacobian_regularizer)
    # ------------------------------------------------------------------

    def _compute_jacobian(self, obs: Any) -> np.ndarray:
        """Compute the Jacobian *J* of ``encoder(obs)`` w.r.t. *obs*.

        Returns an ndarray of shape ``(embed_dim, obs_dim)``.
        """
        state_np = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        obs_tensor = torch.as_tensor(state_np, device=self.device, dtype=torch.float32)
        obs_tensor = obs_tensor.clone().detach().requires_grad_(True)

        obs_type = getattr(self.world_model, "obs_type", "vector")
        model_obs = obs_tensor - 0.5 if obs_type == "RGB" else obs_tensor

        embed = self.world_model.encoder(model_obs)  # (1, embed_dim)
        embed_dim = embed.shape[-1]
        obs_dim = obs_tensor.shape[-1]

        # Row-by-row backward (reference repo uses per-output grad accumulation)
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
        # Column L2 norms → influence per observation dimension
        influence = np.linalg.norm(jacobian, axis=0, ord=2)  # (obs_dim,)
        return np.argsort(influence)[-k:][::-1]

    # ------------------------------------------------------------------
    # Public transform (monkey-patched into genetic.transform)
    # ------------------------------------------------------------------

    def transform(self, state: Any) -> Any:
        """Sparse perturbation of the most encoder-sensitive observation dims.

        Falls back to random noise if the Jacobian is degenerate or fails.
        """
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

            # Track sparsity: fraction of dims perturbed
            self._jacobian_sparsity_accum.append(len(top_dims) / max(obs_dim, 1))

            perturbation = np.zeros_like(state_np)
            for j in top_dims:
                col_sum = float(np.sum(jacobian[:, j]))
                perturbation[j] = self.epsilon * np.sign(col_sum)

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

    @property
    def avg_jacobian_sparsity(self) -> float:
        """Average fraction of observation dimensions perturbed."""
        if not self._jacobian_sparsity_accum:
            return 0.0
        return float(np.mean(self._jacobian_sparsity_accum))
