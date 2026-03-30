"""E2: Jacobian-Guided Sparse Perturbation (encoding channel) — DMControl Cheetah-run (RGB).

Uses the Jacobian of the world-model encoder to identify which observation
dimensions most influence the latent representation, then perturbs ONLY those
dimensions.  Adapted for RGB observations of shape ``(3, 64, 64)``.

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
    """Perturb only the observation pixels that the encoder is most sensitive to.

    For RGB obs ``(3, 64, 64)``, the Jacobian has shape ``(embed_dim, 12288)``.
    Column norms identify the most influential *pixel-channel* positions.

    Parameters
    ----------
    world_model:
        DreamerV3 world model (must expose ``.encoder``).
    epsilon:
        Magnitude of the additive perturbation applied to selected dimensions.
    top_k:
        Number of pixel-channel dimensions to perturb (most sensitive first).
    max_embed_dims:
        Limit Jacobian computation to the first N embedding dimensions for speed.
    device:
        Torch device string; must be ``"cpu"`` for the CPU dev container.
    """

    def __init__(
        self,
        world_model: Any,
        *,
        epsilon: float = 0.05,
        top_k: int = 128,
        max_embed_dims: int = 64,
        device: str = "cpu",
    ) -> None:
        self.world_model = world_model
        self.epsilon = float(epsilon)
        self.top_k = int(top_k)
        self._max_embed_dims = int(max_embed_dims)
        self.device = torch.device(device)
        self._fallback_count: int = 0
        self._jacobian_sparsity_accum: list[float] = []

    # ------------------------------------------------------------------
    # Jacobian computation
    # ------------------------------------------------------------------

    def _compute_jacobian(self, obs: Any) -> np.ndarray:
        """Compute the Jacobian *J* of ``encoder(obs)`` w.r.t. *obs*.

        For RGB input ``(3, 64, 64)`` returns shape ``(effective_embed_dim, 12288)``.
        """
        state_np = np.asarray(obs, dtype=np.float32)
        obs_shape = state_np.shape  # (3, 64, 64)
        obs_tensor = torch.as_tensor(
            state_np.reshape(1, *obs_shape), device=self.device, dtype=torch.float32,
        )
        obs_tensor = obs_tensor.clone().detach().requires_grad_(True)

        model_obs = obs_tensor - 0.5  # DreamerV3 RGB preprocessing

        embed = self.world_model.encoder(model_obs)
        embed_dim = embed.shape[-1]
        obs_dim = obs_tensor[0].numel()

        effective_embed_dim = min(embed_dim, self._max_embed_dims)

        jacobian = torch.zeros(effective_embed_dim, obs_dim, device=self.device)
        for i in range(effective_embed_dim):
            if obs_tensor.grad is not None:
                obs_tensor.grad.zero_()
            embed[0, i].backward(retain_graph=True)
            if obs_tensor.grad is not None:
                jacobian[i] = obs_tensor.grad.reshape(1, -1)[0].clone()

        return jacobian.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Dimension selection
    # ------------------------------------------------------------------

    def _select_sensitive_obs_dims(self, jacobian: np.ndarray, k: int) -> np.ndarray:
        obs_dim = jacobian.shape[1]
        k = min(k, obs_dim)
        influence = np.linalg.norm(jacobian, axis=0, ord=2)
        return np.argsort(influence)[-k:][::-1]

    # ------------------------------------------------------------------
    # Public transform
    # ------------------------------------------------------------------

    def transform(self, state: Any) -> Any:
        """Sparse perturbation of the most encoder-sensitive pixel-channel dims."""
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

            perturbation = np.zeros(obs_dim, dtype=np.float32)
            for j in top_dims:
                col_sum = float(np.sum(jacobian[:, j]))
                perturbation[j] = self.epsilon * np.sign(col_sum)

            return self._apply_perturbation(state, perturbation.reshape(state_np.shape))

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
        return np.clip(new_state_np, 0.0, 1.0)

    @staticmethod
    def _random_fallback(state: Any) -> Any:
        state_np = np.asarray(state, dtype=np.float32)
        noise = np.random.normal(0, 0.02, size=state_np.shape).astype(np.float32)
        return np.clip(state_np + noise, 0.0, 1.0)

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    @property
    def avg_jacobian_sparsity(self) -> float:
        if not self._jacobian_sparsity_accum:
            return 0.0
        return float(np.mean(self._jacobian_sparsity_accum))
