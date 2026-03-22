"""SAM-guided mutation for MuJoCo vector observations (full-vector fallback)."""

from __future__ import annotations

import numpy as np

from experiments_sam.adapters.sam_mutation import SAMGuidedMutator


class SAMGuidedMutatorMujoco(SAMGuidedMutator):
    """Same SAM gradient path; fallback perturbs all state dimensions."""

    @staticmethod
    def _random_fallback(state):
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
