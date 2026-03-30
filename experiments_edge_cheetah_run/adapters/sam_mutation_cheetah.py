"""SAM-guided mutation for DMControl Cheetah-run pixel observations."""

from __future__ import annotations

import numpy as np

from experiments_sam.adapters.sam_mutation import SAMGuidedMutator


class SAMGuidedMutatorCheetah(SAMGuidedMutator):
    """Same as SAMGuidedMutator except random fallback for image states."""

    @staticmethod
    def _random_fallback(state):
        state_np = np.asarray(state, dtype=np.float32)
        noise = np.random.normal(0, 0.02, size=state_np.shape).astype(np.float32)
        noisy = state_np + noise
        if hasattr(state, "dtype") and getattr(state, "dtype", None) == np.uint8:
            return np.clip(noisy, 0, 255).astype(np.uint8)
        return np.clip(noisy, 0.0, 1.0)
