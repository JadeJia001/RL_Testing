"""Base protocol definitions for agent adapters."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class AgentProtocol(Protocol):
    """Agent contract used by STARLA evaluation and search loops."""

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, dict[str, Any]]:
        """Return selected action and optional model metadata."""

    def get_action_probabilities(self, obs: Any) -> np.ndarray:
        """Return action probability distribution for one observation."""

    def get_q_values(self, obs: Any) -> np.ndarray:
        """Return estimated Q-values for one observation."""
