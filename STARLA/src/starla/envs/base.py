"""Base protocol definitions for environment adapters."""

from __future__ import annotations

from typing import Any, Protocol


class EnvProtocol(Protocol):
    """Environment contract used by STARLA for replayable rollouts."""

    def reset(self) -> Any:
        """Reset environment and return initial observation."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Take one environment step."""

    def get_state(self) -> Any:
        """Return serializable full environment state."""

    def set_state(self, state: Any) -> Any:
        """Restore environment state and return current observation."""
