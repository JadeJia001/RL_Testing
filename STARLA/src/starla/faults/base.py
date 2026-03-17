"""Base abstractions for user-defined fault oracles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FaultOracle(ABC):
    """User-provided fault logic decoupled from concrete environments."""

    @abstractmethod
    def is_functional_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        """Return True when the episode violates functional expectations."""

    @abstractmethod
    def is_reward_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        """Return True when the episode reward behavior is faulty."""

    @abstractmethod
    def get_fault_thresholds(self) -> dict[str, Any]:
        """Expose thresholds/parameters used by the oracle."""
