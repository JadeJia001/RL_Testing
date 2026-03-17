"""Abstraction interfaces for mapping concrete states to abstract classes."""

from __future__ import annotations

from typing import Any, Protocol


class AbstractionStrategy(Protocol):
    """Protocol for state abstraction strategies."""

    def abstract(self, state: Any) -> tuple[Any, ...]:
        """Map a concrete state to an abstract state identifier."""

    def are_same_class(self, state1: Any, state2: Any) -> bool:
        """Return True if two states belong to the same abstract class."""

