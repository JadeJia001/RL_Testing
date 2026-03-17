"""Q-value based abstraction strategy."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from starla.abstraction.base import AbstractionStrategy
from starla.agents.base import AgentProtocol


class QValueAbstraction(AbstractionStrategy):
    """Discretize Q-values by granularity and use bins as abstract class IDs."""

    def __init__(self, agent: AgentProtocol, granularity: float = 1.0):
        if granularity <= 0:
            raise ValueError("granularity must be > 0")
        self.agent = agent
        self.granularity = granularity

    def abstract(self, state: Any) -> tuple[Any, ...]:
        if isinstance(state, str) and state == "done":
            return ("end",)
        q_values = np.asarray(self.agent.get_q_values(state), dtype=float).reshape(-1)
        return tuple(int(ceil(value / self.granularity)) for value in q_values)

    def are_same_class(self, state1: Any, state2: Any) -> bool:
        return self.abstract(state1) == self.abstract(state2)

