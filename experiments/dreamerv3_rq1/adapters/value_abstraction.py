"""Dreamer value-based abstraction strategy for RQ1 experiments."""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_STARLA_SRC = _ROOT / "STARLA" / "src"
if str(_STARLA_SRC) not in sys.path:
    sys.path.insert(0, str(_STARLA_SRC))

from starla.abstraction.base import AbstractionStrategy
from starla.adapters.dreamerv3.agent import DreamerV3Agent


class DreamerValueAbstraction(AbstractionStrategy):
    """Discretize Dreamer pseudo Q/value vectors into abstraction bins."""

    def __init__(self, agent: DreamerV3Agent, granularity: float = 1.0):
        if granularity <= 0:
            raise ValueError("granularity must be > 0")
        self.agent = agent
        self.granularity = granularity

    def abstract(self, state: Any) -> tuple[Any, ...]:
        if isinstance(state, str) and state == "done":
            return ("end",)
        values = np.asarray(self.agent.get_q_values(state), dtype=float).reshape(-1)
        return tuple(int(ceil(value / self.granularity)) for value in values)

    def are_same_class(self, state1: Any, state2: Any) -> bool:
        return self.abstract(state1) == self.abstract(state2)
