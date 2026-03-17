"""Episode encoding utilities for ML-based fault prediction."""

from __future__ import annotations

from typing import Any

import numpy as np

from starla.abstraction.base import AbstractionStrategy
from starla.core.candidate import Episode


class EpisodeEncoder:
    """Encode episodes as binary presence/absence vectors of abstract states."""

    def __init__(self, abstract_states: list[tuple[Any, ...]], abstraction: AbstractionStrategy):
        self.abstract_states = list(abstract_states)
        self.abstraction = abstraction
        self._index = {state: i for i, state in enumerate(self.abstract_states)}

    def encode(self, episode: Episode) -> np.ndarray:
        record = np.zeros(len(self.abstract_states), dtype=float)
        for transition in episode:
            state = transition[0]
            abstract_state = self.abstraction.abstract(state)
            if abstract_state == ("end",):
                continue
            index = self._index.get(abstract_state)
            if index is not None:
                record[index] = 1.0
        return record

    def encode_batch(self, episodes: list[Episode]) -> np.ndarray:
        if not episodes:
            return np.zeros((0, len(self.abstract_states)), dtype=float)
        return np.vstack([self.encode(episode) for episode in episodes])

