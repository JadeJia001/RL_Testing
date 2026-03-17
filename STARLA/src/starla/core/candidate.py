"""Candidate representation used by STARLA genetic search."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

Episode = list[tuple[Any, ...]]


class Candidate:
    """Container for one candidate episode and its search metadata."""

    def __init__(self, episode: Episode | np.ndarray):
        if isinstance(episode, (np.ndarray, np.generic)):
            raw_episode = episode.tolist()
        else:
            raw_episode = episode
        self._episode: Episode = raw_episode
        self._objective_values: list[float] = []
        self._objectives_covered: list[int] = []
        self._crowding_distance: float = 0.0
        self._uncertainty: list[float] = []
        self._start_state: Any = 0
        self._information: list[Any] = []
        self._mutation: bool = False

    @property
    def episode(self) -> Episode:
        return self._episode

    @episode.setter
    def episode(self, value: Episode) -> None:
        self._episode = value

    @property
    def objective_values(self) -> list[float]:
        return self._objective_values

    @objective_values.setter
    def objective_values(self, values: list[float]) -> None:
        self._objective_values = values

    @property
    def objectives_covered(self) -> list[int]:
        return self._objectives_covered

    @property
    def crowding_distance(self) -> float:
        return self._crowding_distance

    @crowding_distance.setter
    def crowding_distance(self, value: float) -> None:
        self._crowding_distance = value

    @property
    def uncertainty(self) -> list[float]:
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, values: list[float]) -> None:
        self._uncertainty = values

    @property
    def start_state(self) -> Any:
        return self._start_state

    @start_state.setter
    def start_state(self, state: Any) -> None:
        self._start_state = deepcopy(state)

    @property
    def information(self) -> list[Any]:
        return self._information

    @property
    def mutation(self) -> bool:
        return self._mutation

    @mutation.setter
    def mutation(self, value: bool) -> None:
        self._mutation = value

    def get_uncertainty_value(self, index: int) -> float:
        return self._uncertainty[index]

    def add_objective_covered(self, objective_index: int) -> None:
        if objective_index not in self._objectives_covered:
            self._objectives_covered.append(objective_index)

    def exists_in_satisfied(self, objective_index: int) -> bool:
        for index in self._objectives_covered:
            if index == objective_index:
                return True
        return False

    def is_objective_covered(self, objective_index: int) -> bool:
        for covered in self._objectives_covered:
            if covered == objective_index:
                return True
        return False

    def add_info(self, new_information: Any) -> None:
        self._information.append(new_information)

