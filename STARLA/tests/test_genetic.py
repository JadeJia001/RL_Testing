from __future__ import annotations

import random
from typing import Any

import numpy as np

from starla.core.candidate import Candidate
from starla.core.genetic import crossover, mutate, transform


class MockAgent:
    def predict(self, obs: Any, deterministic: bool = True) -> tuple[int, dict[str, Any]]:
        _ = deterministic
        if isinstance(obs, np.ndarray) and obs[0] > 0:
            return 1, {}
        return 0, {}

    def get_action_probabilities(self, obs: Any) -> np.ndarray:
        _ = obs
        return np.array([0.6, 0.4])

    def get_q_values(self, obs: Any) -> np.ndarray:
        _ = obs
        return np.array([1.0, 0.5])


class MockEnv:
    def __init__(self) -> None:
        self._counter = 0
        self._obs = np.array([0.0, 0.0])

    def reset(self) -> np.ndarray:
        self._counter = 0
        self._obs = np.array([0.0, 0.0])
        return self._obs

    def set_state(self, state: Any) -> np.ndarray:
        self._obs = np.array(state, dtype=float)
        return self._obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._counter += 1
        self._obs = np.array([self._counter * 0.1, float(action)])
        terminated = self._counter >= 8
        info = {"mem": [("s", i) for i in range(self._counter + 1)]}
        return self._obs, 1.0, terminated, False, info


def _make_candidate(seed: float) -> Candidate:
    episode = []
    for i in range(6):
        episode.append((np.array([seed + i * 0.1, 0.0]), i % 2))
    episode.append(("done", 6.0))
    candidate = Candidate(episode)
    candidate.start_state = np.array([seed, 0.0])
    candidate.objective_values = [seed + 1.0, 0.2, 0.3]
    return candidate


def test_transform_noise_range() -> None:
    np.random.seed(0)
    state = np.array([2.0, 1.0, -1.0])
    transformed = transform(state, noise_low=0.95, noise_high=1.05)
    ratio = transformed[0] / state[0]
    assert 0.95 <= ratio <= 1.05
    assert transformed[1] == state[1]
    assert transformed[2] == state[2]


def test_mutate_with_mock_agent_env() -> None:
    random.seed(1)
    np.random.seed(1)
    parent = _make_candidate(0.0)
    child, mutation_count = mutate(parent, MockAgent(), MockEnv(), mutation_rate=1.0)
    assert mutation_count == 1
    assert isinstance(child, Candidate)
    assert len(child.episode) > 0


def test_crossover_with_mock_agent() -> None:
    random.seed(2)
    population = [_make_candidate(0.0), _make_candidate(0.5), _make_candidate(1.0)]
    off1, off2 = crossover(
        population,
        MockAgent(),
        abstraction_fn=lambda s: (round(float(np.asarray(s)[0]), 1),),
        objective_uncovered=[0, 1, 2],
    )
    assert isinstance(off1, Candidate)
    assert isinstance(off2, Candidate)
    assert off1.episode[-1][0] == "done"
    assert off2.episode[-1][0] == "done"

