from __future__ import annotations

import numpy as np

from starla.abstraction.qvalue_abstraction import QValueAbstraction


class MockAgent:
    def get_q_values(self, state):  # noqa: ANN001
        base = float(np.asarray(state)[0])
        return np.array([base, base + 0.6])

    def predict(self, obs, deterministic=True):  # noqa: ANN001, ANN201
        _ = (obs, deterministic)
        return 0, {}

    def get_action_probabilities(self, obs):  # noqa: ANN001
        _ = obs
        return np.array([0.5, 0.5])


def test_qvalue_abstraction_maps_with_ceil() -> None:
    abstraction = QValueAbstraction(MockAgent(), granularity=1.0)
    cls = abstraction.abstract(np.array([1.2]))
    assert cls == (2, 2)


def test_qvalue_abstraction_same_class() -> None:
    abstraction = QValueAbstraction(MockAgent(), granularity=1.0)
    assert abstraction.are_same_class(np.array([1.1]), np.array([1.2]))
    assert not abstraction.are_same_class(np.array([1.1]), np.array([2.2]))

