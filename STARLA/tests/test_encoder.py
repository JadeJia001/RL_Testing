from __future__ import annotations

import numpy as np

from starla.ml.encoder import EpisodeEncoder


class MockAbstraction:
    def abstract(self, state):  # noqa: ANN001
        if isinstance(state, str) and state == "done":
            return ("end",)
        return (int(np.asarray(state)[0]),)

    def are_same_class(self, state1, state2):  # noqa: ANN001
        return self.abstract(state1) == self.abstract(state2)


def test_episode_encoder_binary_vector() -> None:
    encoder = EpisodeEncoder(abstract_states=[(0,), (1,), (2,)], abstraction=MockAbstraction())
    episode = [(np.array([0]), 0), (np.array([2]), 1), ("done", 2.0)]
    encoded = encoder.encode(episode)
    assert np.array_equal(encoded, np.array([1.0, 0.0, 1.0]))


def test_episode_encoder_batch_shape() -> None:
    encoder = EpisodeEncoder(abstract_states=[(0,), (1,), (2,)], abstraction=MockAbstraction())
    episodes = [
        [(np.array([0]), 0), ("done", 1.0)],
        [(np.array([1]), 1), ("done", 1.0)],
    ]
    batch = encoder.encode_batch(episodes)
    assert batch.shape == (2, 3)

