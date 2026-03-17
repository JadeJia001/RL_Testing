"""Fitness functions extracted from the original STARLA notebook."""

from __future__ import annotations

from typing import Any

import numpy as np

from starla.agents.base import AgentProtocol
from starla.core.candidate import Episode


def compute_reward_fitness(episode: Episode) -> float:
    """
    Reward fitness from the notebook: episode length minus terminal marker.
    """
    return float(len(episode) - 1)


def compute_confidence_fitness(
    episode: Episode,
    agent: AgentProtocol,
    mode: str = "m",
) -> float:
    """
    Confidence fitness as average action-probability margin/ratio.

    - `mode='m'`: highest - second highest
    - `mode='r'`: highest / second highest
    """
    confidence_level = 0.0
    for i in range(len(episode)):
        if i == (len(episode) - 1):
            if episode[i][0] == "done":
                return confidence_level / float(episode[i][1])
            raise AssertionError("last state is not done , reward")

        prob = np.asarray(agent.get_action_probabilities(episode[i][0]), dtype=float).reshape(-1)
        high1 = int(prob.argmax())
        first = prob[high1]
        temp = prob.copy()
        temp[high1] = 0.0
        high2 = int(temp.argmax())
        second = prob[high2]

        if mode == "r":
            confidence_level += float(first / second)
        if mode == "m":
            confidence_level += float(first - second)

    print("WARNING nothing returned", episode)
    raise AssertionError("confidence fitness did not terminate correctly")


def compute_ml_probability(ml_model: Any, binary_episode: Any) -> float:
    """
    Return non-fault probability from ML model, matching notebook behavior.
    """
    return float(ml_model.predict_proba(binary_episode)[0][0])

