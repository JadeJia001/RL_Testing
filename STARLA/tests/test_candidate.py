from __future__ import annotations

import numpy as np

from starla.core.candidate import Candidate
from starla.core.sorting import dominates


def test_candidate_create_and_setters() -> None:
    episode = [(np.array([0.1, 0.2]), 0), (np.array([0.3, 0.4]), 1), ("done", 2.0)]
    candidate = Candidate(episode)

    assert len(candidate.episode) == 3
    candidate.objective_values = [10.0, 0.2, 0.8]
    assert candidate.objective_values == [10.0, 0.2, 0.8]

    candidate.add_objective_covered(1)
    assert candidate.exists_in_satisfied(1)
    assert candidate.is_objective_covered(1)

    candidate.crowding_distance = 1.25
    assert candidate.crowding_distance == 1.25


def test_candidate_domination_relation() -> None:
    better = Candidate([(np.array([0.0]), 0), ("done", 1.0)])
    worse = Candidate([(np.array([0.0]), 0), ("done", 1.0)])
    better.objective_values = [1.0, 1.0, 1.0]
    worse.objective_values = [2.0, 2.0, 2.0]

    assert dominates(better.objective_values, worse.objective_values, [0, 1, 2])
    assert not dominates(worse.objective_values, better.objective_values, [0, 1, 2])

