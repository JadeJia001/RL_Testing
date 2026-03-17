from __future__ import annotations

import random

from starla.core.candidate import Candidate
from starla.core.sorting import dominates, fast_non_dominated_sort, tournament_selection


def _cand(vals: list[float]) -> Candidate:
    c = Candidate([("done", 0.0)])
    c.objective_values = vals
    return c


def test_dominates_correctness() -> None:
    assert dominates([1.0, 2.0], [2.0, 3.0], [0, 1])
    assert not dominates([2.0, 3.0], [1.0, 2.0], [0, 1])
    assert not dominates([1.0, 3.0], [2.0, 2.0], [0, 1])


def test_fast_non_dominated_sort_fronts() -> None:
    # Front 1: a, b (互不支配) ; Front 2: c (被 a/b 支配)
    a = _cand([1.0, 3.0])
    b = _cand([3.0, 1.0])
    c = _cand([4.0, 4.0])
    fronts = fast_non_dominated_sort([a, b, c], [0, 1])
    assert len(fronts) >= 2
    assert set(fronts[0]) == {a, b}
    assert fronts[1] == [c]


def test_tournament_selection_returns_candidate() -> None:
    random.seed(7)
    population = [_cand([1.0, 1.0]), _cand([2.0, 2.0]), _cand([3.0, 3.0])]
    winner = tournament_selection(population, size=3, objectives=[0, 1])
    assert isinstance(winner, Candidate)
    assert winner in population

