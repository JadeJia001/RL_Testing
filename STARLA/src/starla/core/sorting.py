"""Sorting and selection operators used by MOSA."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from starla.core.candidate import Candidate


def _resolve_objective_indices(values_length: int, objectives: Sequence[int]) -> list[int]:
    if len(objectives) == 0:
        return list(range(values_length))
    return [int(index) for index in objectives]


def dominates(values_a: Sequence[float], values_b: Sequence[float], objectives: Sequence[int]) -> bool:
    """Return True if `values_a` dominates `values_b` on the selected objectives."""
    indices = _resolve_objective_indices(min(len(values_a), len(values_b)), objectives)
    dominates_a = False
    dominates_b = False
    for objective_index in indices:
        f_a = values_a[objective_index]
        f_b = values_b[objective_index]
        if f_a < f_b:
            dominates_a = True
        if f_b < f_a:
            dominates_b = True
        if dominates_a and dominates_b:
            break
    if dominates_a == dominates_b:
        return False
    if dominates_a:
        return True
    return False


def fast_non_dominated_sort(
    population: list[Candidate],
    objectives: Sequence[int],
) -> list[list[Candidate]]:
    """Sort population into non-dominated fronts using pymoo's NDS API."""
    if not population:
        return []
    indices = _resolve_objective_indices(len(population[0].objective_values), objectives)
    objective_matrix = np.array(
        [[candidate.objective_values[idx] for idx in indices] for candidate in population],
        dtype=float,
    )
    fronts_indices = NonDominatedSorting().do(objective_matrix)
    return [[population[i] for i in front] for front in fronts_indices]


def preference_sort(population: list[Candidate], size: int, objectives: Sequence[int]) -> list[Candidate | list[Candidate]]:
    """
    Preference sort as used in notebook MOSA.

    Returns a mixed list: first preferred candidates, then non-dominated fronts.
    """
    _ = size  # kept for API compatibility with notebook signature
    if not population:
        return []
    remaining = list(population)
    ordered: list[Candidate | list[Candidate]] = []
    objective_indices = _resolve_objective_indices(len(remaining[0].objective_values), objectives)

    for objective_index in objective_indices:
        best = remaining[0]
        best_value = best.objective_values[objective_index]
        for candidate in remaining:
            value = candidate.objective_values[objective_index]
            if value < best_value:
                best = candidate
                best_value = value
        ordered.append(best)
        remaining.remove(best)
        if not remaining:
            break

    if remaining:
        ordered.extend(fast_non_dominated_sort(remaining, objective_indices))
    return ordered


def compute_crowding_distance(front: list[Candidate]) -> np.ndarray:
    """Compute NSGA-II crowding distance for one front using pymoo 0.6.x APIs."""
    if not front:
        return np.array([], dtype=float)

    # Ensure imports stay aligned with modern pymoo APIs.
    _ = RankAndCrowding(crowding_func="cd")

    objective_matrix = np.array([candidate.objective_values for candidate in front], dtype=float)
    crowding_function = get_crowding_function("cd")
    return np.asarray(crowding_function.do(objective_matrix, n_remove=0), dtype=float)


def tournament_selection(population: list[Candidate], size: int, objectives: Sequence[int]) -> Candidate:
    """Select the most dominating candidate from a random tournament."""
    if not population:
        raise ValueError("Population must not be empty for tournament selection.")
    tournament: list[Candidate] = []
    for _ in range(size):
        index = random.randint(0, len(population) - 1)
        tournament.append(population[index])

    best = tournament[0]
    for candidate_a in tournament:
        for candidate_b in tournament:
            if dominates(candidate_a.objective_values, candidate_b.objective_values, objectives):
                best = candidate_a
    return best

