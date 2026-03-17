"""MOSA multi-objective search engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import random
import time
from typing import Any, Callable, Protocol

from starla.agents.base import AgentProtocol
from starla.config import StarlaConfig
from starla.core.candidate import Candidate, Episode
from starla.core.fitness import (
    compute_confidence_fitness,
    compute_ml_probability,
    compute_reward_fitness,
)
from starla.core.genetic import crossover, mutate
from starla.core.sorting import compute_crowding_distance, preference_sort, tournament_selection
from starla.envs.base import EnvProtocol
from starla.faults.base import FaultOracle


class _PredictProbaProtocol(Protocol):
    def predict_proba(self, x: Any) -> Any:
        ...


@dataclass(slots=True)
class SearchResult:
    archive: list[Candidate]
    all_generations: list[list[Candidate]]
    mutation_count: int
    elapsed_time: float


class MOSAEngine:
    """Type-safe MOSA engine extracted from the original notebook workflow."""

    def __init__(
        self,
        config: StarlaConfig,
        agent: AgentProtocol,
        env: EnvProtocol,
        fault_oracle: FaultOracle,
        abstraction: Any,
        ml_predictor: _PredictProbaProtocol,
    ) -> None:
        self.config = config
        self.agent = agent
        self.env = env
        self.fault_oracle = fault_oracle
        self.abstraction = abstraction
        self.ml_predictor = ml_predictor
        self.logger = logging.getLogger(__name__)

    def _abstract_state(self, state: Any) -> Any:
        if hasattr(self.abstraction, "abstract"):
            return self.abstraction.abstract(state)
        if hasattr(self.abstraction, "abstract_state"):
            return self.abstraction.abstract_state(state)
        if callable(self.abstraction):
            return self.abstraction(state)
        return state

    def _abstract_episode(self, episode: Episode) -> Any:
        if hasattr(self.abstraction, "encode"):
            return self.abstraction.encode(episode)
        if hasattr(self.abstraction, "encode_episode"):
            return self.abstraction.encode_episode(episode)
        if hasattr(self.abstraction, "transform_episode"):
            return self.abstraction.transform_episode(episode)
        if callable(self.abstraction):
            return self.abstraction(episode)
        return episode

    def _evaluate_candidate(self, candidate: Candidate) -> None:
        episode = candidate.episode
        objective_1 = compute_reward_fitness(episode)
        objective_2 = compute_confidence_fitness(episode, self.agent, mode="m")
        binary_episode = self._abstract_episode(episode)
        objective_3 = compute_ml_probability(self.ml_predictor, binary_episode)

        values = [objective_1, objective_2, objective_3]
        candidate.objective_values = values[: self.config.num_objectives]

    def _evaluate_population(self, population: list[Candidate]) -> None:
        for candidate in population:
            self._evaluate_candidate(candidate)

    def _update_archive(
        self,
        population: list[Candidate],
        objective_uncovered: list[int],
        archive: list[Candidate],
        initial_population: list[Candidate],
    ) -> None:
        # Merged behavior of notebook update_archive + Build_Archive:
        # objective-wise best satisfying candidates + duplicate filtering by objective vector.
        thresholds = self.config.objective_thresholds
        for objective_index in range(self.config.num_objectives):
            for candidate in population:
                objective_values = candidate.objective_values
                if objective_values[objective_index] <= thresholds[objective_index]:
                    replacement_index = -1
                    for i, archived in enumerate(archive):
                        if archived.exists_in_satisfied(objective_index):
                            replacement_index = i
                            break

                    if replacement_index >= 0:
                        archived = archive[replacement_index]
                        if archived.objective_values[objective_index] > objective_values[objective_index]:
                            candidate.add_objective_covered(objective_index)
                            archive[replacement_index] = candidate
                            if objective_index in objective_uncovered:
                                objective_uncovered.remove(objective_index)
                    else:
                        duplicate = False
                        for archived in archive:
                            if archived.objective_values == objective_values:
                                duplicate = True
                                break
                        if not duplicate:
                            for base in initial_population:
                                if base.objective_values == objective_values:
                                    duplicate = True
                                    break
                        if not duplicate:
                            candidate.add_objective_covered(objective_index)
                            archive.append(candidate)
                            if objective_index in objective_uncovered:
                                objective_uncovered.remove(objective_index)

    def _generate_offspring(
        self,
        population: list[Candidate],
        objective_uncovered: list[int],
    ) -> tuple[list[Candidate], int]:
        offspring: list[Candidate] = []
        mutation_count = 0
        size = self.config.population_size

        while len(offspring) < size:
            probability_crossover = random.uniform(0, 1)
            if probability_crossover <= self.config.crossover_probability:
                parent1, parent2 = crossover(
                    population,
                    self.agent,
                    self._abstract_state,
                    objective_uncovered,
                )
                mutation_rate_1 = self.config.mutation_rate_factor / max(1, len(parent1.episode))
                mutation_rate_2 = self.config.mutation_rate_factor / max(1, len(parent2.episode))
                child1, cnt1 = mutate(parent1, self.agent, self.env, mutation_rate_1)
                child2, cnt2 = mutate(parent2, self.agent, self.env, mutation_rate_2)
                mutation_count += cnt1 + cnt2
                offspring.append(child1)
                if len(offspring) < size:
                    offspring.append(child2)
                continue

            parent = tournament_selection(population, self.config.tournament_size, objective_uncovered)
            mutation_rate = self.config.mutation_rate_factor / max(1, len(parent.episode))
            child, cnt = mutate(parent, self.agent, self.env, mutation_rate)
            mutation_count += cnt
            offspring.append(child)

        return offspring[:size], mutation_count

    @staticmethod
    def _sort_worse(front: list[Candidate]) -> list[Candidate]:
        return sorted(front, key=lambda candidate: candidate.objective_values[0], reverse=True)

    def run(self, initial_population: list[Candidate]) -> SearchResult:
        start = time.perf_counter()
        archive: list[Candidate] = []
        all_generations: list[list[Candidate]] = []
        mutation_count = 0

        objective_uncovered = list(range(self.config.num_objectives))
        population = list(initial_population)
        self._evaluate_population(population)
        self._update_archive(population, objective_uncovered, archive, initial_population)
        all_generations.append(deepcopy(population))

        for generation_index in range(self.config.num_generations):
            elapsed = time.perf_counter() - start
            if self.config.time_budget_seconds is not None and elapsed >= self.config.time_budget_seconds:
                self.logger.info("Stopping at generation %s due to time budget", generation_index)
                break

            self.logger.debug("Generation %s, archive size=%s", generation_index + 1, len(archive))
            offspring, new_mutations = self._generate_offspring(population, objective_uncovered)
            mutation_count += new_mutations

            self._evaluate_population(offspring)
            self._update_archive(offspring, objective_uncovered, archive, initial_population)
            all_generations.append(deepcopy(offspring))

            combined = deepcopy(population)
            combined.extend(offspring)
            fronts_or_candidates = preference_sort(
                combined,
                self.config.population_size,
                objective_uncovered,
            )

            if len(objective_uncovered) == 0:
                self.logger.info("All objectives covered at generation %s", generation_index + 1)
                break

            next_population: list[Candidate] = []
            index = 0
            while len(next_population) <= self.config.population_size and index < len(fronts_or_candidates):
                current = fronts_or_candidates[index]
                if isinstance(current, Candidate):
                    if len(next_population) + 1 > self.config.population_size:
                        break
                    next_population.append(current)
                else:
                    if len(next_population) + len(current) > self.config.population_size:
                        break
                    next_population.extend(current)
                index += 1

            while len(next_population) < self.config.population_size and index < len(fronts_or_candidates):
                current = fronts_or_candidates[index]
                if isinstance(current, Candidate):
                    next_population.append(current)
                    index += 1
                    continue

                sorted_front = self._sort_worse(deepcopy(current))
                crowding = compute_crowding_distance(sorted_front)
                for candidate_index, candidate in enumerate(sorted_front):
                    candidate.crowding_distance = float(crowding[candidate_index])
                sorted_front.sort(key=lambda candidate: candidate.crowding_distance, reverse=True)

                for candidate in sorted_front:
                    next_population.append(candidate)
                    if len(next_population) >= self.config.population_size:
                        break
                index += 1

            population = next_population[: self.config.population_size]

        elapsed_time = time.perf_counter() - start
        return SearchResult(
            archive=archive,
            all_generations=all_generations,
            mutation_count=mutation_count,
            elapsed_time=elapsed_time,
        )

