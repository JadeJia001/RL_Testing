"""Core genetic operators extracted from the original STARLA notebook."""

from __future__ import annotations

from copy import deepcopy
import random
from typing import Any, Callable

import numpy as np

from starla.agents.base import AgentProtocol
from starla.core.candidate import Candidate, Episode
from starla.envs.base import EnvProtocol


def transform(state: Any, noise_low: float = 0.95, noise_high: float = 1.05) -> Any:
    """Apply multiplicative noise to the first state dimension."""
    position = state[0]
    noise = np.random.uniform(low=noise_low, high=noise_high)
    new_position = position * noise
    new_state = deepcopy(state)
    new_state[0] = new_position
    return new_state


def _normalize_action(action: Any) -> Any:
    """Discrete: single scalar -> int. Continuous: vector -> ndarray for env.step."""
    arr = np.asarray(action)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    return arr


def _step_env(env: EnvProtocol, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = bool(terminated or truncated)
        return obs, float(reward), done, info
    if len(step_result) == 4:
        obs, reward, done, info = step_result  # legacy envs in notebook
        return obs, float(reward), bool(done), info
    raise ValueError("Unsupported env.step return signature.")


def mutate(
    parent: Candidate,
    agent: AgentProtocol,
    env: EnvProtocol,
    mutation_rate: float,
    *,
    reexecute_rollout_limit: int = 2000,
) -> tuple[Candidate, int]:
    """
    Mutate one parent with probability `mutation_rate`.

    Returns `(candidate, mutation_count)` where `mutation_count` is 0 or 1.
    """
    chance = random.uniform(0, 1)
    if chance > mutation_rate:
        return parent, 0

    parent_episode = deepcopy(parent.episode)
    if len(parent_episode) < 3:
        raise AssertionError("parent in mutation is shorter than 3")

    mutpoint = random.randint(3, (len(parent_episode) - 3))
    new_state = transform(parent_episode[mutpoint][0])
    predicted_action, _ = agent.predict(new_state, deterministic=True)
    pred = _normalize_action(predicted_action)
    orig = parent_episode[mutpoint][1]
    try:
        lured = pred != int(orig)
    except (TypeError, ValueError):
        lured = not np.array_equal(np.asarray(pred), np.asarray(orig))
    if lured:
        print("Mutation lured the agent ... ")

    new_parent_episode = parent_episode[:mutpoint]
    new_parent_episode.append((new_state, "Mut"))
    new_candidate = Candidate(new_parent_episode)
    new_candidate.start_state = parent.start_state

    re_executed_episode = re_execute(
        agent,
        env,
        new_candidate,
        max_followup_steps=reexecute_rollout_limit,
    )
    re_executed_candidate = Candidate(re_executed_episode)
    re_executed_candidate.start_state = new_candidate.start_state
    re_executed_candidate.information.extend(deepcopy(parent.information))
    re_executed_candidate.add_info(["mutation is done! ", "mutpoint was:", mutpoint])
    re_executed_candidate.mutation = True
    return re_executed_candidate, 1


def _dominates(
    value_from_pop: list[float],
    value_from_archive: list[float],
    objective_uncovered: list[int],
) -> bool:
    dominates_f1 = False
    dominates_f2 = False
    for each_objective in objective_uncovered:
        f1 = value_from_pop[each_objective]
        f2 = value_from_archive[each_objective]
        if f1 < f2:
            dominates_f1 = True
        if f2 < f1:
            dominates_f2 = True
        if dominates_f1 and dominates_f2:
            break
    if dominates_f1 == dominates_f2:
        return False
    if dominates_f1:
        return True
    return False


def _select_best(
    tournament_candidates: list[Candidate], objective_uncovered: list[int]
) -> Candidate:
    best = tournament_candidates[0]
    for candidate1 in tournament_candidates:
        for candidate2 in tournament_candidates:
            if _dominates(
                candidate1.objective_values,
                candidate2.objective_values,
                objective_uncovered,
            ):
                best = candidate1
    return best


def _tournament_selection(
    population: list[Candidate], size: int, objective_uncovered: list[int]
) -> Candidate:
    tournament_candidates: list[Candidate] = []
    for _ in range(size):
        index = random.randint(0, len(population) - 1)
        tournament_candidates.append(population[index])
    return _select_best(tournament_candidates, objective_uncovered)


def crossover(
    population: list[Candidate],
    agent: AgentProtocol,
    abstraction_fn: Callable[[Any], Any],
    objective_uncovered: list[int],
) -> tuple[Candidate, Candidate]:
    """Crossover adapted from `Crossover_improved_v2` with injected abstraction function."""
    found_match = False
    while not found_match:
        parent = _tournament_selection(population, 10, objective_uncovered)
        parent1 = deepcopy(parent.episode)
        parent1_start_point = deepcopy(parent.start_state)
        if len(parent1) < 4:
            raise AssertionError("input of crossover is shorter than expected")

        matches_list: list[int] = []
        crosspoint = random.randint(1, (len(parent1) - 3))
        abs_class = list(np.atleast_1d(abstraction_fn(parent1[crosspoint][0])))

        for _ in range(50):
            index = random.randint(0, len(population) - 1)
            random_candidate = deepcopy(population[index])
            random_candidate_data = random_candidate.episode
            random_candidate_start = random_candidate.start_state
            for state_index in range(1, len(random_candidate_data) - 3):
                random_ab = list(
                    np.atleast_1d(abstraction_fn(random_candidate_data[state_index][0]))
                )
                if random_ab == abs_class:
                    matches_list.append(state_index)
                    found_match = True
            if found_match:
                break

    index_match = random.randint(0, len(matches_list) - 1)
    matchpoint = matches_list[index_match]
    match_candidate = deepcopy(random_candidate)
    match = deepcopy(random_candidate_data)
    match_start = deepcopy(random_candidate_start)

    offspring1: Episode = deepcopy(parent1[:crosspoint])
    offspring1 += deepcopy(match[matchpoint:])
    offspring1[-1] = ("done", (len(offspring1) - 1))
    candidate1 = Candidate(offspring1)
    candidate1.start_state = parent1_start_point
    candidate1.information.extend(deepcopy(parent.information))
    candidate1.add_info(["crossover is Done!", "the crossover point is:", crosspoint])

    offspring2: Episode = deepcopy(match[:matchpoint])
    offspring2 += deepcopy(parent1[crosspoint:])
    offspring2[-1] = ("done", (len(offspring2) - 1))
    candidate2 = Candidate(offspring2)
    candidate2.start_state = match_start
    candidate2.information.extend(deepcopy(match_candidate.information))
    candidate2.add_info(["crossover is Done!", "the crossover point is:", matchpoint])

    if len(offspring1) < 4:
        print(offspring1)
        raise AssertionError(
            "created offspring 1 in crossover is shorter than expected"
        )
    if len(offspring2) < 4:
        print(offspring2)
        raise AssertionError(
            "created offspring 2 in crossover is shorter than expected"
        )

    return candidate1, candidate2


def re_execute(
    agent: AgentProtocol,
    env: EnvProtocol,
    candidate: Candidate,
    *,
    max_followup_steps: int = 2000,
) -> Episode:
    """
    Re-execute candidate trajectory from its saved start state.

    `max_followup_steps` caps rollout after replaying the candidate prefix (default 2000).
    """
    env.reset()
    obs = env.set_state(deepcopy(candidate.start_state))
    episode = candidate.episode
    steps_to_mut_point = len(episode)
    episode_reward = 0.0
    done = False
    info: dict[str, Any] = {}

    for i in range(steps_to_mut_point):
        _, _ = agent.predict(obs, deterministic=True)
        action_selected = episode[i][1]
        if isinstance(action_selected, str) and action_selected == "Mut":
            action_selected, _ = agent.predict(episode[i][0], deterministic=True)
        obs, reward, done, info = _step_env(env, _normalize_action(action_selected))
        episode_reward += reward
        if done:
            break

    for _ in range(max_followup_steps):
        if done:
            break
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, info = _step_env(env, _normalize_action(action))
        episode_reward += reward

    if not done:
        # Episode did not terminate within max_followup_steps;
        # build a synthetic terminal from the last observation.
        if info and "mem" in info:
            mem = info["mem"]
            mem.append(("done", episode_reward))
            return mem
        raise AssertionError("Episode did not terminate during re-execution")

    if "mem" not in info:
        raise KeyError("Expected 'mem' in env info during re_execute")
    return info["mem"]
