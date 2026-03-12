"""Run Path B experiment: DreamerV3 + STARLA with SAM-guided mutation."""

from __future__ import annotations

import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "/workspaces/RL_Testing")
sys.path.insert(0, "/workspaces/RL_Testing/DI-engine")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA/src")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA")

from Examples.cartpole.fault_oracle import CartPoleFaultOracle
from experiments.dreamerv3_rq1.train_dreamerv3 import META_PATH, load_dreamerv3
from experiments_sam.adapters.sam_mutation import SAMGuidedMutator
from starla.adapters.dreamerv3.agent import DreamerV3Agent
from starla.config import StarlaConfig
from starla.core.candidate import Episode
from starla.core.mosa import SearchResult
from starla.envs.gymnasium_adapter import GymnasiumEnv
from starla.faults.base import FaultOracle
from starla.runner import MOSAEngine as RunnerMOSAEngine
from starla.runner import StarlaRunner

SEED = 42
POPULATION_SIZE = 12
NUM_GENERATIONS = 8
TIME_BUDGET_SECONDS = 180.0
TRAINING_EPISODES = 10
RANDOM_EPISODES = 10
MUTATION_RATE_FACTOR = 5.0
SAM_RHO = 0.1
RESULTS_PATH = Path("/workspaces/RL_Testing/experiments_sam/results/results.json")


class GenericGymFaultOracle(FaultOracle):
    """Fallback oracle for non-CartPole gym environments."""

    def __init__(self, env_id: str):
        self.env_id = env_id
        probe_env = gym.make(env_id)
        self._max_steps = int(getattr(probe_env.spec, "max_episode_steps", 200) or 200)
        reward_threshold = getattr(probe_env.spec, "reward_threshold", None)
        probe_env.close()

        self._functional_min_length = max(5, int(0.1 * self._max_steps))
        if reward_threshold is None:
            self._reward_threshold = float(0.3 * self._max_steps)
        else:
            self._reward_threshold = float(0.5 * reward_threshold)

    def is_functional_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        if not episode:
            return True
        transition_count = max(0, len(episode) - 1)
        if transition_count < self._functional_min_length:
            return True
        for item in episode[:-1]:
            obs = item[0]
            if isinstance(obs, str):
                continue
            obs_arr = np.asarray(obs, dtype=float)
            if not np.isfinite(obs_arr).all():
                return True
        return False

    def is_reward_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        if not episode:
            return True
        terminal = episode[-1]
        if terminal[0] != "done":
            return True
        return float(terminal[1]) < self._reward_threshold

    def get_fault_thresholds(self) -> dict[str, Any]:
        return {
            "env_id": self.env_id,
            "functional_min_length": self._functional_min_length,
            "reward_threshold": self._reward_threshold,
        }


def _select_env_and_oracle() -> tuple[str, FaultOracle]:
    env_id = "CartPole-v1"
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
            backend = str(meta.get("backend", "")).lower()
            if backend == "minigrid":
                env_id = "MiniGrid-Empty-8x8-v0"
            elif backend == "dmc2gym":
                env_id = "dmc2gym_cartpole_balance"
        except Exception:
            env_id = "CartPole-v1"

    if "cartpole" in env_id.lower():
        return env_id, CartPoleFaultOracle()
    return env_id, GenericGymFaultOracle(env_id)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _normalize_action(action: Any) -> Any:
    arr = np.asarray(action)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    return arr


def _collect_agent_episodes(
    agent: DreamerV3Agent,
    env_id: str,
    n: int,
    deterministic: bool,
    seed: int,
    min_transitions: int = 6,
) -> list[Episode]:
    env = GymnasiumEnv(env_id)
    episodes: list[Episode] = []
    ep_idx = 0
    attempts = 0
    max_attempts = max(20, n * 10)
    while len(episodes) < n and attempts < max_attempts:
        attempts += 1
        env.env.reset(seed=seed + ep_idx)
        obs = env.reset()
        agent.reset_state()
        done = False
        total_reward = 0.0
        episode: Episode = []
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            env_action = _normalize_action(action)
            episode.append((deepcopy(obs), env_action))
            obs, reward, terminated, truncated, _ = env.step(env_action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        episode.append(("done", total_reward))
        if len(episode) - 1 >= min_transitions:
            episodes.append(episode)
        ep_idx += 1
    env.env.close()
    if len(episodes) < n:
        raise RuntimeError(
            f"Unable to collect enough valid episodes for STARLA. "
            f"needed={n}, got={len(episodes)}, min_transitions={min_transitions}"
        )
    return episodes


def _collect_random_episodes(env_id: str, n: int, seed: int) -> list[Episode]:
    env = gym.make(env_id)
    episodes: list[Episode] = []
    for ep_idx in range(n):
        obs, _ = env.reset(seed=seed + ep_idx)
        done = False
        total_reward = 0.0
        episode: Episode = []
        while not done:
            action = env.action_space.sample()
            saved_action = int(action) if np.asarray(action).size == 1 else np.asarray(action).tolist()
            episode.append((deepcopy(obs), saved_action))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
        episode.append(("done", total_reward))
        episodes.append(episode)
    env.close()
    return episodes


def _run_starla_with_sam_mutation(runner: StarlaRunner, sam_mutator: SAMGuidedMutator) -> tuple[Any, SearchResult]:
    import starla.core.genetic as genetic_module
    import starla.core.mosa as mosa_module

    captured: dict[str, SearchResult] = {}
    original_transform = genetic_module.transform
    original_run = RunnerMOSAEngine.run
    original_mutate = mosa_module.mutate

    def _wrapped_run(self: RunnerMOSAEngine, initial_population: list[Any]) -> SearchResult:
        result = original_run(self, initial_population)
        captured["result"] = result
        return result

    def _safe_mutate(parent: Any, agent: Any, env: Any, mutation_rate: float):
        episode = getattr(parent, "episode", [])
        if len(episode) < 7:
            return parent, 0
        return original_mutate(parent, agent, env, mutation_rate)

    genetic_module.transform = sam_mutator.transform
    RunnerMOSAEngine.run = _wrapped_run
    mosa_module.mutate = _safe_mutate
    try:
        report = runner.run()
    finally:
        genetic_module.transform = original_transform
        RunnerMOSAEngine.run = original_run
        mosa_module.mutate = original_mutate

    if "result" not in captured:
        raise RuntimeError("Failed to capture STARLA search result.")
    return report, captured["result"]


def main() -> None:
    start_time = time.perf_counter()
    _set_global_seed(SEED)
    env_id, fault_oracle = _select_env_and_oracle()
    print(f"[Path B] env={env_id}, oracle={fault_oracle.__class__.__name__}")

    try:
        policy, world_model = load_dreamerv3(seed=SEED)
    except Exception as err:
        raise RuntimeError(
            "Failed to load DreamerV3 checkpoint. "
            "Please ensure checkpoints exist and dependencies are installed. "
            "Try: pip install transformers tensorboardX easydict gym minigrid dmc2gym dm-control"
        ) from err

    # DI-engine world_model.heads can be ModuleDict-like without .get in some setups.
    if hasattr(world_model, "heads") and not hasattr(world_model.heads, "get"):
        def _heads_get(key: str, default: Any = None) -> Any:
            return world_model.heads[key] if key in world_model.heads else default
        setattr(world_model.heads, "get", _heads_get)

    agent = DreamerV3Agent(policy=policy, world_model=world_model)
    sam_mutator = SAMGuidedMutator(
        world_model=world_model,
        actor=agent.actor,
        critic=agent.value_head,
        rho=SAM_RHO,
        device=str(agent.device),
    )
    env_adapter = GymnasiumEnv(env_id)

    training_episodes = _collect_agent_episodes(agent, env_id, TRAINING_EPISODES, deterministic=True, seed=SEED)
    random_seed_episodes = _collect_agent_episodes(
        agent, env_id, RANDOM_EPISODES, deterministic=False, seed=SEED + 10_000
    )

    config = StarlaConfig(
        population_size=POPULATION_SIZE,
        num_generations=NUM_GENERATIONS,
        crossover_probability=0.75,
        mutation_rate_factor=MUTATION_RATE_FACTOR,
        abstraction_granularity=1.0,
        tournament_size=3,
        num_objectives=3,
        objective_thresholds=[70.0, 0.06, 0.05],
        random_seed=SEED,
        time_budget_seconds=TIME_BUDGET_SECONDS,
    )

    runner = StarlaRunner(config=config, agent=agent, env=env_adapter, fault_oracle=fault_oracle)
    runner.prepare_data(training_episodes, random_seed_episodes)
    report, raw_result = _run_starla_with_sam_mutation(runner, sam_mutator)

    total_budget_episodes = int(raw_result.mutation_count + len(report.archive))
    random_eval_episodes = _collect_random_episodes(env_id, n=total_budget_episodes, seed=SEED + 20_000)
    random_functional_faults = sum(1 for ep in random_eval_episodes if fault_oracle.is_functional_fault(ep))
    random_reward_faults = sum(1 for ep in random_eval_episodes if fault_oracle.is_reward_fault(ep))

    output = {
        "experiment": "Path B - SAM Sharpness-Guided Mutation",
        "sam_rho": SAM_RHO,
        "sam_fallback_count": int(sam_mutator.fallback_count),
        "env": env_id,
        "agent": "DreamerV3",
        "seed": SEED,
        "total_budget_episodes": total_budget_episodes,
        "starla_functional_faults": int(report.found_faults["functional_faults"]),
        "starla_reward_faults": int(report.found_faults["reward_faults"]),
        "starla_archive_size": int(len(report.archive)),
        "random_functional_faults": int(random_functional_faults),
        "random_reward_faults": int(random_reward_faults),
        "random_total_episodes": int(len(random_eval_episodes)),
        "num_generations": NUM_GENERATIONS,
        "time_seconds": float(time.perf_counter() - start_time),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[Path B] results saved to: {RESULTS_PATH}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
