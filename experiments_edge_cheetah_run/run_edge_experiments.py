"""Run edge-perturbation experiments for DMControl Cheetah-run: baseline + E1(SAM) + E2(Jacobian) + E3(Uncertainty) + E4(Temporal).

Five experiments per seed, each targeting a different edge of the DreamerV3
pipeline.  Produces per-experiment JSON files and a combined vulnerability
profile report.

Usage (inside the CPU dev container)::

    cd /workspaces/RL_Testing
    python -m experiments_edge_cheetah_run.run_edge_experiments --seeds 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, "/workspaces/RL_Testing")
sys.path.insert(0, "/workspaces/RL_Testing/DI-engine")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA/src")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA")

from experiments_edge_cheetah_run.envs.cheetah_pixel_env import (
    _make_dmcontrol_pixel_env,
    preprocess_cheetah_rgb_obs_for_dreamer,
    register_cheetah_env,
)
register_cheetah_env()

from experiments_edge_cheetah_run.train_dreamerv3_cheetah import (
    load_dreamerv3_cheetah_run,
    CHECKPOINT_DIR_CHEETAH_RUN,
)
from experiments_edge_cheetah_run.adapters.sam_mutation_cheetah import SAMGuidedMutatorCheetah
from experiments_edge_cheetah_run.adapters.e2_jacobian_mutation import JacobianGuidedMutator
from experiments_edge_cheetah_run.adapters.e3_uncertainty_mutation import UncertaintyGuidedMutator
from experiments_edge_cheetah_run.adapters.e4_temporal_action import TemporalActionAgent
from starla.adapters.dreamerv3.agent import DreamerV3Agent
from starla.config import StarlaConfig
from starla.core.candidate import Episode
from starla.core.mosa import SearchResult
from starla.faults.base import FaultOracle
from starla.runner import MOSAEngine as RunnerMOSAEngine
from starla.runner import StarlaRunner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RESULTS_DIR = Path("/workspaces/RL_Testing/experiments_edge_cheetah_run/results")
DREAMERV3_CKPT_DIR = Path(
    os.environ.get("DREAMERV3_CHEETAH_RUN_CKPT_DIR", str(CHECKPOINT_DIR_CHEETAH_RUN))
)
ENV_LABEL = "cheetah-run (DMControl)"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    population_size: int
    num_generations: int
    time_budget_seconds: float
    training_episodes: int
    random_episodes: int
    mutation_rate_factor: float
    objective_thresholds: tuple[float, float, float]


# ---------------------------------------------------------------------------
# Custom fault oracle for DMControl Cheetah-run (not a Gymnasium env)
# ---------------------------------------------------------------------------
class CheetahRunFaultOracle(FaultOracle):
    """Fault oracle for DMControl Cheetah-run with reward_threshold=400.0."""

    def __init__(self) -> None:
        self._max_steps = 1000
        self._reward_threshold = 400.0
        self._functional_min_length = max(5, int(0.1 * self._max_steps))

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
            "env_id": "cheetah-run (DMControl)",
            "functional_min_length": self._functional_min_length,
            "reward_threshold": self._reward_threshold,
        }


# ---------------------------------------------------------------------------
# DMControl env adapter for STARLA runner
# ---------------------------------------------------------------------------
class _CheetahRunEnvAdapter:
    """Thin adapter for STARLA's runner to use DMControl Cheetah-run pixel env."""

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed
        self._env: Any = None
        self._last_obs: Any = None
        self._mem: list[tuple[Any, Any]] = []
        self._episode_reward: float = 0.0

    def _ensure_env(self) -> None:
        if self._env is None:
            self._env = _make_dmcontrol_pixel_env("cheetah", "run", seed=self._seed)

    def reset(self) -> np.ndarray:
        self._ensure_env()
        time_step = self._env.reset()
        obs = preprocess_cheetah_rgb_obs_for_dreamer(time_step.observation)
        self._last_obs = deepcopy(obs)
        self._mem = []
        self._episode_reward = 0.0
        return obs

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._ensure_env()
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._last_obs is not None:
            saved_action = action_arr.astype(np.float32).tolist()
            self._mem.append((deepcopy(self._last_obs), saved_action))
        time_step = self._env.step(action_arr)
        obs = preprocess_cheetah_rgb_obs_for_dreamer(time_step.observation)
        rew = float(time_step.reward) if time_step.reward is not None else 0.0
        done = time_step.last()
        self._episode_reward += rew
        if done:
            self._mem.append((deepcopy(obs), -1))
            self._mem.append(("done", self._episode_reward))
        self._last_obs = deepcopy(obs)
        info: dict[str, Any] = {"mem": deepcopy(self._mem)}
        return obs, rew, done, False, info

    def get_state(self) -> Any:
        return deepcopy(self._last_obs)

    def set_state(self, state: Any) -> Any:
        # DMControl has no native state save/restore; reset as fallback
        obs = self.reset()
        return obs

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Cheetah-run edge-perturbation experiments.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds")
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument("--time-budget-seconds", type=float, default=600.0)
    parser.add_argument("--training-episodes", type=int, default=14)
    parser.add_argument("--random-episodes", type=int, default=14)
    parser.add_argument("--mutation-rate-factor", type=float, default=5.0)
    parser.add_argument(
        "--objective-thresholds", type=str, default="400.0,0.8,0.8",
        help="Comma-separated MOSA thresholds (3 values)",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DREAMERV3_CKPT_DIR)
    parser.add_argument("--sam-rho", type=float, default=0.1, help="SAM rho for E1")
    parser.add_argument("--e2-epsilon", type=float, default=0.05)
    parser.add_argument("--e2-top-k", type=int, default=128)
    parser.add_argument("--e3-epsilon", type=float, default=0.05)
    parser.add_argument("--e3-n-samples", type=int, default=5)
    parser.add_argument("--e4-variant", type=str, default="delay", choices=["delay", "freeze", "smooth"])
    parser.add_argument("--e4-delay", type=int, default=1)
    parser.add_argument("--e4-alpha", type=float, default=0.5)
    parser.add_argument("--e4-freeze-prob", type=float, default=0.3)
    return parser.parse_args()


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collect_agent_episodes(
    agent: Any,
    n: int,
    deterministic: bool,
    seed: int,
    min_transitions: int = 6,
) -> list[Episode]:
    """Collect episodes from agent in DMControl Cheetah-run (RGB continuous)."""
    env = _make_dmcontrol_pixel_env("cheetah", "run", seed=seed)
    episodes: list[Episode] = []
    ep_idx = 0
    attempts = 0
    max_attempts = max(20, n * 10)
    while len(episodes) < n and attempts < max_attempts:
        attempts += 1
        time_step = env.reset()
        obs = preprocess_cheetah_rgb_obs_for_dreamer(time_step.observation)
        if hasattr(agent, "reset_state"):
            agent.reset_state()
        total_reward = 0.0
        episode: Episode = []
        while not time_step.last():
            action, _ = agent.predict(obs, deterministic=deterministic)
            action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
            episode.append((obs.copy(), action_arr.astype(np.float32).copy()))
            time_step = env.step(action_arr)
            obs = preprocess_cheetah_rgb_obs_for_dreamer(time_step.observation)
            total_reward += float(time_step.reward) if time_step.reward is not None else 0.0
        episode.append((obs.copy(), -1))
        episode.append(("done", total_reward))
        if len(episode) - 1 >= min_transitions:
            episodes.append(episode)
        ep_idx += 1
    env.close()
    if len(episodes) < n:
        raise RuntimeError(
            f"Unable to collect enough valid Cheetah-run episodes: needed={n}, got={len(episodes)}"
        )
    return episodes


def _collect_random_episodes(n: int, seed: int) -> list[Episode]:
    """Collect random-policy episodes in Cheetah-run."""
    env = _make_dmcontrol_pixel_env("cheetah", "run", seed=seed)
    episodes: list[Episode] = []
    for ep_idx in range(n):
        time_step = env.reset()
        obs = preprocess_cheetah_rgb_obs_for_dreamer(time_step.observation)
        total_reward = 0.0
        episode: Episode = []
        while not time_step.last():
            action_arr = np.random.uniform(-1.0, 1.0, size=(6,)).astype(np.float64)
            episode.append((obs.copy(), action_arr.astype(np.float32).copy()))
            time_step = env.step(action_arr)
            obs = preprocess_cheetah_rgb_obs_for_dreamer(time_step.observation)
            total_reward += float(time_step.reward) if time_step.reward is not None else 0.0
        episode.append((obs.copy(), -1))
        episode.append(("done", total_reward))
        episodes.append(episode)
    env.close()
    return episodes


def _build_config(exp_cfg: ExperimentConfig) -> StarlaConfig:
    return StarlaConfig(
        population_size=exp_cfg.population_size,
        num_generations=exp_cfg.num_generations,
        crossover_probability=0.75,
        mutation_rate_factor=exp_cfg.mutation_rate_factor,
        abstraction_granularity=1.0,
        tournament_size=3,
        num_objectives=3,
        objective_thresholds=list(exp_cfg.objective_thresholds),
        random_seed=exp_cfg.seed,
        time_budget_seconds=exp_cfg.time_budget_seconds,
    )


# ---------------------------------------------------------------------------
# STARLA runner with monkey-patching
# ---------------------------------------------------------------------------
def _run_starla(
    runner: StarlaRunner,
    mutator: Any | None,
) -> tuple[Any, SearchResult]:
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

    RunnerMOSAEngine.run = _wrapped_run  # type: ignore[assignment]
    mosa_module.mutate = _safe_mutate  # type: ignore[assignment]
    if mutator is not None:
        genetic_module.transform = mutator.transform  # type: ignore[assignment]

    try:
        report = runner.run()
    finally:
        genetic_module.transform = original_transform
        RunnerMOSAEngine.run = original_run  # type: ignore[assignment]
        mosa_module.mutate = original_mutate  # type: ignore[assignment]

    if "result" not in captured:
        raise RuntimeError("Failed to capture STARLA search result.")
    return report, captured["result"]


def _count_functional_faults(
    episodes: list[Episode],
    fault_oracle: FaultOracle,
) -> int:
    return sum(1 for ep in episodes if fault_oracle.is_functional_fault(ep))


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------
def _run_single_experiment(
    *,
    experiment_id: str,
    mode: str,
    agent: Any,
    fault_oracle: FaultOracle,
    exp_cfg: ExperimentConfig,
    mutator: Any | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    env_adapter = _CheetahRunEnvAdapter(seed=exp_cfg.seed)

    training_episodes = _collect_agent_episodes(
        agent, exp_cfg.training_episodes, deterministic=True, seed=exp_cfg.seed,
    )
    random_seed_episodes = _collect_agent_episodes(
        agent, exp_cfg.random_episodes, deterministic=False, seed=exp_cfg.seed + 10_000,
    )

    runner = StarlaRunner(
        config=_build_config(exp_cfg), agent=agent, env=env_adapter, fault_oracle=fault_oracle,
    )
    runner.prepare_data(training_episodes, random_seed_episodes)
    report, raw_result = _run_starla(runner, mutator=mutator)

    archive_episodes = [c.episode for c in report.archive]
    all_search_episodes: list[Episode] = []
    for gen in report.generations:
        for c in gen:
            all_search_episodes.append(c.episode)

    total_searched = len(all_search_episodes)
    search_func_faults = _count_functional_faults(all_search_episodes, fault_oracle)
    search_reward_faults = sum(1 for ep in all_search_episodes if fault_oracle.is_reward_fault(ep))
    search_func_fault_rate = search_func_faults / max(1, total_searched)
    search_reward_fault_rate = search_reward_faults / max(1, total_searched)

    total_budget_episodes = int(raw_result.mutation_count + len(report.archive))
    random_eval_episodes = _collect_random_episodes(
        n=max(total_budget_episodes, total_searched), seed=exp_cfg.seed + 20_000,
    )
    starla_functional_faults = _count_functional_faults(archive_episodes, fault_oracle)
    random_functional_faults = _count_functional_faults(random_eval_episodes, fault_oracle)
    random_reward_faults = sum(1 for ep in random_eval_episodes if fault_oracle.is_reward_fault(ep))
    random_func_fault_rate = random_functional_faults / max(1, len(random_eval_episodes))
    random_reward_fault_rate = random_reward_faults / max(1, len(random_eval_episodes))

    result = {
        "experiment_id": experiment_id,
        "mode": mode,
        "env": ENV_LABEL,
        "agent": "DreamerV3",
        "seed": exp_cfg.seed,
        "total_budget_episodes": total_budget_episodes,
        "starla_functional_faults": int(starla_functional_faults),
        "starla_reward_faults": int(report.found_faults["reward_faults"]),
        "starla_archive_size": int(len(report.archive)),
        "search_total_episodes": total_searched,
        "search_func_faults": search_func_faults,
        "search_reward_faults": search_reward_faults,
        "search_func_fault_rate": round(search_func_fault_rate, 4),
        "search_reward_fault_rate": round(search_reward_fault_rate, 4),
        "random_functional_faults": int(random_functional_faults),
        "random_reward_faults": int(random_reward_faults),
        "random_total_episodes": int(len(random_eval_episodes)),
        "random_func_fault_rate": round(random_func_fault_rate, 4),
        "random_reward_fault_rate": round(random_reward_fault_rate, 4),
        "num_generations": exp_cfg.num_generations,
        "time_seconds": float(time.perf_counter() - start_time),
    }
    if extra_metrics:
        result.update(extra_metrics)
    return result


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------
def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------
def _build_comparison_report(
    results: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    edge_names = {
        "exp0": "baseline",
        "exp1": "E1_perception",
        "exp2": "E2_encoding",
        "exp3": "E3_prediction",
        "exp4": "E4_execution",
    }
    edges: dict[str, Any] = {}
    for key, label in edge_names.items():
        r = results[key]
        edges[label] = {
            "search_reward_fault_rate": r["search_reward_fault_rate"],
            "search_func_fault_rate": r["search_func_fault_rate"],
            "starla_reward_faults": r["starla_reward_faults"],
            "time_seconds": r["time_seconds"],
        }

    profile: dict[str, float] = {}
    for key, label in edge_names.items():
        if key == "exp0":
            continue
        profile[f"{label}_fault_rate"] = results[key]["search_reward_fault_rate"]

    sorted_edges = sorted(profile.items(), key=lambda kv: kv[1], reverse=True)
    weakest = sorted_edges[0][0].replace("_fault_rate", "") if sorted_edges else "unknown"
    strongest = sorted_edges[-1][0].replace("_fault_rate", "") if sorted_edges else "unknown"

    return {
        "experiment": "edge_perturbation_comparison",
        "env": ENV_LABEL,
        "seed": seed,
        "edges": edges,
        "vulnerability_profile": profile,
        "weakest_edge": weakest,
        "strongest_edge": strongest,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("seeds is empty.")

    obj_thresh = tuple(float(v) for v in args.objective_thresholds.split(","))
    if len(obj_thresh) != 3:
        raise ValueError("--objective-thresholds must have exactly 3 comma-separated values.")

    fault_oracle = CheetahRunFaultOracle()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EdgeExp-CheetahRun] env={ENV_LABEL}, seeds={seeds}")
    print(f"[EdgeExp-CheetahRun] results_dir={results_dir}")

    for seed in seeds:
        exp_cfg = ExperimentConfig(
            seed=seed,
            population_size=args.population_size,
            num_generations=args.num_generations,
            time_budget_seconds=args.time_budget_seconds,
            training_episodes=args.training_episodes,
            random_episodes=args.random_episodes,
            mutation_rate_factor=args.mutation_rate_factor,
            objective_thresholds=obj_thresh,
        )

        _set_global_seed(seed)
        print(f"\n[EdgeExp-CheetahRun][seed={seed}] Loading DreamerV3 checkpoint ...")
        try:
            policy, world_model = load_dreamerv3_cheetah_run(
                checkpoint_dir=args.checkpoint_dir, seed=seed,
            )
        except Exception as err:
            raise RuntimeError(
                "Failed to load Cheetah-run DreamerV3 checkpoint. "
                "Ensure checkpoints exist at the specified directory."
            ) from err

        if hasattr(world_model, "heads") and not hasattr(world_model.heads, "get"):
            def _heads_get(key: str, default: Any = None) -> Any:
                return world_model.heads[key] if key in world_model.heads else default
            setattr(world_model.heads, "get", _heads_get)

        all_results: dict[str, dict[str, Any]] = {}

        # ---------------------------------------------------------------
        # Exp 0: Baseline (starla_only)
        # ---------------------------------------------------------------
        print(f"[EdgeExp-CheetahRun][seed={seed}] Running exp0_baseline ...")
        _set_global_seed(seed)
        agent_baseline = DreamerV3Agent(policy=policy, world_model=world_model)
        all_results["exp0"] = _run_single_experiment(
            experiment_id="exp0_baseline",
            mode="starla_only",
            agent=agent_baseline,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
        )
        _dump_json(results_dir / f"exp0_baseline_seed_{seed}.json", all_results["exp0"])
        print(f"[EdgeExp-CheetahRun][seed={seed}] exp0_baseline done: "
              f"reward_fault_rate={all_results['exp0']['search_reward_fault_rate']}")

        # ---------------------------------------------------------------
        # Exp 1: E1 SAM (perception channel)
        # ---------------------------------------------------------------
        print(f"[EdgeExp-CheetahRun][seed={seed}] Running exp1_e1_sam ...")
        _set_global_seed(seed)
        agent_e1 = DreamerV3Agent(policy=policy, world_model=world_model)
        sam_mutator = SAMGuidedMutatorCheetah(
            world_model=world_model,
            actor=agent_e1.actor,
            critic=agent_e1.value_head,
            rho=args.sam_rho,
            device=str(agent_e1.device),
        )
        all_results["exp1"] = _run_single_experiment(
            experiment_id="exp1_e1_sam",
            mode="sam",
            agent=agent_e1,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
            mutator=sam_mutator,
            extra_metrics={
                "sam_rho": args.sam_rho,
                "sam_fallback_count": 0,
            },
        )
        all_results["exp1"]["sam_fallback_count"] = sam_mutator.fallback_count
        _dump_json(results_dir / f"exp1_e1_sam_seed_{seed}.json", all_results["exp1"])
        print(f"[EdgeExp-CheetahRun][seed={seed}] exp1_e1_sam done: "
              f"reward_fault_rate={all_results['exp1']['search_reward_fault_rate']}, "
              f"fallbacks={sam_mutator.fallback_count}")

        # ---------------------------------------------------------------
        # Exp 2: E2 Jacobian (encoding channel)
        # ---------------------------------------------------------------
        print(f"[EdgeExp-CheetahRun][seed={seed}] Running exp2_e2_jacobian ...")
        _set_global_seed(seed)
        agent_e2 = DreamerV3Agent(policy=policy, world_model=world_model)
        jacobian_mutator = JacobianGuidedMutator(
            world_model=world_model,
            epsilon=args.e2_epsilon,
            top_k=args.e2_top_k,
            device=str(agent_e2.device),
        )
        all_results["exp2"] = _run_single_experiment(
            experiment_id="exp2_e2_jacobian",
            mode="jacobian",
            agent=agent_e2,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
            mutator=jacobian_mutator,
            extra_metrics={
                "e2_epsilon": args.e2_epsilon,
                "e2_top_k": args.e2_top_k,
                "jacobian_fallback_count": 0,
                "avg_jacobian_sparsity": 0.0,
            },
        )
        all_results["exp2"]["jacobian_fallback_count"] = jacobian_mutator.fallback_count
        all_results["exp2"]["avg_jacobian_sparsity"] = round(jacobian_mutator.avg_jacobian_sparsity, 4)
        _dump_json(results_dir / f"exp2_e2_jacobian_seed_{seed}.json", all_results["exp2"])
        print(f"[EdgeExp-CheetahRun][seed={seed}] exp2_e2_jacobian done: "
              f"reward_fault_rate={all_results['exp2']['search_reward_fault_rate']}, "
              f"fallbacks={jacobian_mutator.fallback_count}, "
              f"sparsity={jacobian_mutator.avg_jacobian_sparsity:.4f}")

        # ---------------------------------------------------------------
        # Exp 3: E3 Uncertainty (prediction channel)
        # ---------------------------------------------------------------
        print(f"[EdgeExp-CheetahRun][seed={seed}] Running exp3_e3_uncertainty ...")
        _set_global_seed(seed)
        agent_e3 = DreamerV3Agent(policy=policy, world_model=world_model)
        uncertainty_mutator = UncertaintyGuidedMutator(
            world_model=world_model,
            epsilon=args.e3_epsilon,
            n_samples=args.e3_n_samples,
            device=str(agent_e3.device),
        )
        all_results["exp3"] = _run_single_experiment(
            experiment_id="exp3_e3_uncertainty",
            mode="uncertainty",
            agent=agent_e3,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
            mutator=uncertainty_mutator,
            extra_metrics={
                "e3_epsilon": args.e3_epsilon,
                "e3_n_samples": args.e3_n_samples,
                "uncertainty_fallback_count": 0,
                "avg_uncertainty_magnitude": 0.0,
            },
        )
        all_results["exp3"]["uncertainty_fallback_count"] = uncertainty_mutator.fallback_count
        all_results["exp3"]["avg_uncertainty_magnitude"] = round(uncertainty_mutator.avg_uncertainty_magnitude, 6)
        _dump_json(results_dir / f"exp3_e3_uncertainty_seed_{seed}.json", all_results["exp3"])
        print(f"[EdgeExp-CheetahRun][seed={seed}] exp3_e3_uncertainty done: "
              f"reward_fault_rate={all_results['exp3']['search_reward_fault_rate']}, "
              f"fallbacks={uncertainty_mutator.fallback_count}, "
              f"magnitude={uncertainty_mutator.avg_uncertainty_magnitude:.6f}")

        # ---------------------------------------------------------------
        # Exp 4: E4 Temporal (execution channel)
        # ---------------------------------------------------------------
        print(f"[EdgeExp-CheetahRun][seed={seed}] Running exp4_e4_temporal ...")
        _set_global_seed(seed)
        agent_e4_inner = DreamerV3Agent(policy=policy, world_model=world_model)
        wrapped_agent = TemporalActionAgent(
            agent_e4_inner,
            variant=args.e4_variant,
            delay=args.e4_delay,
            alpha=args.e4_alpha,
            freeze_prob=args.e4_freeze_prob,
        )
        all_results["exp4"] = _run_single_experiment(
            experiment_id="exp4_e4_temporal",
            mode="temporal",
            agent=wrapped_agent,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
            mutator=None,
            extra_metrics={
                "e4_variant": args.e4_variant,
                "e4_delay": args.e4_delay,
                "e4_alpha": args.e4_alpha,
                "e4_freeze_prob": args.e4_freeze_prob,
                "action_perturbation_count": 0,
            },
        )
        all_results["exp4"]["action_perturbation_count"] = wrapped_agent.perturbation_count
        _dump_json(results_dir / f"exp4_e4_temporal_seed_{seed}.json", all_results["exp4"])
        print(f"[EdgeExp-CheetahRun][seed={seed}] exp4_e4_temporal done: "
              f"reward_fault_rate={all_results['exp4']['search_reward_fault_rate']}, "
              f"perturbations={wrapped_agent.perturbation_count}")

        # ---------------------------------------------------------------
        # Comparison report
        # ---------------------------------------------------------------
        comparison = _build_comparison_report(all_results, seed)
        _dump_json(results_dir / f"edge_comparison_report_seed_{seed}.json", comparison)
        print(f"\n[EdgeExp-CheetahRun][seed={seed}] === Vulnerability Profile ===")
        for edge_label, rate in comparison["vulnerability_profile"].items():
            print(f"  {edge_label}: {rate}")
        print(f"  Weakest edge:  {comparison['weakest_edge']}")
        print(f"  Strongest edge: {comparison['strongest_edge']}")
        print(f"[EdgeExp-CheetahRun][seed={seed}] Reports saved to {results_dir}")

    print("\n[EdgeExp-CheetahRun] All seeds complete.")


if __name__ == "__main__":
    main()
