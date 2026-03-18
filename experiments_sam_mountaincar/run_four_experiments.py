"""Run four STARLA/SAM experiments with rho sweep and feasibility report."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, "/workspaces/RL_Testing")
sys.path.insert(0, "/workspaces/RL_Testing/DI-engine")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA/src")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA")

from experiments.dreamerv3_rq1.train_dreamerv3 import (
    CHECKPOINT_DIR_MOUNTAINCAR,
    load_dreamerv3,
)
from experiments_sam_mountaincar.adapters.sam_mutation import SAMGuidedMutator
from starla.adapters.dreamerv3.agent import DreamerV3Agent
from starla.config import StarlaConfig
from starla.core.candidate import Episode
from starla.core.mosa import SearchResult
from starla.envs.gymnasium_adapter import GymnasiumEnv
from starla.faults.base import FaultOracle
from starla.runner import MOSAEngine as RunnerMOSAEngine
from starla.runner import StarlaRunner


DEFAULT_RESULTS_DIR = Path("/workspaces/RL_Testing/experiments_sam_mountaincar/results")
DEFAULT_RHO_CANDIDATES = [0.01, 0.03, 0.05, 0.1, 0.2]

# Mountain Car checkpoint dir; override via env DREAMERV3_MOUNTAINCAR_CKPT_DIR
DREAMERV3_CKPT_DIR = Path(
    os.environ.get("DREAMERV3_MOUNTAINCAR_CKPT_DIR", str(CHECKPOINT_DIR_MOUNTAINCAR))
)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STARLA/SAM four-experiment feasibility suite.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated seeds for multi-seed aggregation, e.g. 42,123,456",
    )
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--num-generations", type=int, default=20)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=300.0,
        help="Per-run STARLA time cap (seconds). 9 runs/seed by design.",
    )
    parser.add_argument("--training-episodes", type=int, default=14)
    parser.add_argument("--random-episodes", type=int, default=14)
    parser.add_argument("--mutation-rate-factor", type=float, default=5.0)
    parser.add_argument(
        "--rho-candidates",
        type=str,
        default=",".join(str(v) for v in DEFAULT_RHO_CANDIDATES),
        help="Comma-separated SAM rho values used for sweep, e.g. 0.01,0.03,0.05,0.1,0.2",
    )
    parser.add_argument(
        "--objective-thresholds",
        type=str,
        default="180.0,0.8,0.8",
        help=(
            "Comma-separated MOSA archive thresholds for the 3 objectives: "
            "reward_fitness (episode length-1), confidence_margin, ml_nonfault_prob. "
            "For Mountain Car: reward -1/step, max 200 steps; reward_fitness=180 ~ reward -180. "
            "Lower = stricter. E.g. '180.0,0.8,0.8'"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Output directory for per-experiment and feasibility report JSON files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DREAMERV3_CKPT_DIR,
        help="DreamerV3 Mountain Car checkpoint directory. Override via DREAMERV3_MOUNTAINCAR_CKPT_DIR env.",
    )
    return parser.parse_args()


def _select_env_and_oracle() -> tuple[str, FaultOracle]:
    env_id = "MountainCar-v0"
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
        episode.append((deepcopy(obs), -1))
        episode.append(("done", total_reward))
        if len(episode) - 1 >= min_transitions:
            episodes.append(episode)
        ep_idx += 1
    env.env.close()
    if len(episodes) < n:
        raise RuntimeError(
            "Unable to collect enough valid episodes for STARLA. "
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
        episode.append((deepcopy(obs), -1))
        episode.append(("done", total_reward))
        episodes.append(episode)
    env.close()
    return episodes


def _run_starla(
    runner: StarlaRunner,
    sam_mutator: SAMGuidedMutator | None,
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

    RunnerMOSAEngine.run = _wrapped_run
    mosa_module.mutate = _safe_mutate
    if sam_mutator is not None:
        genetic_module.transform = sam_mutator.transform

    try:
        report = runner.run()
    finally:
        genetic_module.transform = original_transform
        RunnerMOSAEngine.run = original_run
        mosa_module.mutate = original_mutate

    if "result" not in captured:
        raise RuntimeError("Failed to capture STARLA search result.")
    return report, captured["result"]


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


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    total_faults = float(result["starla_functional_faults"] + result["starla_reward_faults"])
    reward_faults = float(result["starla_reward_faults"])
    fallback_penalty = -float(result.get("sam_fallback_count", 0))
    time_penalty = -float(result["time_seconds"])
    return (total_faults, reward_faults, fallback_penalty, time_penalty)


def _count_functional_faults(
    episodes: list[Episode],
    fault_oracle: FaultOracle,
    *,
    mode: str,
) -> int:
    if mode == "default":
        return sum(1 for episode in episodes if fault_oracle.is_functional_fault(episode))
    if mode == "legacy":
        checker = getattr(fault_oracle, "is_functional_fault_legacy", None)
        if callable(checker):
            return sum(1 for episode in episodes if checker(episode))
        return sum(1 for episode in episodes if fault_oracle.is_functional_fault(episode))
    if mode == "window":
        checker = getattr(fault_oracle, "is_functional_fault_window", None)
        if callable(checker):
            return sum(1 for episode in episodes if checker(episode))
        return sum(1 for episode in episodes if fault_oracle.is_functional_fault(episode))
    if mode == "strict":
        checker = getattr(fault_oracle, "is_functional_fault_strict", None)
        if callable(checker):
            return sum(1 for episode in episodes if checker(episode))
        return sum(1 for episode in episodes if fault_oracle.is_functional_fault(episode))
    raise ValueError(f"Unsupported functional fault mode: {mode}")


def _run_single_experiment(
    *,
    experiment_id: str,
    mode: str,
    sam_rho: float | None,
    policy: Any,
    world_model: Any,
    env_id: str,
    fault_oracle: FaultOracle,
    exp_cfg: ExperimentConfig,
) -> dict[str, Any]:
    if mode not in {"starla_only", "sam"}:
        raise ValueError(f"Unsupported mode: {mode}")

    start_time = time.perf_counter()
    _set_global_seed(exp_cfg.seed)
    agent = DreamerV3Agent(policy=policy, world_model=world_model)
    env_adapter = GymnasiumEnv(env_id)

    training_episodes = _collect_agent_episodes(
        agent, env_id, exp_cfg.training_episodes, deterministic=True, seed=exp_cfg.seed
    )
    random_seed_episodes = _collect_agent_episodes(
        agent, env_id, exp_cfg.random_episodes, deterministic=False, seed=exp_cfg.seed + 10_000
    )

    sam_mutator: SAMGuidedMutator | None = None
    if mode == "sam":
        if sam_rho is None:
            raise ValueError("sam mode requires a rho value.")
        sam_mutator = SAMGuidedMutator(
            world_model=world_model,
            actor=agent.actor,
            critic=agent.value_head,
            rho=float(sam_rho),
            device=str(agent.device),
        )

    runner = StarlaRunner(config=_build_config(exp_cfg), agent=agent, env=env_adapter, fault_oracle=fault_oracle)
    runner.prepare_data(training_episodes, random_seed_episodes)
    report, raw_result = _run_starla(runner, sam_mutator=sam_mutator)
    archive_episodes = [candidate.episode for candidate in report.archive]

    all_search_episodes: list[Episode] = []
    for gen in report.generations:
        for candidate in gen:
            all_search_episodes.append(candidate.episode)

    total_searched = len(all_search_episodes)
    search_func_faults = _count_functional_faults(all_search_episodes, fault_oracle, mode="default")
    search_reward_faults = sum(1 for ep in all_search_episodes if fault_oracle.is_reward_fault(ep))
    search_func_fault_rate = search_func_faults / max(1, total_searched)
    search_reward_fault_rate = search_reward_faults / max(1, total_searched)

    total_budget_episodes = int(raw_result.mutation_count + len(report.archive))
    random_eval_episodes = _collect_random_episodes(env_id, n=max(total_budget_episodes, total_searched), seed=exp_cfg.seed + 20_000)
    starla_functional_faults = _count_functional_faults(archive_episodes, fault_oracle, mode="default")
    starla_functional_faults_legacy = _count_functional_faults(archive_episodes, fault_oracle, mode="legacy")
    starla_functional_faults_window = _count_functional_faults(archive_episodes, fault_oracle, mode="window")
    random_functional_faults = _count_functional_faults(random_eval_episodes, fault_oracle, mode="default")
    random_functional_faults_legacy = _count_functional_faults(random_eval_episodes, fault_oracle, mode="legacy")
    random_functional_faults_window = _count_functional_faults(random_eval_episodes, fault_oracle, mode="window")
    random_reward_faults = sum(1 for ep in random_eval_episodes if fault_oracle.is_reward_fault(ep))
    random_func_fault_rate = random_functional_faults / max(1, len(random_eval_episodes))
    random_reward_fault_rate = random_reward_faults / max(1, len(random_eval_episodes))

    return {
        "experiment_id": experiment_id,
        "mode": mode,
        "sam_rho": None if sam_rho is None else float(sam_rho),
        "sam_fallback_count": int(sam_mutator.fallback_count if sam_mutator else 0),
        "functional_fault_definition_active": "strict" if hasattr(fault_oracle, "is_functional_fault_strict") else ("window" if hasattr(fault_oracle, "is_functional_fault_window") else "default"),
        "env": env_id,
        "agent": "DreamerV3",
        "seed": exp_cfg.seed,
        "total_budget_episodes": total_budget_episodes,
        "starla_functional_faults": int(starla_functional_faults),
        "starla_functional_faults_legacy": int(starla_functional_faults_legacy),
        "starla_functional_faults_window": int(starla_functional_faults_window),
        "starla_reward_faults": int(report.found_faults["reward_faults"]),
        "starla_archive_size": int(len(report.archive)),
        "search_total_episodes": total_searched,
        "search_func_faults": search_func_faults,
        "search_reward_faults": search_reward_faults,
        "search_func_fault_rate": round(search_func_fault_rate, 4),
        "search_reward_fault_rate": round(search_reward_fault_rate, 4),
        "random_functional_faults": int(random_functional_faults),
        "random_functional_faults_legacy": int(random_functional_faults_legacy),
        "random_functional_faults_window": int(random_functional_faults_window),
        "random_reward_faults": int(random_reward_faults),
        "random_total_episodes": int(len(random_eval_episodes)),
        "random_func_fault_rate": round(random_func_fault_rate, 4),
        "random_reward_fault_rate": round(random_reward_fault_rate, 4),
        "num_generations": exp_cfg.num_generations,
        "time_seconds": float(time.perf_counter() - start_time),
    }


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _generate_feasibility_report(
    *,
    env_id: str,
    best_rho: float,
    sweep_results: list[dict[str, Any]],
    exp_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exp1 = exp_results["exp1_starla_only"]
    exp2 = exp_results["exp2_sam_best_rho"]
    exp3 = exp_results["exp3_sam_half_best"]
    exp4 = exp_results["exp4_sam_double_best"]

    all_sam = [exp2, exp3, exp4]
    baseline_rate = exp1.get("search_func_fault_rate", 0.0)
    any_sam_better = any(
        r.get("search_func_fault_rate", 0.0) > baseline_rate
        for r in all_sam
    )
    all_stable = all(r.get("sam_fallback_count", 0) == 0 for r in all_sam)
    all_outputs_non_empty = all(r["total_budget_episodes"] > 0 for r in exp_results.values())

    if any_sam_better and all_stable:
        next_step = "scale_up_budget_and_multi_seed_validation"
    elif any_sam_better:
        next_step = "investigate_fallback_then_scale"
    else:
        next_step = "increase_budget_or_adjust_rho_candidates"

    return {
        "experiment": "STARLA/SAM feasibility",
        "env": env_id,
        "best_rho": float(best_rho),
        "rho_sweep": sweep_results,
        "experiments": exp_results,
        "conclusion": {
            "is_pipeline_runnable": bool(all_outputs_non_empty),
            "is_sam_stable": bool(all_stable),
            "is_sam_better_than_starla_only": bool(any_sam_better),
            "recommended_next_step": next_step,
        },
    }


def _aggregate_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_reports:
        raise ValueError("seed_reports is empty.")

    def _mean_std(values: list[float]) -> dict[str, float]:
        mean_value = float(statistics.mean(values))
        std_value = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        return {"mean": mean_value, "std": std_value}

    experiment_ids = [
        "exp1_starla_only",
        "exp2_sam_best_rho",
        "exp3_sam_half_best",
        "exp4_sam_double_best",
    ]
    metric_names = [
        "starla_functional_faults",
        "starla_reward_faults",
        "search_total_episodes",
        "search_func_faults",
        "search_reward_faults",
        "search_func_fault_rate",
        "search_reward_fault_rate",
        "random_functional_faults",
        "random_reward_faults",
        "random_func_fault_rate",
        "random_reward_fault_rate",
        "total_budget_episodes",
        "time_seconds",
    ]

    per_experiment_summary: dict[str, Any] = {}
    for exp_id in experiment_ids:
        rows = [report["experiments"][exp_id] for report in seed_reports]
        summary = {
            "sample_size": len(rows),
            "sam_rho_values": [row.get("sam_rho") for row in rows],
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary[metric] = _mean_std(values)
        per_experiment_summary[exp_id] = summary

    baseline_rate = per_experiment_summary["exp1_starla_only"]["search_func_fault_rate"]["mean"]
    sam_rates = []
    for exp_id in ("exp2_sam_best_rho", "exp3_sam_half_best", "exp4_sam_double_best"):
        sam_rates.append(per_experiment_summary[exp_id]["search_func_fault_rate"]["mean"])
    is_sam_better = any(rate > baseline_rate for rate in sam_rates)

    best_rhos = [float(report["best_rho"]) for report in seed_reports]
    all_stable = all(report["conclusion"]["is_sam_stable"] for report in seed_reports)
    all_runnable = all(report["conclusion"]["is_pipeline_runnable"] for report in seed_reports)

    return {
        "experiment": "STARLA/SAM feasibility (multi-seed aggregate)",
        "seed_count": len(seed_reports),
        "best_rho_stats": _mean_std(best_rhos),
        "best_rho_values_per_seed": best_rhos,
        "per_experiment_summary": per_experiment_summary,
        "per_seed_reports": seed_reports,
        "conclusion": {
            "is_pipeline_runnable": bool(all_runnable),
            "is_sam_stable": bool(all_stable),
            "is_sam_better_than_starla_only": bool(is_sam_better),
            "recommended_next_step": (
                "increase_budget_or_adjust_rho_candidates"
                if not is_sam_better
                else "scale_up_budget_and_multi_seed_validation"
            ),
        },
    }


def main() -> None:
    args = _parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("seeds is empty.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds contains duplicates.")
    rho_candidates = [float(x.strip()) for x in args.rho_candidates.split(",") if x.strip()]
    if not rho_candidates:
        raise ValueError("rho_candidates is empty.")
    if any(v <= 0 for v in rho_candidates):
        raise ValueError("All rho candidates must be positive.")

    env_id, fault_oracle = _select_env_and_oracle()
    print(f"[FourExps] env={env_id}, oracle={fault_oracle.__class__.__name__}")

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    max_wall_clock_bound = len(seeds) * 9 * args.time_budget_seconds
    print(
        f"[FourExps] seeds={seeds}, rho_candidates={rho_candidates}, "
        f"worst_case_seconds={max_wall_clock_bound:.1f}, "
        f"objective_thresholds={args.objective_thresholds}"
    )
    if max_wall_clock_bound > 2 * 3600:
        print("[FourExps][WARN] worst-case budget exceeds 2 hours. Reduce seeds or time-budget-seconds.")

    per_seed_reports: list[dict[str, Any]] = []
    for seed in seeds:
        obj_thresh = tuple(float(v) for v in args.objective_thresholds.split(","))
        if len(obj_thresh) != 3:
            raise ValueError("--objective-thresholds must have exactly 3 comma-separated values.")
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
        try:
            policy, world_model = load_dreamerv3(
                checkpoint_dir=args.checkpoint_dir,
                seed=seed,
            )
        except Exception as err:
            raise RuntimeError(
                "Failed to load DreamerV3 checkpoint. "
                "Please ensure checkpoints exist and dependencies are installed. "
                "Try: pip install transformers tensorboardX easydict gym minigrid dmc2gym dm-control"
            ) from err

        if hasattr(world_model, "heads") and not hasattr(world_model.heads, "get"):

            def _heads_get(key: str, default: Any = None) -> Any:
                return world_model.heads[key] if key in world_model.heads else default

            setattr(world_model.heads, "get", _heads_get)

        sweep_results: list[dict[str, Any]] = []
        for rho in rho_candidates:
            result = _run_single_experiment(
                experiment_id=f"sweep_rho_{rho}",
                mode="sam",
                sam_rho=rho,
                policy=policy,
                world_model=world_model,
                env_id=env_id,
                fault_oracle=fault_oracle,
                exp_cfg=exp_cfg,
            )
            sweep_results.append(result)

        best = max(sweep_results, key=_rank_key)
        best_rho = float(best["sam_rho"])
        half_rho = float(best_rho * 0.5)
        double_rho = float(best_rho * 2.0)

        exp_results: dict[str, dict[str, Any]] = {}
        exp_results["exp1_starla_only"] = _run_single_experiment(
            experiment_id="exp1_starla_only",
            mode="starla_only",
            sam_rho=None,
            policy=policy,
            world_model=world_model,
            env_id=env_id,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
        )
        exp_results["exp2_sam_best_rho"] = _run_single_experiment(
            experiment_id="exp2_sam_best_rho",
            mode="sam",
            sam_rho=best_rho,
            policy=policy,
            world_model=world_model,
            env_id=env_id,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
        )
        exp_results["exp3_sam_half_best"] = _run_single_experiment(
            experiment_id="exp3_sam_half_best",
            mode="sam",
            sam_rho=half_rho,
            policy=policy,
            world_model=world_model,
            env_id=env_id,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
        )
        exp_results["exp4_sam_double_best"] = _run_single_experiment(
            experiment_id="exp4_sam_double_best",
            mode="sam",
            sam_rho=double_rho,
            policy=policy,
            world_model=world_model,
            env_id=env_id,
            fault_oracle=fault_oracle,
            exp_cfg=exp_cfg,
        )

        report = _generate_feasibility_report(
            env_id=env_id,
            best_rho=best_rho,
            sweep_results=sweep_results,
            exp_results=exp_results,
        )
        per_seed_reports.append(report)

        _dump_json(results_dir / f"rho_sweep_results_seed_{seed}.json", {"rho_sweep": sweep_results, "best_rho": best_rho})
        _dump_json(results_dir / f"exp1_starla_only_seed_{seed}.json", exp_results["exp1_starla_only"])
        _dump_json(results_dir / f"exp2_sam_best_rho_seed_{seed}.json", exp_results["exp2_sam_best_rho"])
        _dump_json(results_dir / f"exp3_sam_half_best_seed_{seed}.json", exp_results["exp3_sam_half_best"])
        _dump_json(results_dir / f"exp4_sam_double_best_seed_{seed}.json", exp_results["exp4_sam_double_best"])
        _dump_json(results_dir / f"feasibility_report_seed_{seed}.json", report)

        print(f"[FourExps][seed={seed}] best_rho={best_rho}, half={half_rho}, double={double_rho}")

    aggregate_report = _aggregate_seed_reports(per_seed_reports)
    _dump_json(results_dir / "feasibility_report.json", aggregate_report)
    print(f"[FourExps] aggregate report saved to: {results_dir / 'feasibility_report.json'}")


if __name__ == "__main__":
    main()
