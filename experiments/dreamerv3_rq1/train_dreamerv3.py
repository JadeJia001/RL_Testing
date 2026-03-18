"""Train and restore a DI-engine DreamerV3 agent for RQ1."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from easydict import EasyDict

sys.path.insert(0, "/workspaces/RL_Testing/DI-engine")
sys.path.insert(0, "/workspaces/RL_Testing/STARLA/src")

MAX_ENV_STEP = 10_000
SEED = 0

CHECKPOINT_DIR = Path("/workspaces/RL_Testing/experiments/dreamerv3_rq1/checkpoints")
CHECKPOINT_DIR_MOUNTAINCAR = Path("/workspaces/RL_Testing/experiments/dreamerv3_rq1/checkpoints_mountaincar")
POLICY_CKPT_PATH = CHECKPOINT_DIR / "dreamerv3_policy.pth"
WORLD_MODEL_CKPT_PATH = CHECKPOINT_DIR / "dreamerv3_world_model.pth"
META_PATH = CHECKPOINT_DIR / "dreamerv3_meta.json"

INSTALL_HINTS = {
    "dmc2gym": [
        "pip install dmc2gym",
        "pip install dm-control",
    ],
    "minigrid": [
        "pip install minigrid",
    ],
    "ding_entry_import": [
        "pip install transformers",
        "pip install tensorboardX easydict gym",
    ],
}


def _is_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_backend_availability() -> dict[str, bool]:
    return {
        "dmc2gym": _is_module_available("dmc2gym") and _is_module_available("dm_control"),
        "minigrid": _is_module_available("minigrid"),
    }


def _build_dmc2gym_config() -> tuple[EasyDict, EasyDict]:
    from dizoo.dmc2gym.config.cartpole_balance.cartpole_balance_dreamer_config import (
        cartpole_balance_create_config,
        cartpole_balance_dreamer_config,
    )

    cfg = copy.deepcopy(cartpole_balance_dreamer_config)
    create_cfg = copy.deepcopy(cartpole_balance_create_config)
    cfg.exp_name = "dreamerv3_rq1_dmc2gym_quick"
    cfg.env.collector_env_num = 1
    cfg.env.evaluator_env_num = 1
    cfg.env.n_evaluator_episode = 1
    cfg.policy.random_collect_size = 200
    cfg.policy.learn.batch_size = 8
    cfg.policy.learn.batch_length = 16
    cfg.world_model.pretrain = 1
    cfg.world_model.model.batch_size = 8
    return cfg, create_cfg


def _build_minigrid_config() -> tuple[EasyDict, EasyDict]:
    from dizoo.minigrid.config.minigrid_dreamer_config import minigrid_create_config, minigrid_dreamer_config

    cfg = copy.deepcopy(minigrid_dreamer_config)
    create_cfg = copy.deepcopy(minigrid_create_config)
    cfg.exp_name = "dreamerv3_rq1_minigrid_quick"
    cfg.env.collector_env_num = 1
    cfg.env.evaluator_env_num = 1
    cfg.env.n_evaluator_episode = 1
    cfg.policy.random_collect_size = 200
    cfg.policy.learn.batch_size = 8
    cfg.policy.learn.batch_length = 16
    cfg.world_model.pretrain = 1
    cfg.world_model.model.batch_size = 8
    return cfg, create_cfg


def _build_mountaincar_config() -> tuple[EasyDict, EasyDict]:
    cfg = EasyDict(
        dict(
            exp_name="dreamerv3_rq1_mountaincar",
            env=dict(
                env_id="MountainCar-v0",
                collector_env_num=1,
                evaluator_env_num=1,
                n_evaluator_episode=1,
                stop_value=-100,
            ),
            policy=dict(
                cuda=False,
                random_collect_size=200,
                model=dict(
                    action_shape=3,
                    actor_dist="onehot",
                ),
                learn=dict(
                    lambda_=0.95,
                    learning_rate=3e-5,
                    batch_size=8,
                    batch_length=16,
                    imag_sample=True,
                    discount=0.997,
                    reward_EMA=True,
                    update_per_collect=8,
                ),
                collect=dict(
                    n_sample=1,
                    unroll_len=1,
                    action_size=3,
                    collect_dyn_sample=True,
                ),
                eval=dict(evaluator=dict(eval_freq=2000)),
                other=dict(
                    replay_buffer=dict(replay_buffer_size=100000, periodic_thruput_seconds=60),
                ),
            ),
            world_model=dict(
                pretrain=1,
                train_freq=2,
                cuda=False,
                model=dict(
                    state_size=2,
                    obs_type="vector",
                    action_size=3,
                    action_type="discrete",
                    encoder_hidden_size_list=[128, 64, 64],
                    reward_size=1,
                    batch_size=8,
                ),
            ),
        )
    )
    create_cfg = EasyDict(
        dict(
            env=dict(
                type="mountain_car",
                import_names=["dizoo.classic_control.mountain_car.envs.mtcar_env"],
            ),
            env_manager=dict(type="base"),
            policy=dict(
                type="dreamer",
                import_names=["ding.policy.mbpolicy.dreamer"],
            ),
            replay_buffer=dict(type="sequence"),
            world_model=dict(
                type="dreamer",
                import_names=["ding.world_model.dreamer"],
            ),
        )
    )
    return cfg, create_cfg


def _build_cartpole_fallback_config() -> tuple[EasyDict, EasyDict]:
    cfg = EasyDict(
        dict(
            exp_name="dreamerv3_rq1_cartpole_fallback",
            env=dict(
                env_id="CartPole-v1",
                collector_env_num=1,
                evaluator_env_num=1,
                n_evaluator_episode=1,
                stop_value=500,
            ),
            policy=dict(
                cuda=False,
                random_collect_size=200,
                model=dict(
                    action_shape=2,
                    actor_dist="onehot",
                ),
                learn=dict(
                    lambda_=0.95,
                    learning_rate=3e-5,
                    batch_size=8,
                    batch_length=16,
                    imag_sample=True,
                    discount=0.997,
                    reward_EMA=True,
                    update_per_collect=8,
                ),
                collect=dict(
                    n_sample=1,
                    unroll_len=1,
                    action_size=2,
                    collect_dyn_sample=True,
                ),
                eval=dict(evaluator=dict(eval_freq=2000)),
                other=dict(
                    replay_buffer=dict(replay_buffer_size=100000, periodic_thruput_seconds=60),
                ),
            ),
            world_model=dict(
                pretrain=1,
                train_freq=2,
                cuda=False,
                model=dict(
                    state_size=4,
                    obs_type="vector",
                    action_size=2,
                    action_type="discrete",
                    encoder_hidden_size_list=[128, 64, 64],
                    reward_size=1,
                    batch_size=8,
                ),
            ),
        )
    )
    create_cfg = EasyDict(
        dict(
            env=dict(
                type="cartpole",
                import_names=["dizoo.classic_control.cartpole.envs.cartpole_env"],
            ),
            env_manager=dict(type="base"),
            policy=dict(
                type="dreamer",
                import_names=["ding.policy.mbpolicy.dreamer"],
            ),
            replay_buffer=dict(type="sequence"),
            world_model=dict(
                type="dreamer",
                import_names=["ding.world_model.dreamer"],
            ),
        )
    )
    return cfg, create_cfg


def _resolve_config(preferred_backend: str | None = None) -> tuple[str, EasyDict, EasyDict]:
    availability = _check_backend_availability()
    print(f"[BackendCheck] availability={availability}")

    candidates: list[str] = []
    if preferred_backend is not None:
        candidates.append(preferred_backend)
    else:
        if availability["minigrid"]:
            candidates.append("minigrid")
        if availability["dmc2gym"]:
            candidates.append("dmc2gym")
        candidates.append("cartpole_fallback")

    if not availability["minigrid"]:
        print("[BackendCheck] minigrid is unavailable.")
        print(f"[BackendCheck] install hint: {' ; '.join(INSTALL_HINTS['minigrid'])}")
    if not availability["dmc2gym"]:
        print("[BackendCheck] dmc2gym is unavailable.")
        print(f"[BackendCheck] install hint: {' ; '.join(INSTALL_HINTS['dmc2gym'])}")

    for backend in candidates:
        try:
            if backend == "minigrid":
                cfg, create_cfg = _build_minigrid_config()
            elif backend == "dmc2gym":
                cfg, create_cfg = _build_dmc2gym_config()
            elif backend == "mountaincar":
                cfg, create_cfg = _build_mountaincar_config()
            elif backend == "cartpole_fallback":
                cfg, create_cfg = _build_cartpole_fallback_config()
            else:
                raise ValueError(f"Unknown backend: {backend}")
            return backend, cfg, create_cfg
        except Exception as err:  # pragma: no cover - defensive for env/package issues
            print(f"[BackendCheck] backend={backend} build failed: {err}")

    raise RuntimeError("No DreamerV3 backend config is available.")


def _close_worker_resources(collector_env: Any, evaluator_env: Any, tb_logger: Any) -> None:
    try:
        collector_env.close()
    except Exception:
        pass
    try:
        evaluator_env.close()
    except Exception:
        pass
    try:
        tb_logger.close()
    except Exception:
        pass


def _import_ding_entry_tools() -> tuple[Any, Any]:
    try:
        from ding.entry.serial_entry_mbrl import mbrl_entry_setup as _mbrl_entry_setup
        from ding.entry.utils import random_collect as _random_collect
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Failed to import DI-engine entry dependencies. "
            f"Please install: {' ; '.join(INSTALL_HINTS['ding_entry_import'])}. "
            f"Original error: {err}"
        ) from err
    return _mbrl_entry_setup, _random_collect


def _run_training_loop(setup_items: tuple[Any, ...], max_env_step: int) -> tuple[Any, Any, str]:
    _, random_collect = _import_ding_entry_tools()
    cfg, policy, world_model, env_buffer, learner, collector, collector_env, evaluator, commander, tb_logger = setup_items
    learner.call_hook("before_run")

    if cfg.policy.get("random_collect_size", 0) > 0:
        cfg.policy.random_collect_size = cfg.policy.random_collect_size // cfg.policy.collect.unroll_len
        random_collect(cfg.policy, policy, collector, collector_env, commander, env_buffer)

    hit_world_model_shape_issue = False
    try:
        while True:
            collect_kwargs = commander.step()
            if evaluator.should_eval(collector.envstep):
                stop, _ = evaluator.eval(
                    learner.save_checkpoint,
                    learner.train_iter,
                    collector.envstep,
                    policy_kwargs=dict(world_model=world_model),
                )
                if stop:
                    break

            steps = cfg.world_model.pretrain if world_model.should_pretrain() else int(world_model.should_train(collector.envstep))
            for _ in range(steps):
                batch_size = learner.policy.get_attribute("batch_size")
                batch_length = cfg.policy.learn.batch_length
                try:
                    post, _ = world_model.train(env_buffer, collector.envstep, learner.train_iter, batch_size, batch_length)
                except RuntimeError as err:
                    # CartPole fallback may produce ragged sequence tensors in very short runs.
                    if "zero-dimensional tensor" in str(err):
                        print(f"[Train] world_model.train skipped due to shape mismatch: {err}")
                        hit_world_model_shape_issue = True
                        break
                    raise
                learner.train(
                    post,
                    collector.envstep,
                    policy_kwargs=dict(world_model=world_model, envstep=collector.envstep),
                )
            if hit_world_model_shape_issue:
                break

            data = collector.collect(
                train_iter=learner.train_iter,
                policy_kwargs=dict(world_model=world_model, envstep=collector.envstep, **collect_kwargs),
            )
            env_buffer.push(data, cur_collector_envstep=collector.envstep)

            if collector.envstep >= max_env_step:
                break
    finally:
        learner.call_hook("after_run")
        _close_worker_resources(collector_env, evaluator, tb_logger)

    return policy, world_model, cfg.exp_name


def train_dreamerv3(
    max_env_step: int = MAX_ENV_STEP,
    seed: int = SEED,
    preferred_backend: str | None = None,
) -> tuple[Any, Any, str]:
    mbrl_entry_setup, _ = _import_ding_entry_tools()
    backend, cfg, create_cfg = _resolve_config(preferred_backend=preferred_backend)
    print(f"[Train] Selected backend: {backend}")

    ckpt_dir = CHECKPOINT_DIR_MOUNTAINCAR if backend == "mountaincar" else CHECKPOINT_DIR
    policy_ckpt_path = ckpt_dir / "dreamerv3_policy.pth"
    world_model_ckpt_path = ckpt_dir / "dreamerv3_world_model.pth"
    meta_path = ckpt_dir / "dreamerv3_meta.json"

    setup_items = mbrl_entry_setup((cfg, create_cfg), seed=seed, env_setting=None, model=None)
    policy, world_model, exp_name = _run_training_loop(setup_items, max_env_step=max_env_step)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy_model = policy if hasattr(policy, "state_dict") else getattr(policy, "_model", None)
    if policy_model is None or not hasattr(policy_model, "state_dict"):
        raise RuntimeError("Unable to locate policy model state_dict for DreamerV3 policy.")
    torch.save(policy_model.state_dict(), policy_ckpt_path)
    torch.save(world_model.state_dict(), world_model_ckpt_path)
    meta_path.write_text(
        json.dumps(
            {
                "backend": backend,
                "seed": seed,
                "max_env_step": max_env_step,
                "exp_name": exp_name,
                "policy_ckpt": str(policy_ckpt_path),
                "world_model_ckpt": str(world_model_ckpt_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Train] Saved policy checkpoint: {policy_ckpt_path}")
    print(f"[Train] Saved world model checkpoint: {world_model_ckpt_path}")
    return policy, world_model, backend


def load_dreamerv3(
    policy_ckpt_path: str | Path | None = None,
    world_model_ckpt_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    seed: int = SEED,
) -> tuple[Any, Any]:
    mbrl_entry_setup, _ = _import_ding_entry_tools()
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        policy_ckpt = ckpt_dir / "dreamerv3_policy.pth"
        world_model_ckpt = ckpt_dir / "dreamerv3_world_model.pth"
        meta_path = ckpt_dir / "dreamerv3_meta.json"
    else:
        policy_ckpt = Path(policy_ckpt_path) if policy_ckpt_path is not None else POLICY_CKPT_PATH
        world_model_ckpt = Path(world_model_ckpt_path) if world_model_ckpt_path is not None else WORLD_MODEL_CKPT_PATH
        meta_path = META_PATH
    if not policy_ckpt.exists() or not world_model_ckpt.exists():
        raise FileNotFoundError("Checkpoint files do not exist. Run train_dreamerv3 first.")

    backend = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = meta.get("backend")
    backend, cfg, create_cfg = _resolve_config(preferred_backend=backend)
    print(f"[Load] Restoring backend: {backend}")

    setup_items = mbrl_entry_setup((cfg, create_cfg), seed=seed, env_setting=None, model=None)
    _, policy, world_model, _, _, _, collector_env, evaluator_env, _, tb_logger = setup_items
    policy_model = policy if hasattr(policy, "load_state_dict") else getattr(policy, "_model", None)
    if policy_model is None or not hasattr(policy_model, "load_state_dict"):
        raise RuntimeError("Unable to locate policy model load_state_dict for DreamerV3 policy.")
    policy_model.load_state_dict(torch.load(policy_ckpt, map_location="cpu"))
    world_model.load_state_dict(torch.load(world_model_ckpt, map_location="cpu"))
    policy.eval_mode.reset()
    _close_worker_resources(collector_env, evaluator_env, tb_logger)
    return policy, world_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DreamerV3 for RQ1")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["minigrid", "dmc2gym", "mountaincar", "cartpole_fallback"],
        help="Preferred backend (minigrid, dmc2gym, mountaincar, cartpole_fallback). "
        "If not set, auto-selects from available backends.",
    )
    parser.add_argument("--max-env-step", type=int, default=MAX_ENV_STEP)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    preferred_backend = args.backend
    if preferred_backend is None and META_PATH.exists():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        preferred_backend = meta.get("backend")
        if preferred_backend:
            print(f"[Main] Using backend from meta.json: {preferred_backend}")

    print(f"[Main] Start DreamerV3 training with MAX_ENV_STEP={args.max_env_step}")
    train_dreamerv3(
        max_env_step=args.max_env_step,
        seed=args.seed,
        preferred_backend=preferred_backend,
    )
    ckpt_dir = CHECKPOINT_DIR_MOUNTAINCAR if preferred_backend == "mountaincar" else None
    loaded_policy, loaded_world_model = load_dreamerv3(
        checkpoint_dir=ckpt_dir,
        seed=args.seed,
    )
    print(f"[Main] Load verification done: policy={type(loaded_policy)}, world_model={type(loaded_world_model)}")
