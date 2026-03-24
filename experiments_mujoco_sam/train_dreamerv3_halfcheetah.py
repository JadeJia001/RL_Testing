"""Train and load DI-engine DreamerV3 for MuJoCo HalfCheetah-v4 (continuous control)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from easydict import EasyDict

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "DI-engine"))
sys.path.insert(0, str(_ROOT / "STARLA" / "src"))

CHECKPOINT_DIR_HALFCHEETAH = _ROOT / "experiments_mujoco_sam" / "checkpoints_halfcheetah"
POLICY_CKPT_PATH_HALFCHEETAH = CHECKPOINT_DIR_HALFCHEETAH / "dreamerv3_policy.pth"
WORLD_MODEL_CKPT_PATH_HALFCHEETAH = CHECKPOINT_DIR_HALFCHEETAH / "dreamerv3_world_model.pth"
META_PATH_HALFCHEETAH = CHECKPOINT_DIR_HALFCHEETAH / "dreamerv3_meta.json"

MAX_ENV_STEP_DEFAULT = 200_000
SEED_DEFAULT = 0

INSTALL_HINTS = {
    "ding_entry_import": [
        "pip install transformers",
        "pip install tensorboardX easydict gym",
    ],
}


def _build_halfcheetah_config() -> tuple[EasyDict, EasyDict]:
    cfg = EasyDict(
        dict(
            exp_name="dreamerv3_halfcheetah",
            env=dict(
                env_id="HalfCheetah-v4",
                norm_obs=dict(use_norm=False),
                norm_reward=dict(use_norm=False),
                action_clip=False,
                delay_reward_step=0,
                replay_path_gif=None,
                save_replay_gif=False,
                action_bins_per_branch=None,
                collector_env_num=1,
                evaluator_env_num=1,
                n_evaluator_episode=1,
                stop_value=4000,
            ),
            policy=dict(
                cuda=True,
                random_collect_size=1000,
                model=dict(
                    action_shape=6,
                    actor_dist="normal",
                ),
                learn=dict(
                    lambda_=0.95,
                    learning_rate=3e-5,
                    batch_size=16,
                    batch_length=64,
                    imag_sample=True,
                    discount=0.997,
                    reward_EMA=True,
                    update_per_collect=8,
                ),
                collect=dict(
                    n_sample=1,
                    unroll_len=1,
                    action_size=6,
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
                cuda=True,
                model=dict(
                    state_size=17,
                    obs_type="vector",
                    action_size=6,
                    action_type="continuous",
                    encoder_hidden_size_list=[256, 128, 64],
                    reward_size=1,
                    batch_size=16,
                ),
            ),
        )
    )
    create_cfg = EasyDict(
        dict(
            env=dict(
                type="mujoco",
                import_names=["dizoo.mujoco.envs.mujoco_env"],
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
                except (RuntimeError, TypeError) as err:
                    if "zero-dimensional tensor" in str(err):
                        print(f"[Train] world_model.train skipped due to shape mismatch: {err}")
                        hit_world_model_shape_issue = True
                        break
                    if "NoneType" in str(err):
                        print(f"[Train] world_model.train skipped: not enough samples in buffer yet.")
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


def train_dreamerv3_halfcheetah(max_env_step: int = MAX_ENV_STEP_DEFAULT, seed: int = SEED_DEFAULT) -> tuple[Any, Any, str]:
    mbrl_entry_setup, _ = _import_ding_entry_tools()
    cfg, create_cfg = _build_halfcheetah_config()
    print("[Train] Using HalfCheetah DreamerV3 config.")

    ckpt_dir = CHECKPOINT_DIR_HALFCHEETAH
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
                "backend": "halfcheetah",
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
    return policy, world_model, "halfcheetah"


def load_dreamerv3_halfcheetah(
    policy_ckpt_path: str | Path | None = None,
    world_model_ckpt_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    seed: int = SEED_DEFAULT,
) -> tuple[Any, Any]:
    mbrl_entry_setup, _ = _import_ding_entry_tools()
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        policy_ckpt = ckpt_dir / "dreamerv3_policy.pth"
        world_model_ckpt = ckpt_dir / "dreamerv3_world_model.pth"
    else:
        policy_ckpt = Path(policy_ckpt_path) if policy_ckpt_path is not None else POLICY_CKPT_PATH_HALFCHEETAH
        world_model_ckpt = Path(world_model_ckpt_path) if world_model_ckpt_path is not None else WORLD_MODEL_CKPT_PATH_HALFCHEETAH

    if not policy_ckpt.exists() or not world_model_ckpt.exists():
        raise FileNotFoundError("Checkpoint files do not exist. Run train_dreamerv3_halfcheetah first.")

    cfg, create_cfg = _build_halfcheetah_config()
    print("[Load] Restoring HalfCheetah DreamerV3.")

    setup_items = mbrl_entry_setup((cfg, create_cfg), seed=seed, env_setting=None, model=None)
    _, policy, world_model, _, _, _, collector_env, evaluator_env, _, tb_logger = setup_items
    policy_model = policy if hasattr(policy, "load_state_dict") else getattr(policy, "_model", None)
    if policy_model is None or not hasattr(policy_model, "load_state_dict"):
        raise RuntimeError("Unable to locate policy model load_state_dict for DreamerV3 policy.")
    use_cuda = bool(cfg.policy.get("cuda", False)) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    policy_model.load_state_dict(torch.load(policy_ckpt, map_location="cpu"))
    world_model.load_state_dict(torch.load(world_model_ckpt, map_location="cpu"))
    policy_model.to(device)
    world_model.to(device)
    if hasattr(policy, "to"):
        policy.to(device)
    elif use_cuda and hasattr(policy, "cuda"):
        policy.cuda()
    policy.eval_mode.reset()
    _close_worker_resources(collector_env, evaluator_env, tb_logger)
    return policy, world_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DreamerV3 on HalfCheetah-v4")
    parser.add_argument("--max-env-step", type=int, default=MAX_ENV_STEP_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()
    print(f"[Main] Start HalfCheetah DreamerV3 training with max_env_step={args.max_env_step}")
    train_dreamerv3_halfcheetah(max_env_step=args.max_env_step, seed=args.seed)
    loaded_policy, loaded_world_model = load_dreamerv3_halfcheetah(seed=args.seed)
    print(f"[Main] Load verification done: policy={type(loaded_policy)}, world_model={type(loaded_world_model)}")
