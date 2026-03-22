"""Train and load DI-engine DreamerV3 for ALE Breakout (RGB, 64x64)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict

_ROOT = Path("/Users/jq/Documents/RL_Testing")
sys.path.insert(0, str(_ROOT / "DI-engine"))
sys.path.insert(0, str(_ROOT / "STARLA" / "src"))

CHECKPOINT_DIR_BREAKOUT = _ROOT / "experiments_atari_sam" / "checkpoints_breakout"
POLICY_CKPT_PATH_BREAKOUT = CHECKPOINT_DIR_BREAKOUT / "dreamerv3_policy.pth"
WORLD_MODEL_CKPT_PATH_BREAKOUT = CHECKPOINT_DIR_BREAKOUT / "dreamerv3_world_model.pth"
META_PATH_BREAKOUT = CHECKPOINT_DIR_BREAKOUT / "dreamerv3_meta.json"

MAX_ENV_STEP_DEFAULT = 500_000
SEED_DEFAULT = 0

INSTALL_HINTS = {
    "atari": [
        "pip install ale-py shimmy",
        "pip install Pillow",
        'pip install "AutoROM[accept-rom-license]"',
        "AutoROM --accept-license",
    ],
    "ding_entry_import": [
        "pip install transformers",
        "pip install tensorboardX easydict gym",
    ],
}


def preprocess_ale_rgb_obs_for_dreamer(obs: Any, *, image_size: int = 64) -> np.ndarray:
    """
    Convert raw Gymnasium ALE RGB (H,W,3) uint8 or CHW float to CHW float32 [0,1] at image_size,
    matching BreakoutALERGBEnv and DreamerV3 (obs - 0.5 in model).
    """
    from PIL import Image

    arr = np.asarray(obs)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == image_size and arr.shape[2] == image_size:
        out = arr.astype(np.float32)
        if out.max() > 1.5:
            out = out / 255.0
        return np.clip(out, 0.0, 1.0)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        pil_img = Image.fromarray(arr).resize((image_size, image_size), Image.LANCZOS)
        resized = np.array(pil_img, dtype=np.float32)
        return np.transpose(resized, (2, 0, 1)) / 255.0
    raise ValueError(f"Unexpected observation for Breakout preprocessing: shape={arr.shape}, dtype={arr.dtype}")


@ENV_REGISTRY.register("breakout_ale_rgb")
class BreakoutALERGBEnv(BaseEnv):
    """Gymnasium ALE Breakout with RGB resized to CHW float32 in [0, 1]."""

    def __init__(self, cfg: dict | None = None) -> None:
        self._cfg = cfg if cfg is not None else {}
        self._init_flag = False
        self._env: gym.Env | None = None
        self._image_size = int(self._cfg.get("image_size", 64))
        self._env_id = str(self._cfg.get("env_id", "ALE/Breakout-v5"))
        low = np.zeros((3, self._image_size, self._image_size), dtype=np.float32)
        high = np.ones((3, self._image_size, self._image_size), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self._action_space = gym.spaces.Discrete(4)
        self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def _preprocess_obs(self, obs: np.ndarray) -> np.ndarray:
        return preprocess_ale_rgb_obs_for_dreamer(obs, image_size=self._image_size)

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id, obs_type="rgb")
            self._init_flag = True
        assert self._env is not None
        if hasattr(self, "_seed"):
            obs, _ = self._env.reset(seed=self._seed)
        else:
            obs, _ = self._env.reset()
        obs = self._preprocess_obs(to_ndarray(obs))
        self._eval_episode_return = 0.0
        return obs

    def close(self) -> None:
        if self._init_flag and self._env is not None:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(seed)

    def step(self, action: Union[int, np.ndarray]) -> BaseEnvTimestep:
        assert self._env is not None
        if isinstance(action, np.ndarray):
            if action.shape == (1,):
                action = int(action.squeeze())
            else:
                action = int(action.item())
        obs, rew, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        obs = self._preprocess_obs(to_ndarray(obs))
        self._eval_episode_return += float(rew)
        rew_arr = to_ndarray([rew]).astype(np.float32)
        if done:
            info = dict(info)
            info["eval_episode_return"] = self._eval_episode_return
        return BaseEnvTimestep(obs, rew_arr, done, info)

    def random_action(self) -> np.ndarray:
        return to_ndarray([self._action_space.sample()], dtype=np.int64)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return f"BreakoutALERGBEnv({self._env_id}, image_size={self._image_size})"


def _build_breakout_fallback_config() -> tuple[EasyDict, EasyDict]:
    cfg = EasyDict(
        dict(
            exp_name="dreamerv3_breakout_ale_rgb",
            env=dict(
                env_id="ALE/Breakout-v5",
                image_size=64,
                collector_env_num=1,
                evaluator_env_num=1,
                n_evaluator_episode=1,
                stop_value=500,
            ),
            policy=dict(
                cuda=False,
                random_collect_size=1000,
                model=dict(
                    action_shape=4,
                    actor_dist="onehot",
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
                    action_size=4,
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
                    state_size=(3, 64, 64),
                    obs_type="RGB",
                    action_size=4,
                    action_type="discrete",
                    encoder_hidden_size_list=[128, 64, 64],
                    reward_size=1,
                    batch_size=16,
                ),
            ),
        )
    )
    create_cfg = EasyDict(
        dict(
            env=dict(
                type="breakout_ale_rgb",
                import_names=["experiments_atari_sam.train_dreamerv3_breakout"],
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


def _build_breakout_config() -> tuple[EasyDict, EasyDict]:
    try:
        from dizoo.atari.config.breakout_dreamer_config import (  # type: ignore[import-not-found]
            breakout_create_config,
            breakout_dreamer_config,
        )

        cfg = copy.deepcopy(breakout_dreamer_config)
        create_cfg = copy.deepcopy(breakout_create_config)
        cfg.env.collector_env_num = 1
        cfg.env.evaluator_env_num = 1
        cfg.env.n_evaluator_episode = 1
        return cfg, create_cfg
    except Exception:
        return _build_breakout_fallback_config()


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


def train_dreamerv3_breakout(max_env_step: int = MAX_ENV_STEP_DEFAULT, seed: int = SEED_DEFAULT) -> tuple[Any, Any, str]:
    mbrl_entry_setup, _ = _import_ding_entry_tools()
    cfg, create_cfg = _build_breakout_config()
    print("[Train] Using Breakout DreamerV3 config.")

    ckpt_dir = CHECKPOINT_DIR_BREAKOUT
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
                "backend": "breakout",
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
    return policy, world_model, "breakout"


def load_dreamerv3_breakout(
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
        policy_ckpt = Path(policy_ckpt_path) if policy_ckpt_path is not None else POLICY_CKPT_PATH_BREAKOUT
        world_model_ckpt = Path(world_model_ckpt_path) if world_model_ckpt_path is not None else WORLD_MODEL_CKPT_PATH_BREAKOUT

    if not policy_ckpt.exists() or not world_model_ckpt.exists():
        raise FileNotFoundError("Checkpoint files do not exist. Run train_dreamerv3_breakout first.")

    cfg, create_cfg = _build_breakout_config()
    print("[Load] Restoring Breakout DreamerV3.")

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
    parser = argparse.ArgumentParser(description="Train DreamerV3 on ALE Breakout")
    parser.add_argument("--max-env-step", type=int, default=MAX_ENV_STEP_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()
    print(f"[Main] Start Breakout DreamerV3 training with max_env_step={args.max_env_step}")
    train_dreamerv3_breakout(max_env_step=args.max_env_step, seed=args.seed)
    loaded_policy, loaded_world_model = load_dreamerv3_breakout(seed=args.seed)
    print(f"[Main] Load verification done: policy={type(loaded_policy)}, world_model={type(loaded_world_model)}")
