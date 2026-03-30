"""DMControl Cheetah-run pixel environment wrapper for DI-engine DreamerV3 and STARLA."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Union

import gymnasium as gym
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_DI_ENGINE = str(_ROOT / "DI-engine")
if _DI_ENGINE not in sys.path:
    sys.path.insert(0, _DI_ENGINE)

IMAGE_SIZE = 64


def preprocess_cheetah_rgb_obs_for_dreamer(obs: Any, *, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Convert DMControl pixel observation to CHW float32 [0,1] at image_size.

    DMControl pixels.Wrapper returns (H, W, 3) uint8 inside a dict keyed by
    ``"pixels"``.  Raw ndarray input is also accepted.
    """
    from PIL import Image

    if isinstance(obs, dict):
        obs = obs.get("pixels", obs.get("observations", next(iter(obs.values()))))

    arr = np.asarray(obs)

    # Already in the target format (CHW float32 at target size)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] == image_size and arr.shape[2] == image_size:
        out = arr.astype(np.float32)
        if out.max() > 1.5:
            out = out / 255.0
        return np.clip(out, 0.0, 1.0)

    # HWC uint8 -> resize -> CHW float32
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        pil_img = Image.fromarray(arr).resize((image_size, image_size), Image.LANCZOS)
        resized = np.array(pil_img, dtype=np.float32)
        return np.transpose(resized, (2, 0, 1)) / 255.0

    raise ValueError(
        f"Unexpected observation for Cheetah pixel preprocessing: shape={arr.shape}, dtype={arr.dtype}"
    )


def _make_dmcontrol_pixel_env(
    domain: str = "cheetah",
    task: str = "run",
    *,
    image_size: int = IMAGE_SIZE,
    seed: int | None = None,
) -> Any:
    """Create a DMControl environment with native pixel observations."""
    import os

    if "MUJOCO_GL" not in os.environ:
        try:
            import torch
            os.environ["MUJOCO_GL"] = "egl" if torch.cuda.is_available() else "osmesa"
        except ImportError:
            os.environ["MUJOCO_GL"] = "osmesa"

    from dm_control import suite
    from dm_control.suite.wrappers import pixels as pixel_wrapper

    env = suite.load(
        domain_name=domain,
        task_name=task,
        task_kwargs={"random": seed} if seed is not None else {},
    )
    env = pixel_wrapper.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={"height": image_size, "width": image_size, "camera_id": 0},
    )
    return env


def register_cheetah_env() -> None:
    """Lazily import DI-engine and register the Cheetah env class.

    Call this once before DI-engine's ``mbrl_entry_setup`` so that
    ``ENV_REGISTRY`` knows about ``"cheetah_dmcontrol_rgb"``.
    """
    from ding.envs import BaseEnv, BaseEnvTimestep
    from ding.torch_utils import to_ndarray
    from ding.utils import ENV_REGISTRY

    try:
        if ENV_REGISTRY.get("cheetah_dmcontrol_rgb") is not None:
            return
    except KeyError:
        pass

    @ENV_REGISTRY.register("cheetah_dmcontrol_rgb")
    class CheetahDMControlRGBEnv(BaseEnv):
        """DMControl Cheetah-run with native pixel observations (CHW float32 [0,1])."""

        def __init__(self, cfg: dict | None = None) -> None:
            self._cfg = cfg if cfg is not None else {}
            self._init_flag = False
            self._env: Any = None
            self._image_size = int(self._cfg.get("image_size", IMAGE_SIZE))
            self._domain = str(self._cfg.get("domain", "cheetah"))
            self._task = str(self._cfg.get("task", "run"))
            low = np.zeros((3, self._image_size, self._image_size), dtype=np.float32)
            high = np.ones((3, self._image_size, self._image_size), dtype=np.float32)
            self._observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
            self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
            self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

        def _preprocess_obs(self, obs: Any) -> np.ndarray:
            return preprocess_cheetah_rgb_obs_for_dreamer(obs, image_size=self._image_size)

        def reset(self) -> np.ndarray:
            if not self._init_flag:
                seed = getattr(self, "_seed", None)
                self._env = _make_dmcontrol_pixel_env(
                    self._domain, self._task,
                    image_size=self._image_size,
                    seed=seed,
                )
                self._init_flag = True
            assert self._env is not None
            time_step = self._env.reset()
            obs = self._preprocess_obs(time_step.observation)
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

        def step(self, action: Union[np.ndarray, list]) -> "BaseEnvTimestep":
            assert self._env is not None
            action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
            time_step = self._env.step(action_arr)
            obs = self._preprocess_obs(time_step.observation)
            rew = float(time_step.reward) if time_step.reward is not None else 0.0
            done = time_step.last()
            self._eval_episode_return += rew
            rew_arr = to_ndarray([rew]).astype(np.float32)
            info: dict[str, Any] = {}
            if done:
                info["eval_episode_return"] = self._eval_episode_return
            return BaseEnvTimestep(obs, rew_arr, done, info)

        def random_action(self) -> np.ndarray:
            return np.random.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)

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
            return f"CheetahDMControlRGBEnv({self._domain}-{self._task}, image_size={self._image_size})"
