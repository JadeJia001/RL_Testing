"""Gymnasium environment adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym


class GymnasiumEnv:
    """Adapter for Gymnasium environments to STARLA EnvProtocol."""

    def __init__(self, env_id: str, **kwargs: Any):
        self.env = gym.make(env_id, **kwargs)
        self._last_obs: Any = None
        self._mem: list[tuple[Any, Any]] = []
        self._episode_reward: float = 0.0

    def reset(self) -> Any:
        obs, _ = self.env.reset()
        self._last_obs = deepcopy(obs)
        self._mem = []
        self._episode_reward = 0.0
        return obs

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward)
        if self._last_obs is not None:
            self._mem.append((deepcopy(self._last_obs), int(action)))
        self._episode_reward += reward_value
        done = bool(terminated or truncated)
        if done:
            self._mem.append((deepcopy(obs), -1))
            self._mem.append(("done", self._episode_reward))
        self._last_obs = deepcopy(obs)
        info = dict(info)
        info["mem"] = deepcopy(self._mem)
        return obs, reward_value, bool(terminated), bool(truncated), info

    def get_state(self) -> Any:
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "ale"):
            # ale-py uses cloneState/restoreState for save/restore.
            ale_obj = unwrapped.ale
            if hasattr(ale_obj, "cloneState"):
                return {"_ale_state": ale_obj.cloneState()}
            raise RuntimeError("ALE environment has .ale but no cloneState(); cannot snapshot state.")
        if hasattr(unwrapped, "state"):
            return deepcopy(unwrapped.state)
        try:
            return deepcopy(unwrapped)
        except Exception:
            return deepcopy(unwrapped.__dict__)

    def set_state(self, state: Any) -> Any:
        unwrapped = self.env.unwrapped
        # ALE state (saved by get_state above)
        if isinstance(state, dict) and "_ale_state" in state:
            ale_obj = unwrapped.ale
            st = state["_ale_state"]
            if hasattr(ale_obj, "restoreState"):
                ale_obj.restoreState(st)
            else:
                raise RuntimeError("ALE environment has .ale but no restoreState(); cannot restore snapshot.")
            if hasattr(unwrapped, "_get_obs"):
                obs = unwrapped._get_obs()
            else:
                obs, _ = self.env.reset()
            self._last_obs = deepcopy(obs)
            self._mem = []
            self._episode_reward = 0.0
            return obs

        if hasattr(unwrapped, "state") and not hasattr(state, "__dict__"):
            unwrapped.state = deepcopy(state)
            current_state = deepcopy(unwrapped.state)
            self._last_obs = deepcopy(current_state)
            self._mem = []
            self._episode_reward = 0.0
            return current_state

        if hasattr(state, "__dict__"):
            unwrapped.__dict__.clear()
            unwrapped.__dict__.update(deepcopy(state.__dict__))
        elif isinstance(state, dict):
            unwrapped.__dict__.update(deepcopy(state))
        else:
            # Fallback: unrecognized format (e.g. raw obs numpy array),
            # reset to episode start — acceptable for fixed-start envs like Breakout
            obs, _ = self.env.reset()
            self._last_obs = deepcopy(obs)
            self._mem = []
            self._episode_reward = 0.0
            return obs

        if hasattr(unwrapped, "state"):
            current_state = deepcopy(unwrapped.state)
            self._last_obs = deepcopy(current_state)
            self._mem = []
            self._episode_reward = 0.0
            return current_state
        obs, _ = self.env.reset()
        self._last_obs = deepcopy(obs)
        self._mem = []
        self._episode_reward = 0.0
        return obs

