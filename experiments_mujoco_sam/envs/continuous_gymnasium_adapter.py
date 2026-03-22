"""Gymnasium adapter wrapper that records continuous actions without calling int(action)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from starla.envs.gymnasium_adapter import GymnasiumEnv


class ContinuousGymnasiumEnv(GymnasiumEnv):
    """GymnasiumEnv patched for continuous action spaces (e.g. MuJoCo)."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward)
        if self._last_obs is not None:
            if np.asarray(action).size == 1:
                saved_action = int(np.asarray(action).reshape(-1)[0])
            else:
                saved_action = np.asarray(action).tolist()
            self._mem.append((deepcopy(self._last_obs), saved_action))
        self._episode_reward += reward_value
        done = bool(terminated or truncated)
        if done:
            self._mem.append((deepcopy(obs), -1))
            self._mem.append(("done", self._episode_reward))
        self._last_obs = deepcopy(obs)
        info = dict(info)
        info["mem"] = deepcopy(self._mem)
        return obs, reward_value, bool(terminated), bool(truncated), info
