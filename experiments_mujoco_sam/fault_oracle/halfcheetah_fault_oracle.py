"""HalfCheetah-v4 fault oracle (reward / step thresholds)."""

from __future__ import annotations

from typing import Any

import numpy as np

from starla.faults.base import FaultOracle


class HalfCheetahFaultOracle(FaultOracle):
    """
    Episode layout: [(obs, action), ..., (last_obs, -1), ("done", total_reward)].
    HalfCheetah-v4 runs up to 1000 steps; episodes typically end with truncated=True.
    Obs index 0 is torso z (roughly 0.5~1.0 when upright).
    """

    _REWARD_THRESHOLD = 100.0
    _MIN_STEPS = 100
    _STRICT_REWARD_THRESHOLD = 500.0
    _FUNCTIONAL_WINDOW_SIZE = 3

    @staticmethod
    def _episode_step_count(episode: list[tuple[Any, ...]]) -> int:
        if len(episode) < 2:
            return 0
        return len(episode) - 2

    @staticmethod
    def _total_reward(episode: list[tuple[Any, ...]]) -> float:
        if not episode:
            return 0.0
        terminal = episode[-1]
        if len(terminal) < 2 or terminal[0] != "done":
            return 0.0
        return float(terminal[1])

    @classmethod
    def _obs_has_nonfinite(cls, obs: Any) -> bool:
        if isinstance(obs, str):
            return False
        try:
            arr = np.asarray(obs, dtype=np.float64)
        except (TypeError, ValueError):
            return True
        return not np.isfinite(arr).all()

    def is_functional_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        if self._episode_step_count(episode) < self._MIN_STEPS:
            return True
        for item in episode[:-1]:
            if self._obs_has_nonfinite(item[0]):
                return True
        return False

    def is_reward_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        if not episode:
            return False
        terminal = episode[-1]
        if terminal[0] != "done":
            return False
        return float(terminal[1]) < self._REWARD_THRESHOLD

    def is_functional_fault_legacy(self, episode: list[tuple[Any, ...]]) -> bool:
        if len(episode) < 2:
            return False
        return self._obs_has_nonfinite(episode[-2][0])

    def is_functional_fault_window(self, episode: list[tuple[Any, ...]]) -> bool:
        if len(episode) < 2:
            return False
        transitions = episode[:-1]
        if not transitions:
            return False
        for item in transitions[-self._FUNCTIONAL_WINDOW_SIZE :]:
            if self._obs_has_nonfinite(item[0]):
                return True
        return False

    def is_functional_fault_strict(self, episode: list[tuple[Any, ...]]) -> bool:
        """
        Stricter gate: short rollout or low return (OR), using a higher reward bar than is_reward_fault.
        """
        steps = self._episode_step_count(episode)
        reward = self._total_reward(episode)
        return steps < self._MIN_STEPS or reward < self._STRICT_REWARD_THRESHOLD

    def get_fault_thresholds(self) -> dict[str, Any]:
        return {
            "reward_fault_threshold": self._REWARD_THRESHOLD,
            "min_steps": self._MIN_STEPS,
            "strict_reward_threshold": self._STRICT_REWARD_THRESHOLD,
            "functional_window_size": self._FUNCTIONAL_WINDOW_SIZE,
        }
