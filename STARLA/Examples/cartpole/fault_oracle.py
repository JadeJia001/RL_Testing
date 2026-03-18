"""CartPole-specific fault oracle example."""

from __future__ import annotations

from typing import Any

from starla.faults.base import FaultOracle


class CartPoleFaultOracle(FaultOracle):
    """Fault criteria matching the original STARLA CartPole case."""

    _POSITION_LIMIT = 2.4
    _ANGLE_LIMIT_RAD = 12.0 * 3.141592653589793 / 180.0
    _FUNCTIONAL_WINDOW_SIZE = 3

    _STRICT_POSITION = 1.5
    _STRICT_ANGLE_RAD = 8.0 * 3.141592653589793 / 180.0

    @classmethod
    def _state_is_out_of_bounds(cls, state: Any) -> bool:
        if isinstance(state, str):
            return False
        try:
            position = float(state[0])
            angle = float(state[2])
        except (TypeError, ValueError, IndexError):
            return False
        return abs(position) > cls._POSITION_LIMIT or abs(angle) > cls._ANGLE_LIMIT_RAD

    @classmethod
    def _state_is_severe_fault(cls, state: Any) -> bool:
        """
        OR logic: fault when position > 1.5 OR angle > 8°.
        Creates differentiation with stricter thresholds than env limits (2.4, 12°).
        """
        if isinstance(state, str):
            return False
        try:
            position = float(state[0])
            angle = float(state[2])
        except (TypeError, ValueError, IndexError):
            return False
        return abs(position) > cls._STRICT_POSITION or abs(angle) > cls._STRICT_ANGLE_RAD

    def is_functional_fault_legacy(self, episode: list[tuple[Any, ...]]) -> bool:
        """Legacy definition: check only the last state before terminal marker."""
        if len(episode) < 2:
            return False
        last_state = episode[-2][0]
        return self._state_is_out_of_bounds(last_state)

    def is_functional_fault_window(self, episode: list[tuple[Any, ...]]) -> bool:
        """
        Window definition: check a tail window of pre-terminal states.
        Uses _state_is_out_of_bounds (OR logic).
        """
        if len(episode) < 2:
            return False
        transitions = episode[:-1]
        if not transitions:
            return False
        for state, _ in transitions[-self._FUNCTIONAL_WINDOW_SIZE :]:
            if self._state_is_out_of_bounds(state):
                return True
        return False

    def is_functional_fault_strict(self, episode: list[tuple[Any, ...]]) -> bool:
        """
        Strict definition (思路1): only fault when BOTH position AND angle are severe.
        Creates differentiation for search_func_fault_rate.
        """
        if len(episode) < 2:
            return False
        transitions = episode[:-1]
        if not transitions:
            return False
        for state, _ in transitions[-self._FUNCTIONAL_WINDOW_SIZE :]:
            if self._state_is_severe_fault(state):
                return True
        return False

    def is_functional_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        return self.is_functional_fault_strict(episode)

    def is_reward_fault(self, episode: list[tuple[Any, ...]]) -> bool:
        if not episode:
            return False
        terminal = episode[-1]
        if terminal[0] != "done":
            return False
        return float(terminal[1]) < 30.0

    def get_fault_thresholds(self) -> dict[str, Any]:
        return {
            "functional_definition_active": "strict",
            "functional_window_size": self._FUNCTIONAL_WINDOW_SIZE,
            "functional_position_limit": self._POSITION_LIMIT,
            "functional_angle_limit_rad": self._ANGLE_LIMIT_RAD,
            "strict_position": self._STRICT_POSITION,
            "strict_angle_rad": self._STRICT_ANGLE_RAD,
            "reward_fault_threshold": 30.0,
        }

