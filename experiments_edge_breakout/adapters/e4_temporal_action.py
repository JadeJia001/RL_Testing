"""E4: Temporal Action Perturbation (execution channel) — Atari Breakout (discrete).

Does NOT perturb the observation.  Instead it corrupts the *action execution
timing*, modelling real-world actuation delays, freezes, and inertia effects.
Actions are discrete integers (4 actions: NOOP, FIRE, RIGHT, LEFT).

Reference
---------
Tessler et al., "Action Robust Reinforcement Learning and Applications in
Continuous Control" (ICML 2019).  arXiv:1901.09184
Adapted from https://github.com/tesslerc/ActionRobustRL
"""

from __future__ import annotations

import random
from typing import Any


class TemporalActionAgent:
    """Wrapper agent that applies temporal action perturbation (discrete).

    Parameters
    ----------
    inner_agent:
        The real agent (e.g. ``DreamerV3Agent``).
    variant:
        ``"delay"`` | ``"freeze"`` | ``"smooth"``.
    delay:
        Number of steps of action delay (variant A only).
    alpha:
        Smoothing factor for variant C (``0 < alpha < 1``; higher = more lag).
    freeze_prob:
        Probability of freezing the previous action at each step (variant B).
    """

    def __init__(
        self,
        inner_agent: Any,
        *,
        variant: str = "delay",
        delay: int = 1,
        alpha: float = 0.5,
        freeze_prob: float = 0.3,
    ) -> None:
        if variant not in ("delay", "freeze", "smooth"):
            raise ValueError(f"Unknown E4 variant: {variant!r}")
        self.inner = inner_agent
        self.variant = variant
        self.delay = int(delay)
        self.alpha = float(alpha)
        self.freeze_prob = float(freeze_prob)
        self._action_history: list[Any] = []
        self._perturbation_count: int = 0

    # ------------------------------------------------------------------
    # AgentProtocol interface
    # ------------------------------------------------------------------

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, dict[str, Any]]:
        action, meta = self.inner.predict(obs, deterministic)
        self._action_history.append(action)

        if self.variant == "delay":
            if len(self._action_history) > self.delay:
                self._perturbation_count += 1
                return self._action_history[-1 - self.delay], meta

        elif self.variant == "freeze":
            if len(self._action_history) >= 2 and random.random() < self.freeze_prob:
                self._perturbation_count += 1
                return self._action_history[-2], meta

        elif self.variant == "smooth":
            if len(self._action_history) >= 2:
                prev = self._action_history[-2]
                curr = action
                smoothed = self.alpha * float(prev) + (1.0 - self.alpha) * float(curr)
                self._perturbation_count += 1
                return int(round(smoothed)), meta

        return action, meta

    def reset_state(self) -> None:
        """Clear action history and reset the inner agent."""
        self._action_history = []
        if hasattr(self.inner, "reset_state"):
            self.inner.reset_state()

    def get_action_probabilities(self, state: Any) -> Any:
        return self.inner.get_action_probabilities(state)

    def get_q_values(self, state: Any) -> Any:
        return self.inner.get_q_values(state)

    @property
    def perturbation_count(self) -> int:
        return self._perturbation_count

    # ------------------------------------------------------------------
    # Forwarded attributes for compatibility with STARLA internals
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        if name in ("inner",):
            raise AttributeError(name)
        return getattr(self.inner, name)
