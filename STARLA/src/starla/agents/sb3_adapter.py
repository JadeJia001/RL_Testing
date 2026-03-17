"""Stable-Baselines3 agent adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SB3Agent:
    """Adapter for SB3 policies (DQN/PPO/A2C) to STARLA AgentProtocol."""

    def __init__(self, model: Any):
        self.model = model

    def predict(self, obs: Any, deterministic: bool = True) -> tuple[Any, dict[str, Any]]:
        action, info = self.model.predict(obs, deterministic=deterministic)
        return action, info

    def get_action_probabilities(self, obs: Any) -> np.ndarray:
        policy = self.model.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            if hasattr(policy, "get_distribution"):
                distribution_wrapper = policy.get_distribution(obs_tensor)
                distribution = distribution_wrapper.distribution
                if hasattr(distribution, "probs"):
                    probs = distribution.probs.detach().cpu().numpy()
                    return np.asarray(probs[0], dtype=float)
                if hasattr(distribution, "logits"):
                    logits = distribution.logits.detach().cpu().numpy()[0]
                    exp = np.exp(logits - np.max(logits))
                    return exp / np.sum(exp)

            if hasattr(policy, "q_net"):
                q_values = policy.q_net(obs_tensor).detach().cpu().numpy()[0]
                exp = np.exp(q_values - np.max(q_values))
                return exp / np.sum(exp)

        raise NotImplementedError(
            "Action probabilities are not available for this SB3 policy type."
        )

    def get_q_values(self, obs: Any) -> np.ndarray:
        policy = self.model.policy
        if not hasattr(policy, "q_net"):
            raise NotImplementedError("Q-values are only available for value-based policies like DQN.")
        obs_tensor, _ = policy.obs_to_tensor(obs)
        with torch.no_grad():
            q_values = policy.q_net(obs_tensor).detach().cpu().numpy()
        return np.asarray(q_values[0], dtype=float)

