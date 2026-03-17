"""Random-forest based fault predictors."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from starla.core.candidate import Episode
from starla.faults.base import FaultOracle
from starla.ml.encoder import EpisodeEncoder


class FaultPredictor:
    """Train and query functional/reward fault probability models."""

    def __init__(self, encoder: EpisodeEncoder):
        self.encoder = encoder
        self.functional_fault_model: RandomForestClassifier | None = None
        self.reward_fault_model: RandomForestClassifier | None = None

    def fit(self, episodes: list[Episode], fault_oracle: FaultOracle) -> "FaultPredictor":
        x = self.encoder.encode_batch(episodes)
        y_functional = np.array(
            [1 if fault_oracle.is_functional_fault(episode) else 0 for episode in episodes],
            dtype=int,
        )
        y_reward = np.array(
            [1 if fault_oracle.is_reward_fault(episode) else 0 for episode in episodes],
            dtype=int,
        )

        x_train_r, _, y_train_r, _ = train_test_split(
            x, y_reward, test_size=0.30, random_state=42
        )
        x_train_f, _, y_train_f, _ = train_test_split(
            x, y_functional, test_size=0.30, random_state=42
        )

        self.reward_fault_model = RandomForestClassifier(
            random_state=0, class_weight="balanced"
        )
        self.reward_fault_model.fit(x_train_r, y_train_r)

        self.functional_fault_model = RandomForestClassifier(
            random_state=0, class_weight="balanced"
        )
        self.functional_fault_model.fit(x_train_f, y_train_f)
        return self

    def predict_functional_fault_proba(self, episode: Episode) -> float:
        if self.functional_fault_model is None:
            raise RuntimeError("Functional fault model is not trained. Call fit() first.")
        x = self.encoder.encode(episode).reshape(1, -1)
        return float(self.functional_fault_model.predict_proba(x)[0][1])

    def predict_reward_fault_proba(self, episode: Episode) -> float:
        if self.reward_fault_model is None:
            raise RuntimeError("Reward fault model is not trained. Call fit() first.")
        x = self.encoder.encode(episode).reshape(1, -1)
        return float(self.reward_fault_model.predict_proba(x)[0][1])

    def predict_proba(self, x: Any) -> np.ndarray:
        """
        Compatibility method for existing ML interface in fitness/MOSA.
        Returns functional-fault model probabilities.
        """
        if self.functional_fault_model is None:
            raise RuntimeError("Functional fault model is not trained. Call fit() first.")
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)
        return self.functional_fault_model.predict_proba(x_array)

