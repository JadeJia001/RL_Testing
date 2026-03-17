"""Configuration models for STARLA."""

from __future__ import annotations

from pydantic import BaseModel


class StarlaConfig(BaseModel):
    """Global configuration for STARLA search and evolution."""

    population_size: int = 50
    num_generations: int = 10
    crossover_probability: float = 0.75
    mutation_rate_factor: float = 1.0
    abstraction_granularity: float = 1.0
    tournament_size: int = 10
    num_objectives: int = 3
    objective_thresholds: list[float]
    random_seed: int | None = None
    time_budget_seconds: float | None = None
