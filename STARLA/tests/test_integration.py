from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
from stable_baselines3 import DQN

from Examples.cartpole.fault_oracle import CartPoleFaultOracle
from starla.agents.sb3_adapter import SB3Agent
from starla.config import StarlaConfig
from starla.envs.gymnasium_adapter import GymnasiumEnv
from starla.runner import StarlaRunner


def _collect_episodes(model: DQN, env_id: str, n: int, deterministic: bool) -> list[list[tuple]]:
    env = gym.make(env_id)
    episodes: list[list[tuple]] = []
    for _ in range(n):
        obs, _ = env.reset(seed=123)
        done = False
        total_reward = 0.0
        ep: list[tuple] = []
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            ep.append((deepcopy(obs), int(action)))
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += float(reward)
            done = bool(terminated or truncated)
        ep.append(("done", total_reward))
        episodes.append(ep)
    env.close()
    return episodes


def test_end_to_end_small_search_cartpole() -> None:
    env_id = "CartPole-v1"
    train_env = gym.make(env_id)
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_starts=1,
        buffer_size=200,
        batch_size=1,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=200)

    training_episodes = _collect_episodes(model, env_id, n=6, deterministic=True)
    random_episodes = _collect_episodes(model, env_id, n=6, deterministic=False)

    config = StarlaConfig(
        population_size=6,
        num_generations=2,
        crossover_probability=0.75,
        mutation_rate_factor=1.0,
        abstraction_granularity=1.0,
        tournament_size=3,
        num_objectives=3,
        objective_thresholds=[70.0, 0.06, 0.05],
        random_seed=42,
        time_budget_seconds=30.0,
    )

    runner = StarlaRunner(
        config=config,
        agent=SB3Agent(model),
        env=GymnasiumEnv(env_id),
        fault_oracle=CartPoleFaultOracle(),
    )
    population = runner.prepare_data(training_episodes, random_episodes)
    report = runner.run()

    assert len(population) > 0
    assert isinstance(report.archive, list)
    assert len(report.generations) >= 1
    assert "run_seconds" in report.timing
    assert report.total_budget["num_generations"] == 2

