"""End-to-end STARLA example for SB3 DQN on CartPole."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
from stable_baselines3 import DQN

from starla.agents.sb3_adapter import SB3Agent
from starla.config import StarlaConfig
from starla.envs.gymnasium_adapter import GymnasiumEnv
from starla.runner import StarlaRunner

from Examples.cartpole.fault_oracle import CartPoleFaultOracle


def collect_episodes(
    model: DQN,
    env_id: str,
    num_episodes: int,
    *,
    deterministic: bool,
) -> list[list[tuple]]:
    episodes: list[list[tuple]] = []
    env = gym.make(env_id)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode: list[tuple] = []
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            episode.append((deepcopy(obs), int(action)))
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += float(reward)
            done = bool(terminated or truncated)
        episode.append(("done", total_reward))
        episodes.append(episode)
    env.close()
    return episodes


def main() -> None:
    env_id = "CartPole-v1"
    train_env = gym.make(env_id)

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_starts=100,
        buffer_size=5000,
        batch_size=64,
        train_freq=4,
        gradient_steps=1,
        verbose=0,
        seed=42,
    )
    model.learn(total_timesteps=5000)

    training_episodes = collect_episodes(model, env_id, num_episodes=40, deterministic=True)
    random_episodes = collect_episodes(model, env_id, num_episodes=40, deterministic=False)

    config = StarlaConfig(
        population_size=30,
        num_generations=8,
        crossover_probability=0.75,
        mutation_rate_factor=1.0,
        abstraction_granularity=1.0,
        tournament_size=10,
        num_objectives=3,
        objective_thresholds=[70.0, 0.06, 0.05],
        random_seed=42,
        time_budget_seconds=60.0,
    )

    runner = StarlaRunner(
        config=config,
        agent=SB3Agent(model),
        env=GymnasiumEnv(env_id),
        fault_oracle=CartPoleFaultOracle(),
    )
    runner.prepare_data(training_episodes, random_episodes)
    report = runner.run()

    print("STARLA run completed.")
    print("Found faults:", report.found_faults)
    print("Archive size:", len(report.archive))
    print("Generations stored:", len(report.generations))
    print("Timing:", report.timing)


if __name__ == "__main__":
    main()

