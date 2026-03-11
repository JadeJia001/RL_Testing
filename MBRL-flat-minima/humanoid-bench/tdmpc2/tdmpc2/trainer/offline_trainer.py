import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from tdmpc2.common.buffer import Buffer
from tdmpc2.trainer.base import Trainer
from tdmpc2.common import math


class OfflineTrainer(Trainer):
    """Trainer class for multi-task offline TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time()

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        results = dict()
        v_s0_values_all_tasks = []  # Collect V(s_0) values across all tasks
        
        for task_idx in tqdm(range(len(self.cfg.tasks)), desc="Evaluating"):
            ep_rewards, ep_successes = [], []
            v_s0_values_task = []  # Collect V(s_0) values for this specific task
            
            for _ in range(self.cfg.eval_episodes):
                obs, done, ep_reward, t = self.env.reset(task_idx)[0], False, 0, 0
                
                # Calculate V(s_0) = sum_i pi(a_i) * min_j Q_j(a_i, s_0) at initial observation
                if t == 0:
                    with torch.no_grad():
                        obs_tensor = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
                        task_tensor = torch.tensor([task_idx], device=self.agent.device)
                        z0 = self.agent.model.encode(obs_tensor, task_tensor)
                        
                        # Sample actions from policy and compute their probabilities
                        num_action_samples = getattr(self.cfg, 'num_action_samples_for_v_s0', 32)
                        
                        # Get policy parameters (mean and log_std)
                        mu, _, _, log_std = self.agent.model.pi(z0, task_tensor)
                        std = log_std.exp()
                        
                        # Sample actions and compute their log probabilities
                        actions_sampled = []
                        log_probs = []
                        q_values_min = []
                        
                        for _ in range(num_action_samples):
                            # Sample action from policy
                            eps = torch.randn_like(mu)
                            action_raw = mu + eps * std
                            
                            # Apply tanh squashing and action masking for multitask
                            action = torch.tanh(action_raw)
                            if self.cfg.multitask:
                                action = action * self.agent.model._action_masks[task_tensor]
                            
                            # Compute log probability (with tanh correction)
                            log_prob = -0.5 * (eps**2 + 2 * log_std + np.log(2 * np.pi)).sum(-1)
                            log_prob = log_prob - (2 * (np.log(2) - action_raw - torch.nn.functional.softplus(-2 * action_raw))).sum(-1)
                            
                            # Get minimum Q-value across ensemble
                            q_all = self.agent.model.Q(z0, action, task_tensor, return_type="all")
                            if self.cfg.num_bins > 1:
                                q_processed = torch.stack([
                                    math.two_hot_inv(q_all[j], self.cfg) 
                                    for j in range(q_all.shape[0])
                                ])
                            else:
                                q_processed = q_all
                            q_min = q_processed.min(dim=0)[0]
                            
                            actions_sampled.append(action)
                            log_probs.append(log_prob)
                            q_values_min.append(q_min)
                        
                        # Convert to tensors
                        log_probs = torch.stack(log_probs)
                        q_values_min = torch.stack(q_values_min)
                        
                        # Compute probabilities (normalize)
                        probs = torch.exp(log_probs - log_probs.max())  # Numerical stability
                        probs = probs / probs.sum()
                        
                        # Compute V(s_0) = sum_i pi(a_i) * min_j Q_j(a_i, s_0)
                        v_s0 = (probs.unsqueeze(-1) * q_values_min).sum(dim=0).mean().item()
                        v_s0_values_task.append(v_s0)
                        v_s0_values_all_tasks.append(v_s0)
                
                while not done:
                    action = self.agent.act(
                        obs, t0=t == 0, eval_mode=True, task=task_idx
                    )
                    obs, reward, done, truncated, info = self.env.step(action)
                    done = done or truncated
                    ep_reward += reward
                    t += 1
                ep_rewards.append(ep_reward)
                ep_successes.append(info["success"])
            
            # Add task-specific metrics
            results.update(
                {
                    f"episode_reward+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_rewards
                    ),
                    f"episode_success+{self.cfg.tasks[task_idx]}": np.nanmean(
                        ep_successes
                    ),
                }
            )
            
            # Add task-specific V(s_0) metrics
            if v_s0_values_task:
                results[f"v_s0+{self.cfg.tasks[task_idx]}"] = np.nanmean(v_s0_values_task)
        
        # Add overall V(s_0) metric across all tasks
        if v_s0_values_all_tasks:
            results["v_s0"] = np.nanmean(v_s0_values_all_tasks)
            
        return results

    def train(self):
        """Train a TD-MPC2 agent."""
        assert self.cfg.multitask and self.cfg.task in {
            "mt30",
            "mt80",
        }, "Offline training only supports multitask training with mt30 or mt80 task sets."

        # Load data
        assert self.cfg.task in self.cfg.data_dir, (
            f"Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, "
            f"please double-check your config."
        )
        fp = Path(os.path.join(self.cfg.data_dir, "*.pt"))
        fps = sorted(glob(str(fp)))
        assert len(fps) > 0, f"No data found at {fp}"
        print(f"Found {len(fps)} files in {fp}")

        # Create buffer for sampling
        _cfg = deepcopy(self.cfg)
        _cfg.episode_length = 101 if self.cfg.task == "mt80" else 501
        _cfg.buffer_size = 550_450_000 if self.cfg.task == "mt80" else 345_690_000
        _cfg.steps = _cfg.buffer_size
        self.buffer = Buffer(_cfg)
        for fp in tqdm(fps, desc="Loading data"):
            td = torch.load(fp)
            assert td.shape[1] == _cfg.episode_length, (
                f"Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, "
                f"please double-check your config."
            )
            for i in range(len(td)):
                self.buffer.add(td[i])
        assert (
            self.buffer.num_eps == self.buffer.capacity
        ), f"Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes."

        print(f"Training agent for {self.cfg.steps} iterations...")
        metrics = {}
        for i in range(self.cfg.steps):
            # Update agent
            train_metrics = self.agent.update(self.buffer)

            # Evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
                metrics = {
                    "iteration": i,
                    "total_time": time() - self._start_time,
                }
                metrics.update(train_metrics)
                if i % self.cfg.eval_freq == 0:
                    metrics.update(self.eval())
                    self.logger.pprint_multitask(metrics, self.cfg)
                    if i > 0:
                        self.logger.save_agent(self.agent, identifier=f"{i}")
                self.logger.log(metrics, "pretrain")

        self.logger.finish(self.agent)
