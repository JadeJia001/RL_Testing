from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from tdmpc2.trainer.base import Trainer
from tdmpc2.common import math


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        return dict(
            step=self._step,
            episode=self._ep_idx,
            total_time=time() - self._start_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        ep_rewards, ep_successes = [], []
        v_s0_values = []  # Collect V(s_0) values from initial observations
        
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
            # Calculate V(s_0) = sum_i pi(a_i) * min_j Q_j(a_i, s_0) at initial observation
            if t == 0:
                with torch.no_grad():
                    obs_tensor = obs.to(self.agent.device, non_blocking=True).unsqueeze(0)
                    z0 = self.agent.model.encode(obs_tensor, None)
                    
                    # Sample actions from policy and compute their probabilities
                    num_action_samples = getattr(self.cfg, 'num_action_samples_for_v_s0', 64)
                    
                    # Get policy parameters (mean and log_std)
                    mu, _, _, log_std = self.agent.model.pi(z0, None)
                    std = log_std.exp()
                    
                    # Sample actions and compute their log probabilities
                    actions_sampled = []
                    log_probs = []
                    q_values_min = []
                    
                    for _ in range(num_action_samples):
                        # Sample action from policy
                        eps = torch.randn_like(mu)
                        action_raw = mu + eps * std
                        
                        # Apply tanh squashing (as done in the model)
                        action = torch.tanh(action_raw)
                        
                        # Compute log probability (with tanh correction)
                        log_prob = -0.5 * (eps**2 + 2 * log_std + np.log(2 * np.pi)).sum(-1)
                        log_prob = log_prob - (2 * (np.log(2) - action_raw - torch.nn.functional.softplus(-2 * action_raw))).sum(-1)
                        
                        # Get minimum Q-value across ensemble
                        q_all = self.agent.model.Q(z0, action, None, return_type="all")
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
                    v_s0_values.append(v_s0)
            
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info["success"])
            if self.cfg.save_video:
                # self.logger.video.save(self._step)
                self.logger.video.save(self._step, key='results/video')
        
        eval_metrics = dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )
        
        # Add V(s_0) to evaluation metrics for wandb logging
        if v_s0_values:
            eval_metrics["v_s0"] = np.nanmean(v_s0_values)
        
        return eval_metrics

    def to_td(self, obs, action=None, reward=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        td = TensorDict(
            dict(
                obs=obs,
                action=action.unsqueeze(0),
                reward=reward.unsqueeze(0),
            ),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics, done, eval_next = {}, True, True
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False

                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td["reward"] for td in self._tds[1:]]
                        ).sum(),
                        episode_success=info["success"],
                    )
                    train_metrics.update(self.common_metrics())

                    results_metrics = {'return': train_metrics['episode_reward'],
                                       'episode_length': len(self._tds[1:]),
                                       'success': train_metrics['episode_success'],
                                       'success_subtasks': info['success_subtasks'],
                                       'step': self._step,}
                
                    self.logger.log(train_metrics, "train")
                    self.logger.log(results_metrics, "results")
                    self._ep_idx = self.buffer.add(torch.cat(self._tds))

                obs = self.env.reset()[0]
                self._tds = [self.to_td(obs)]

            # Collect experience
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds) == 1)
            else:
                action = self.env.rand_act()
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            self._tds.append(self.to_td(obs, action, reward))

            # Update agent
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)

            self._step += 1

        self.logger.finish(self.agent)
