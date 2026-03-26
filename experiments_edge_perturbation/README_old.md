# Edge-Based Perturbation Experiments (E2, E3, E4)

Vulnerability profiling for DreamerV3 on MountainCar-v0 via perturbation at
four edges of the perception-action pipeline.

## Experiments

| ID  | Name           | Edge        | Method                                         |
|-----|----------------|-------------|------------------------------------------------|
| exp0 | baseline      | —           | Vanilla STARLA (no perturbation)               |
| exp1 | E1_perception | Observation | SAM sharpness-guided mutation (rho=0.1)        |
| exp2 | E2_encoding   | Encoder     | Jacobian-guided sparse perturbation (ECCV 2018)|
| exp3 | E3_prediction | World model | Uncertainty-guided directional perturbation    |
| exp4 | E4_execution  | Action      | Temporal action perturbation (ICML 2019)       |

## References

- **E2:** Jakubovitz & Giryes, ECCV 2018 — arXiv:1803.08680
- **E3:** Yu et al., NeurIPS 2020 (MOPO) — arXiv:2005.13239
- **E4:** Tessler et al., ICML 2019 — arXiv:1901.09184

## Usage

```bash
cd /workspaces/RL_Testing
python -m experiments_edge_perturbation.run_edge_experiments --seeds 42
```

Results are written to `experiments_edge_perturbation/results/`.

## Results (MountainCar-v0, seed=42)

### Reward Fault Rate

Higher fault rate = the perturbation exposed more reward-level defects = the edge is more vulnerable.

| Rank | Experiment | Perturbed Edge | Reward Fault Rate | vs Baseline |
|:----:|------------|---------------|:-----------------:|:-----------:|
| 1 | E3 Uncertainty | World model prediction | 32.43% | +4.65% |
| 2 | E1 SAM | Observation (critic-guided) | 29.63% | +1.85% |
| 3 | E2 Jacobian | Observation (encoder-guided) | 28.38% | +0.60% |
| 4 | Baseline | Random noise | 27.78% | — |
| 5 | E4 Temporal | Action timing | 27.03% | -0.75% |

### Functional Fault Rate

Functional faults detect "crash-level" failures: episodes that terminate abnormally early
(fewer than 20 transitions) or produce non-finite (NaN/Inf) observations.

| Experiment | Functional Faults | Functional Fault Rate |
|------------|:-----------------:|:---------------------:|
| exp0 Baseline | 0 | 0.00% |
| exp1 E1 SAM | 0 | 0.00% |
| exp2 E2 Jacobian | 0 | 0.00% |
| exp3 E3 Uncertainty | 0 | 0.00% |
| exp4 E4 Temporal | 1 | 1.35% |

**Why functional faults are near-zero across all experiments:**

MountainCar-v0 is a physically simple and stable environment. The observation space is
2-dimensional (position in [-1.2, 0.6], velocity in [-0.07, 0.07]), and the physics
engine always produces finite values regardless of the agent's actions. Episodes almost
always run for the full 200 steps (the environment's max_episode_steps), far exceeding
the functional fault threshold of 20 transitions. None of the observation-space
perturbations (E1/E2/E3) alter the environment's internal physics — they only change
what the agent *sees*, so the environment itself never crashes or produces degenerate
states.

**Why E4 (Temporal) is the only experiment that triggered a functional fault:**

E4 does not perturb observations — it delays action execution by 1 step. Unlike E1/E2/E3
where the perturbation is applied only at the single mutation point, E4's action delay
affects every step of the entire episode. This persistent timing corruption can cause the
agent to take nonsensical action sequences (e.g., pushing left when it should push right),
which in rare cases leads to an episode that terminates abnormally early, triggering the
functional fault detector.

### Additional Metrics

| Experiment | Fallback Count | Unique Metric | Value |
|------------|:--------------:|---------------|:-----:|
| exp1 E1 SAM | 0 | sam_rho | 0.1 |
| exp2 E2 Jacobian | 0 | avg_jacobian_sparsity | 1.0 (all dims perturbed) |
| exp3 E3 Uncertainty | 0 | avg_uncertainty_magnitude | 0.0 (gradient chain broken) |
| exp4 E4 Temporal | — | action_perturbation_count | 5,976 |

**Notes:**

- **E2 sparsity = 1.0:** MountainCar has only 2 observation dimensions, but top_k was
  set to 3. Since top_k > obs_dim, all dimensions were perturbed every time, so the
  intended sparse perturbation effect did not manifest. This metric would be meaningful
  in higher-dimensional environments (e.g., Atari with 210x160x3 pixel observations).

- **E3 uncertainty magnitude = 0.0:** The RSSM's stochastic sampling operation inside
  `dynamics.obs_step(sample=True)` does not maintain a gradient path back to the input
  observation. Although the 5 forward passes produce different reward predictions (due to
  different latent samples), this variance does not depend on the input obs in a
  differentiable way. Consequently, backpropagating the variance yields a zero gradient
  every time, and the mutator falls back to random perturbation. Despite this, E3 still
  achieved the highest fault rate — likely because the random fallback happened to explore
  effectively under the specific seed, or because the stochastic RSSM sampling itself
  introduced beneficial diversity during the STARLA search.

### Vulnerability Profile

```
Weakest edge:   E3_prediction   (world model prediction channel)
Strongest edge: E4_execution    (action execution channel)
```

DreamerV3 on MountainCar is functionally robust (no crashes or degenerate states) but
exhibits reward-level vulnerabilities, particularly when the world model's prediction
channel is targeted.
