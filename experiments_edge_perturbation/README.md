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
