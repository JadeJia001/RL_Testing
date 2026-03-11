# Quick SAM Usage Guide

## Basic SAM Usage

```bash
# Enable SAM with default settings (rho=0.05, adaptive=False)
env_name=atari100k-alien run_name=my_sam_exp python3 main.py --use_sam

# Enable SAM with custom rho
env_name=atari100k-alien run_name=my_sam_exp python3 main.py --use_sam --rho 0.1

# Enable SAM with adaptive mode
env_name=atari100k-alien run_name=my_sam_exp python3 main.py --use_sam --use_adaptive

# Full SAM configuration
env_name=atari100k-alien run_name=my_sam_exp python3 main.py --use_sam --rho 0.1 --use_adaptive
```

## With Wandb Integration

```bash
# SAM with wandb (automatic naming: group and run will include game name and "SAM_rho_0.1_adaptive")
env_name=atari100k-alien run_name=my_sam_exp python3 main.py \
  --use_sam --rho 0.1 --use_adaptive \
  --wandb --wandb_project "twister_sam" --wandb_entity "your_username"
```

## Visual Feedback

When SAM is enabled, you'll see:

```
🔥 SAM OPTIMIZER ENABLED FOR WORLD MODEL 🔥
SAM Configuration:
  - rho: 0.1
  - adaptive: True
==================================================
```

And during compilation:

```
🔥 USING SAM OPTIMIZER FOR WORLD MODEL 🔥
SAM Parameters: rho=0.1, adaptive=True
```

## Command Line Arguments

- `--use_sam`: Enable SAM optimizer for world model
- `--rho FLOAT`: Rho value (default: 0.05)
- `--use_adaptive`: Enable adaptive SAM (default: False)

## Notes

- SAM will approximately double training time due to dual forward passes
- Only applies to the world model optimizer (actor and critic still use Adam)
- Fully compatible with all existing TWISTER features
- Wandb runs will automatically be named with game name and SAM parameters for easy identification

## Wandb Naming Examples

- **Without SAM**: Group `my_experiment/atari100k-alien`, Run `seed_42_my_experiment/atari100k-alien`
- **With SAM**: Group `my_experiment/atari100k-alien_SAM_rho_0.1_adaptive`, Run `seed_42_my_experiment/atari100k-alien_SAM_rho_0.1_adaptive` 