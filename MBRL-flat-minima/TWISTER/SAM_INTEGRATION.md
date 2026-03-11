# SAM Optimizer Integration for TWISTER World Model

This document describes the implementation of Sharpness-Aware Minimization (SAM) optimizer integration for the TWISTER world model.

## Overview

The SAM optimizer has been integrated to replace the default Adam optimizer for the world model when enabled via command line arguments. This implementation provides:

- ✅ **Seamless SAM Integration**: Drop-in replacement for world model optimizer
- ✅ **Clear User Feedback**: Large message when SAM is enabled
- ✅ **Wandb Integration**: Automatic naming with SAM parameters
- ✅ **Configurable Parameters**: Customizable rho and adaptive settings
- ✅ **Backward Compatibility**: Existing code continues to work unchanged

## Command Line Arguments

### New SAM-related Arguments

```bash
--use_sam                    # Enable SAM optimizer for world model
--rho FLOAT                  # Rho value (default: 0.05)
--use_adaptive               # Enable adaptive SAM (default: False)
```

### Usage Examples

```bash
# Basic SAM usage with default parameters
env_name=atari100k-alien run_name=sam_experiment python3 main.py --use_sam

# Custom rho value
env_name=atari100k-alien run_name=sam_experiment python3 main.py --use_sam --rho 0.1

# SAM with adaptive and custom rho
env_name=atari100k-alien run_name=sam_experiment python3 main.py --use_sam --rho 0.1 --use_adaptive

# Full training with wandb logging
env_name=atari100k-alien run_name=sam_experiment python3 main.py \
  --use_sam --rho 0.1 --use_adaptive \
  --wandb --wandb_project "twister_sam" --wandb_entity "your_username" \
  --mode training --save_checkpoints
```

## Implementation Details

### 1. Configuration Integration (`configs/twister.py`)

- **SAM Parameter Extraction**: Parses command line arguments for SAM settings
- **Clear Visual Feedback**: Displays prominent message when SAM is enabled
- **Config Integration**: Adds SAM parameters to model configuration

```python
# SAM activation message
🔥 SAM OPTIMIZER ENABLED FOR WORLD MODEL 🔥
SAM Configuration:
  - rho: 0.1
  - adaptive: True
==================================================
```

### 2. Model Integration (`nnet/models/twister.py`)

#### Configuration Parameters
- `config.use_sam`: Boolean flag for SAM usage
- `config.rho`: Rho value for SAM (default: 0.05)
- `config.use_adaptive`: Adaptive SAM flag (default: False)

#### Optimizer Selection
The `compile()` method automatically selects between Adam and SAM:

```python
if self.config.use_sam:
    # SAM optimizer with Adam as base
    world_model_optimizer = SAM(
        params=[...],
        base_optimizer=optimizers.Adam,
        rho=self.config.rho,
        adaptive=self.config.use_adaptive,
        weight_decay=self.config.opt_weight_decay
    )
else:
    # Regular Adam optimizer
    world_model_optimizer = optimizers.Adam(...)
```

### 3. Training Step Override

Custom `train_step` method in `WorldModel` class handles SAM's two-step optimization:

1. **First Step**: Standard forward-backward pass to compute gradients
2. **SAM Step**: Uses closure function for second forward-backward pass
3. **Gradient Scaling**: Properly handles mixed precision training

### 4. Wandb Integration (`nnet/models/model.py`)

Automatic naming convention when SAM is used:

- **Group Name**: `<experiment_name>/<game_name>_SAM_rho_<rho_value>[_adaptive]`
- **Run Name**: `[seed_<seed>_]<group_name>`

Examples:
- **Without SAM**: 
  - Group: `my_experiment/atari100k-alien`
  - Run: `seed_42_my_experiment/atari100k-alien`

- **With SAM (rho=0.1)**: 
  - Group: `my_experiment/atari100k-alien_SAM_rho_0.1`
  - Run: `seed_42_my_experiment/atari100k-alien_SAM_rho_0.1`

- **With SAM adaptive (rho=0.05)**: 
  - Group: `my_experiment/atari100k-alien_SAM_rho_0.05_adaptive`
  - Run: `seed_42_my_experiment/atari100k-alien_SAM_rho_0.05_adaptive`

- **Just game name with SAM**: 
  - Group: `atari100k-alien_SAM_rho_0.1_adaptive`
  - Run: `atari100k-alien_SAM_rho_0.1_adaptive`

## Technical Implementation

### SAM Optimizer Features

- **Base Optimizer**: Uses Adam as the base optimizer
- **Rho Parameter**: Controls perturbation magnitude (default: 0.05)
- **Adaptive Mode**: Uses parameter-dependent perturbation scaling
- **Mixed Precision**: Full compatibility with gradient scaling

### Training Loop Integration

The SAM optimizer requires a closure function that performs:
1. Zero gradients
2. Forward pass
3. Loss computation
4. Backward pass

This is seamlessly handled in the custom `train_step` method.

### Memory and Performance

- **Memory Usage**: Approximately 2x forward passes per training step
- **Computation**: ~2x training time due to dual forward-backward passes
- **Compatibility**: Works with all existing TWISTER features

## Verification

The implementation has been tested with:

- ✅ Compilation without SAM (baseline)
- ✅ Compilation with SAM
- ✅ SAM message display
- ✅ Parameter parsing
- ✅ Wandb naming convention

## Benefits of SAM for World Models

1. **Improved Generalization**: SAM seeks parameters in flatter loss regions
2. **Better Convergence**: More robust optimization for complex world models
3. **Enhanced Stability**: Reduced sensitivity to initialization and hyperparameters

## Notes and Considerations

- **Training Time**: Expect ~2x longer training time due to dual forward passes
- **Memory Usage**: Slightly increased memory usage for gradient computation
- **Hyperparameter Tuning**: Start with rho=0.05, adjust based on results
- **Adaptive Mode**: Consider enabling for very large models or when using different learning rates for different parameter groups

## Migration Guide

Existing TWISTER users can enable SAM by simply adding `--use_sam` to their command line:

```bash
# Original command
env_name=atari100k-alien run_name=my_experiment python3 main.py --wandb

# With SAM
env_name=atari100k-alien run_name=my_experiment python3 main.py --wandb --use_sam
```

No code changes are required for existing configurations or scripts. 