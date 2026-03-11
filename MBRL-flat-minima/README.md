# MBRL-flat-minima

Official code repository for the NeurIPS paper: **"Improving Model-Based Reinforcement Learning by Converging to Flatter Minima"**

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
  - [TWISTER](#twister)
  - [TD-MPC2](#td-mpc2)
  - [HumanoidBench](#humanoidbench)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

## Overview

This repository contains the implementation and experimental code for our NeurIPS paper on improving model-based reinforcement learning (MBRL) by converging to flatter minima. Our work demonstrates that optimization strategies targeting flatter loss landscapes can significantly enhance the performance and generalization of model-based RL agents.

The repository includes three main experimental frameworks:
1. **TWISTER** - Transformer-based World Models with Contrastive Predictive Coding
2. **TD-MPC2** - Scalable, Robust World Models for Continuous Control
3. **HumanoidBench** - Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation

## Repository Structure

```
MBRL-flat-minima/
├── TWISTER/              # Atari 100k and DMControl experiments
├── tdmpc2/               # Continuous control experiments
├── humanoid-bench/       # Humanoid robot experiments
└── README.md            # This file
```

Each experiment directory is a self-contained environment with its own dependencies, configuration files, and training scripts.

## Experiments

### TWISTER

**Learning Transformer-based World Models with Contrastive Predictive Coding**

TWISTER implements a Transformer-based model-based RL algorithm using action-conditioned Contrastive Predictive Coding (AC-CPC) to learn high-level feature representations. This experiment framework is used for:
- **Atari 100k Benchmark**: 26 discrete action space games
- **DeepMind Control Suite**: Continuous control tasks

**Key Features:**
- Transformer-based world model architecture
- Contrastive predictive coding for representation learning
- Support for both discrete and continuous action spaces
- Integrated SAM (Sharpness-Aware Minimization) optimization

**Location:** `TWISTER/`

### TD-MPC2

**Temporal Difference Learning for Model Predictive Control (Version 2)**

TD-MPC2 is a scalable, robust model-based RL algorithm that demonstrates strong performance across 104 continuous control tasks spanning multiple domains. This framework supports:
- **DMControl**: 39 tasks including custom environments
- **Meta-World**: 50 manipulation tasks
- **ManiSkill2**: 5 robotic manipulation tasks
- **MyoSuite**: 10 musculoskeletal control tasks

**Key Features:**
- Single set of hyperparameters across all tasks
- Multi-task offline RL capabilities
- Support for both state and pixel observations
- Scalable model architectures (1M to 317M parameters)

**Location:** `tdmpc2/`

### HumanoidBench

**Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation**

HumanoidBench provides a comprehensive benchmark for humanoid robot control with 15 whole-body manipulation tasks and 12 locomotion tasks. This framework includes:
- H1 humanoid robot with dexterous hands
- Unitree G1 robot variants
- Support for hierarchical policies
- Multiple sensing modalities (proprio, visual, tactile)

**Key Features:**
- 27+ benchmark tasks for humanoid control
- Low-level skill policies for reaching and manipulation
- Support for TD-MPC2, DreamerV3, SAC, and PPO algorithms
- Hierarchical policy training capabilities

**Location:** `humanoid-bench/`

## Installation

**IMPORTANT:** Each experiment has its own dependencies and installation requirements. You **must** navigate into each experiment directory and follow its specific installation instructions.

### General Prerequisites

- Linux operating system (tested on Ubuntu 18.04+)
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM for single-task, 24GB+ for multi-task)
- Python 3.8+ (specific version requirements vary by experiment)
- Git

### Per-Experiment Installation

#### 1. TWISTER Installation

Navigate to the TWISTER directory and run the installation script:

```bash
cd TWISTER
./install.sh
```

This will set up a conda environment with all necessary dependencies for Atari and DMControl experiments.

**For detailed installation instructions, see:** [TWISTER/README.md](TWISTER/README.md)

#### 2. TD-MPC2 Installation

Navigate to the tdmpc2 directory and choose your installation method:

**Option A: Using Docker (Recommended)**
```bash
cd tdmpc2
cd docker && docker build . -t <user>/tdmpc2:1.0.1
```

**Option B: Using Conda**
```bash
cd tdmpc2
conda env create -f docker/environment.yaml
```

**Additional Setup for Specific Domains:**
- **ManiSkill2**: Download assets with `python -m mani_skill2.utils.download_asset all`
- **Meta-World**: Download MuJoCo license from [https://www.tdmpc2.com/files/mjkey.txt](https://www.tdmpc2.com/files/mjkey.txt)

**For detailed installation instructions, see:** [tdmpc2/README.md](tdmpc2/README.md)

#### 3. HumanoidBench Installation

Navigate to the humanoid-bench directory and install the package:

```bash
cd humanoid-bench

# Create and activate conda environment
conda create -n humanoidbench python=3.11
conda activate humanoidbench

# Install HumanoidBench
pip install -e .

# Install JAX (GPU or CPU version)
pip install "jax[cuda12]==0.4.28"  # For GPU
# OR
pip install "jax[cpu]==0.4.28"     # For CPU

# Install framework-specific requirements
pip install -r requirements_jaxrl.txt    # For SAC
pip install -r requirements_dreamer.txt  # For DreamerV3
pip install -r requirements_tdmpc.txt    # For TD-MPC2
```

**For detailed installation instructions, see:** [humanoid-bench/README.md](humanoid-bench/README.md)

## Usage

Each experiment directory contains its own training and evaluation scripts with specific command-line arguments and configurations.

### Running TWISTER Experiments

```bash
cd TWISTER

# Train on Atari 100k
env_name=atari100k-alien run_name=atari100k python3 main.py

# Train on DMControl
env_name=dmc-Acrobot-swingup run_name=dmc python3 main.py

# Evaluate trained model
env_name=atari100k-alien run_name=atari100k python3 main.py --load_last --mode evaluation
```

### Running TD-MPC2 Experiments

```bash
cd tdmpc2

# Train on single task
python train.py task=dog-run steps=7000000

# Train on multi-task dataset
python train.py task=mt80 model_size=48 batch_size=1024

# Evaluate checkpoint
python evaluate.py task=dog-run checkpoint=/path/to/checkpoint.pt save_video=true
```

### Running HumanoidBench Experiments

```bash
cd humanoid-bench

# Set task
export TASK="h1hand-walk-v0"

# Train with TD-MPC2
python -m tdmpc2.train disable_wandb=False exp_name=tdmpc task=humanoid_${TASK} seed=0

# Train with DreamerV3
python -m embodied.agents.dreamerv3.train --configs humanoid_benchmark --method dreamer --logdir logs --task humanoid_${TASK} --seed 0

# Train with SAC
python ./jaxrl_m/examples/mujoco/run_mujoco_sac.py --env_name ${TASK} --seed 0

# Test environment
python -m humanoid_bench.test_env --env ${TASK}
```

**For detailed usage instructions and hyperparameters, refer to the README in each experiment directory.**

## Flat Minima Integration

Our paper introduces methods to improve MBRL by converging to flatter minima. The integration of Sharpness-Aware Minimization (SAM) and other flatness-seeking optimization strategies are documented in:

- **TWISTER:** See [TWISTER/SAM_INTEGRATION.md](TWISTER/SAM_INTEGRATION.md) and [TWISTER/SAM_USAGE_GUIDE.md](TWISTER/SAM_USAGE_GUIDE.md)
- **TD-MPC2:** See [tdmpc2/SAM_USAGE_EXAMPLE.md](tdmpc2/SAM_USAGE_EXAMPLE.md)
- **HumanoidBench:** SAM integration is available across all supported training frameworks

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@inproceedings{MBRL-flat-minima,
  title={Improving Model-Based Reinforcement Learning by Converging to Flatter Minima},
  author={[Authors]},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

Please also consider citing the original papers for the frameworks used:

**TWISTER:**
```bibtex
@inproceedings{burchilearning,
  title={Learning Transformer-based World Models with Contrastive Predictive Coding},
  author={Burchi, Maxime and Timofte, Radu},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

**TD-MPC2:**
```bibtex
@inproceedings{hansen2024tdmpc2,
  title={TD-MPC2: Scalable, Robust World Models for Continuous Control},
  author={Nicklas Hansen and Hao Su and Xiaolong Wang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

**HumanoidBench:**
```bibtex
@article{sferrazza2024humanoidbench,
    title={HumanoidBench: Simulated Humanoid Benchmark for Whole-Body Locomotion and Manipulation},
    author={Carmelo Sferrazza and Dun-Ming Huang and Xingyu Lin and Youngwoon Lee and Pieter Abbeel},
    journal={arXiv Preprint arxiv:2403.10506},
    year={2024}
}
```

## License

This project contains code from multiple sources, each with their own licenses:
- **TWISTER**: See [TWISTER/LICENSE](TWISTER/LICENSE)
- **TD-MPC2**: See [tdmpc2/LICENSE](tdmpc2/LICENSE)
- **HumanoidBench**: See [humanoid-bench/LICENSE](humanoid-bench/LICENSE)

Please refer to the LICENSE file in each experiment directory for specific licensing information.

## Repository Tracking

This repository tracks the main project at: [https://github.com/autonlab/MBRL-flat-minima.git](https://github.com/autonlab/MBRL-flat-minima.git)

The three experiment directories (TWISTER, tdmpc2, humanoid-bench) are included as regular directories (not submodules) to simplify the installation and usage process.

## Acknowledgments

We thank the authors of TWISTER, TD-MPC2, and HumanoidBench for their excellent codebases that formed the foundation for our experiments.

## Contact

For questions about this work, please open an issue on the [GitHub repository](https://github.com/autonlab/MBRL-flat-minima).
