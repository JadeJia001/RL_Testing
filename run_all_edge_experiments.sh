#!/usr/bin/env bash
###############################################################################
# run_all_edge_experiments.sh
#
# 一键完成: 训练3个DreamerV3 checkpoint + 运行3组 edge perturbation 实验
#
# 使用方法:
#   conda activate rl_testing
#   cd /path/to/RL_Testing
#   bash run_all_edge_experiments.sh
#
# 前置条件:
#   1. conda env create -f environment_gpu.yml
#   2. conda activate rl_testing
#   3. pip install -e ./STARLA[dev]
#   4. pip install -e ./DI-engine    # 或 pip install DI-engine
#   5. AutoROM --accept-license
###############################################################################
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# 确保 PYTHONPATH 包含项目根目录、DI-engine、STARLA
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/DI-engine:${PROJECT_ROOT}/STARLA/src:${PYTHONPATH:-}"

echo "============================================================"
echo " Project root: $PROJECT_ROOT"
echo " Python:       $(python --version)"
echo " PyTorch CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================================"

###############################################################################
# 可调参数 (通过环境变量覆盖)
###############################################################################
# 训练步数
BREAKOUT_TRAIN_STEPS="${BREAKOUT_TRAIN_STEPS:-500000}"
HALFCHEETAH_TRAIN_STEPS="${HALFCHEETAH_TRAIN_STEPS:-200000}"
CHEETAH_RUN_TRAIN_STEPS="${CHEETAH_RUN_TRAIN_STEPS:-100000}"

# 实验参数
SEEDS="${SEEDS:-42}"
POPULATION_SIZE="${POPULATION_SIZE:-20}"
NUM_GENERATIONS="${NUM_GENERATIONS:-20}"
TIME_BUDGET="${TIME_BUDGET:-600.0}"
TRAINING_EPISODES="${TRAINING_EPISODES:-14}"
RANDOM_EPISODES="${RANDOM_EPISODES:-14}"

echo ""
echo "=== Training Config ==="
echo "  Breakout steps:     $BREAKOUT_TRAIN_STEPS"
echo "  HalfCheetah steps:  $HALFCHEETAH_TRAIN_STEPS"
echo "  Cheetah-run steps:  $CHEETAH_RUN_TRAIN_STEPS"
echo ""
echo "=== Experiment Config ==="
echo "  Seeds:              $SEEDS"
echo "  Population:         $POPULATION_SIZE"
echo "  Generations:        $NUM_GENERATIONS"
echo "  Time budget:        $TIME_BUDGET s"
echo ""

###############################################################################
# Phase 1: 训练 DreamerV3 Checkpoints
###############################################################################
echo "============================================================"
echo " Phase 1: Training DreamerV3 Checkpoints"
echo "============================================================"

# --- 1a. Breakout ---
BREAKOUT_CKPT_DIR="$PROJECT_ROOT/experiments_atari_sam/checkpoints_breakout"
if [ -f "$BREAKOUT_CKPT_DIR/dreamerv3_policy.pth" ]; then
    echo "[Breakout] Checkpoint already exists, skipping training."
else
    echo "[Breakout] Training DreamerV3 (${BREAKOUT_TRAIN_STEPS} steps)..."
    python experiments_atari_sam/train_dreamerv3_breakout.py \
        --max-env-step "$BREAKOUT_TRAIN_STEPS" --seed 0
    echo "[Breakout] Training complete."
fi

# --- 1b. HalfCheetah ---
HALFCHEETAH_CKPT_DIR="$PROJECT_ROOT/experiments_mujoco_sam/checkpoints_halfcheetah"
if [ -f "$HALFCHEETAH_CKPT_DIR/dreamerv3_policy.pth" ]; then
    echo "[HalfCheetah] Checkpoint already exists, skipping training."
else
    echo "[HalfCheetah] Training DreamerV3 (${HALFCHEETAH_TRAIN_STEPS} steps)..."
    python experiments_mujoco_sam/train_dreamerv3_halfcheetah.py \
        --max-env-step "$HALFCHEETAH_TRAIN_STEPS" --seed 0
    echo "[HalfCheetah] Training complete."
fi

# --- 1c. Cheetah-run (DMControl) ---
CHEETAH_RUN_CKPT_DIR="$PROJECT_ROOT/experiments_edge_cheetah_run/checkpoints_cheetah_run"
if [ -f "$CHEETAH_RUN_CKPT_DIR/dreamerv3_policy.pth" ]; then
    echo "[Cheetah-run] Checkpoint already exists, skipping training."
else
    echo "[Cheetah-run] Training DreamerV3 (${CHEETAH_RUN_TRAIN_STEPS} steps)..."
    python experiments_edge_cheetah_run/train_dreamerv3_cheetah.py \
        --max-env-step "$CHEETAH_RUN_TRAIN_STEPS" --seed 0
    echo "[Cheetah-run] Training complete."
fi

echo ""
echo "[Phase 1] All checkpoints ready."
echo ""

###############################################################################
# Phase 2: 运行 Edge Perturbation 实验
###############################################################################
echo "============================================================"
echo " Phase 2: Running Edge Perturbation Experiments"
echo "============================================================"

# --- 2a. Breakout Edge Experiments ---
echo ""
echo "[Breakout] Running edge perturbation experiments..."
python -m experiments_edge_breakout.run_edge_experiments \
    --seeds "$SEEDS" \
    --population-size "$POPULATION_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --time-budget-seconds "$TIME_BUDGET" \
    --training-episodes "$TRAINING_EPISODES" \
    --random-episodes "$RANDOM_EPISODES" \
    --mutation-rate-factor 5.0 \
    --objective-thresholds 15.0,0.8,0.8 \
    --sam-rho 0.1 \
    --e2-epsilon 0.05 --e2-top-k 128 \
    --e3-epsilon 0.05 --e3-n-samples 5 \
    --e4-variant delay --e4-delay 1 --e4-alpha 0.5 --e4-freeze-prob 0.3
echo "[Breakout] Edge experiments complete."

# --- 2b. HalfCheetah Edge Experiments ---
echo ""
echo "[HalfCheetah] Running edge perturbation experiments..."
python -m experiments_edge_halfcheetah.run_edge_experiments \
    --seeds "$SEEDS" \
    --population-size "$POPULATION_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --time-budget-seconds "$TIME_BUDGET" \
    --training-episodes "$TRAINING_EPISODES" \
    --random-episodes "$RANDOM_EPISODES" \
    --mutation-rate-factor 5.0 \
    --objective-thresholds 1000.0,0.8,0.8 \
    --sam-rho 0.1 \
    --e2-epsilon 0.05 --e2-top-k 5 \
    --e3-epsilon 0.05 --e3-n-samples 5 \
    --e4-variant delay --e4-delay 1 --e4-alpha 0.5 --e4-freeze-prob 0.3
echo "[HalfCheetah] Edge experiments complete."

# --- 2c. Cheetah-run Edge Experiments ---
echo ""
echo "[Cheetah-run] Running edge perturbation experiments..."
python -m experiments_edge_cheetah_run.run_edge_experiments \
    --seeds "$SEEDS" \
    --population-size "$POPULATION_SIZE" \
    --num-generations "$NUM_GENERATIONS" \
    --time-budget-seconds "$TIME_BUDGET" \
    --training-episodes "$TRAINING_EPISODES" \
    --random-episodes "$RANDOM_EPISODES" \
    --mutation-rate-factor 5.0 \
    --objective-thresholds 400.0,0.8,0.8 \
    --sam-rho 0.1 \
    --e2-epsilon 0.05 --e2-top-k 128 \
    --e3-epsilon 0.05 --e3-n-samples 5 \
    --e4-variant delay --e4-delay 1 --e4-alpha 0.5 --e4-freeze-prob 0.3
echo "[Cheetah-run] Edge experiments complete."

###############################################################################
# Done
###############################################################################
echo ""
echo "============================================================"
echo " ALL DONE"
echo "============================================================"
echo " Results:"
echo "   Breakout:     $PROJECT_ROOT/experiments_edge_breakout/results/"
echo "   HalfCheetah:  $PROJECT_ROOT/experiments_edge_halfcheetah/results/"
echo "   Cheetah-run:  $PROJECT_ROOT/experiments_edge_cheetah_run/results/"
echo "============================================================"
