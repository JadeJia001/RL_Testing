#!/bin/bash

# All Atari100k Games Parallel Training Script
# Runs each selected game with 4 seeds in parallel on 4 GPUs
# Comment out games you don't want to run on this machine

# =============================================================================
# ATARI GAMES CONFIGURATION
# Comment out (add # at the beginning) games you don't want to run
# =============================================================================

ATARI_GAMES=(
    "alien"
    "amidar"
    "assault"
    "asterix"
    "bank_heist"
    "battle_zone"
    "boxing"
    "breakout"
    "chopper_command"
    "crazy_climber"
    "demon_attack"
    "freeway"
    "frostbite"
    "gopher"
    "hero"
    "jamesbond"
    "kangaroo"
    "krull"
    "kung_fu_master"
    "ms_pacman"
    "pong"
    "private_eye"
    "qbert"
    "road_runner"
    "seaquest"
    "up_n_down"
)

# Example: To run only a subset, comment out unwanted games like this:
# ATARI_GAMES=(
#     "alien"
#     "amidar"
#     # "assault"      # Commented out - won't run
#     # "asterix"      # Commented out - won't run
#     "bank_heist"
#     # ... etc
# )

# =============================================================================
# CONFIGURATION
# =============================================================================

# Run configuration
RUN_NAME="atari100k_sweep_$(date +%Y%m%d_%H%M%S)"  # Timestamp for unique run name
SEEDS=(42 123 456 789)  # 4 different seeds
GPUS=(0 1 2 3)  # GPU IDs to use (assuming 4 GPUs per machine)

# Wandb configuration
WANDB_ENTITY="stablegradients"
WANDB_PROJECT="twister"

# Training configuration (no checkpoints, no media)
SAVE_CHECKPOINTS=""  # Disabled
LOG_MEDIA=""  # Disabled

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

echo "=================================="
echo "All Atari100k Games Training Script"
echo "=================================="
echo "Selected games: ${#ATARI_GAMES[@]}"
echo "Games to run:"
for game in "${ATARI_GAMES[@]}"; do
    echo "  - $game"
done
echo ""
echo "Run name: $RUN_NAME"
echo "Seeds per game: ${SEEDS[@]}"
echo "GPUs: ${GPUS[@]}"
echo "Wandb Entity: $WANDB_ENTITY"
echo "Wandb Project: $WANDB_PROJECT"
echo "=================================="

# Function to run training for a specific game and seed on a specific GPU
run_game_training() {
    local game=$1
    local gpu_id=$2
    local seed=$3
    local env_name="atari100k-${game}"
    
    echo "🚀 Starting $game (seed $seed) on GPU $gpu_id..."
    
    # Set environment variables and run training
    CUDA_VISIBLE_DEVICES=$gpu_id \
    env_name=$env_name \
    run_name=$RUN_NAME \
    python3 main.py \
        --wandb \
        --wandb_entity "$WANDB_ENTITY" \
        --wandb_project "$WANDB_PROJECT" \
        --seed $seed \
        $SAVE_CHECKPOINTS \
        $LOG_MEDIA \
        --mode training \
        --verbose_progress_bar 1
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ $game (seed $seed) on GPU $gpu_id completed successfully"
    else
        echo "❌ $game (seed $seed) on GPU $gpu_id failed"
    fi
    
    return $exit_code
}

# Function to run all seeds for a single game in parallel
run_game_all_seeds() {
    local game=$1
    echo ""
    echo "Starting game: $game"
    echo "Running 4 seeds in parallel on 4 GPUs..."
    
    local game_pids=()
    
    # Launch all 4 seeds for this game in parallel
    for i in "${!SEEDS[@]}"; do
        seed=${SEEDS[$i]}
        gpu_id=${GPUS[$i]}
        
        run_game_training $game $gpu_id $seed &
        game_pids+=($!)
        
        # Small delay to avoid overwhelming
        sleep 1
    done
    
    # Wait for all seeds of this game to complete
    local failed_seeds=0
    for i in "${!game_pids[@]}"; do
        pid=${game_pids[$i]}
        seed=${SEEDS[$i]}
        
        wait $pid
        exit_code=$?
        
        if [ $exit_code -ne 0 ]; then
            ((failed_seeds++))
        fi
    done
    
    if [ $failed_seeds -eq 0 ]; then
        echo "🎉 Game $game completed successfully (all 4 seeds)"
    else
        echo "⚠️  Game $game completed with $failed_seeds failed seeds"
    fi
    
    return $failed_seeds
}

# Main execution
echo ""
echo "Starting training for ${#ATARI_GAMES[@]} games..."
echo "Each game will run 4 seeds in parallel"
echo ""

total_games=${#ATARI_GAMES[@]}
completed_games=0
failed_games=0

# Process each game sequentially (but seeds within each game run in parallel)
for game in "${ATARI_GAMES[@]}"; do
    echo "[$((completed_games + failed_games + 1))/$total_games] Processing game: $game"
    
    run_game_all_seeds $game
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        ((completed_games++))
    else
        ((failed_games++))
    fi
    
    echo "Progress: $((completed_games + failed_games))/$total_games games processed"
    
    # Small break between games
    sleep 2
done

# Final summary
echo ""
echo "=================================="
echo "FINAL SUMMARY"
echo "=================================="
echo "Total games: $total_games"
echo "Successful games: $completed_games"
echo "Failed games: $failed_games"
echo "Total experiments: $((total_games * 4)) (4 seeds per game)"

if [ $failed_games -eq 0 ]; then
    echo ""
    echo "🎉 All games completed successfully!"
    echo ""
    echo "Results available at:"
    echo "- Wandb: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
    echo "- Local logs: callbacks/$RUN_NAME/"
else
    echo ""
    echo "⚠️  Some games failed. Check wandb for details."
fi

echo ""
echo "Run completed at: $(date)"
echo "==================================" 