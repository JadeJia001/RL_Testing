# Machine Setup Examples for Atari100k Training

This file shows how to configure `run_all_atari_parallel.sh` for different machines running subsets of Atari games.

## Machine 1: First 10 games

Edit the `ATARI_GAMES` array in `run_all_atari_parallel.sh`:

```bash
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
    # "demon_attack"     # Commented out for other machines
    # "freeway"          # Commented out for other machines
    # "frostbite"        # Commented out for other machines
    # "gopher"           # Commented out for other machines
    # "hero"             # Commented out for other machines
    # "jamesbond"        # Commented out for other machines
    # "kangaroo"         # Commented out for other machines
    # "krull"            # Commented out for other machines
    # "kung_fu_master"   # Commented out for other machines
    # "ms_pacman"        # Commented out for other machines
    # "pong"             # Commented out for other machines
    # "private_eye"      # Commented out for other machines
    # "qbert"            # Commented out for other machines
    # "road_runner"      # Commented out for other machines
    # "seaquest"         # Commented out for other machines
    # "up_n_down"        # Commented out for other machines
)
```

## Machine 2: Next 8 games

```bash
ATARI_GAMES=(
    # "alien"            # Commented out - running on machine 1
    # "amidar"           # Commented out - running on machine 1
    # "assault"          # Commented out - running on machine 1
    # "asterix"          # Commented out - running on machine 1
    # "bank_heist"       # Commented out - running on machine 1
    # "battle_zone"      # Commented out - running on machine 1
    # "boxing"           # Commented out - running on machine 1
    # "breakout"         # Commented out - running on machine 1
    # "chopper_command"  # Commented out - running on machine 1
    # "crazy_climber"    # Commented out - running on machine 1
    "demon_attack"
    "freeway"
    "frostbite"
    "gopher"
    "hero"
    "jamesbond"
    "kangaroo"
    "krull"
    # "kung_fu_master"   # Commented out for machine 3
    # "ms_pacman"        # Commented out for machine 3
    # "pong"             # Commented out for machine 3
    # "private_eye"      # Commented out for machine 3
    # "qbert"            # Commented out for machine 3
    # "road_runner"      # Commented out for machine 3
    # "seaquest"         # Commented out for machine 3
    # "up_n_down"        # Commented out for machine 3
)
```

## Machine 3: Last 8 games

```bash
ATARI_GAMES=(
    # "alien"            # Commented out - running on machine 1
    # "amidar"           # Commented out - running on machine 1
    # "assault"          # Commented out - running on machine 1
    # "asterix"          # Commented out - running on machine 1
    # "bank_heist"       # Commented out - running on machine 1
    # "battle_zone"      # Commented out - running on machine 1
    # "boxing"           # Commented out - running on machine 1
    # "breakout"         # Commented out - running on machine 1
    # "chopper_command"  # Commented out - running on machine 1
    # "crazy_climber"    # Commented out - running on machine 1
    # "demon_attack"     # Commented out - running on machine 2
    # "freeway"          # Commented out - running on machine 2
    # "frostbite"        # Commented out - running on machine 2
    # "gopher"           # Commented out - running on machine 2
    # "hero"             # Commented out - running on machine 2
    # "jamesbond"        # Commented out - running on machine 2
    # "kangaroo"         # Commented out - running on machine 2
    # "krull"            # Commented out - running on machine 2
    "kung_fu_master"
    "ms_pacman"
    "pong"
    "private_eye"
    "qbert"
    "road_runner"
    "seaquest"
    "up_n_down"
)
```

## Quick Configuration Tips

1. **Copy the script to each machine** and edit the `ATARI_GAMES` array
2. **Comment out games** by adding `#` at the beginning of the line
3. **Verify your selection** by running `./run_all_atari_parallel.sh` - it will show you which games are selected before starting
4. **All games will be grouped together in wandb** regardless of which machine runs them

## Expected Results

- **Total experiments**: 26 games × 4 seeds = 104 experiments
- **Machine 1**: 10 games × 4 seeds = 40 experiments  
- **Machine 2**: 8 games × 4 seeds = 32 experiments
- **Machine 3**: 8 games × 4 seeds = 32 experiments
- **Wandb project**: All results under `stablegradients/twister`
- **Grouping**: Each game will have its 4 seeds grouped together 