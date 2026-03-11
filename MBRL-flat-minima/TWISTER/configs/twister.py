import nnet
import os
import json
import sys

# Extract params from filename
env_name = os.environ["env_name"]
print("TWISTER selected env_name: {}".format(env_name))

# Extract SAM parameters from command line arguments
sam_config = {}
if "--use_sam" in sys.argv:
    sam_config["use_sam"] = True
    
    # Extract rho value
    if "--rho" in sys.argv:
        rho_idx = sys.argv.index("--rho") + 1
        if rho_idx < len(sys.argv):
            sam_config["rho"] = float(sys.argv[rho_idx])
    else:
        sam_config["rho"] = 0.05  # default value
    
    # Extract adaptive setting
    sam_config["use_adaptive"] = "--use_adaptive" in sys.argv
    
    print("🔥 SAM OPTIMIZER ENABLED FOR WORLD MODEL 🔥")
    print(f"SAM Configuration:")
    print(f"  - rho: {sam_config['rho']}")
    print(f"  - adaptive: {sam_config['use_adaptive']}")
    print("=" * 50)

# Override Config
override_config = os.environ.get("override_config", {})
if isinstance(override_config, str):
    override_config = json.loads(override_config)

# Add SAM configuration to override_config
override_config.update(sam_config)

print("override_config:", override_config)

# Check if media logging is disabled
log_media = os.environ.get("log_media", "false").lower() == "true"
if not log_media:
    # Disable video saving by ensuring episode_saving_path is None for both training and evaluation
    if "train_env_params" not in override_config:
        override_config["train_env_params"] = {}
    if "eval_env_params" not in override_config:
        override_config["eval_env_params"] = {}
    
    # Explicitly set episode_saving_path to None to disable video saving
    if "episode_saving_path" not in override_config["train_env_params"]:
        override_config["train_env_params"]["episode_saving_path"] = None
    if "episode_saving_path" not in override_config["eval_env_params"]:
        override_config["eval_env_params"]["episode_saving_path"] = None

# Model
model = nnet.models.TWISTER(env_name=env_name, override_config=override_config)
model.compile()

# Training
epochs = model.config.epochs
epoch_length = model.config.epoch_length

# Callback Path
if os.environ.get("run_name", False):
    callback_path = "callbacks/{}/{}".format(os.environ["run_name"], env_name)
else:
    callback_path = "callbacks/{}".format(env_name)

# Replay Buffer
training_dataset = nnet.datasets.ReplayBuffer(
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L
)
model.set_replay_buffer(training_dataset)

# Evaluation Dataset
evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_episodes)
