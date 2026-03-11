# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Solve dm_control bug
import os
os.environ["MUJOCO_GL"] = "egl"

# PyTorch
import torch

# Functions
import functions

# Other
import os
import argparse
import importlib
import warnings
import random
import numpy as np

# Disable Warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):

    ###############################################################################
    # Init
    ###############################################################################
    
    # Set seed if provided
    if args.seed is not None:
        set_seed(args.seed)
        print(f"Set random seed to: {args.seed}")

    # Print Mode
    print("Mode: {}".format(args.mode))

    # Set environment variable for media logging
    os.environ["log_media"] = str(args.log_media).lower()

    # Load Config
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))

    # Load Model
    model = functions.load_model(args)

    # Load Dataset
    dataset_train, dataset_eval = functions.load_datasets(args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.mode == "training":

        model.fit(
            dataset_train=dataset_train, 
            epochs=getattr(args.config, "epochs", 1000), 
            dataset_eval=dataset_eval, 
            initial_epoch=int(args.checkpoint.split("_")[2]) if args.checkpoint != None else 0, 
            callback_path=args.config.callback_path,
            accumulated_steps=getattr(args.config, "accumulated_steps", 1),
            eval_period_step=getattr(args.config, "eval_period_step", args.eval_period_step),
            eval_period_epoch=getattr(args.config, "eval_period_epoch", args.eval_period_epoch),
            saving_period_epoch=getattr(args.config, "saving_period_epoch", args.saving_period_epoch) if args.save_checkpoints else None,
            log_figure_period_step=getattr(args.config, "log_figure_period_step", args.log_figure_period_step) if args.log_media else None,
            log_figure_period_epoch=getattr(args.config, "log_figure_period_epoch", args.log_figure_period_epoch) if args.log_media else None,
            step_log_period=args.step_log_period,
            detect_anomaly=getattr(args.config, "detect_anomaly", args.detect_anomaly),
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            wandb_logging=args.wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            seed=args.seed,
            verbose_progress_bar=args.verbose_progress_bar,
            keep_last_k=args.keep_last_k
        )

    # Evaluation
    elif args.mode == "evaluation":

        model._evaluate(
            dataset_eval, 
            writer=None,
            recompute_metrics=getattr(args.config, "recompute_metrics", False),
            verbose_progress_bar=args.verbose_progress_bar,
        )

    # Pass
    elif args.mode == "pass":
        pass

if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/twister.py",                                       help="Python configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training", choices=["training", "evaluation", "pass"],     help="Mode: training, validation-clean, test-clean, eval_time-dev-clean, ...")
    parser.add_argument("-i", "--checkpoint",           type=str,   default=None,                                                       help="Load model from checkpoint name")
    parser.add_argument("--cpu",                        action="store_true",                                                            help="Load model on cpu")
    parser.add_argument("--load_last",                  action="store_true",                                                            help="Load last model checkpoint")
    parser.add_argument("--wandb",                      action="store_true",                                                            help="Initialize wandb logging")
    parser.add_argument("--wandb_project",              type=str,   default="nnet",                                                     help="Wandb project name")
    parser.add_argument("--wandb_entity",               type=str,   default=None,                                                       help="Wandb entity/username")
    parser.add_argument("--verbose_progress_bar",       type=int,   default=1,                                                          help="Verbose level of progress bar display")
    parser.add_argument("--seed",                       type=int,   default=None,                                                       help="Random seed for reproducibility")
    parser.add_argument("--save_checkpoints",           action="store_true",                                                            help="Enable checkpoint saving")
    parser.add_argument("--log_media",                  action="store_true",                                                            help="Enable media (images/videos) logging")

    # Training
    parser.add_argument("--saving_period_epoch",        type=int,   default=1,                                                          help="Model saving every 'n' epochs")
    parser.add_argument("--log_figure_period_step",     type=int,   default=None,                                                       help="Log figure every 'n' steps")
    parser.add_argument("--log_figure_period_epoch",    type=int,   default=1,                                                          help="Log figure every 'n' epochs")
    parser.add_argument("--step_log_period",            type=int,   default=100,                                                        help="Training step log period")
    parser.add_argument("--keep_last_k",                type=int,   default=3,                                                          help="Keep last k checkpoints")

    # Eval
    parser.add_argument("--eval_period_epoch",          type=int,   default=5,                                                          help="Model evaluation every 'n' epochs")
    parser.add_argument("--eval_period_step",           type=int,   default=None,                                                       help="Model evaluation every 'n' steps")

    # Info
    parser.add_argument("--show_dict",                  action="store_true",                                                            help="Show model dict summary")
    parser.add_argument("--show_modules",               action="store_true",                                                            help="Show model named modules")
    
    # Debug
    parser.add_argument("--detect_anomaly",             action="store_true",                                                            help="Enable or disable the autograd anomaly detection")
    
    # SAM Optimizer
    parser.add_argument("--use_sam",                     action="store_true",                                                            help="Replace world model optimizer with sam based optimizer")
    parser.add_argument("--rho",                        type=float, default=0.05,                                                       help="Rho value to be used in the optimizer")
    parser.add_argument("--use_adaptive",               action="store_true",                                                            help="Whether to use adaptive=true in sam or not")
    
    # Parse Args
    args = parser.parse_args()

    # Run main
    main(args)
