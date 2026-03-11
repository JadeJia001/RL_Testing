#!/bin/bash

export HYDRA_FULL_ERROR=1
# SAM rho ablation study

CUDA_VISIBLE_DEVICES=0 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.00-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=0 sam_rho=0.00 &
CUDA_VISIBLE_DEVICES=1 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.00-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=1 sam_rho=0.00 &
CUDA_VISIBLE_DEVICES=2 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.00-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=2 sam_rho=0.00 &
CUDA_VISIBLE_DEVICES=3 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.00-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=3 sam_rho=0.00 &


CUDA_VISIBLE_DEVICES=0 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.005-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=0 sam_rho=0.005 &
CUDA_VISIBLE_DEVICES=1 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.005-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=1 sam_rho=0.005 &
CUDA_VISIBLE_DEVICES=2 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.005-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=2 sam_rho=0.005 &
CUDA_VISIBLE_DEVICES=3 python -m tdmpc2.train disable_wandb=False wandb_entity=stablegradients exp_name=tdmpc-sam-rho-0.005-value-calc-fig-1 task=humanoid_h1hand-run-v0 seed=3 sam_rho=0.005  
