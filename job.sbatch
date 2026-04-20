#!/bin/bash
#SBATCH --job-name=20260416211900.eed90347
#SBATCH --output=/gpfs/home1/rvalperga/icil-rlbench/.smanager/scripts/icil.pretrain_perceiver_encoder_decoder_ddp/logs/20260416211900.eed90347.out
#SBATCH --error=/gpfs/home1/rvalperga/icil-rlbench/.smanager/scripts/icil.pretrain_perceiver_encoder_decoder_ddp/logs/20260416211900.eed90347.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

# Preamble (from .smanager/preamble.sh if found)

source ~/miniforge3/bin/activate
conda activate icil-rlbench

cd /gpfs/home1/rvalperga/icil-rlbench

source env_snellius.sh
source env_snellius_coppeliasim.sh

export WANDB_PROJECT="icil-perceiver-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/gpfs/home1/rvalperga/icil-rlbench
export ICIL_EVAL_OUTPUT_DIR=/gpfs/home1/rvalperga/icil-rlbench/outputs

export DISPLAY=:$((100 + SLURM_JOB_ID % 5000))


# Change to working directory
cd /gpfs/home1/rvalperga/icil-rlbench

# Run the Python script
torchrun --standalone --nproc_per_node=4 /gpfs/home1/rvalperga/icil-rlbench/icil/pretrain_perceiver_encoder_decoder_ddp.py --config=configs/pretrain_perceiver_encoder_decoder.py --config.model.encoder_name=traj_perceiver_v2 --config.model.policy.context_attention_mode=two_ctx --config.train.batch_size=32 --config.model.traj_perceiver_v2.tokenize_frames_chunked=False --config.wandb.enable=True --config.wandb.n_sample_steps=10000