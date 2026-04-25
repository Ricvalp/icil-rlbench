#!/bin/bash
#SBATCH --job-name=direct_reg_supernode_absolute
#SBATCH --output=/home/rvalper/icil-rlbench/hipster/logs/direct_reg_supernode_absolute.%j.out
#SBATCH --error=/home/rvalper/icil-rlbench/hipster/logs/direct_reg_supernode_absolute.%j.err
#SBATCH --partition=performance
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH -n 128

source ~/miniconda3/bin/activate
conda activate icil-rlbench

cd /home/rvalper/icil-rlbench

source env_hipster.sh

export WANDB_PROJECT="icil-direct-regression-pretrain"
export WANDB_ENTITY="ricvalp"
export WANDB_MODE="online"

export PYTHONPATH=/home/rvalper/icil-rlbench

cd /home/rvalper/icil-rlbench

torchrun --standalone --nproc_per_node=4 \
  icil/pretrain_direct_regression_perceiver_encoder_decoder_ddp.py \
  --config=configs/pretrain_direct_regression_perceiver_encoder_decoder.py \
  --config.data.task_sampling=variation_uniform \
  --config.train.batch_size=32 \
  --config.dataset.K=2 \
  --config.model.encoder_name=traj_supernode_perceiver_v2 \
  --config.dataset.action_representation=absolute \
  --config.model.direct_regression.context_attention_mode=two_ctx \
  --config.model.direct_regression.loss_type=l1 \
  --config.model.direct_regression.attention_backend=manual \
  --config.model.traj_supernode_perceiver_v2.use_mask_id=True \
  --config.model.traj_supernode_perceiver_v2.use_mask_embedding=False \
  --config.model.traj_supernode_perceiver_v2.use_mask_instance_quota=True \
  --config.model.traj_supernode_perceiver_v2.supernode_sampling_mode=fast_random \
  --config.model.traj_supernode_perceiver_v2.tokenize_frames_chunked=True \
  --config.model.traj_supernode_perceiver_v2.compress_demo_latents=True \
  --config.model.traj_supernode_perceiver_v2.demo_supernodes=64 \
  --config.model.traj_supernode_perceiver_v2.query_supernodes=128 \
  --config.model.traj_supernode_perceiver_v2.neighbors_per_supernode=64 \
  --config.model.traj_supernode_perceiver_v2.demo_supernode_refine_layers=2 \
  --config.model.traj_supernode_perceiver_v2.query_supernode_refine_layers=2 \
  --config.model.traj_supernode_perceiver_v2.compress_supernodes_demo=False \
  --config.model.traj_supernode_perceiver_v2.compress_supernodes_query=False \
  --config.model.traj_supernode_perceiver_v2.m_traj_tokens=64 \
  --config.model.traj_supernode_perceiver_v2.traj_perceiver_layers=2 \
  --config.model.traj_supernode_perceiver_v2.include_traj_tokens=True \
  --config.model.traj_supernode_perceiver_v2.use_demo_id_embed=True \
  --config.model.traj_supernode_perceiver_v2.use_cond_state_as_traj_fallback=False
