# ICIL RLBench Perceiver

Repo for training, profiling, fine-tuning, and evaluating ICIL diffusion policies on cached RLBench point-cloud demonstrations.

## Included Workflows
- Single-GPU pretraining: `icil/pretrain_perceiver_encoder_decoder.py`
- DDP pretraining: `icil/pretrain_perceiver_encoder_decoder_ddp.py`
- Single-GPU profiling: `icil/profiling/profile_pretrain_perceiver_encoder_decoder.py`
- DDP profiling: `icil/profiling/profile_pretrain_perceiver_encoder_decoder_ddp.py`
- MAML fine-tuning from pretrained checkpoints: `icil/maml_perceiver_encoder_decoder.py`
- Single-task RLBench evaluation: `icil/eval/eval_single_task_perceiver.py`
- Test-time training / adaptation evaluation: `icil/eval/eval_perceiver_encoder_decoder_ttt.py`
- Cached-dataset inspection: `icil/inspect_icil_pretrain_cached_dataset.py`

## Setup
```bash
conda activate icil-rlbench
cd icil-rlbench
pip install -e .
source env.sh
```

`setup.py` installs the core training dependencies. Optional extras:
```bash
# wandb logging + sample plots
pip install -e ".[wandb]"

# dataset inspection script
pip install -e ".[inspect]"

# profiling progress bars
pip install -e ".[profile]"
```

## Environment Variables
- `ICIL_CACHE_ROOT`: cached RLBench dense dataset root used by training and eval configs
- `ICIL_OUTPUT_PARENT_DIR`: parent directory for training run outputs
- `ICIL_CHECKPOINT_PARENT_DIR`: parent directory for training checkpoints
- `ICIL_EVAL_OUTPUT_DIR`: parent directory for eval outputs
- `ICIL_PROFILE_TRACE_DIR`: profiling trace output directory
- `ICIL_PRETRAIN_PROFILE_TRACE_FILE`: profiling trace filename stem

If `cfg.conditioning.cache_root` is empty in eval configs, eval falls back to `checkpoint["config"]["data"]["cache_root"]`.

## Pretrain
Single GPU:
```bash
python -m icil.pretrain_perceiver_encoder_decoder \
  --config=configs/pretrain_perceiver_encoder_decoder.py
```

DDP:
```bash
torchrun --standalone --nproc_per_node=2 -m icil.pretrain_perceiver_encoder_decoder_ddp \
  --config=configs/pretrain_perceiver_encoder_decoder.py
```

CPU smoke test:
```bash
python -m icil.pretrain_perceiver_encoder_decoder \
  --config=configs/pretrain_perceiver_encoder_decoder.py \
  --config.device=cpu \
  --config.train.num_steps=1 \
  --config.train.batch_size=1
```

Important model/config knobs now exposed in `configs/pretrain_perceiver_encoder_decoder.py`:
- `cfg.model.policy.context_attention_mode = "single" | "two_ctx"`
- `cfg.model.policy.grad_checkpoint_dit`
- `cfg.model.perceiver_demo_query.checkpoint_demo_memory`
- `cfg.model.perceiver_demo_query.checkpoint_build_demo_memory`
- `cfg.model.perceiver_demo_query.checkpoint_frame_tokenizer`
- `cfg.model.perceiver_demo_query.tokenize_frames_chunked`
- the same checkpointing/chunking flags under `cfg.model.traj_perceiver.*`

## Profile
Single process:
```bash
python -m icil.profiling.profile_pretrain_perceiver_encoder_decoder \
  --train_config=configs/pretrain_perceiver_encoder_decoder.py \
  --profile_config=configs/profile_pretrain_perceiver_encoder_decoder.py
```

DDP:
```bash
torchrun --standalone --nproc_per_node=2 -m icil.profiling.profile_pretrain_perceiver_encoder_decoder_ddp \
  --train_config=configs/pretrain_perceiver_encoder_decoder.py \
  --profile_config=configs/profile_pretrain_perceiver_encoder_decoder.py
```

Profiling exports:
- Perfetto / Chrome trace: `*.json`
- Memory timeline JSON: `*.memory.json`
- Memory timeline HTML: `*.memory.html`
- Memory plot: `*.memory.png`
- Activation shape dump: `*.activation_shapes.json`
- Activation shape text summary: `*.activation_shapes.txt`

## MAML Fine-Tuning
Train from scratch or fine-tune a pretrained checkpoint:
```bash
python -m icil.maml_perceiver_encoder_decoder \
  --config=configs/maml_perceiver_encoder_decoder.py \
  --config.finetune.pretrained_checkpoint=/path/to/pretrain_checkpoint.pt
```

Quick debug run:
```bash
python -m icil.maml_perceiver_encoder_decoder \
  --config=configs/debug_maml_perceiver_encoder_decoder.py
```

Checkpoint-driven behavior in the MAML trainer:
- model config is rebuilt from the loaded checkpoint
- `dataset.L`, `dataset.T_obs`, `dataset.H`, `dataset.stride`, `data.tasks`, and `data.exclude_tasks` are taken from the checkpoint
- if `cfg.dataset.K = 0`, it resolves to `K_pretrain + 1`
- if `cfg.maml.outer_context_size = 0`, it resolves to `K_pretrain`

The inner loop uses leave-one-out diffusion-loss updates on the fast params. The trainer logs the outer loss plus the same sample diagnostics used in pretraining, and also logs the average inner-loop fast-gradient norm.

## Evaluate in Simulation
### Standard single-task perceiver eval
```bash
PYTHONUNBUFFERED=1 \
COPPELIASIM_ROOT="$HOME/CoppeliaSim" \
LD_LIBRARY_PATH="$HOME/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/CoppeliaSim" \
QT_QPA_PLATFORM=xcb \
DISPLAY=:99 \
python -u -m icil.eval.eval_single_task_perceiver \
  --config=configs/eval_single_task_perceiver.py \
  --config.checkpoint_path=/path/to/perceiver_checkpoint.pt \
  --config.task.name=put_item_in_drawer \
  --config.task.variation=0 \
  --config.task.num_eval_episodes=10 \
  --config.sim.headless=True
```

This script supports:
- `cfg.conditioning.support_source = "cache" | "live"`
- cache-root override through `ICIL_CACHE_ROOT`
- checkpoint-driven dataset config when `cfg.dataset.use_checkpoint_dataset_config = True`

### TTT eval: adapt fast params before rollout
```bash
PYTHONUNBUFFERED=1 \
COPPELIASIM_ROOT="$HOME/CoppeliaSim" \
LD_LIBRARY_PATH="$HOME/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/CoppeliaSim" \
QT_QPA_PLATFORM=xcb \
DISPLAY=:99 \
python -u -m icil.eval.eval_perceiver_encoder_decoder_ttt \
  --config=configs/eval_perceiver_encoder_decoder_ttt.py \
  --config.checkpoint_path=/path/to/perceiver_checkpoint.pt \
  --config.task.name=put_item_in_drawer \
  --config.task.variation=0 \
  --config.task.num_eval_episodes=10 \
  --config.sim.headless=True
```

TTT eval behavior:
- loads a pretrained checkpoint and rebuilds the model from the checkpoint config
- selects fast params from configurable groups under `cfg.ttt.include_*`
- performs leave-one-out diffusion-loss gradient updates on those fast params before rollout
- currently supports `cfg.conditioning.support_source = "cache"` only
- if `cfg.dataset.K = 0`, it resolves the support pool size to `K_pretrain + 1`
- if `cfg.ttt.outer_context_size = 0`, rollout uses `K_pretrain` support demos, matching the pretraining conditioning count

TTT eval outputs:
- per-episode inner-loss history: `ttt_episode_XXXX.inner_losses.json`
- per-episode log-scale loss plot: `ttt_episode_XXXX.inner_losses.png`
- videos from one or more cameras using `cfg.video.cameras`
- `mp4` and/or `gif` outputs using `cfg.video.formats`

### DP3 policy eval
```bash
PYTHONUNBUFFERED=1 \
COPPELIASIM_ROOT="$HOME/CoppeliaSim" \
LD_LIBRARY_PATH="$HOME/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/CoppeliaSim" \
QT_QPA_PLATFORM=xcb \
DISPLAY=:99 \
python -u -m icil.eval.eval_single_task_dp3 \
  --config=configs/eval_single_task_dp3.py \
  --config.checkpoint_path=/path/to/dp3_checkpoint.pt \
  --config.task.name=put_item_in_drawer \
  --config.task.variation=0 \
  --config.task.num_eval_episodes=10 \
  --config.sim.headless=True
```

If your CoppeliaSim installation is not at `$HOME/CoppeliaSim`, replace that path in the commands.

## Inspect Cached Batches
```bash
python -m icil.inspect_icil_pretrain_cached_dataset \
  --cache-root /mnt/external_storage/robotics/rlbench/icil_rlbench/.rlbench_cache_dense \
  --output-dir inspect_cached_dataset \
  --num-batches 3 \
  --batch-size 2 \
  --samples-per-batch 2 \
  --num-workers 0 \
  --H 64 \
  --stride 3
```

## Legacy Full RLBench Setup
The previous full RLBench / PyRep setup file is kept at:
- `legacy/setup_rlbench_full.py`

Use it only if you want to restore raw RLBench generation and caching workflows.

## Optional: Headless RLBench Generation
Known working command:
```bash
conda activate icil-rlbench

PYTHONUNBUFFERED=1 \
COPPELIASIM_ROOT="$HOME/CoppeliaSim" \
LD_LIBRARY_PATH="$HOME/CoppeliaSim:${LD_LIBRARY_PATH:-}" \
QT_QPA_PLATFORM_PLUGIN_PATH="$HOME/CoppeliaSim" \
QT_QPA_PLATFORM=xcb \
DISPLAY=:99 \
python -u -m rlbench.dataset_generator_pc \
  --save_path /mnt/external_storage/robotics/rlbench/icil_rlbench \
  --episodes_per_task 15 \
  --variations 15 \
  --image_size 128 128 \
  --renderer opengl \
  --processes 4 \
  --tasks beat_the_buzz change_channel ...
```

Dense caching:
```bash
python -m icil.cache_dense_icil_dataset \
  --root-raw /mnt/external_storage/robotics/rlbench/icil_rlbench \
  --root-cache /mnt/external_storage/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v2 \
  --num-points 4096 \
  --num-workers 16
```
