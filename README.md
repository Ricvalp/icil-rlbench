# ICIL Perceiver Pretraining

Minimal repo for pretraining and profiling the ICIL Perceiver encoder-decoder diffusion policy from an already cached dense dataset.

## Included Components
- `icil/pretrain_perceiver_encoder_decoder.py`
- `icil/profile_pretrain_perceiver_encoder_decoder.py`
- `icil/inspect_icil_pretrain_cached_dataset.py`
- `icil/models/perceiver_encoder_decoder.py`
- `icil/datasets/in_context_imitation_learning/`
- `configs/pretrain_perceiver_encoder_decoder.py`
- `configs/profile_pretrain_perceiver_encoder_decoder.py`
- `env.sh`

## Setup (Pretraining Only)
```bash
conda activate icil-rlbench
cd icil-rlbench
pip install -e .
source env.sh
```

`setup.py` now installs only pretraining dependencies:
- `numpy`
- `torch`
- `absl-py`
- `ml-collections`
- `h5py`

Optional extras:
```bash
# wandb logging + sample plots
pip install -e ".[wandb]"

# dataset inspection script
pip install -e ".[inspect]"

# profiling script progress bars
pip install -e ".[profile]"
```

## Pretrain
```bash
python -m icil.pretrain_perceiver_encoder_decoder \
  --config=configs/pretrain_perceiver_encoder_decoder.py
```

Smoke test:
```bash
python -m icil.pretrain_perceiver_encoder_decoder \
  --config=configs/pretrain_perceiver_encoder_decoder.py \
  --config.device=cpu \
  --config.train.num_steps=1 \
  --config.train.batch_size=1
```

## Profile (Perfetto JSON)
```bash
python -m icil.profile_pretrain_perceiver_encoder_decoder \
  --train_config=configs/pretrain_perceiver_encoder_decoder.py \
  --profile_config=configs/profile_pretrain_perceiver_encoder_decoder.py
```

Trace output path is controlled by:
- `ICIL_PROFILE_TRACE_DIR`
- `ICIL_PRETRAIN_PROFILE_TRACE_FILE`

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

## Evaluate in Simulation (Perceiver and DP3)
### Perceiver policy
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

### DP3 policy
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

## Legacy Full RLBench Setup
The previous full RLBench/PyRep setup file is kept at:
- `legacy/setup_rlbench_full.py`

Use it only if you want to restore raw RLBench generation/caching workflows.

## Optional: Headless RLBench Generation (Known Working Command)
If you need to regenerate raw RLBench data on a headless machine, the following command has been verified to work:

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

If your CoppeliaSim installation is not at `$HOME/CoppeliaSim`, replace that path in the command.








python -m icil.cache_dense_icil_dataset \
--root-raw /mnt/external_storage/robotics/rlbench/icil_rlbench \
--root-cache /mnt/external_storage/robotics/rlbench/icil_rlbench/.rlbench_cache_dense_v2 \
--num-points 4096 \
--num-workers 16