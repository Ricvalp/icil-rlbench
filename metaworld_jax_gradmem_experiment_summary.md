# MetaWorld JAX GradMem Experiment Summary

This document summarizes the MetaWorld/JAX query-memory MAML experiments tried so far, what was implemented, what worked, what failed, and what should be tried next. It is written to be consumed by an LLM/code agent as context for future work.

## TL;DR

We implemented a JAX MetaWorld direct-regression MAML policy where a decoder predicts action chunks from query observations plus trainable memory tokens. Inner-loop gradient descent updates only memory tokens using support demonstrations. The system can learn strong in-distribution adaptation on training task families, and correct same-family support often beats wrong-family support on train. However, it does not reliably generalize to held-out MetaWorld task families. On held-out tasks, same-family adaptation is often no better than no adaptation and sometimes worse; wrong-family adaptation is often close to or better than same-family. This suggests the current method learns train-family-specific memory-writing behavior rather than a transferable support-conditioned control rule.

The most important next direction is probably not another small hyperparameter sweep. The likely missing piece is a better state/support representation, especially object-centric tokenization or an explicit support encoder. Current flat 39D state plus gradient-only memory writes appears insufficient for OOD task-family generalization.

## Current Code/Experiment Line

Relevant directories and scripts:

- JAX query-memory code: `icil_jax_query_memory/`
- MetaWorld data/cache pipeline: `icil_metaworld/`
- MAML training script: `icil_jax_query_memory/maml_metaworld_query_memory_write_read_direct_regression.py`
- Main MetaWorld WRITE/READ config: `configs/jax_metaworld_maml_query_memory_write_read_direct_regression.py`
- Wrong-support-margin config: `configs/jax_metaworld_maml_query_memory_write_read_wrong_support_margin_direct_regression.py`
- MSE diagnostic: `diagnostics/jax_metaworld_adaptation_mse_diagnostic.py`
- Rollout diagnostic: `diagnostics/jax_metaworld_adaptation_rollout_diagnostic.py`
- Snellius sbatches for current proposed follow-ups: `sbatch/snellius/maml_metaworld_*.sbatch`
- JAX checkpoint fetch helper: `./get_jax_ckpt`

## Dataset Semantics

The current MetaWorld cache used for most experiments is:

```text
ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT=/mnt/external_storage/robotics/metaworld/icil_metaworld/ml45_goal_train_50x1
ICIL_METAWORLD_ML45_GOAL_TEST_CACHE_ROOT=/mnt/external_storage/robotics/metaworld/icil_metaworld/ml45_goal_test_50x1
```

Important semantics:

- The cache contains scripted-policy demonstrations from MetaWorld.
- We moved away from sampling multiple byte-identical episodes from the same task instance.
- The intended meta-task is now a MetaWorld task family, not an exact task instance.
- Support and query are sampled from the same task family but usually different task instances/goals:

```text
sample_same_task_name=True
sample_same_task_instance=False
```

- This means the model is asked to adapt from support demos from the same family but different goal/instance.
- For OOD evaluation, train uses ML45 train task families, test uses held-out ML45 test task families.

### State Inputs

For goal-cache datasets, the model sees a 39D state unless goal is zeroed:

```text
36D raw MetaWorld obs_model + 3D goal
```

If `query_zero_goal=True`, the last 3 goal dimensions in the query state are set to zero. If `support_zero_goal=False`, the support demonstrations still include goal coordinates.

Query-goal dropout is different from zeroing:

```text
query_zero_goal=False
query_goal_dropout_rate > 0
```

This keeps the goal in the stored query state but randomly zeros the query goal during training with the configured probability.

## Model/Training Setup

The core model is a JAX direct action chunk regression policy:

- No expensive RLBench-style encoder.
- Query observation/state is tokenized by lightweight modules.
- Memory tokens are trainable slow parameters and become fast parameters during inner adaptation.
- Inner loop updates memory tokens only.
- Outer loop trains the full model from scratch.
- Output is direct action chunk regression, not diffusion.

WRITE/READ setup:

- `mode='write'`: used for support loss during inner adaptation.
- `mode='read'`: used for query action prediction after memory adaptation.
- Some runs use separate WRITE and READ heads.
- `write_use_support_obs=True` means support observation/state tokens are explicitly passed into the WRITE path, not only learned write query tokens.

Typical important knobs:

```bash
--config.maml.inner_loss_mode=write
--config.maml.inner_steps=2
--config.maml.inner_lr=1e-1
--config.model.query_memory_direct_regression.write_use_support_obs=True
--config.data.sample_same_task_name=True
--config.data.sample_same_task_instance=False
```

## Features Implemented

Already implemented:

- Hidden query goal:

```bash
--config.data.query_zero_goal=True
```

- Stochastic query-goal dropout:

```bash
--config.data.query_zero_goal=False
--config.maml.query_goal_dropout_rate=0.5  # or similar
```

- Memory conditioning modes:

```bash
--config.model.query_memory_direct_regression.memory_conditioning_mode=film
--config.model.query_memory_direct_regression.memory_conditioning_mode=adaln
```

- Goal-prediction auxiliary head:

```bash
--config.maml.goal_prediction_loss_weight=1.0
```

This predicts query goal xyz after inner adaptation and adds the auxiliary loss to the meta-loss.

- Wrong-support margin:

```bash
--config.maml.use_wrong_support_margin=True
--config.maml.wrong_support_margin=0.01
--config.maml.wrong_support_margin_weight=1.0
```

This tries to enforce correct-support adapted query loss lower than wrong-support adapted query loss.

- Memory contrast losses, though these have not produced a clear fix so far.

- Attention logging:

```text
train/attn_memory_entropy
train/attn_memory_max
train/attn_query_entropy
train/attn_query_max
```

Important: attention entropy is normalized by `log(num_tokens)`. `attn_memory_entropy ~= 1` means nearly uniform memory attention. If `attn_memory_max ~= 1/N`, attention is diffuse.

## Metrics and Diagnostics

The main diagnostic script compares:

```text
no_adaptation
same_family_adaptation
wrong_family_adaptation
```

where:

- `no_adaptation`: memory is initial memory.
- `same_family_adaptation`: memory adapted using support from the same target task family.
- `wrong_family_adaptation`: memory adapted using support from a different task family.

The printed metrics include:

```text
mse      raw unweighted MSE over action chunks
l1       raw unweighted L1 over action chunks
xyz_mse  MSE over first 3 action dimensions
xyz_l1   L1 over first 3 action dimensions
gripper_l1 L1 over remaining non-xyz dims for MetaWorld action_dim=4 this is gripper/action dim 3
best condition counts: number of diagnostic batches where each condition has lowest MSE
```

Important caution:

- WandB `train/read_loss_after` is the training loss on current training batches.
- For these configs, it is usually L1 via `_action_chunk_loss`, not raw MSE.
- Diagnostic `L1` is closer to WandB than diagnostic `MSE`.
- If a run trained only on an aligned subset, compare WandB to `aligned train subset same-family L1`, not `ML45 train all`.
- `ML45 train all` includes train-cache tasks outside the aligned subset and can look much worse than WandB.

Wrong-support ranking accuracy:

```text
train/wrong_support_ranking_accuracy = fraction of tasks where read_loss_wrong_support > read_loss_after
```

Values:

- `1.0`: correct support always beats wrong support.
- `0.5`: no clear separation.
- `0.0`: wrong support is better than correct support.

## Main Empirical Pattern

The main pattern is consistent across many runs:

1. In-distribution adaptation works.
2. Same-family support often strongly improves train/held-in task-family metrics.
3. Wrong-family support is often worse on train, especially with wrong-support margin.
4. Held-out task-family generalization is weak or absent.
5. On held-out families, same-family adaptation is often close to wrong-family adaptation or worse than no adaptation.
6. This suggests the model learns train-family-specific memory updates, not a general support-conditioned control mechanism.

## Evaluated Checkpoints and Results

The following summaries are from MSE diagnostic runs. `L1` is the metric closest to training loss for L1 configs. `MSE` is raw squared error and can be dominated by outliers/action scale.

### `01vpcu2i` older/no-new-conditioning hidden-query-goal run

Relevant config observed:

- `query_zero_goal=True`
- no `memory_conditioning_mode` field in older checkpoints
- no goal auxiliary field in older checkpoints

#### `01vpcu2i/step_0005000.pkl`

```text
ML45 train all:
  no MSE 1.413 L1 0.577
  same MSE 1.428 L1 0.565
  wrong MSE 1.426 L1 0.597

ML45 test all:
  no MSE 2.219 L1 0.911
  same MSE 2.265 L1 0.908
  wrong MSE 2.284 L1 0.940

aligned train subset:
  no MSE 4.542 L1 0.438
  same MSE 4.576 L1 0.383
  wrong MSE 4.778 L1 0.461

held-out aligned test subset:
  no MSE 2.354 L1 0.942
  same MSE 2.432 L1 0.944
  wrong MSE 2.395 L1 0.962
```

Interpretation: weak early train adaptation; no held-out benefit.

#### `01vpcu2i/step_0007000.pkl`

```text
ML45 train all:
  no MSE 1.557 L1 0.605
  same MSE 1.528 L1 0.587
  wrong MSE 1.577 L1 0.634

ML45 test all:
  no MSE 2.292 L1 0.940
  same MSE 2.390 L1 0.958
  wrong MSE 2.406 L1 0.980

aligned train subset:
  no MSE 6.872 L1 0.515
  same MSE 6.817 L1 0.444
  wrong MSE 6.975 L1 0.528

held-out aligned test subset:
  no MSE 2.387 L1 0.947
  same MSE 2.543 L1 0.991
  wrong MSE 2.471 L1 0.984
```

Interpretation: adaptation starts improving train, but held-out already worsens.

#### `01vpcu2i/step_0014000.pkl`

```text
ML45 train all:
  no MSE 1.187 L1 0.521
  same MSE 1.184 L1 0.497
  wrong MSE 1.182 L1 0.541

ML45 test all:
  no MSE 2.210 L1 0.913
  same MSE 2.289 L1 0.918
  wrong MSE 2.345 L1 0.963

aligned train subset:
  no MSE 0.928 L1 0.182
  same MSE 0.879 L1 0.125
  wrong MSE 1.007 L1 0.223

held-out aligned test subset:
  no MSE 2.194 L1 0.905
  same MSE 2.318 L1 0.931
  wrong MSE 2.349 L1 0.973
```

Interpretation: strong train/aligned adaptation, but no held-out generalization.

### `grin7arj/step_0013000.pkl`

```text
ML45 train all:
  no MSE 0.911 L1 0.228
  same MSE 0.948 L1 0.234
  wrong MSE 0.951 L1 0.252

ML45 test all:
  no MSE 2.179 L1 0.842
  same MSE 2.119 L1 0.837
  wrong MSE 2.097 L1 0.836

aligned train subset:
  no MSE 7.194 L1 0.507
  same MSE 7.146 L1 0.510
  wrong MSE 7.107 L1 0.537

held-out aligned test subset:
  no MSE 1.798 L1 0.762
  same MSE 1.762 L1 0.771
  wrong MSE 1.764 L1 0.775
```

Interpretation: not a clean same-family memory signal. Wrong-family often close/better.

### `xqwkns8c` AdaLN + hidden query goal + goal auxiliary

Relevant config observed:

```text
query_zero_goal=True
support_zero_goal=False
query_goal_dropout_rate=0.0
memory_conditioning_mode=adaln
goal_prediction_loss_weight=1.0
```

#### `xqwkns8c/step_0006000.pkl`

```text
ML45 train all:
  no MSE 1.830 L1 0.592
  same MSE 1.749 L1 0.623
  wrong MSE 1.490 L1 0.591

ML45 test all:
  no MSE 2.056 L1 0.888
  same MSE 2.007 L1 0.845
  wrong MSE 2.000 L1 0.851

aligned train subset:
  no MSE 5.034 L1 0.421
  same MSE 5.078 L1 0.372
  wrong MSE 5.597 L1 0.490

held-out aligned test subset:
  no MSE 2.123 L1 0.900
  same MSE 2.046 L1 0.852
  wrong MSE 2.055 L1 0.865
```

Interpretation: one of the more promising runs. Held-out aligned same-family improves versus no by L1/MSE, but wrong-family remains close.

#### `xqwkns8c/step_0008000.pkl`

```text
ML45 train all:
  no MSE 1.825 L1 0.563
  same MSE 1.318 L1 0.482
  wrong MSE 1.854 L1 0.550

ML45 test all:
  no MSE 2.116 L1 0.877
  same MSE 2.086 L1 0.852
  wrong MSE 2.084 L1 0.857

aligned train subset:
  no MSE 4.207 L1 0.340
  same MSE 4.044 L1 0.306
  wrong MSE 4.706 L1 0.377

held-out aligned test subset:
  no MSE 2.129 L1 0.876
  same MSE 2.074 L1 0.845
  wrong MSE 2.058 L1 0.849
```

Interpretation: still one of the best. Train/all improved a lot. Held-out aligned same-family improves over no, but wrong-family is slightly better by MSE/count. Not clean task-specific adaptation.

### `4vj9kkm2/step_0020000.pkl`

Relevant config observed:

```text
data.tasks=()  # all train tasks
query_zero_goal=False
memory_conditioning_mode=None
inner_steps=3
inner_lr=0.1
```

```text
ML45 train all:
  no MSE 0.538 L1 0.232
  same MSE 0.442 L1 0.151
  wrong MSE 0.585 L1 0.260

ML45 test all:
  no MSE 1.797 L1 0.708
  same MSE 1.798 L1 0.700
  wrong MSE 1.806 L1 0.744

aligned train subset:
  no MSE 1.716 L1 0.363
  same MSE 1.709 L1 0.276
  wrong MSE 1.794 L1 0.404

held-out aligned test subset:
  no MSE 1.834 L1 0.795
  same MSE 1.879 L1 0.780
  wrong MSE 1.838 L1 0.819
```

Interpretation: strong in-distribution adaptation. Test-all small L1 gain but no convincing OOD adaptation.

### `7gmlglaa` FiLM + hidden query goal + goal auxiliary

Relevant config observed:

```text
query_zero_goal=True
query_goal_dropout_rate=0.0
memory_conditioning_mode=film
goal_prediction_loss_weight=1.0
```

#### `7gmlglaa/step_0010000.pkl`

```text
ML45 train all:
  no MSE 5.163 L1 1.040
  same MSE 2.178 L1 0.612
  wrong MSE 8.474 L1 1.015

ML45 test all:
  no MSE 2.049 L1 0.917
  same MSE 2.218 L1 0.906
  wrong MSE 2.260 L1 0.970

aligned train subset:
  no MSE 4.815 L1 0.594
  same MSE 3.812 L1 0.267
  wrong MSE 6.125 L1 0.833

held-out aligned test subset:
  no MSE 1.984 L1 0.879
  same MSE 2.211 L1 0.894
  wrong MSE 2.164 L1 0.940
```

Interpretation: very strong train adaptation, no held-out transfer.

#### `7gmlglaa/step_0011000.pkl`

```text
ML45 train all:
  no MSE 4.718 L1 0.968
  same MSE 1.182 L1 0.452
  wrong MSE 8.218 L1 0.931

ML45 test all:
  no MSE 1.930 L1 0.862
  same MSE 2.110 L1 0.862
  wrong MSE 2.083 L1 0.916

aligned train subset:
  no MSE 5.051 L1 0.545
  same MSE 5.065 L1 0.304
  wrong MSE 5.747 L1 0.766

held-out aligned test subset:
  no MSE 1.846 L1 0.827
  same MSE 2.094 L1 0.853
  wrong MSE 1.995 L1 0.881
```

Interpretation: train specialization gets stronger; held-out gets worse. Clear over-specialization to train families.

### `1dmtamec/step_0010000.pkl` FiLM + query-goal dropout + goal auxiliary

Relevant config observed:

```text
query_zero_goal=False
query_goal_dropout_rate=0.5
memory_conditioning_mode=film
goal_prediction_loss_weight=1.0
```

```text
ML45 train all:
  no MSE 4.391 L1 0.980
  same MSE 1.238 L1 0.458
  wrong MSE 2.915 L1 0.788

ML45 test all:
  no MSE 2.017 L1 0.896
  same MSE 2.277 L1 0.916
  wrong MSE 2.169 L1 0.927

aligned train subset:
  no MSE 4.569 L1 0.494
  same MSE 4.206 L1 0.299
  wrong MSE 4.653 L1 0.558

held-out aligned test subset:
  no MSE 1.929 L1 0.855
  same MSE 2.327 L1 0.930
  wrong MSE 2.024 L1 0.881
```

Interpretation: actual stochastic query-goal dropout did not solve OOD. Strong train adaptation, worse held-out.

### `ppwj0qqa/step_0012000.pkl` FiLM + hidden query goal, no goal auxiliary

Relevant config observed:

```text
query_zero_goal=True
query_goal_dropout_rate=0.0
memory_conditioning_mode=film
goal_prediction_loss_weight=0.0
```

```text
ML45 train all:
  no MSE 1.615 L1 0.562
  same MSE 1.264 L1 0.491
  wrong MSE 1.636 L1 0.575

ML45 test all:
  no MSE 2.449 L1 0.947
  same MSE 2.549 L1 0.949
  wrong MSE 2.554 L1 0.988

aligned train subset:
  no MSE 2.159 L1 0.330
  same MSE 2.234 L1 0.195
  wrong MSE 2.479 L1 0.414

held-out aligned test subset:
  no MSE 2.358 L1 0.908
  same MSE 2.514 L1 0.930
  wrong MSE 2.415 L1 0.951
```

Interpretation: removing goal auxiliary did not improve held-out generalization.

## What Seems To Work

### In-distribution memory adaptation

The model can learn to use support demos on training task families. Many runs show:

```text
same_family_adaptation << no_adaptation
same_family_adaptation << wrong_family_adaptation
```

on ML45 train and especially aligned train subsets.

Examples:

- `7gmlglaa/step_0011000`: train all same L1 `0.452` vs no `0.968`, wrong `0.931`.
- `1dmtamec/step_0010000`: train all same L1 `0.458` vs no `0.980`, wrong `0.788`.
- `xqwkns8c/step_0008000`: train all same MSE `1.318` vs no `1.825`, wrong `1.854`.

### Wrong-support margin can enforce train-family separation

With wrong-support margin enabled, wrong support often becomes worse than same support on train. This confirms the machinery is doing something, but it may also encourage task-family discrimination that does not transfer.

### FiLM and AdaLN both can train

Both `film` and `adaln` memory conditioning modes can produce in-distribution gains.

- `xqwkns8c`: AdaLN, hidden query goal, goal auxiliary, best-looking held-out behavior but still not clean.
- `7gmlglaa`: FiLM, hidden query goal, goal auxiliary, very strong train adaptation but poor held-out.

## What Does Not Work Yet

### Full OOD task-family generalization

Across evaluated checkpoints, held-out ML45 test task families do not reliably benefit from same-family adaptation.

Common held-out pattern:

```text
same-family adaptation ~= wrong-family adaptation
or
same-family adaptation worse than no adaptation
```

This is the core failure.

### Fully hidden query goal alone

`query_zero_goal=True` forces the model to rely on support for goal/task information, but it does not by itself produce OOD generalization. It often increases train adaptation while hurting test.

### Query-goal dropout alone

Actual stochastic dropout was tested in `1dmtamec`:

```text
query_zero_goal=False
query_goal_dropout_rate=0.5
```

It produced strong train adaptation but poor held-out results.

### Goal auxiliary alone

Goal auxiliary was active in `xqwkns8c`, `7gmlglaa`, and `1dmtamec`. It did not solve held-out generalization. Removing it in `ppwj0qqa` also did not solve the issue.

### More train specialization

Later checkpoints can get better train metrics but worse held-out behavior. Example: `7gmlglaa` from `step_0010000` to `step_0011000` got much stronger on train and worse on held-out.

## Working Hypothesis

The current flat-state + gradient-only memory-write setup is learning a train-family-specific adaptation code rather than a general algorithm for reading support demonstrations and applying them to new task families.

Likely contributing factors:

1. Flat 39D MetaWorld state has task-family-dependent semantics.
2. Gradient-only WRITE may learn family-specific shortcuts.
3. Wrong-support margin may encourage discrimination among train families rather than transferable structure.
4. Hidden query goal creates a hard problem: for unseen families, support must reveal how goals map to behavior, but the representation may not support this abstraction.
5. The model has no explicit object-centric inductive bias.

## Experiments Proposed/Written After These Findings

Four Snellius sbatch files were written for experiments that do not require new implementation:

### 1. Near-OOD sibling split

File:

```text
sbatch/snellius/maml_metaworld_nearood_memadaln_goal_aux_wrong_margin_1gpu.sbatch
```

Actually patched to use:

```text
gpu:2
train.batch_size=32
```

Purpose: test if the method can generalize to closely related task families before asking full ML45 OOD.

Train task list:

```text
button variants
door-open/door-close
drawer-open/drawer-close
handle variants
```

Expected evaluation should target sibling held-out tasks such as:

```text
door-lock-v3
door-unlock-v3
box-close-v3
hand-insert-v3
```

### 2. No wrong-support margin

File:

```text
sbatch/snellius/maml_metaworld_memadaln_aligned_goal_aux_no_wrong_margin_1gpu.sbatch
```

Also patched to:

```text
gpu:2
train.batch_size=32
```

Purpose: test whether wrong-support margin is causing train-family discrimination and hurting transfer.

### 3. Soft memory update

File:

```text
sbatch/snellius/maml_metaworld_memadaln_aligned_goal_aux_soft_memory_1gpu.sbatch
```

Key changes:

```text
inner_lr=3e-2
inner_steps=4
memory_update_clip_norm=0.05
```

Purpose: reduce saturated memory writes and force smoother adaptation.

### 4. High query-goal dropout without wrong-support margin

File:

```text
sbatch/snellius/maml_metaworld_memadaln_aligned_goal_dropout08_goal_aux_no_wrong_margin_1gpu.sbatch
```

Key changes:

```text
query_zero_goal=False
query_goal_dropout_rate=0.8
use_wrong_support_margin=False
```

Purpose: avoid the fully-hidden-goal hard setting while still forcing support dependence often.

## Experiments Still Not Implemented But Probably Important

### 1. Object-centric tokenizer

Most important unimplemented idea.

Instead of feeding a flat 39D vector to an MLP, tokenize the state as semantically meaningful tokens:

```text
hand position token
gripper token
object/handle/button token(s)
goal token if available
relative vector tokens: object-hand, goal-object, goal-hand
```

Reason: flat state slots are likely not semantically consistent across MetaWorld task families. OOD generalization probably requires object-centric inductive bias.

### 2. Explicit support encoder

Currently memory is written by inner-loop gradients only. Add a cheap explicit support encoder:

```text
support obs/action chunks -> support embedding -> initialize/bias memory tokens
```

Then still perform memory MAML updates. This may make support content easier to use than pure gradient descent.

### 3. Query-goal dropout curriculum

Currently only fixed dropout exists. Implement a schedule:

```text
query_goal_dropout_rate: 0.0 -> 0.8 over training
```

Reason: visible goal helps learn general control early; dropout later forces support use.

### 4. Memory delta regularization loss

Currently we can reduce `memory_update_clip_norm`, but there is no explicit penalty on memory update magnitude. Add:

```text
meta_loss += lambda_delta * ||adapted_memory - initial_memory||^2
```

Reason: prevent memory updates from becoming saturated train-family codes.

## Recommended Next Decision Logic

After the four new sbatches finish:

1. Evaluate each checkpoint on:

```text
ML45 train all
ML45 test all
aligned train subset
held-out aligned test subset
near-OOD sibling test subset if applicable
```

2. For each, compare primarily:

```text
same-family L1 vs no-adaptation L1
same-family L1 vs wrong-family L1
best condition counts
```

3. If near-OOD fails, stop broad OOD claims and debug representation.

4. If no-wrong-margin improves held-out same-family vs wrong-family, reduce/avoid margin in future.

5. If soft memory improves held-out transfer, add explicit memory-delta regularization.

6. If high dropout helps, implement dropout curriculum.

7. If all fail, implement object-centric tokenizer before more sweeps.

## Short Prompt For Future LLM

We are training JAX MetaWorld direct-regression memory-MAML policies. The model predicts action chunks from query observations plus memory tokens. Inner-loop updates memory tokens using support demos. MetaWorld cache uses ML45 goal train/test splits, 39D state = 36D obs + 3D goal. We sample support/query from same task family but different task instances. We compare no adaptation, same-family support adaptation, and wrong-family support adaptation.

Current result: train-family adaptation works strongly, but held-out task-family generalization mostly fails. Same-family support often beats wrong support on train, but on held-out ML45 test same-family is often close to wrong-family or worse than no adaptation. Film/AdaLN memory conditioning, hidden query goal, query-goal dropout, wrong-support margin, and goal auxiliary have not solved OOD generalization. The best-ish run so far is `xqwkns8c/step_0008000.pkl` with AdaLN + hidden query goal + goal auxiliary; it improves held-out aligned test somewhat, but wrong-family remains close/better by MSE/count.

Likely issue: flat 39D state and gradient-only memory writes learn train-family-specific memory codes. Next serious implementation should be object-centric state tokenization and/or an explicit support encoder. Before that, run the four existing Snellius follow-up sbatches: near-OOD sibling split, no wrong-support margin, soft memory update, and high query-goal dropout.
