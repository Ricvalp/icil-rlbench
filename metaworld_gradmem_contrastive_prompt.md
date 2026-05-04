# Codex prompt: Meta-World GradMem / contrastive MAML experiments

You are working in the `icil-rlbench` repository. The immediate target is the low-dimensional Meta-World branch, especially `icil_jax_query_memory`, not the RLBench point-cloud encoder path. The goal is to test whether GradMem-style writable memory can learn **task-specific** information from support demonstrations in a low-dimensional environment where perception is not the main bottleneck.

This is **not** an in-context point-cloud tokenizer experiment. Do not use the RLBench Perceiver or supernode encoders in this pass. Keep the code focused on Meta-World state/action trajectories, encoderless memory tokens, and MAML/FOMAML-style inner-loop updates.

## Background and problem

Current observation: in encoderless GradMem-style experiments, memory tokens are learnable parameters updated in the inner loop with gradient descent on a WRITE loss, followed by a READ loss on the query action chunk. Test-time memory updates improve performance, but they improve performance even when the support demos come from a wrong/unrelated task. This suggests that the inner update may be a **generic calibration update**, not a task-specific memory write.

We want to test contrastive meta-learning ideas to force the memory update to become task-specific.

Meta-World is used as a fast diagnostic environment:

- low-dimensional observations, no point clouds;
- no task ID or explicit goal vector fed to the model by default;
- support demonstrations should define the task;
- query observation should provide current state/control information, not the task label;
- action chunks are trained with direct L1/Huber imitation loss.

Use existing repo conventions and utilities wherever possible. Do not rewrite the full training stack unless necessary.

---

## Implementation target

Implement two or three small experimental variants, controlled by config flags, in the existing Meta-World / query-memory training path.

The expected data format is approximately:

```text
support_obs:      [B, K, T_support, obs_dim]
support_actions:  [B, K, T_support, action_dim]
query_obs:        [B, T_obs, obs_dim]
target_action:    [B, H, action_dim]
```

The model should expose or already contain:

```text
M0: learned memory tokens
WRITE(M0, support) -> M'
READ(query_obs, M') -> predicted action chunk
```

Do not require a support encoder for this pass.

---

# Variant 1: wrong-support READ margin

This is the highest-priority implementation.

For every query task in a meta-batch, sample:

- correct support from the same task instance/task family;
- wrong support from a different task, preferably a hard negative when available.

Run WRITE twice:

```text
M_pos = WRITE(M0, correct_support)
M_neg = WRITE(M0, wrong_support)
```

Then compute query prediction energies:

```text
E_pos = L1_or_Huber(READ(query_obs, M_pos), target_action)
E_neg = L1_or_Huber(READ(query_obs, M_neg), target_action)
```

Add a margin ranking loss:

```text
L_rank = max(0, margin + E_pos - E_neg)
```

Total loss:

```text
L = E_pos + lambda_rank * L_rank
```

Purpose: correct support memory should predict the query action better than wrong support memory. This directly targets the current failure mode where wrong support updates are also useful.

Implementation notes:

- Add config flags for `use_wrong_support_margin`, `lambda_rank`, `rank_margin`.
- Add a loader/sampler option for wrong support.
- Prefer hard negatives when possible. Examples: drawer-open vs drawer-close, button variants, push vs pull, reach vs pick-place.
- Log `E_pos`, `E_neg`, `E_neg - E_pos`, and ranking accuracy.

---

# Variant 2: contrastive memory-update objective

Implement a ConML-style contrastive objective over memory updates.

For each task in a meta-batch, sample two different support subsets from the same task:

```text
S_i_a, S_i_b
```

Run WRITE independently:

```text
M_i_a = WRITE(M0, S_i_a)
M_i_b = WRITE(M0, S_i_b)
```

Construct representations from the **memory update residual**, not from raw memory:

```text
Delta_i_a = M_i_a - M0
Delta_i_b = M_i_b - M0
z_i_a = normalize(proj(pool(Delta_i_a)))
z_i_b = normalize(proj(pool(Delta_i_b)))
```

Use InfoNCE / supervised contrastive loss:

- positive pairs: two support subsets from the same task;
- negative pairs: support subsets from different tasks in the meta-batch.

Total loss:

```text
L = L_read + lambda_contrast * L_contrast
```

Purpose: same-task supports should induce similar memory updates, while different tasks should induce different memory updates. This discourages a generic task-independent update.

Implementation notes:

- Add config flags: `use_memory_contrast`, `lambda_contrast`, `contrast_temperature`, `contrast_on_delta=true`.
- Pooling can be mean pooling over memory tokens initially.
- Projection head can be a small MLP.
- Log within-task similarity, between-task similarity, contrastive accuracy, and norm of `DeltaM`.

---

# Variant 3: generic/task-specific memory split

This is optional, but useful if Variant 1 or 2 shows that there is a useful generic update plus a weak task-specific residual.

Split memory into two banks:

```text
M_global: generic writable memory
M_task:   task-specific writable memory
```

Both can be updated by WRITE, but apply contrastive losses only to `M_task` or to `DeltaM_task`.

READ receives both:

```text
READ(query_obs, concat(M_global, M_task))
```

Purpose: allow the model to keep useful generic adaptation while forcing part of memory to encode task-specific information.

Implementation notes:

- Add config flags for `split_memory`, `num_global_memory_tokens`, `num_task_memory_tokens`.
- Keep this implementation minimal. If it complicates the code too much, skip it for now.

---

## WRITE loss recommendations

Use direct imitation losses, not diffusion:

```text
L_write = weighted L1/Huber on support action chunks
```

Optionally weight early action steps more heavily than later steps, because receding-horizon control depends most on the first few predicted actions.

Do not feed task IDs, task names, or explicit goals to WRITE or READ unless explicitly enabled as an ablation.

For Meta-World observations, store raw observations for debugging, but use a filtered `model_obs` by default. The filtered observation should not include explicit task IDs or explicit goal coordinates unless the config enables them.

---

## Required diagnostics and logging

Log the following for every variant:

```text
write_loss_before
write_loss_after
read_loss_before_update
read_loss_after_update
memory_update_norm = ||M' - M0|| / ||M0||
policy_output_delta = ||READ(query, M') - READ(query, M0)||
correct_support_loss
wrong_support_loss
wrong_minus_correct_loss
ranking_accuracy
within_task_memory_similarity
between_task_memory_similarity
```

Also add evaluation modes:

```text
correct support
wrong support
null/no support
zero WRITE steps
1 WRITE step
5 WRITE steps
```

The success criterion is not only higher rollout success. The method should show that correct support updates are better than wrong support updates.

---

## Integration expectations

- Keep existing MAML/FOMAML code paths as much as possible.
- Add config-driven branches rather than duplicating entire trainers.
- Prefer minimal changes in `icil_jax_query_memory` first.
- Do not touch RLBench point-cloud encoders in this pass.
- Keep the implementation compatible with future encoder-based support memory, but do not implement that encoder now.

## Suggested first experiments

1. Baseline encoderless GradMem:
   - no contrastive loss;
   - correct support only;
   - 0, 1, 5 WRITE steps.

2. Wrong-support margin:
   - enable Variant 1;
   - compare correct, wrong, and null support.

3. Memory-update contrast:
   - enable Variant 2;
   - inspect within-task vs between-task memory update similarity.

If wrong-support margin does not separate correct and wrong support in Meta-World, do not scale back to RLBench yet. Debug the memory-writing objective first.
