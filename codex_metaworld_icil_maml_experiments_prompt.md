# Codex Implementation Prompt: MetaWorld ICIL with Object-Centric Memory MAML/FOMAML

## Repository context

You are working in the `icil-rlbench` repository. The current relevant codepath is the JAX/MetaWorld GradMem-style implementation under:

```text
icil_jax_query_memory/
```

The repository already contains dataset creation/loading for the MetaWorld experiments, JAX MAML/FOMAML training code, direct-regression action chunk losses, logging, and diagnostic scripts. Reuse the existing code as much as possible. Do not rewrite data loading or training infrastructure unless necessary.

This implementation is for a paper about **in-context imitation learning (ICIL) with memory test-time training**. The main hypothesis to test is:

> A robot policy can infer a task/goal from support demonstrations, write this information into memory tokens using a self-supervised or imitation WRITE loss, and use the adapted memory at READ time to predict actions for a new query state.

The experiments should make it clear when MAML/FOMAML-style memory writing works, when it does not, and whether an explicit support encoder is necessary.

This is **not** a standard multi-task policy with task IDs. Avoid task IDs unless explicitly requested for oracle/debug baselines. The goal is to evaluate whether demonstrations and/or memory updates are used.

---

## Scientific motivation

The previous pure GradMem-style experiments used flat MetaWorld state vectors and learned memory tokens updated by gradient descent. Results suggested that:

1. Pure no-encoder gradient-written memory struggles on unseen task families.
2. Flat 39D state likely gives poor cross-task inductive bias.
3. Hiding the query goal while sampling support/query from different task instances may make the task underdetermined.
4. MAML/FOMAML should be evaluated under clean task semantics.

The new implementation should therefore introduce:

1. **Object-centric state tokens** instead of flat state vectors.
2. Three explicit task settings: A, B, and C.
3. An explicit **support encoder** that maps demonstrations to memory tokens.
4. Optional memory MAML/FOMAML refinement on top of encoder-produced memory.
5. Baselines that separate query-only performance, forward support conditioning, no-encoder memory writing, and memory test-time training.

---

# High-level experimental settings

Implement three settings. They should share as much code as possible.

## Setting A: Same-instance hidden-goal ICIL

### Purpose

Test whether demonstrations can specify a hidden goal/task instance.

### Semantics

```text
support and query share the same task family and the same goal/task instance
query goal is hidden
support demonstrations specify the goal through their trajectories
```

This is the cleanest support-necessary ICIL setting.

### Required sampling

For each meta-sample:

```text
family/task name: same for support and query
instance/goal: same for support and query
support demos: K trajectories from the same goal instance
query demo: held-out trajectory from the same goal instance, different start if possible
query goal visibility: hidden
support goal visibility: configurable
```

Two variants are useful:

```text
A-clean:
    support goal hidden too. Goal must be inferred from support trajectory behavior.

A-debug:
    support goal visible, query goal hidden. This is easier and tests whether support can carry goal information into memory.
```

### Expected behavior

A query-only policy without goal should fail or perform poorly. A support-conditioned policy should improve. Wrong support should hurt.

---

## Setting B: Same-family different-goal ICIL

### Purpose

Test whether support demonstrations specify the task family/mode while the query supplies the specific goal instance.

### Semantics

```text
support and query share the same task family
support and query have different goal/task instances
query goal is visible
support teaches the family/dynamics/mode
```

Do **not** hide the query goal in this setting. If support and query have different goals, hiding the query goal makes the target underdetermined.

### Required sampling

For each meta-sample:

```text
family/task name: same for support and query
instance/goal: different between support and query
support demos: K trajectories from support goal instances
query demo: trajectory from query goal instance
query goal visibility: visible
support goal visibility: visible
```

### Expected behavior

A query-only policy with goal may already do well on seen families. Support should help mainly when the family/mode is ambiguous or for generalization. Memory updates may provide small refinement, not necessarily a large gain.

---

## Setting C: Held-out family ICIL / OOD family adaptation

### Purpose

Test whether the method can adapt to task families not seen during training.

### Semantics

```text
train on some MetaWorld task families
test on held-out task families
support demos come from the held-out family at test time
query goal is visible by default
support should identify the family/control mode
```

This is the hardest setting. Do not expect no-encoder GradMem to solve it. The main method should use the support encoder.

### Required sampling

Use ML-style splits when possible:

```text
train families: configured list, e.g. ML10 train tasks or ML45 train tasks
held-out families: configured list, e.g. ML10/ML45 test tasks
support demos at eval: K demonstrations from held-out family
query demos at eval: different trajectories from held-out family
query goal visibility: visible by default
```

Also optionally evaluate:

```text
C-hidden-goal same-instance variant:
    held-out family, support/query same instance, query goal hidden.
```

But this should be a secondary experiment because it is harder and less standard.

---

# Object-centric representation

Implement object-centric state tokenization and use it for **all** settings A, B, and C.

The goal is to avoid feeding a flat MetaWorld state vector as one undifferentiated vector. Instead, parse the state into semantically meaningful entities and relations.

## Required module

Create a module similar to:

```text
icil_jax_query_memory/models/object_centric_state.py
```

or an equivalent location consistent with the repo structure.

Implement:

```python
class ObjectCentricStateTokenizer:
    def __call__(self, obs, *, goal_visible: bool, config) -> (tokens, mask, aux)
```

JAX/Flax naming can follow repository style.

## Inputs

Use the existing MetaWorld observation tensors from the dataloader. The parser must be configurable because MetaWorld observation layouts can vary by wrapper/version.

The tokenizer should parse at least:

```text
hand / end-effector position
possibly gripper open state
object position(s)
goal position, if available and visible
previous state components, if useful
```

If the current dataset uses a known flat layout, implement the parser for that layout and add clear config fields:

```text
obs_layout.hand_pos_slice
obs_layout.gripper_slice
obs_layout.obj1_pos_slice
obs_layout.obj2_pos_slice
obs_layout.goal_pos_slice
obs_layout.has_obj2
obs_layout.goal_available
```

Avoid hard-coding undocumented indices without config names.

## Tokens to create

At minimum create these tokens:

```text
hand token:              MLP([hand_pos, gripper_open]) + role_hand
object_1 token:          MLP(object_1_pos) + role_object_1
object_2 token:          MLP(object_2_pos) + role_object_2, masked if absent
goal token:              MLP(goal_pos) + role_goal, only if goal_visible
object_1_to_hand token:  MLP(object_1_pos - hand_pos) + role_rel_obj1_hand
object_2_to_hand token:  MLP(object_2_pos - hand_pos) + role_rel_obj2_hand, if present
goal_to_hand token:      MLP(goal_pos - hand_pos) + role_rel_goal_hand, if goal_visible
goal_to_obj1 token:      MLP(goal_pos - object_1_pos) + role_rel_goal_obj1, if goal_visible
goal_to_obj2 token:      MLP(goal_pos - object_2_pos) + role_rel_goal_obj2, if goal_visible and obj2 present
```

All tokens should be projected to `d_model`.

For temporal observations, add time embeddings:

```text
token = token + role_embedding + time_embedding
```

Return:

```text
tokens: [B, T_obs, N_tokens, d] or flattened [B, T_obs * N_tokens, d]
mask:   [B, T_obs, N_tokens] or flattened [B, T_obs * N_tokens]
aux:    parsed fields for losses/diagnostics
```

## Goal visibility

The tokenizer must support:

```text
goal_visible=True
    include goal token and goal relation tokens

goal_visible=False
    remove/mask goal token and all goal relation tokens
```

Do not zero out goal values and leave a token with a role embedding unless explicitly configured. Prefer masking/removing the token so the model cannot infer goal visibility from a constant token unless that is intended.

---

# Action representation and loss

Use the existing action chunk representation and direct regression loss.

Main loss:

```text
L_action = weighted L1 or Huber over action chunks
```

Make position, rotation, and gripper weights configurable:

```text
loss.position_weight
loss.rotation_weight
loss.gripper_weight
loss.horizon_decay
```

If actions include quaternions, handle sign ambiguity if currently relevant:

```text
quat_loss = min(||q - q*||, ||q + q*||)
```

If current code already has an action loss, reuse it but expose these weights.

---

# Architectures to implement

Implement four model families. These are needed for coherent ablations.

## Model 0: Query-only baseline

### Purpose

Measure whether the query observation alone solves the task.

### Inputs

```text
query object-centric tokens
no support
no memory updates
```

### Architecture

```text
query tokens -> query encoder -> action decoder -> action chunk
```

Memory/support input should be either empty or a learned null token.

### Use in settings

Run in A, B, and C.

Expected:

```text
Setting A hidden-goal: should be weak
Setting B visible-goal: may be strong
Setting C held-out family: useful baseline
```

---

## Model 1: No-encoder GradMem / learned memory tokens

### Purpose

Mechanistic diagnostic for pure gradient-written memory.

### Inputs

```text
learned memory tokens M0, shared across samples
support trajectories/actions only through WRITE gradient
query object-centric tokens at READ
```

### WRITE

Use a shared decoder backbone with `mode="write"` and WRITE action/reconstruction head.

Do not feed support observations directly into WRITE, unless explicitly configured for an ablation.

WRITE query tokens should contain abstract indices only:

```text
demo_id embedding
chunk_start or normalized time embedding
action_slot embedding
optional family/mode embedding only for oracle/debug, not main
```

WRITE target:

```text
support action chunks
optionally support object-centric quantities
```

Update only memory tokens in the inner loop:

```text
M' = M - alpha * grad_M L_write(M)
```

### READ

```text
query object-centric tokens + adapted memory M' -> query action chunk
```

### FOMAML caveat

If using separate WRITE and READ heads under FOMAML, add explicit WRITE auxiliary loss to train the WRITE head:

```text
L_outer = L_read_after + beta_write_aux * L_write_after
```

Otherwise the WRITE head may not receive useful gradients.

### Use in settings

Run mainly in Setting A as a diagnostic. Also run a small version in B/C if cheap, but do not make it the main model for C.

---

## Model 2: Support encoder memory, no test-time update

### Purpose

Primary amortized ICIL baseline.

This model asks whether an explicit support encoder can infer the task/goal from demonstrations without MAML.

### Inputs

```text
support object-centric observation tokens
support action chunks or full trajectories
query object-centric tokens
```

### Support encoder architecture

Implement a support encoder that maps K demonstrations to memory tokens:

```text
support demos -> support encoder -> M0 [B, M_mem, d_model]
```

Use object-centric observation tokens and action tokens.

Recommended implementation:

1. **Per-timestep tokenization**

For each support timestep or chunk:

```text
state tokens = object-centric tokens(obs_t)
action token(s) = MLP(action_t or action_chunk)
time embedding = normalized time within demo
demo embedding = support demo index
```

2. **Frame/chunk encoder**

Compress each timestep/chunk to one or a few tokens:

```text
state/action tokens -> small Transformer or MLP pooling -> frame token(s)
```

3. **Demo temporal encoder**

For each demo independently:

```text
frame tokens over time -> Transformer/GRU/MLP-attention -> demo tokens
```

Prefer Transformer/attention pooling first because repository already uses transformer-style blocks.

4. **Set/demo aggregator**

Aggregate across K support demos:

```text
concat demo tokens + demo-id embeddings
learned memory queries cross-attend to demo tokens
output M0 [B, M_mem, d]
```

This is a Perceiver-style support encoder. Use learned memory queries and cross-attention from memory queries to support tokens, followed by optional self-attention over memory.

Suggested config:

```text
M_mem = 32 or 64
support_d_model = policy d_model
support_encoder_layers = 2 or 4
support_encoder_heads = 4 or 8
support_memory_self_attn_layers = 1 or 2
```

### READ

```text
query object-centric tokens + M0 -> query action chunk
```

No inner-loop update.

### Use in settings

Run in A, B, and C. This is the most important baseline/main model before MAML.

---

## Model 3: Support encoder memory + MAML/FOMAML memory update

### Purpose

Main paper model for memory test-time training.

### Inputs

Same as Model 2.

### Initial memory

```text
M0 = SupportEncoder(C)
```

### WRITE

Use the shared decoder backbone with `mode="write"` to reconstruct support information from memory.

Prefer WRITE targets:

```text
support action chunks
support trajectory chunks
object-centric support quantities, e.g. hand/object/goal relative vectors
```

Do not pass raw support observation tokens through a direct shortcut in WRITE unless doing an ablation.

Update memory only:

```text
M' = M0 - alpha * grad_M L_write(M0)
```

Configurable:

```text
write_steps in {0, 1, 2, 5, 10}
inner_lr in {1e-3, 3e-3, 1e-2, 3e-2}
full_maml vs fomaml
memory_grad_clip
layer_norm_after_update
```

### READ

```text
query object-centric tokens + M' -> query action chunk
```

### Outer loss

```text
L_outer = L_read_after + beta_write_aux * L_write_after + beta_rank * L_rank
```

Where `L_rank` is optional wrong-support contrastive ranking loss.

### Use in settings

Run in A, B, and C. For C, this is the main TTT method.

---

# Shared decoder with WRITE/READ mode

If not already implemented, add a shared decoder module that accepts:

```python
mode: Literal["write", "read"]
query_tokens: [B, Nq, d]
memory_tokens: [B, M, d]
action_slot_tokens or decoder input tokens
```

### READ mode

```text
query_tokens = real object-centric query observation tokens
task/memory tokens = M0 or M'
output = query action chunk
head = read_action_head
```

### WRITE mode

```text
query_tokens = learned demo/time/action-slot WRITE tokens
memory_tokens = M during inner loop
output = support reconstruction target
head = write_head or shared head depending config
```

### Heads

Support these configs:

```text
shared_read_write_head=True
    WRITE and READ both use same action head. Good if targets are same action representation.

shared_read_write_head=False
    same decoder backbone, separate write_head and read_head. Default for stability.
```

If FOMAML and separate heads are used, ensure WRITE head receives gradients via explicit `L_write` auxiliary.

---

# Memory-conditioned query modulation

Implement an optional READ architecture that uses memory to modulate query tokens before action decoding.

This should be configurable:

```text
read_memory_conditioning = {cross_attn, film, adaln, cross_attn_plus_film}
```

For FiLM/AdaLN:

```text
m = pool(M')
scale, shift = MLP(m)
query_tokens = LayerNorm(query_tokens) * (1 + scale) + shift
```

Then pass modulated query tokens to decoder.

This is important because memory should specify intent/task while query tokens specify geometry/state.

---

# Wrong-support contrastive/ranking loss

Implement wrong-support sampling and ranking loss for Models 2 and 3.

For each query:

```text
C+ = correct support
C- = wrong support
M+ = SupportEncoder(C+) or WRITE(C+)
M- = SupportEncoder(C-) or WRITE(C-)
E+ = action_loss(policy(query, M+), target_action)
E- = action_loss(policy(query, M-), target_action)
L_rank = max(0, margin + E+ - E-)
```

Total:

```text
L = E+ + beta_rank * L_rank
```

Wrong support types:

```text
random_wrong_family
same_family_wrong_goal
hard_sibling_family
same_family_wrong_instance
```

Use structured negatives as much as possible. Random negatives are useful but can be too easy.

---

# Diagnostics and metrics

Implement the following metrics for every train/eval run.

## Loss metrics

```text
read_loss_before_update
read_loss_after_update
write_loss_before_update
write_loss_after_update
rank_loss
```

## Memory metrics

```text
memory_grad_norm
memory_update_norm_abs
memory_update_norm_relative = ||M' - M0|| / ||M0||
memory_token_norm_before
memory_token_norm_after
```

## Output sensitivity metrics

```text
action_delta_update = ||policy(query, M') - policy(query, M0)||
action_delta_wrong_support = ||policy(query, M_correct) - policy(query, M_wrong)||
```

## Support intervention metrics

Evaluate each checkpoint under:

```text
correct support
wrong support
null support / query-only
shuffled support
```

Report:

```text
loss_correct
loss_wrong
loss_null
context_intervention_gap_loss = loss_wrong - loss_correct
success_correct
success_wrong
success_null
context_intervention_gap_success = success_correct - success_wrong
```

This is central to the paper. Do not only report success with correct support.

---

# Experiment scripts/configs

Implement separate scripts or config entry points for each setting. Reuse common training/eval code.

Preferred structure:

```text
configs/metaworld_icil/setting_a_same_instance_hidden_goal.yaml
configs/metaworld_icil/setting_b_same_family_visible_goal.yaml
configs/metaworld_icil/setting_c_heldout_family_visible_goal.yaml

scripts/train_metaworld_icil.py
scripts/eval_metaworld_icil.py
scripts/diagnose_support_intervention.py
```

If the existing repo uses Python config files rather than YAML, follow the repo style. The key is that A/B/C are separate reproducible configs.

---

# Required experiment matrix

## Setting A: same-instance hidden-goal

Run:

```text
A0_query_only_hidden_goal
A1_no_encoder_gradmem_memory_only
A2_support_encoder_no_update
A3_support_encoder_fomaml_memory_update
A4_support_encoder_full_maml_memory_update, if feasible
A5_oracle_query_goal_visible, upper bound
```

Goal:

```text
show that query-only fails when goal is hidden,
support encoder helps,
memory WRITE update optionally improves over encoder-only.
```

This is the most important setting for proving support necessity.

---

## Setting B: same-family different-goal, query goal visible

Run:

```text
B0_query_only_visible_goal
B1_support_encoder_no_update
B2_support_encoder_fomaml_memory_update
B3_support_encoder_full_maml_memory_update, if feasible
B4_no_encoder_gradmem, optional diagnostic
```

Goal:

```text
show whether support improves family/mode inference beyond visible query goal.
```

Expect query-only may be strong. That is fine. The key metric is whether support helps on low-data or family-generalization cases.

---

## Setting C: held-out family ICIL

Run:

```text
C0_query_only_visible_goal
C1_support_encoder_no_update
C2_support_encoder_fomaml_memory_update
C3_support_encoder_full_maml_memory_update, if feasible
C4_wrong_support/null support intervention eval
```

Optional:

```text
C5_same_instance_hidden_goal held-out family variant
```

Goal:

```text
show whether support encoder and memory TTT help adapt to unseen families.
```

This is the hardest result. Do not rely on no-encoder GradMem as the main model here.

---

# Baselines

Implement/report these baselines consistently.

## Query-only baseline

No support, no memory. Same query encoder and action decoder where possible.

This is essential because some settings are solvable from query alone.

## Null-support baseline

Use support path but replace support with learned null memory or zeros.

This checks whether support conditioning is actually used.

## Wrong-support baseline

Use support from a different task/family/goal.

This checks whether the model is sensitive to the content of support.

## Forward support encoder baseline

Support encoder produces memory `M0`, no MAML update.

This is the main baseline against which memory TTT must improve.

## No-encoder GradMem baseline

Learned memory tokens only, updated by WRITE loss. Use as diagnostic, especially in Setting A.

## Oracle goal-visible baseline

For hidden-goal Setting A, evaluate query-only with goal visible as an upper bound.

---

# Logging

Reuse existing wandb/logging infrastructure.

Log:

```text
all losses
all memory metrics
support intervention metrics
setting name
family names / split names
query_goal_visible
support_goal_visible
sample_same_task_instance
sample_same_task_name
model_type
memory_update_type
write_steps
inner_lr
beta_write_aux
beta_rank
```

Save per-task/family results, not only aggregate numbers.

---

# Go/no-go criteria

Codex should add comments/config notes for the following expected interpretations.

## For Setting A

This setting should show support necessity. If:

```text
support_encoder_no_update <= query_only_hidden_goal
```

then support inference is not working.

If:

```text
memory_update does not improve support_encoder_no_update
```

then MAML/GradMem is not useful yet, but the encoder-based ICIL path may still be valid.

## For Setting B

Query-only may be strong. Success is measured by:

```text
support-conditioned improvement on ambiguous/low-data examples
wrong-support gap
held-out goal robustness
```

## For Setting C

If all models fail on held-out families, do not hide the query goal. First verify visible-goal support encoder works.

---

# Implementation priorities

Implement in this order:

1. Object-centric state tokenizer.
2. Setting A/B/C dataloader/config semantics.
3. Query-only baseline.
4. Support encoder memory model without MAML.
5. Support intervention diagnostics: correct/wrong/null support.
6. Memory WRITE update on encoder-produced memory.
7. No-encoder GradMem diagnostic.
8. Ranking loss and FiLM/AdaLN memory conditioning.
9. Full MAML if second-order is feasible; otherwise FOMAML + write auxiliary.

Do not start with full MAML before the support encoder no-update model and diagnostics are working.

---

# Notes on correctness

Be careful with Setting A vs B:

```text
Setting A:
    support/query same instance
    query goal hidden is valid

Setting B:
    support/query different goal
    query goal must be visible
```

If query goal is hidden in Setting B, the task may be impossible because support does not specify the query goal.

Be careful with FOMAML and separate WRITE heads:

```text
If create_graph=False and WRITE head is separate,
READ loss does not train WRITE head through the inner update.
Therefore add WRITE auxiliary loss or use shared head.
```

Be careful to evaluate with wrong/null support. Otherwise a model may look successful while ignoring support.

