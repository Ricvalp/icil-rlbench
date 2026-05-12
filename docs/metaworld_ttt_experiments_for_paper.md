# MetaWorld Test-Time Training Experiments for the Paper

This note summarizes the MetaWorld-only experimental setup that should replace the older RLBench-centric experiment plan. It is intended as source material for updating the paper's experiment section. The central paper story is **robotic imitation policies trained for test-time training (TTT) via meta-learning**, not merely feed-forward in-context conditioning. The experimental narrative should therefore emphasize where inner-loop adaptation helps, while also reporting strong non-TTT support-conditioning baselines honestly.

## High-Level Experimental Message

We evaluate whether a policy can use a small set of in-context demonstrations at test time to adapt its internal memory and improve closed-loop control. The experiments are organized into three settings of increasing difficulty:

| Setting | Name | Main Question | Difficulty |
|---|---|---|---|
| A | Hidden-goal, fixed-goal/random-start | Can support demonstrations reveal a hidden task goal and enable control from new starts? | Easiest, strongest simulation evidence |
| B | Seen-family, different-goal | Can support from a seen task family condition/adapt the policy to a different goal instance? | Medium, strong offline MSE evidence |
| C | Held-out-family, different-goal | Does the trained policy adapt to unseen MetaWorld task families? | Hardest, current simulation evidence weak |

The most paper-useful result so far is Setting A: for an A3 support-encoder + FOMAML checkpoint, same-support adaptation improves closed-loop success from about `5%` to `90%`, while wrong-support adaptation is much lower (`15%`). This directly supports the claim that test-time adaptation on demonstrations can substantially improve a MAML-trained robotic policy.

The important caveat is that a strong feed-forward support encoder baseline (A2) also achieves high Setting A success (`~85%` in the current 20-episode run). Therefore, the strongest defensible claim at present is:

> Test-time adaptation dramatically improves the MAML-trained policy over its unadapted state and is support-specific; in the easiest hidden-goal setting it matches or slightly improves over a strong support-encoder-only baseline.

Avoid claiming, unless additional results support it, that TTT uniformly dominates feed-forward in-context conditioning.

## Dataset and Task Construction

### Benchmark

All current paper experiments use **MetaWorld**, not RLBench. MetaWorld is substantially simpler and more controllable for in-context imitation learning experiments. We use scripted expert policies from MetaWorld to generate demonstration caches.

The policy sees low-dimensional object-centric state, not images or point clouds. The environment action is a 4D continuous action: 3D end-effector delta/control plus gripper command.

### Observation Used by the Model

The raw MetaWorld observation is represented as a 39D state. We use an object-centric parser over this state. The slices used by the object-centric encoder are:

| Quantity | Slice | Meaning |
|---|---:|---|
| Hand position | `0:3` | End-effector xyz |
| Gripper scalar | `3:4` | Gripper/opening state |
| Object 1 position | `4:7` | Primary object xyz |
| Object 2 position | `11:14` | Secondary object xyz, if meaningful |
| Goal position | `36:39` | Explicit target xyz when visible |

The model can hide the goal by zeroing/masking the `36:39` goal slice before tokenization. In Setting A, both support and query goals are hidden for the main runs. In Settings B/C, the goal is visible in support and query for the main runs.

### Demonstration Chunks

The dataset supplies chunks rather than full trajectories to the training loop:

| Parameter | Value | Meaning |
|---|---:|---|
| `K` | 4 | Number of support demonstration chunks per task episode/task specification |
| `T_obs` | 2 | Number of observation timesteps in each query/support chunk |
| `H` | 8 | Action horizon predicted by the policy |
| `stride` | 1 | State chunk stride |
| `action_stride` | 1 | Action chunk stride |
| `action_representation` | `absolute` | Direct action chunk regression target |

During meta-training, an outer batch contains multiple task episodes/specifications. For each task, the inner loop uses support chunks and the outer/read loss uses query chunks.

### Cache Variants

We currently rely on three cache roots:

| Cache | Used For | Semantics |
|---|---|---|
| `ICIL_METAWORLD_ML45_FIXED_GOAL_RANDOM_START_CACHE_ROOT` | Setting A | Same goal/task instance with randomized starts; filtered to 16 task families where fixed-goal/random-start generation actually changes trajectories |
| `ICIL_METAWORLD_ML45_GOAL_CACHE_ROOT` | Setting B | ML45 train task families; one successful episode per task instance/goal |
| `ICIL_METAWORLD_ML45_GOAL_TEST_CACHE_ROOT` | Setting C | ML45 held-out test task families; one successful episode per task instance/goal |

### Why Setting A Uses Only 16 Task Families

When generating fixed-goal/random-start demonstrations, many MetaWorld task families remain byte-identical across repeated episodes because the reset vector cannot be cleanly split into a fixed goal and independently randomized start. We diagnosed this with `diagnose_same_instance_cache`. For Setting A we therefore restrict to the 16 non-identical families:

```text
basketball-v3
coffee-pull-v3
coffee-push-v3
peg-insert-side-v3
pick-place-v3
pick-place-wall-v3
push-back-v3
push-v3
push-wall-v3
reach-v3
reach-wall-v3
shelf-place-v3
soccer-v3
stick-pull-v3
stick-push-v3
sweep-into-v3
```

This matters for paper validity: Setting A is not all of ML45; it is a curated subset where hidden-goal/random-start evaluation is meaningful.

## Settings A, B, and C

### Setting A: Hidden Goal, Same Goal, Randomized Starts

**Purpose:** Test whether demonstrations reveal a hidden goal and whether TTT can adapt the policy to that hidden goal.

- Train/evaluate on fixed-goal/random-start cache.
- Support and query are from the same task family and same task instance/goal.
- Support and query have different randomized starts.
- Explicit goal xyz is hidden in both support and query: `support_zero_goal=True`, `query_zero_goal=True`.
- Evaluation also uses fixed-goal/random-start reset in simulation, not the default MetaWorld frozen reset.

This is the cleanest setting for demonstrating support-specific hidden-goal adaptation in closed-loop simulation.

### Setting B: Seen Task Families, Different Goal Instances

**Purpose:** Test in-distribution adaptation/conditioning across goals within task families seen during training.

- Train/evaluate on ML45 train task families.
- Support and query come from the same task family but different task instances/goals: `sample_same_task_instance=False`.
- Goal is visible in support and query: `support_zero_goal=False`, `query_zero_goal=False`.
- This tests whether support helps condition/adapt the policy beyond query-only information in seen families.

This setting currently gives strong offline MSE evidence for support-specific adaptation, especially for B1/B2/B3, but simulation numbers need to be collected/reported selectively.

### Setting C: Held-Out Task Families

**Purpose:** Test few-shot/OOD generalization to MetaWorld task families not seen during training.

- No separate C training run is required for the current evaluation protocol.
- We evaluate B-trained checkpoints on `ICIL_METAWORLD_ML45_GOAL_TEST_CACHE_ROOT`.
- Support and query come from the same held-out task family but different task instances/goals.
- Goal is visible in support and query.

This is the hardest setting. Current offline MSE suggests C3 is the cleanest OOD candidate, but closed-loop success remains low. If reported, C should be framed as an OOD stress test, not the main success result.

## Architecture Family

All reported current experiments use the JAX MetaWorld object-centric query-memory model.

### Query/Object-Centric Encoder

Each query state chunk is converted into object-centric tokens. Conceptually:

```text
hand_token        = MLP_hand([hand_xyz, gripper])
object1_token     = MLP_obj(object1_xyz)
object2_token     = MLP_obj(object2_xyz)
goal_token        = MLP_goal(goal_xyz), masked/zeroed when hidden
obj1_hand_token   = MLP_rel(object1_xyz - hand_xyz)
obj2_hand_token   = MLP_rel(object2_xyz - hand_xyz)
goal_hand_token   = MLP_rel(goal_xyz - hand_xyz), masked when hidden
goal_obj1_token   = MLP_rel(goal_xyz - object1_xyz), masked when hidden
goal_obj2_token   = MLP_rel(goal_xyz - object2_xyz), masked when hidden
```

The implementation uses the object-centric encoder configured by `model.query_encoder_name='object_centric_state'`, with `d_model=256`.

### Query-Memory Direct Regression Decoder

The decoder predicts an action chunk directly:

```text
query state tokens + memory tokens -> action chunk [H, action_dim]
```

Main configuration:

| Component | Value |
|---|---:|
| Model width | `d_model=256` |
| Attention heads | `n_heads=4` |
| Decoder layers | `4` |
| Memory tokens | `64` |
| Action horizon | `H=8` |
| Conditioning mode | `cross_attn_plus_film` |
| Action loss | L1/direct action regression |

The decoder uses memory via cross-attention plus FiLM/AdaLN-style conditioning. This is important: the memory has an explicit route to modulate the action decoder, not only a weak appended-token path.

### Memory Tokens

The policy has a learned base memory:

```text
M_base = learned memory token matrix
```

Depending on the experiment variant, this base memory can be used alone, combined with a support encoder, and/or adapted by MAML/FOMAML inner-loop gradient steps.

### Support Encoder

For support-encoder variants, support demonstration chunks are encoded into memory-shaped tokens. The default memory initialization is additive:

```text
M0 = M_base + SupportEncoder(support_chunks)
```

The support encoder receives support state chunks and support action chunks. It is chunk-based rather than full-trajectory-based. This is a pragmatic choice: it reuses the training dataloader and makes the experiments faster. It is still defensible because the support context is explicitly a set of sampled demonstration segments.

Support encoder configuration:

| Component | Value |
|---|---:|
| Width | `d_model=256` |
| Attention heads | `4` |
| Output memory tokens | `64` |
| Support encoder layers | `2` |
| Memory self-attention layers | `1` |
| Max support chunks | `256` |

### Test-Time Training / Inner Loop

For MAML/FOMAML variants, the inner loop adapts only the memory tokens. Model weights, decoder weights, object-centric encoder weights, and support encoder weights are fixed at test time.

The inner update uses support chunks and the write loss:

```text
M0 -> inner write/read computation on support -> gradient wrt memory -> M1
```

Main inner-loop configuration:

| Parameter | Value |
|---|---:|
| Inner steps | `2` for MAML/FOMAML variants |
| Inner learning rate | `3e-2` |
| Max grad norm / memory update clip | `1.0` |
| Inner loss mode | `write` |
| First-order flag | `True` for FOMAML, `False` for full MAML |

A key diagnostic is whether the same checkpoint performs better with inner updates enabled than with `inner_steps=0`. That is the cleanest way to isolate TTT from support encoding.

## Model Variants

The same variant numbers are reused across settings. For Setting C, the current protocol evaluates B-trained checkpoints on held-out C data.

| Variant | Name | Memory Initialization | Inner Update | Purpose |
|---|---|---|---|---|
| `*0` | Query-only baseline | `M_base` | None | Measures query-only policy performance |
| `*1` | No-encoder GradMem/FOMAML | `M_base` | FOMAML | Tests whether TTT alone works without support encoder |
| `*2` | Support encoder, no update | `M_base + SupportEncoder(C)` | None | Strong feed-forward in-context conditioning baseline |
| `*3` | Support encoder + FOMAML | `M_base + SupportEncoder(C)` | First-order memory update | Main TTT model with support-conditioned initialization |
| `*4` | Support encoder + full MAML | `M_base + SupportEncoder(C)` | Second-order memory update | Tests whether full MAML improves over FOMAML |

Setting-specific examples:

| Experiment | Meaning |
|---|---|
| A0 | Hidden-goal fixed-goal/random-start query-only baseline |
| A1 | Hidden-goal no-encoder memory TTT baseline |
| A2 | Hidden-goal support encoder only |
| A3 | Hidden-goal support encoder + FOMAML |
| A4 | Hidden-goal support encoder + full MAML |
| B0 | Seen-family visible-goal query-only baseline |
| B1 | Seen-family support encoder only |
| B2 | Seen-family support encoder + FOMAML |
| B3 | Seen-family support encoder + full MAML |
| B4 | Seen-family no-encoder memory TTT baseline |
| C0-C4 | Same trained B0-B4 checkpoints evaluated on held-out ML45 test families |

## Metrics

### Offline MSE/L1

Offline metrics evaluate predicted action chunks against cached expert action chunks. We report:

- `MSE`: mean squared error over action chunks.
- `L1`: mean absolute error over action chunks.
- Best-condition counts over evaluation batches.

For each checkpoint we evaluate three memory/adaptation conditions:

| Condition | Meaning |
|---|---|
| `no_adaptation` | Base memory only; no support-conditioned initialization and no inner update |
| `same_family_adaptation` | Correct support from same task family/goal semantics |
| `wrong_family_adaptation` | Incorrect support from a different task family; support-specificity control |

Important caveat: offline action MSE can be misleading because short action chunks can be locally predictable even when closed-loop behavior fails. Simulation success is the primary robotics metric.

### Simulation Success

Closed-loop evaluation reports:

- success rate over episodes
- mean return
- max reward
- videos/GIFs for qualitative inspection

Use success rate as the main result. Dense reward/return is secondary because MetaWorld reward shaping can be misleading; wrong-support runs can have nontrivial return without solving the task.

## Current Preliminary Results

These numbers are from the latest quick MSE sweep with `num_batches=16`, `batch_size=64`.

### Offline MSE Table: Setting A

| Setting | Run | Checkpoint | MSE no | MSE same | MSE wrong | Best counts |
|---|---|---:|---:|---:|---:|---|
| A | A0 | `7hy5ph2l/step_0100000` | **0.0749** | 0.0749 | 0.0749 | no 16 |
| A | A1 | `zrcu9r0y/step_0030000` | 0.2413 | 0.2177 | **0.2160** | mixed |
| A | A2 | `psa4nu0u/step_0052000` | 3.8071 | **0.1790** | 1.9218 | same 16 |
| A | A3 | `izt0zuhl/step_0032000` | 2.4989 | **0.1980** | 2.0931 | same 16 |
| A | A4 | `63f7osmk/step_0022000` | 2.2422 | **0.2163** | 2.2060 | same 16 |

Interpretation:

- A2, A3, and A4 are support-specific: same support is much better than wrong support.
- A3/A4 are improving, but offline MSE does not yet show a decisive advantage over A2.
- A1 is not useful for TTT-only evidence: same and wrong are effectively tied.
- A0's low MSE should not be overinterpreted because query-only policies can fit local action chunks while failing closed-loop hidden-goal control.

### Offline MSE Table: Setting B

| Setting | Run | Checkpoint | MSE no | MSE same | MSE wrong | Best counts |
|---|---|---:|---:|---:|---:|---|
| B | B0 | `o6emiqtj/step_0100000` | **0.1615** | 0.1615 | 0.1615 | no 16 |
| B | B1 | `gbzubxl9/step_0058000` | 5.9221 | **0.1230** | 7.3258 | same 16 |
| B | B2 | `5rxlzxyl/step_0036000` | 10.9258 | **0.5717** | 7.9246 | same 16 |
| B | B3 | `j6vptohf/step_0038000` | 12.2091 | **0.5491** | 7.7367 | same 16 |
| B | B4 | `pb093u2p/step_0036000` | 0.7101 | **0.6968** | 0.7010 | mixed |

Interpretation:

- B1 is currently the strongest seen-family offline model.
- B2 and B3 show clear support-specific adaptation and are improving.
- B3 is slightly better than B2 in this latest sweep, which is useful for the full-MAML story.
- B4 is weak: no/same/wrong are too close.

### Offline MSE Table: Setting C

| Setting | Run | Checkpoint | MSE no | MSE same | MSE wrong | Best counts |
|---|---|---:|---:|---:|---:|---|
| C | C0 | `o6emiqtj/step_0100000` | 3.5633 | 3.5633 | 3.5633 | no 16 |
| C | C1 | `gbzubxl9/step_0058000` | 11.6411 | 2.3753 | **1.4529** | wrong better |
| C | C2 | `5rxlzxyl/step_0036000` | 2.4185 | **1.4069** | 1.4223 | same 9, wrong 7 |
| C | C3 | `j6vptohf/step_0038000` | 6.1986 | **1.3657** | 1.5299 | same 13, wrong 3 |
| C | C4 | `pb093u2p/step_0036000` | **6.0427** | 6.3349 | 6.1556 | no/wrong better |

Interpretation:

- C3 is currently the cleanest held-out/OOD offline result: same support beats wrong support and wins 13/16 batches.
- C2 has a slightly weaker support-specific gap.
- C1 is not clean: wrong support is better in aggregate.
- Simulation for C3 currently shows only weak improvement and low absolute success, so C should not be the main closed-loop claim.

### Current Simulation Results

Current reported closed-loop results are partial and should be rerun with more episodes before final tables.

| Setting | Checkpoint | Condition | Success | Notes |
|---|---|---|---:|---|
| A3 | `izt0zuhl/step_0032000` | no adaptation | 0.05 | Base memory only, almost fails |
| A3 | `izt0zuhl/step_0032000` | same support adaptation | **0.90** | Strong TTT result |
| A3 | `izt0zuhl/step_0032000` | wrong support adaptation | 0.15 | Support-specificity control |
| A2 | `psa4nu0u/step_0042000` | same support encoder only | 0.85 | Strong feed-forward support baseline |
| C3 | `j6vptohf/step_0038000` | no adaptation | 0.10 | Held-out-family sim remains hard |
| C3 | `j6vptohf/step_0038000` | same support adaptation | 0.20 | Weak improvement, low absolute success |

Important: for A3, we should also run the patched `inner_steps=0` evaluation. This isolates support encoder initialization from gradient-based TTT within the same checkpoint:

```text
A3 with inner_steps=0: support encoder active, no gradient update
A3 with inner_steps=2: support encoder active + FOMAML update
```

This is the cleanest ablation for showing that TTT adds value beyond support encoding inside the same trained policy.

## Recommended Paper Tables

Use a small number of clear tables. Do not dump every intermediate checkpoint.

### Table 1: Setting A Closed-Loop Success

This should be the main robotics result.

Columns:

| Method | Support encoder | TTT | Correct support success | Wrong support success | No-adapt success |
|---|---|---|---:|---:|---:|
| A0 Query-only | No | No | n/a | n/a | ... |
| A1 GradMem/FOMAML only | No | Yes | ... | ... | ... |
| A2 Support encoder | Yes | No | ... | ... | ... |
| A3 Support encoder + FOMAML | Yes | Yes | **...** | ... | ... |
| A4 Support encoder + full MAML | Yes | Yes | ... | ... | ... |

Recommended emphasis:

- Bold the highest correct-support success.
- Also bold/mark the largest correct-vs-wrong support gap if different.
- Include confidence intervals or standard error if running 50/100 episodes.

This table supports the main claim if A3/A4 show strong gains over no-adapt and wrong-support, and ideally match/beat A2.

### Table 2: Setting A Inner-Step Ablation for A3

This is the most direct TTT table.

| Checkpoint | Inner steps | Support encoder | Success | Interpretation |
|---|---:|---|---:|---|
| A3 | 0 | Yes | ... | Feed-forward support memory only |
| A3 | 2 | Yes | **0.90** | FOMAML TTT enabled |

If `inner_steps=2` is much better than `inner_steps=0`, this directly supports TTT beyond support encoding. If they are similar, frame the result as support-conditioned ICIL rather than TTT dominance.

### Table 3: Offline MSE Across A/B/C

Use only representative methods, not all variants if space is limited:

| Setting | Query-only | Support encoder | Support encoder + FOMAML | Support encoder + full MAML | Wrong-support control |
|---|---:|---:|---:|---:|---:|
| A | A0 | A2 | A3 | A4 | wrong-support MSE |
| B | B0 | B1 | B2 | B3 | wrong-support MSE |
| C | C0 | C1 | C2 | C3 | wrong-support MSE |

Recommended choice:

- Report same-support MSE as the main number.
- Include wrong-support MSE either as a separate column or in parentheses.
- Bold best same-support MSE per setting.
- For C, be cautious: C3 is the cleanest support-specific result, not necessarily the absolute best in every metric.

### Table 4: Held-Out Family Simulation, If Included

Only include if we have enough episodes and support-specific gain is real.

| Method | No-adapt success | Same-support success | Wrong-support success |
|---|---:|---:|---:|
| C1 support encoder only | ... | ... | ... |
| C2 support encoder + FOMAML | ... | ... | ... |
| C3 support encoder + full MAML | ... | ... | ... |

If success remains low, present this as an OOD stress test rather than a headline result.

## Recommended Experiment Section Structure

### 1. Experimental Goal

State that the aim is to evaluate whether meta-trained robotic imitation policies can improve at test time from demonstration support sets via memory-only MAML/FOMAML updates.

Suggested wording:

> We study whether a policy can use a small set of in-context demonstrations not only as feed-forward conditioning, but as test-time training data for adapting a compact memory state. All inner-loop updates are applied only to memory tokens, leaving the policy network fixed.

### 2. MetaWorld Dataset and Task Splits

Describe:

- MetaWorld scripted-policy demonstrations.
- 39D low-dimensional state and 4D actions.
- Chunked support/query construction.
- ML45 train/test family split.
- Fixed-goal/random-start cache for Setting A.
- Why Setting A uses 16 non-identical task families.

### 3. Model Architecture

Describe the object-centric query encoder, memory tokens, support encoder, and decoder.

Explicitly distinguish:

- feed-forward support conditioning: `M0 = M_base + SupportEncoder(C)`
- TTT: gradient updates to memory tokens from support chunks

This distinction is crucial for the paper because support conditioning alone is a strong baseline.

### 4. Settings A/B/C

Present A/B/C in a compact table. The most important dimensions are:

| Setting | Train families | Eval families | Support/query relation | Goal visibility | Main claim |
|---|---|---|---|---|---|
| A | ML45 train subset | Same subset | Same goal, different start | Hidden | hidden-goal TTT |
| B | ML45 train | ML45 train | Same family, different goal | Visible | in-distribution adaptation |
| C | ML45 train | ML45 test | Held-out family, different goal | Visible | OOD adaptation |

### 5. Baselines and Ablations

List A0-A4/B0-B4/C0-C4. Make clear which are trained models and which are evaluation settings.

Important nuance:

- C rows are held-out evaluations of the B-trained models, not separately trained on test families.

### 6. Results

Recommended ordering:

1. Setting A simulation success table: strongest result.
2. A3 inner-step ablation: direct TTT evidence.
3. Offline A/B/C MSE: broader trend and support-specificity.
4. Optional C simulation: OOD remains hard.

### 7. Discussion and Limitations

Be direct:

- TTT clearly helps within the MAML-trained A3 policy in Setting A.
- Feed-forward support encoding is a very strong baseline.
- Current no-encoder GradMem/FOMAML is weak.
- Held-out task-family closed-loop generalization remains limited.
- Offline MSE does not always predict closed-loop success.

This honesty improves credibility and prevents reviewers from interpreting the work as overclaiming.

## What Still Needs to Be Run Before Finalizing Tables

### Must Run

1. A0 simulation on Setting A.
2. A2 simulation with 50 or 100 episodes.
3. A3 simulation with 50 or 100 episodes.
4. A3 `inner_steps=0` simulation with 50 or 100 episodes.
5. A4 simulation once it trains longer.

### Useful If Time Allows

1. B1/B2/B3 simulation on Setting B.
2. C2/C3 simulation with more episodes.
3. Per-task success breakdown for Setting A and C.
4. Qualitative GIF panels: no-adapt vs same-support vs wrong-support.

## Current Best Candidate Checkpoints

| Purpose | Checkpoint |
|---|---|
| A2 support encoder only | `eval_checkpoints/psa4nu0u/step_0052000.pkl` |
| A3 support encoder + FOMAML | `eval_checkpoints/izt0zuhl/step_0032000.pkl` |
| A4 support encoder + full MAML | `eval_checkpoints/63f7osmk/step_0022000.pkl` |
| B1 support encoder only | `eval_checkpoints/gbzubxl9/step_0058000.pkl` |
| B2 support encoder + FOMAML | `eval_checkpoints/5rxlzxyl/step_0036000.pkl` |
| B3 support encoder + full MAML | `eval_checkpoints/j6vptohf/step_0038000.pkl` |
| C3 OOD candidate | `eval_checkpoints/j6vptohf/step_0038000.pkl` evaluated with Setting C |

## Suggested Claims Based on Current Evidence

Safe claims:

1. Meta-learned memory adaptation can substantially improve closed-loop robotic imitation performance in a hidden-goal MetaWorld setting.
2. The improvement is support-specific: correct support greatly outperforms wrong support.
3. A support encoder is a strong in-context conditioning mechanism and should be included as a baseline.
4. Full MAML/FOMAML variants show clear offline support-specific adaptation in seen-family and held-out-family settings.
5. Held-out-family simulation remains challenging; offline improvement does not yet guarantee robust closed-loop OOD success.

Claims to avoid unless more evidence arrives:

1. TTT universally outperforms feed-forward support encoding.
2. No-encoder memory-gradient adaptation alone is sufficient.
3. Strong OOD task-family simulation generalization.
4. Offline MSE directly predicts closed-loop success.

## Suggested Paper Framing

The best framing is not "TTT always beats in-context conditioning." The best framing is:

> We introduce a memory-based meta-learning formulation for robotic imitation in which test-time updates adapt only compact memory tokens. In MetaWorld hidden-goal tasks, memory adaptation from demonstrations produces large support-specific gains in closed-loop success. Feed-forward support encoding is a strong baseline, but MAML/FOMAML-style memory adaptation provides a principled test-time training mechanism and yields the clearest within-model improvement. More difficult held-out-family settings expose remaining limitations of OOD robotic meta-imitation.

This positions the results as credible and useful even if A2 remains competitive with A3/A4.
