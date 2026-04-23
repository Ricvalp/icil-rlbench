# Direct Regression Action Head

This document describes the direct-regression action head used by the `chunk`
policy in this repository.

The goal is to make the architecture easy to reimplement in another project.
This writeup focuses on the policy head itself, but it also describes the
minimum encoder and task-conditioning interfaces that the head expects.

Relevant source files in this repo:

- `supernode_tokenizer/models/policies/chunk_decoder_policy.py`
- `supernode_tokenizer/models/common/blocks.py`
- `supernode_tokenizer/models/condition/task_conditioner.py`
- `supernode_tokenizer/trainers/train_chunk.py`
- `configs/train_chunk_policy.py`


## 1. What This Head Does

The direct-regression action head predicts an entire future action chunk in one
forward pass.

It takes:

- an encoded observation memory
- a task-conditioning vector
- a fixed action horizon `H`

and outputs:

- `H` future action vectors of dimension `A`

In tensor form:

- input observation tokens: `[B, M, D]`
- output predicted action chunk: `[B, H, A]`

This is different from a diffusion action head, which starts from noise and
iteratively denoises an action chunk over many sampling steps. The direct
regression head is a single-pass decoder.


## 2. High-Level Structure

The policy has four logical parts:

1. Task conditioning
2. Observation encoding into memory tokens
3. A decoder initialized from learned action queries
4. A final linear projection to actions

At a high level:

```text
obs window + task id
    -> task embedding + task tokens
    -> observation encoder
    -> memory = concat(observation_tokens, task_tokens)
    -> learned action queries
    -> N layers of:
         self-attn on queries
         cross-attn from queries to memory
         MLP
    -> linear projection
    -> action chunk [B, H, A]
```


## 3. Inputs and Outputs

### Inputs

The head expects the following batched tensors:

- `task_ids`: `[B]`
- `obs_xyz`: `[B, T_obs, N, 3]`
- `obs_state`: `[B, T_obs, S]`
- `obs_valid`: `[B, T_obs, N]` optional
- `obs_rgb`: `[B, T_obs, N, 3]` optional
- `obs_mask_id`: `[B, T_obs, N]` optional

The head itself does not care how the encoder turns these into tokens. It only
assumes the encoder returns:

- `tokens`: `[B, M_obs, D]`
- optional `token_mask`: `[B, M_obs]`

### Outputs

- predicted action chunk: `[B, H, A]`

where:

- `B` = batch size
- `T_obs` = number of observed frames
- `N` = number of points per frame
- `S` = low-dimensional robot state dimension
- `M_obs` = number of observation tokens produced by the encoder
- `D` = model width
- `H` = action horizon
- `A` = action dimension


## 4. Task Conditioning

Task conditioning in this repo is class-conditioned, not language-conditioned.

The task conditioner takes a discrete task id and returns:

- `task_emb`: `[B, D]`
- `task_tokens`: `[B, T_task, D]`

where `T_task` is a small fixed number of learned task tokens.

The implementation used here is:

```python
class TaskConditioner(nn.Module):
    def __init__(self, num_tasks: int, d_model: int, n_task_tokens: int, dropout: float = 0.0):
        super().__init__()
        self.task_embed = nn.Embedding(num_tasks, d_model)
        self.task_tokens = nn.Parameter(torch.randn(n_task_tokens, d_model) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, task_ids: torch.Tensor):
        task_emb = self.proj(self.task_embed(task_ids.long()))          # [B, D]
        task_tokens = self.task_tokens.unsqueeze(0).expand(task_ids.shape[0], -1, -1)
        task_tokens = self.drop(task_tokens + task_emb.unsqueeze(1))   # [B, T_task, D]
        return task_emb, task_tokens
```

The direct-regression head uses both outputs:

- `task_emb` is the conditioning vector for AdaLN inside each decoder block
- `task_tokens` are appended to encoder memory


## 5. Observation Memory

The action head does not encode raw observations by itself. It delegates that to
an observation encoder.

The required encoder interface is roughly:

```python
class ObservationEncoder(nn.Module):
    def forward(
        self,
        *,
        obs_xyz,
        obs_state,
        task_emb,
        obs_valid=None,
        obs_rgb=None,
        obs_mask_id=None,
        return_debug=False,
    ):
        return ObservationEncoderOutput(tokens=tokens, token_mask=token_mask)
```

The policy then builds memory as:

```python
task_emb, task_tokens = task_conditioner(task_ids)
enc_out = encoder(
    obs_xyz=obs_xyz,
    obs_state=obs_state,
    task_emb=task_emb,
    obs_valid=obs_valid,
    obs_rgb=obs_rgb,
    obs_mask_id=obs_mask_id,
)
memory = torch.cat([enc_out.tokens, task_tokens], dim=1)
```

If the encoder provides a token mask, the task tokens are marked valid and
appended to that mask:

```python
task_mask = torch.ones(task_tokens.shape[:2], device=task_tokens.device, dtype=torch.bool)
memory_mask = torch.cat([enc_out.token_mask.bool(), task_mask], dim=1)
```

So the final decoder memory is:

- `memory`: `[B, M_obs + T_task, D]`
- `memory_mask`: `[B, M_obs + T_task]` or `None`


## 6. Decoder Initialization With Learned Action Queries

The direct-regression head does not start from previous actions or from noise.
Instead, it uses one learned query per future action slot.

There are two learned parameters:

- `action_queries`: `[H, D]`
- `action_slot_embed`: `[H, D]`

At runtime:

```python
h = action_queries.unsqueeze(0).expand(B, -1, -1) + action_slot_embed.unsqueeze(0)
```

So the decoder state starts as:

- `h`: `[B, H, D]`

Each row of `h` corresponds to one future step in the predicted action chunk.


## 7. Decoder Block

The head uses a stack of task-conditioned cross-attention blocks.

In this repo, one block contains:

1. task-conditioned self-attention over the current action queries
2. task-conditioned cross-attention from action queries to observation memory
3. task-conditioned MLP

The conditioning mechanism is an identity-initialized AdaLN:

```python
class IdentityTaskAdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift
```

That identity initialization matters: at initialization, the block behaves like
plain pre-normalized attention/MLP, and the task conditioning is learned
gradually.

The full decoder block is:

```python
class TaskConditionedCrossAttentionBlock(nn.Module):
    def __init__(self, d, n_heads, cond_dim, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.adaln1 = IdentityTaskAdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout)
        self.adaln2 = IdentityTaskAdaLN(d, cond_dim)
        self.cross_attn = CrossAttention(d, n_heads, dropout)
        self.adaln3 = IdentityTaskAdaLN(d, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cond, memory, memory_mask=None):
        x = x + self.drop(self.self_attn(self.adaln1(x, cond)))
        x = x + self.drop(self.cross_attn(self.adaln2(x, cond), memory, kv_mask=memory_mask))
        x = x + self.drop(self.mlp(self.adaln3(x, cond)))
        return x
```

The direct-regression head applies `n_layers` of this block:

```python
for blk in self.decoder:
    h = blk(h, task_emb, memory, memory_mask)
```


## 8. Output Projection

After the decoder stack, each future slot representation is projected
independently to an action vector:

```python
pred = self.out(h)   # [B, H, A]
```

where:

- `h`: `[B, H, D]`
- `out`: `Linear(D, A)`
- `pred`: `[B, H, A]`


## 9. Minimal End-to-End Head

This is the closest minimal version of the direct-regression head from this
repo, with the encoder abstracted out:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectRegressionChunkHead(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        task_conditioner: nn.Module,
        d_model: int,
        action_dim: int,
        horizon: int,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        loss_type: str = "l1",
    ):
        super().__init__()
        self.encoder = encoder
        self.task_conditioner = task_conditioner
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.loss_type = str(loss_type)

        self.action_queries = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.action_slot_embed = nn.Parameter(torch.randn(self.horizon, d_model) * 0.02)
        self.decoder = nn.ModuleList(
            [
                TaskConditionedCrossAttentionBlock(
                    d=d_model,
                    n_heads=n_heads,
                    cond_dim=d_model,
                    mlp_mult=mlp_mult,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Linear(d_model, action_dim)

    def _build_memory(
        self,
        *,
        task_ids,
        obs_xyz,
        obs_state,
        obs_valid=None,
        obs_rgb=None,
        obs_mask_id=None,
    ):
        task_emb, task_tokens = self.task_conditioner(task_ids)
        enc_out = self.encoder(
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            task_emb=task_emb,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
        )
        memory = torch.cat([enc_out.tokens, task_tokens], dim=1)
        memory_mask = None
        if getattr(enc_out, "token_mask", None) is not None:
            task_mask = torch.ones(task_tokens.shape[:2], device=task_tokens.device, dtype=torch.bool)
            memory_mask = torch.cat([enc_out.token_mask.bool(), task_mask], dim=1)
        return memory, memory_mask, task_emb

    def forward(
        self,
        *,
        task_ids,
        obs_xyz,
        obs_state,
        obs_valid=None,
        obs_rgb=None,
        obs_mask_id=None,
    ):
        memory, memory_mask, task_emb = self._build_memory(
            task_ids=task_ids,
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
        )
        batch = task_ids.shape[0]
        h = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
        h = h + self.action_slot_embed.unsqueeze(0)
        for blk in self.decoder:
            h = blk(h, task_emb, memory, memory_mask)
        return self.out(h)

    def compute_loss(self, batch):
        pred = self(
            task_ids=batch["task_id"],
            obs_xyz=batch["obs_xyz"],
            obs_state=batch["obs_state"],
            obs_valid=batch.get("obs_valid"),
            obs_rgb=batch.get("obs_rgb"),
            obs_mask_id=batch.get("obs_mask_id"),
        )
        target = batch["target_action"]
        if self.loss_type.lower() == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = F.l1_loss(pred, target)
        return {
            "loss": loss,
            "l1": F.l1_loss(pred, target),
            "mse": F.mse_loss(pred, target),
            "pred_action": pred,
        }
```


## 10. How It Is Trained in This Repo

### Data target

The dataset returns:

- observation window built from `T_obs` frames
- future action chunk of length `H`

The training item contains:

- `task_id`
- `obs_xyz`
- `obs_state`
- `obs_valid`
- optional `obs_rgb`
- optional `obs_mask_id`
- `target_action`

The target action shape is:

- `target_action`: `[H, A]` per sample
- batched to `[B, H, A]`

The dataset in this repo constructs that chunk by:

- selecting an observation start time `t0`
- taking `T_obs` observation frames
- taking the next `H` action steps after the last observed frame
- clipping the action indices at the episode end

So this is standard fixed-horizon chunk prediction.

### Loss

Default training uses L1 regression:

```python
loss = F.l1_loss(pred, target)
```

The config allows switching to MSE:

```python
cfg.model.chunk_decoder.loss_type = "mse"
```

Even when training with L1, the trainer also logs:

- `l1`
- `mse`

### Optimizer and training defaults

From this repo's default config:

- optimizer: `AdamW`
- learning rate: `1e-4`
- betas: `(0.9, 0.95)`
- weight decay: `1e-4`
- gradient clipping: `1.0`
- AMP enabled

Default chunk-training settings:

- `num_steps = 300000`
- `batch_size = 128`
- `val_batch_size = 32`
- `num_workers = 8`

### Training loop

The chunk head is trained exactly as a supervised sequence regressor:

```python
optimizer.zero_grad(set_to_none=True)

with autocast():
    out = model.compute_loss(batch)
    loss = out["loss"]

loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

There is no:

- behavior cloning with autoregressive teacher forcing
- diffusion schedule
- iterative rollout inside training
- value loss
- actor-critic objective

It is plain imitation learning over action chunks.


## 11. Inference

Inference is also simple:

```python
pred = model(
    task_ids=task_ids,
    obs_xyz=obs_xyz,
    obs_state=obs_state,
    obs_valid=obs_valid,
    obs_rgb=obs_rgb,
    obs_mask_id=obs_mask_id,
)  # [B, H, A]
```

The repo exposes this as:

```python
sample_actions(...)
```

but for the chunk head, `sample_actions()` is just a direct call to the forward
pass. There is no stochastic sampling loop.

In rollout evaluation, the environment typically executes only the first few
actions of the predicted chunk, then replans from the updated observation.


## 12. Default Hyperparameters Used Here

For the direct-regression head itself:

- `d_model = 512`
- `n_heads = 8`
- `n_layers = 8`
- `mlp_mult = 4`
- `dropout = 0.0`
- `horizon = 16`
- `loss_type = "l1"`

Task-conditioning defaults:

- `n_task_tokens = 4`


## 13. What Is Specific to This Repo vs Reusable

Reusable ideas:

- learned action query decoder
- task-conditioned self+cross-attention stack
- appending task tokens to encoder memory
- direct chunk regression with L1 or MSE

Repo-specific parts:

- the exact point-cloud encoders
- RLBench dataset formatting
- discrete task-id conditioning
- low-dimensional state layout

If you reimplement this elsewhere, the head itself only needs:

- `memory` tokens from an encoder
- a conditioning vector `task_emb`
- optional memory mask
- future horizon `H`
- action dimension `A`


## 14. Difference From the Diffusion Head

The diffusion head in this repo keeps the same encoder and task-conditioning
strategy, but replaces the direct chunk decoder with:

- noisy action inputs
- timestep conditioning
- diffusion scheduler
- iterative denoising at inference

The direct-regression head removes all of that complexity. It is just:

```text
encoded observation memory -> action queries -> decoder -> action chunk
```

That is why it is the easier head to port into another codebase.
