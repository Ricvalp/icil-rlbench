from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from icil.models.policies.policy import Policy


def _resolve_timesteps(
    *,
    batch: Dict[str, torch.Tensor],
    num_train_timesteps: int,
) -> torch.Tensor:
    x0 = batch["target_action"]
    batch_size = int(x0.shape[0])
    device = x0.device
    provided = batch.get("timesteps", None)
    if provided is None:
        return torch.randint(
            low=0,
            high=int(num_train_timesteps),
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )

    t = provided.to(device=device, dtype=torch.long)
    if t.ndim == 0:
        t = t.view(1)
    if t.shape == (1,) and batch_size > 1:
        t = t.expand(batch_size)
    if t.shape != (batch_size,):
        raise ValueError(
            f"timesteps must have shape ({batch_size},), got {tuple(t.shape)}."
        )
    return t


def _resolve_noise(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    x0 = batch["target_action"]
    batch_size = int(x0.shape[0])
    provided = batch.get("noise", None)
    if provided is None:
        return torch.randn_like(x0)

    noise = provided.to(device=x0.device, dtype=x0.dtype)
    if noise.ndim == 2:
        noise = noise.unsqueeze(0)
    if noise.shape[0] == 1 and batch_size > 1:
        noise = noise.expand(batch_size, -1, -1)
    if noise.shape != x0.shape:
        raise ValueError(
            f"noise must have shape {tuple(x0.shape)}, got {tuple(noise.shape)}."
        )
    return noise


def compute_policy_loss(
    policy: Policy,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    x0 = batch["target_action"]
    t = _resolve_timesteps(
        batch=batch,
        num_train_timesteps=int(policy.noise_scheduler.config.num_train_timesteps),
    )
    noise = _resolve_noise(batch)
    x_t = policy.noise_scheduler.add_noise(x0, noise, t)

    model_out = policy.predict_model_output(
        x_t=x_t,
        t=t,
        cond_xyz=batch.get("cond_xyz", None),
        query_xyz=batch["query_xyz"],
        query_state=batch["query_state"],
        cond_state=batch.get("cond_state", None),
        cond_traj=batch.get("cond_traj", None),
        cond_traj_mask=batch.get("cond_traj_mask", None),
        cond_rgb=batch.get("cond_rgb", None),
        query_rgb=batch.get("query_rgb", None),
        cond_mask_id=batch.get("cond_mask_id", None),
        query_mask_id=batch.get("query_mask_id", None),
        cond_valid=batch.get("cond_valid", None),
        query_valid=batch.get("query_valid", None),
    )

    pred_type = str(policy.noise_scheduler.config.prediction_type)
    if pred_type == "epsilon":
        target = noise
    elif pred_type == "sample":
        target = x0
    elif pred_type == "v_prediction":
        if hasattr(policy.noise_scheduler, "get_velocity"):
            target = policy.noise_scheduler.get_velocity(x0, noise, t)
        else:
            alpha_t = policy.noise_scheduler.alphas_cumprod[t].sqrt().to(x0.device)
            sigma_t = (1.0 - policy.noise_scheduler.alphas_cumprod[t]).sqrt().to(x0.device)
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            target = alpha_t * noise - sigma_t * x0
    else:
        raise ValueError(f"Unsupported prediction type {pred_type}.")

    loss = F.mse_loss(model_out, target)
    return {
        "loss": loss,
        "mse": loss.detach(),
        "t_mean": t.float().mean().detach(),
    }


class PolicyLossWrapper(nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return compute_policy_loss(self.policy, batch)["loss"]
