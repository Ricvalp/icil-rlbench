from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

try:
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
except Exception as exc:  # pragma: no cover - dependency/runtime environment specific
    raise ImportError(
        "icil.models.dp3 requires diffusers. Install it with `pip install diffusers`."
    ) from exc


def dict_apply(
    x: Dict[str, torch.Tensor],
    func: Callable[[torch.Tensor], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in x.items():
        out[k] = dict_apply(v, func) if isinstance(v, dict) else func(v)
    return out


class ModuleAttrMixin(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._dummy_variable = nn.Parameter(torch.empty(()))

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.parameters())).dtype


class _IdentityFieldNormalizer:
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class IdentityNormalizer:
    def normalize(self, x: Union[Dict[str, torch.Tensor], torch.Tensor]) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        return x

    def __getitem__(self, key: str) -> _IdentityFieldNormalizer:
        del key
        return _IdentityFieldNormalizer()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        del state_dict


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: Sequence[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules: List[nn.Module] = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []
    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
    if output_dim > 0:
        last_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        del use_projection  # Kept for parity with original signature.
        block_channel = [64, 128, 256, 512]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"Unsupported final_norm={final_norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        return self.final_projection(x)


class PointNetEncoderXYZ(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"PointNetEncoderXYZ expects in_channels=3, got {in_channels}.")
        block_channel = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"Unsupported final_norm={final_norm}")
        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, dim=1)[0]
        return self.final_projection(x)


class DP3Encoder(nn.Module):
    """
    Minimal adaptation of 3D-Diffusion-Policy's DP3Encoder.
    Expects:
      - observations["point_cloud"]: [B, N, 3] or [B, N, 6]
      - observations["agent_pos"]:   [B, state_dim]
    """

    def __init__(
        self,
        *,
        state_dim: int,
        out_channel: int = 256,
        state_mlp_size: Sequence[int] = (64, 64),
        state_mlp_activation_fn: type[nn.Module] = nn.ReLU,
        pointcloud_in_channels: int = 3,
        pointcloud_out_channels: int = 256,
        pointcloud_use_layernorm: bool = True,
        pointcloud_final_norm: str = "layernorm",
        pointcloud_use_projection: bool = True,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
    ) -> None:
        super().__init__()
        self.point_cloud_key = "point_cloud"
        self.state_key = "agent_pos"
        self.n_output_channels = int(out_channel)
        self.use_pc_color = bool(use_pc_color)

        if pointnet_type != "pointnet":
            raise NotImplementedError(f"Unsupported pointnet_type={pointnet_type}")

        in_channels = 6 if self.use_pc_color else 3
        if pointcloud_in_channels is not None:
            in_channels = int(pointcloud_in_channels)

        if self.use_pc_color:
            self.extractor = PointNetEncoderXYZRGB(
                in_channels=in_channels,
                out_channels=int(pointcloud_out_channels),
                use_layernorm=bool(pointcloud_use_layernorm),
                final_norm=str(pointcloud_final_norm),
                use_projection=bool(pointcloud_use_projection),
            )
        else:
            self.extractor = PointNetEncoderXYZ(
                in_channels=in_channels,
                out_channels=int(pointcloud_out_channels),
                use_layernorm=bool(pointcloud_use_layernorm),
                final_norm=str(pointcloud_final_norm),
                use_projection=bool(pointcloud_use_projection),
            )

        if len(state_mlp_size) == 0:
            raise ValueError("state_mlp_size must be non-empty.")
        if len(state_mlp_size) == 1:
            net_arch: Sequence[int] = []
        else:
            net_arch = state_mlp_size[:-1]
        state_out_dim = int(state_mlp_size[-1])
        self.state_mlp = nn.Sequential(
            *create_mlp(int(state_dim), state_out_dim, net_arch, state_mlp_activation_fn)
        )
        self.n_output_channels = int(pointcloud_out_channels) + state_out_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        state = observations[self.state_key]
        if points.dim() != 3:
            raise ValueError(f"point_cloud must be [B,N,C], got {tuple(points.shape)}")
        if state.dim() != 2:
            raise ValueError(f"agent_pos must be [B,S], got {tuple(state.shape)}")
        pn_feat = self.extractor(points)
        state_feat = self.state_mlp(state)
        return torch.cat([pn_feat, state_feat], dim=-1)

    def output_shape(self) -> int:
        return int(self.n_output_channels)


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class CrossAttention(nn.Module):
    def __init__(self, in_dim: int, cond_dim: int, out_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key = self.key_proj(cond)
        value = self.value_proj(cond)
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        condition_type: str = "film",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        self.condition_type = condition_type

        cond_channels = out_channels
        if condition_type == "film":
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif condition_type == "add":
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange("batch t -> batch t 1"),
            )
        elif condition_type == "cross_attention_add":
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == "cross_attention_film":
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == "mlp_film":
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange("batch t -> batch t 1"),
            )
        else:
            raise NotImplementedError(f"condition_type={condition_type} not implemented.")

        self.out_channels = out_channels
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.blocks[0](x)
        if cond is not None:
            if self.condition_type == "film":
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == "add":
                embed = self.cond_encoder(cond)
                out = out + embed
            elif self.condition_type == "cross_attention_add":
                embed = self.cond_encoder(x.permute(0, 2, 1), cond).permute(0, 2, 1)
                out = out + embed
            elif self.condition_type == "cross_attention_film":
                embed = self.cond_encoder(x.permute(0, 2, 1), cond).permute(0, 2, 1)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == "mlp_film":
                embed = self.cond_encoder(cond)
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type={self.condition_type} not implemented.")
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        local_cond_dim: Optional[int] = None,
        global_cond_dim: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 3,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
    ) -> None:
        super().__init__()
        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + (0 if global_cond_dim is None else int(global_cond_dim))
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        self.local_cond_encoder: Optional[nn.ModuleList] = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        condition_type=condition_type,
                    ),
                    ConditionalResidualBlock1D(
                        dim_in,
                        dim_out,
                        cond_dim=cond_dim,
                        kernel_size=kernel_size,
                        n_groups=n_groups,
                        condition_type=condition_type,
                    ),
                ]
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    condition_type=condition_type,
                ),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            condition_type=condition_type,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [B, T, C] -> [B, C, T]
        sample = einops.rearrange(sample, "b h t -> b t h")

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        timestep_embed = self.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            if self.condition_type == "cross_attention":
                timestep_embed = timestep_embed.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
        else:
            global_feature = timestep_embed

        h_local: List[torch.Tensor] = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, "b h t -> b t h")
            resnet, resnet2 = self.local_cond_encoder
            x_local = resnet(local_cond, global_feature)
            h_local.append(x_local)
            x_local = resnet2(local_cond, global_feature)
            h_local.append(x_local)

        x = sample
        h: List[torch.Tensor] = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature) if self.use_mid_condition else mid_module(x)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                # Preserved from original implementation.
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b t h -> b h t")


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        max_n_obs_steps: int = 2,
        fix_obs_steps: bool = True,
        action_visible: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.obs_dim = int(obs_dim)
        self.max_n_obs_steps = int(max_n_obs_steps)
        self.fix_obs_steps = bool(fix_obs_steps)
        self.action_visible = bool(action_visible)

    @torch.no_grad()
    def forward(self, shape: Tuple[int, int, int], seed: Optional[int] = None) -> torch.Tensor:
        device = self.device
        B, T, D = shape
        if D != (self.action_dim + self.obs_dim):
            raise ValueError(
                f"Expected feature dim={self.action_dim + self.obs_dim}, got {D}."
            )

        rng = torch.Generator(device=device)
        if seed is not None:
            rng.manual_seed(int(seed))

        dim_mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[..., : self.action_dim] = True
        is_obs_dim = ~is_action_dim

        if self.fix_obs_steps:
            obs_steps = torch.full((B,), self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1,
                high=self.max_n_obs_steps + 1,
                size=(B,),
                generator=rng,
                device=device,
            )

        steps = torch.arange(0, T, device=device).reshape(1, T).expand(B, T)
        obs_mask = (steps.T < obs_steps).T.reshape(B, T, 1).expand(B, T, D)
        obs_mask = obs_mask & is_obs_dim

        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1,
                torch.tensor(0, dtype=obs_steps.dtype, device=obs_steps.device),
            )
            action_mask = (steps.T < action_steps).T.reshape(B, T, 1).expand(B, T, D)
            action_mask = action_mask & is_action_dim
            return obs_mask | action_mask
        return obs_mask


@dataclass
class DP3Config:
    horizon: int = 8
    n_action_steps: int = 8
    n_obs_steps: int = 2

    # Scheduler config (close to original DP3 defaults).
    scheduler_type: str = "ddim"  # "ddim" | "ddpm"
    num_train_timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "sample"  # "epsilon" | "sample" | "v_prediction"
    clip_sample: bool = True
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    num_inference_steps: Optional[int] = 10

    # Conditional U-Net / DP3 encoder.
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 128
    down_dims: Tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    condition_type: str = "film"
    use_down_condition: bool = True
    use_mid_condition: bool = True
    use_up_condition: bool = True
    encoder_output_dim: int = 64

    # Point cloud encoder.
    use_pc_color: bool = False
    pointnet_type: str = "pointnet"
    pointcloud_in_channels: int = 3
    pointcloud_use_layernorm: bool = True
    pointcloud_final_norm: str = "layernorm"
    pointcloud_use_projection: bool = True
    pointcloud_out_channels: int = 64
    state_mlp_size: Tuple[int, ...] = (64, 64)


def _build_scheduler(cfg: DP3Config) -> Union[DDIMScheduler, DDPMScheduler]:
    stype = str(cfg.scheduler_type).lower()
    if stype == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=int(cfg.num_train_timesteps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            beta_schedule=str(cfg.beta_schedule),
            clip_sample=bool(cfg.clip_sample),
            prediction_type=str(cfg.prediction_type),
        )
    if stype == "ddim":
        return DDIMScheduler(
            num_train_timesteps=int(cfg.num_train_timesteps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            beta_schedule=str(cfg.beta_schedule),
            clip_sample=bool(cfg.clip_sample),
            set_alpha_to_one=bool(cfg.set_alpha_to_one),
            steps_offset=int(cfg.steps_offset),
            prediction_type=str(cfg.prediction_type),
        )
    raise ValueError(f"Unsupported scheduler_type={cfg.scheduler_type}")


class DP3Policy(ModuleAttrMixin):
    """
    Self-contained DP3 port adapted to ICIL batch keys.

    Expected training batch keys:
      - query_xyz:     [B, T_obs, N, 3]
      - query_state:   [B, T_obs, S]
      - target_action: [B, H, A]
      - optional query_rgb: [B, T_obs, N, 3] when cfg.use_pc_color=True

    Conditioning support episodes are accepted by sample_actions signature but not
    used by vanilla DP3 (single-task behavior).
    """

    def __init__(self, *, cfg: DP3Config, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.horizon = int(cfg.horizon)
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.obs_as_global_cond = bool(cfg.obs_as_global_cond)
        self.condition_type = str(cfg.condition_type)
        self.use_pc_color = bool(cfg.use_pc_color)

        self.obs_encoder = DP3Encoder(
            state_dim=int(state_dim),
            out_channel=int(cfg.encoder_output_dim),
            state_mlp_size=cfg.state_mlp_size,
            pointcloud_in_channels=int(cfg.pointcloud_in_channels),
            pointcloud_out_channels=int(cfg.pointcloud_out_channels),
            pointcloud_use_layernorm=bool(cfg.pointcloud_use_layernorm),
            pointcloud_final_norm=str(cfg.pointcloud_final_norm),
            pointcloud_use_projection=bool(cfg.pointcloud_use_projection),
            use_pc_color=bool(cfg.use_pc_color),
            pointnet_type=str(cfg.pointnet_type),
        )
        obs_feature_dim = int(self.obs_encoder.output_shape())
        self.obs_feature_dim = obs_feature_dim

        input_dim = self.action_dim + obs_feature_dim
        global_cond_dim: Optional[int] = None
        if self.obs_as_global_cond:
            input_dim = self.action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * self.n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=int(cfg.diffusion_step_embed_dim),
            down_dims=cfg.down_dims,
            kernel_size=int(cfg.kernel_size),
            n_groups=int(cfg.n_groups),
            condition_type=self.condition_type,
            use_down_condition=bool(cfg.use_down_condition),
            use_mid_condition=bool(cfg.use_mid_condition),
            use_up_condition=bool(cfg.use_up_condition),
        )
        self.noise_scheduler = _build_scheduler(cfg)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if self.obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer: Any = IdentityNormalizer()
        self.num_inference_steps = (
            int(cfg.num_inference_steps)
            if cfg.num_inference_steps is not None
            else int(self.noise_scheduler.config.num_train_timesteps)
        )

    def set_normalizer(self, normalizer: Any) -> None:
        self.normalizer = normalizer

    def _normalize_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        try:
            maybe = self.normalizer.normalize(obs)
            if isinstance(maybe, dict):
                return maybe
        except Exception:
            pass
        return obs

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        try:
            return self.normalizer["action"].normalize(action)
        except Exception:
            return action

    def _unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        try:
            return self.normalizer["action"].unnormalize(action)
        except Exception:
            return action

    def _obs_from_query(
        self,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        query_rgb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_pc_color:
            if query_rgb is None:
                raise ValueError("cfg.use_pc_color=True but query_rgb is missing.")
            point_cloud = torch.cat([query_xyz, query_rgb], dim=-1)
        else:
            point_cloud = query_xyz[..., :3]
        return {
            "point_cloud": point_cloud,
            "agent_pos": query_state,
        }

    def _build_global_cond(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        value = next(iter(nobs.values()))
        if value.shape[1] < self.n_obs_steps:
            raise ValueError(
                f"Need at least n_obs_steps={self.n_obs_steps} observations, got {value.shape[1]}."
            )
        batch_size = int(value.shape[0])
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs)
        if "cross_attention" in self.condition_type:
            return nobs_features.reshape(batch_size, self.n_obs_steps, -1)
        return nobs_features.reshape(batch_size, -1)

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        *,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        inference_steps: Optional[int] = None,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        return_trace: bool = False,
        trace_steps: Optional[int] = None,
    ) -> Any:
        scheduler = self.noise_scheduler
        steps = self.num_inference_steps if inference_steps is None else int(inference_steps)
        total = int(scheduler.config.num_train_timesteps)
        steps = max(1, min(steps, total))

        try:
            scheduler.set_timesteps(steps, device=condition_data.device)
        except TypeError:
            scheduler.set_timesteps(steps)

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        trace_x0: List[torch.Tensor] = []
        trace_t: List[int] = []
        capture_idx: Optional[set[int]] = None
        if return_trace:
            if trace_steps is None or int(trace_steps) <= 0 or int(trace_steps) >= steps:
                capture_idx = set(range(steps))
            else:
                n = int(trace_steps)
                if n == 1:
                    capture_idx = {steps - 1}
                else:
                    capture_idx = {
                        int(round(i * (steps - 1) / float(n - 1)))
                        for i in range(n)
                    }

        step_sig = inspect.signature(scheduler.step).parameters
        for i, t in enumerate(scheduler.timesteps):
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                sample=trajectory,
                timestep=t,
                local_cond=local_cond,
                global_cond=global_cond,
            )

            step_kwargs: Dict[str, Any] = {}
            if "eta" in step_sig:
                step_kwargs["eta"] = float(eta)
            if "use_clipped_model_output" in step_sig:
                step_kwargs["use_clipped_model_output"] = bool(use_clipped_model_output)
            if "generator" in step_sig:
                step_kwargs["generator"] = generator

            step_out = scheduler.step(model_output, t, trajectory, **step_kwargs)
            pred_original = getattr(step_out, "pred_original_sample", None)
            if isinstance(step_out, tuple):
                trajectory = step_out[0]
            else:
                trajectory = step_out.prev_sample

            if return_trace and capture_idx is not None and i in capture_idx:
                x0_hat = pred_original if pred_original is not None else trajectory
                trace_x0.append(x0_hat.detach())
                trace_t.append(int(t.item() if torch.is_tensor(t) else t))

        trajectory[condition_mask] = condition_data[condition_mask]
        if return_trace:
            if len(trace_x0) == 0:
                trace_x0 = [trajectory.detach()]
                trace_t = [0]
            return trajectory, {
                "x0_hat": torch.stack(trace_x0, dim=0),
                "timesteps": torch.tensor(trace_t, device=trajectory.device, dtype=torch.long),
            }
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        value = next(iter(nobs.values()))
        batch_size, to = value.shape[:2]
        da = self.action_dim
        horizon = self.horizon

        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            global_cond = self._build_global_cond(nobs)
            cond_data = torch.zeros((batch_size, horizon, da), device=self.device, dtype=self.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :to, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, to, -1)
            cond_data = torch.zeros((batch_size, horizon, da + self.obs_feature_dim), device=self.device, dtype=self.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :to, da:] = nobs_features
            cond_mask[:, :to, da:] = True

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        naction_pred = nsample[..., :da]
        action_pred = self._unnormalize_action(naction_pred)
        start = to - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs = self._obs_from_query(
            query_xyz=batch["query_xyz"],
            query_state=batch["query_state"],
            query_rgb=batch.get("query_rgb", None),
        )
        nobs = self._normalize_obs(obs)
        nactions = self._normalize_action(batch["target_action"])

        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = int(nactions.shape[0])
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            global_cond = self._build_global_cond(nobs)
        else:
            horizon = int(trajectory.shape[1])
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        condition_mask = self.mask_generator(tuple(trajectory.shape))
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            low=0,
            high=int(self.noise_scheduler.config.num_train_timesteps),
            size=(batch_size,),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        pred_type = str(self.noise_scheduler.config.prediction_type)
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            if hasattr(self.noise_scheduler, "get_velocity"):
                target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
            else:
                alpha_t = self.noise_scheduler.alphas_cumprod[timesteps].sqrt().to(trajectory.device)
                sigma_t = (1.0 - self.noise_scheduler.alphas_cumprod[timesteps]).sqrt().to(trajectory.device)
                alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
                sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
                target = alpha_t * noise - sigma_t * trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.to(loss.dtype)
        loss = loss.reshape(batch_size, -1).mean(dim=1).mean()

        return loss, {
            "bc_loss": loss.detach(),
            "mse": loss.detach(),
            "t_mean": timesteps.float().mean().detach(),
        }

    def forward_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss, info = self.compute_loss(batch)
        return {
            "loss": loss,
            "mse": info["mse"],
            "t_mean": info["t_mean"],
        }

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        cond_xyz: torch.Tensor,
        cond_state: torch.Tensor,
        query_xyz: torch.Tensor,
        query_state: torch.Tensor,
        action_horizon: int,
        cond_rgb: Optional[torch.Tensor] = None,
        query_rgb: Optional[torch.Tensor] = None,
        cond_mask_id: Optional[torch.Tensor] = None,
        query_mask_id: Optional[torch.Tensor] = None,
        cond_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        inference_steps: Optional[int] = None,
        eta: float = 0.0,
        clip_x0: Optional[float] = None,
        return_trace: bool = False,
        trace_steps: Optional[int] = None,
    ) -> Any:
        # Vanilla DP3 is conditioned on query observations only.
        del cond_xyz, cond_state, cond_rgb, cond_mask_id, query_mask_id, cond_valid, query_valid

        obs = self._obs_from_query(query_xyz=query_xyz, query_state=query_state, query_rgb=query_rgb)
        nobs = self._normalize_obs(obs)
        if not self.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        batch_size = int(query_xyz.shape[0])
        horizon = int(action_horizon)
        da = self.action_dim
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            global_cond = self._build_global_cond(nobs)
            cond_data = torch.zeros((batch_size, horizon, da), device=query_xyz.device, dtype=query_xyz.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs).reshape(batch_size, self.n_obs_steps, -1)
            cond_data = torch.zeros(
                (batch_size, horizon, da + self.obs_feature_dim),
                device=query_xyz.device,
                dtype=query_xyz.dtype,
            )
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, : self.n_obs_steps, da:] = nobs_features
            cond_mask[:, : self.n_obs_steps, da:] = True

        sample_out = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            inference_steps=inference_steps,
            eta=float(eta),
            use_clipped_model_output=(clip_x0 is not None),
            return_trace=return_trace,
            trace_steps=trace_steps,
        )

        if isinstance(sample_out, tuple):
            nsample, trace = sample_out
        else:
            nsample, trace = sample_out, None

        naction_pred = nsample[..., :da]
        action_pred = self._unnormalize_action(naction_pred)
        if clip_x0 is not None:
            action_pred = action_pred.clamp(-float(clip_x0), float(clip_x0))

        if not return_trace:
            return action_pred

        if trace is not None:
            trace_x0 = trace["x0_hat"][..., :da]
            trace_x0 = self._unnormalize_action(trace_x0)
            if clip_x0 is not None:
                trace_x0 = trace_x0.clamp(-float(clip_x0), float(clip_x0))
            trace["x0_hat"] = trace_x0
        return action_pred, trace

