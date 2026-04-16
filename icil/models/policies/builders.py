from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

from icil.models.encoders import (
    Conv3dDemoQueryEncoder,
    Conv3dDemoQueryEncoderConfig,
    ContextEncoder,
    PerceiverDemoQueryEncoder,
    PerceiverDemoQueryEncoderConfig,
    PerceiverDemoQueryEncoderV2,
    PerceiverDemoQueryEncoderV2Config,
    TrajConv3DConfig,
    TrajPerceiverConfig,
    TrajPerceiverV2Config,
    TrajectoryConv3DEncoder,
    TrajectoryPerceiverEncoder,
    TrajectoryPerceiverEncoderV2,
)
from icil.models.policies.policy import Policy, PolicyConfig


@dataclass
class PolicyBuilderConfig:
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    encoder_name: str = "perceiver_demo_query"
    conv3d_demo_query: Conv3dDemoQueryEncoderConfig = field(default_factory=Conv3dDemoQueryEncoderConfig)
    perceiver_demo_query: PerceiverDemoQueryEncoderConfig = field(default_factory=PerceiverDemoQueryEncoderConfig)
    perceiver_demo_query_v2: PerceiverDemoQueryEncoderV2Config = field(default_factory=PerceiverDemoQueryEncoderV2Config)
    traj_conv3d: TrajConv3DConfig = field(default_factory=TrajConv3DConfig)
    traj_perceiver: TrajPerceiverConfig = field(default_factory=TrajPerceiverConfig)
    traj_perceiver_v2: TrajPerceiverV2Config = field(default_factory=TrajPerceiverV2Config)


ContextEncoderBuilder = Callable[[PolicyBuilderConfig, int, int], ContextEncoder]


def _build_conv3d_demo_query_encoder(
    cfg: PolicyBuilderConfig,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    return Conv3dDemoQueryEncoder(
        cfg=cfg.conv3d_demo_query,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def _build_perceiver_demo_query_encoder(
    cfg: PolicyBuilderConfig,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    return PerceiverDemoQueryEncoder(
        cfg=cfg.perceiver_demo_query,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def _build_perceiver_demo_query_v2_encoder(
    cfg: PolicyBuilderConfig,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    return PerceiverDemoQueryEncoderV2(
        cfg=cfg.perceiver_demo_query_v2,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def _build_traj_perceiver_encoder(
    cfg: PolicyBuilderConfig,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    return TrajectoryPerceiverEncoder(
        cfg=cfg.traj_perceiver,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def _build_traj_perceiver_v2_encoder(
    cfg: PolicyBuilderConfig,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    return TrajectoryPerceiverEncoderV2(
        cfg=cfg.traj_perceiver_v2,
        state_dim=state_dim,
        action_dim=action_dim,
    )


def _build_traj_conv3d_encoder(
    cfg: PolicyBuilderConfig,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    return TrajectoryConv3DEncoder(
        cfg=cfg.traj_conv3d,
        state_dim=state_dim,
        action_dim=action_dim,
    )


_ENCODER_BUILDERS: Dict[str, ContextEncoderBuilder] = {
    "conv3d_demo_query": _build_conv3d_demo_query_encoder,
    "perceiver_demo_query": _build_perceiver_demo_query_encoder,
    "perceiver_demo_query_v2": _build_perceiver_demo_query_v2_encoder,
    "traj_conv3d": _build_traj_conv3d_encoder,
    "traj_perceiver": _build_traj_perceiver_encoder,
    "traj_perceiver_v2": _build_traj_perceiver_v2_encoder,
}


def available_context_encoders() -> Tuple[str, ...]:
    return tuple(sorted(_ENCODER_BUILDERS.keys()))


def register_context_encoder_builder(name: str, builder: ContextEncoderBuilder) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("Encoder name must be a non-empty string.")
    _ENCODER_BUILDERS[key] = builder


def validate_builder_config(cfg: PolicyBuilderConfig) -> None:
    if cfg.encoder_name not in _ENCODER_BUILDERS:
        raise ValueError(
            f"Unknown encoder_name='{cfg.encoder_name}'. "
            f"Available: {', '.join(available_context_encoders())}"
        )

    policy_d = int(cfg.policy.d_model)
    if int(cfg.policy.n_heads) <= 0 or policy_d % int(cfg.policy.n_heads) != 0:
        raise ValueError(
            f"Invalid policy heads config: d_model={cfg.policy.d_model}, n_heads={cfg.policy.n_heads}."
        )

    if cfg.encoder_name == "conv3d_demo_query":
        enc_d = int(cfg.conv3d_demo_query.d_model)
    elif cfg.encoder_name == "perceiver_demo_query":
        enc_d = int(cfg.perceiver_demo_query.d_model)
    elif cfg.encoder_name == "perceiver_demo_query_v2":
        enc_d = int(cfg.perceiver_demo_query_v2.d_model)
    elif cfg.encoder_name == "traj_conv3d":
        enc_d = int(cfg.traj_conv3d.d_model)
    elif cfg.encoder_name == "traj_perceiver":
        enc_d = int(cfg.traj_perceiver.d_model)
    elif cfg.encoder_name == "traj_perceiver_v2":
        enc_d = int(cfg.traj_perceiver_v2.d_model)
    else:  # pragma: no cover - guarded above
        enc_d = policy_d

    if enc_d != policy_d:
        raise ValueError(
            f"d_model mismatch between policy ({policy_d}) and encoder '{cfg.encoder_name}' ({enc_d})."
        )


def build_context_encoder(
    cfg: PolicyBuilderConfig,
    *,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    validate_builder_config(cfg)
    return _ENCODER_BUILDERS[cfg.encoder_name](cfg, state_dim, action_dim)


def build_policy(
    cfg: PolicyBuilderConfig,
    *,
    state_dim: int,
    action_dim: int,
) -> Policy:
    context_encoder = build_context_encoder(cfg, state_dim=state_dim, action_dim=action_dim)
    return Policy(
        cfg=cfg.policy,
        context_encoder=context_encoder,
        state_dim=state_dim,
        action_dim=action_dim,
    )
