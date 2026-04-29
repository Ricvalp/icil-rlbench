from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from icil.models.encoders import ContextEncoder
from icil.models.encoders.dp3_query_frame_encoder import (
    DP3QueryFrameEncoder,
    DP3QueryFrameEncoderConfig,
)
from icil.models.encoders.simple_query_point_encoder import (
    SimpleQueryPointEncoder,
    SimpleQueryPointEncoderConfig,
)
from icil.models.policies.query_memory_direct_regression_policy import (
    QueryMemoryDirectRegressionPolicy,
    QueryMemoryDirectRegressionPolicyConfig,
)


@dataclass
class QueryMemoryDirectRegressionBuilderConfig:
    model_family: str = 'query_memory_direct_regression'
    query_memory_direct_regression: QueryMemoryDirectRegressionPolicyConfig = field(
        default_factory=QueryMemoryDirectRegressionPolicyConfig
    )
    query_encoder_name: str = 'simple_query_point_encoder'
    simple_query_point_encoder: SimpleQueryPointEncoderConfig = field(default_factory=SimpleQueryPointEncoderConfig)
    dp3_query_frame_encoder: DP3QueryFrameEncoderConfig = field(default_factory=DP3QueryFrameEncoderConfig)


_QUERY_ENCODER_BUILDERS: dict[str, Callable[[QueryMemoryDirectRegressionBuilderConfig, int, int], ContextEncoder]] = {
    'simple_query_point_encoder': lambda cfg, state_dim, action_dim: SimpleQueryPointEncoder(
        cfg=cfg.simple_query_point_encoder,
        state_dim=state_dim,
        action_dim=action_dim,
    ),
    'dp3_query_frame_encoder': lambda cfg, state_dim, action_dim: DP3QueryFrameEncoder(
        cfg=cfg.dp3_query_frame_encoder,
        state_dim=state_dim,
        action_dim=action_dim,
    ),
}


def available_query_memory_encoders() -> tuple[str, ...]:
    return tuple(sorted(_QUERY_ENCODER_BUILDERS.keys()))


def validate_query_memory_builder_config(cfg: QueryMemoryDirectRegressionBuilderConfig) -> None:
    if str(cfg.query_encoder_name) not in _QUERY_ENCODER_BUILDERS:
        raise ValueError(
            f"Unknown query_encoder_name={cfg.query_encoder_name!r}. "
            f"Available: {', '.join(available_query_memory_encoders())}"
        )
    policy_d = int(cfg.query_memory_direct_regression.d_model)
    if policy_d % int(cfg.query_memory_direct_regression.n_heads) != 0:
        raise ValueError(
            'Query-memory direct-regression head has incompatible d_model/n_heads: '
            f'{policy_d} / {int(cfg.query_memory_direct_regression.n_heads)}.'
        )
    encoder_d = (
        int(cfg.simple_query_point_encoder.d_model)
        if str(cfg.query_encoder_name) == 'simple_query_point_encoder'
        else int(cfg.dp3_query_frame_encoder.d_model)
    )
    if encoder_d != policy_d:
        raise ValueError(
            f'd_model mismatch between query encoder ({encoder_d}) and direct decoder ({policy_d}).'
        )



def build_query_memory_context_encoder(
    cfg: QueryMemoryDirectRegressionBuilderConfig,
    *,
    state_dim: int,
    action_dim: int,
) -> ContextEncoder:
    validate_query_memory_builder_config(cfg)
    return _QUERY_ENCODER_BUILDERS[str(cfg.query_encoder_name)](cfg, state_dim, action_dim)



def build_query_memory_direct_regression_policy(
    cfg: QueryMemoryDirectRegressionBuilderConfig,
    *,
    state_dim: int,
    action_dim: int,
) -> QueryMemoryDirectRegressionPolicy:
    context_encoder = build_query_memory_context_encoder(
        cfg,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    return QueryMemoryDirectRegressionPolicy(
        cfg=cfg.query_memory_direct_regression,
        context_encoder=context_encoder,
        state_dim=state_dim,
        action_dim=action_dim,
    )



def _get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)



def _none_config() -> object:
    return object()



def build_query_memory_builder_config_from_configdict(
    cfg: Any,
    *,
    as_bool: Callable[[Any], bool] = bool,
) -> QueryMemoryDirectRegressionBuilderConfig:
    sentinel = _none_config()
    policy_raw = _get(cfg, 'query_memory_direct_regression', sentinel)
    if policy_raw is sentinel:
        raise ValueError('Expected cfg.query_memory_direct_regression for query-memory direct-regression models.')

    policy_cfg = QueryMemoryDirectRegressionPolicyConfig(
        d_model=int(_get(policy_raw, 'd_model', 512)),
        n_heads=int(_get(policy_raw, 'n_heads', 8)),
        decoder_layers=int(_get(policy_raw, 'decoder_layers', 8)),
        decoder_mlp_mult=int(_get(policy_raw, 'decoder_mlp_mult', 4)),
        dropout=float(_get(policy_raw, 'dropout', 0.0)),
        grad_checkpoint_decoder=as_bool(_get(policy_raw, 'grad_checkpoint_decoder', False)),
        context_attention_mode=str(_get(policy_raw, 'context_attention_mode', 'two_ctx')),
        attention_backend=str(_get(policy_raw, 'attention_backend', 'manual')),
        loss_type=str(_get(policy_raw, 'loss_type', 'l1')),
        horizon=int(_get(policy_raw, 'horizon', 16)),
        conditioner_mlp_mult=int(_get(policy_raw, 'conditioner_mlp_mult', 2)),
        conditioner_dropout=float(_get(policy_raw, 'conditioner_dropout', 0.0)),
        memory_num_tokens=int(_get(policy_raw, 'memory_num_tokens', 128)),
    )

    simple_raw = _get(cfg, 'simple_query_point_encoder', SimpleNamespace())
    simple_cfg = SimpleQueryPointEncoderConfig(
        d_model=int(_get(simple_raw, 'd_model', policy_cfg.d_model)),
        use_rgb=as_bool(_get(simple_raw, 'use_rgb', True)),
        use_mask_id=as_bool(_get(simple_raw, 'use_mask_id', False)),
        mask_hash_buckets=int(_get(simple_raw, 'mask_hash_buckets', 2048)),
        use_gripper_point_features=as_bool(_get(simple_raw, 'use_gripper_point_features', False)),
        gripper_xyz_state_start=int(_get(simple_raw, 'gripper_xyz_state_start', 0)),
        max_T_obs=int(_get(simple_raw, 'max_T_obs', 16)),
        add_state_token=as_bool(_get(simple_raw, 'add_state_token', True)),
    )

    dp3_raw = _get(cfg, 'dp3_query_frame_encoder', SimpleNamespace())
    hidden_dims = _get(dp3_raw, 'state_mlp_hidden_dims', (64,))
    if isinstance(hidden_dims, list):
        hidden_dims = tuple(int(v) for v in hidden_dims)
    elif isinstance(hidden_dims, tuple):
        hidden_dims = tuple(int(v) for v in hidden_dims)
    else:
        hidden_dims = (int(hidden_dims),)
    dp3_cfg = DP3QueryFrameEncoderConfig(
        d_model=int(_get(dp3_raw, 'd_model', policy_cfg.d_model)),
        pointcloud_out_channels=int(_get(dp3_raw, 'pointcloud_out_channels', 256)),
        pointcloud_use_layernorm=as_bool(_get(dp3_raw, 'pointcloud_use_layernorm', True)),
        pointcloud_final_norm=str(_get(dp3_raw, 'pointcloud_final_norm', 'layernorm')),
        use_rgb=as_bool(_get(dp3_raw, 'use_rgb', True)),
        use_mask_id=as_bool(_get(dp3_raw, 'use_mask_id', False)),
        mask_hash_buckets=int(_get(dp3_raw, 'mask_hash_buckets', 2048)),
        mask_embed_dim=int(_get(dp3_raw, 'mask_embed_dim', 8)),
        use_gripper_point_features=as_bool(_get(dp3_raw, 'use_gripper_point_features', False)),
        gripper_xyz_state_start=int(_get(dp3_raw, 'gripper_xyz_state_start', 0)),
        state_mlp_hidden_dims=hidden_dims,
        state_feat_dim=int(_get(dp3_raw, 'state_feat_dim', 64)),
        max_T_obs=int(_get(dp3_raw, 'max_T_obs', 16)),
    )

    builder_cfg = QueryMemoryDirectRegressionBuilderConfig(
        query_memory_direct_regression=policy_cfg,
        query_encoder_name=str(_get(cfg, 'query_encoder_name', 'simple_query_point_encoder')),
        simple_query_point_encoder=simple_cfg,
        dp3_query_frame_encoder=dp3_cfg,
    )
    validate_query_memory_builder_config(builder_cfg)
    return builder_cfg
