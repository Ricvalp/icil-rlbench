from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable

from icil.models.encoders import (
    Conv3dDemoQueryEncoderConfig,
    PerceiverDemoQueryEncoderConfig,
    PerceiverDemoQueryEncoderV2Config,
    PerceiverDemoQuerySupernodeEncoderV2Config,
    TrajConv3DConfig,
    TrajPerceiverConfig,
    TrajPerceiverV2Config,
    TrajSupernodePerceiverV2Config,
)
from icil.models.policies.builders import PolicyBuilderConfig
from icil.models.policies.policy import PolicyConfig


def _get(obj: Any, name: str, default: Any) -> Any:
    return getattr(obj, name, default)


def _none_config() -> object:
    return object()


_ENCODER_CONFIG_KEYS = (
    "conv3d_demo_query",
    "perceiver_demo_query",
    "perceiver_demo_query_v2",
    "perceiver_demo_query_supernode_v2",
    "traj_conv3d",
    "traj_perceiver",
    "traj_perceiver_v2",
    "traj_supernode_perceiver_v2",
)


def inherit_missing_encoder_attention_backend(model_cfg: Any) -> Any:
    """Copy the global policy attention backend into encoder configs when absent.

    Checkpoints store the raw ml_collections config, where the global knob lives
    under model.policy.attention_backend. Training config construction already
    uses that knob for encoders; eval reconstruction should do the same.
    """
    if not isinstance(model_cfg, dict):
        return model_cfg
    policy_cfg = model_cfg.get("policy", {})
    if not isinstance(policy_cfg, dict) or "attention_backend" not in policy_cfg:
        return model_cfg

    backend = policy_cfg["attention_backend"]
    out = dict(model_cfg)
    for key in _ENCODER_CONFIG_KEYS:
        encoder_cfg = out.get(key, None)
        if isinstance(encoder_cfg, dict) and "attention_backend" not in encoder_cfg:
            encoder_cfg = dict(encoder_cfg)
            encoder_cfg["attention_backend"] = backend
            out[key] = encoder_cfg
    return out


def _coerce_like(value: Any, default: Any, *, as_bool: Callable[[Any], bool]) -> Any:
    if isinstance(default, bool):
        return as_bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, tuple) and isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _config_dataclass_from_raw(
    cls: type,
    raw: Any,
    *,
    as_bool: Callable[[Any], bool],
    fallback_values: dict[str, Any] | None = None,
) -> Any:
    default_obj = cls()
    fallback_values = fallback_values or {}
    kwargs = {}
    for field in fields(default_obj):
        default = fallback_values.get(field.name, getattr(default_obj, field.name))
        value = _get(raw, field.name, default)
        kwargs[field.name] = _coerce_like(value, default, as_bool=as_bool)
    return cls(**kwargs)


def build_policy_builder_config_from_configdict(
    cfg: Any,
    *,
    as_bool: Callable[[Any], bool] = bool,
) -> PolicyBuilderConfig:
    policy_cfg_raw = cfg.policy
    conv3d_cfg_raw = _get(cfg, "conv3d_demo_query", _none_config())
    perceiver_cfg_raw = cfg.perceiver_demo_query
    perceiver_v2_cfg_raw = _get(cfg, "perceiver_demo_query_v2", _none_config())
    perceiver_supernode_v2_cfg_raw = _get(cfg, "perceiver_demo_query_supernode_v2", _none_config())
    traj_conv3d_cfg_raw = _get(cfg, "traj_conv3d", _none_config())
    traj_cfg_raw = cfg.traj_perceiver
    traj_v2_cfg_raw = _get(cfg, "traj_perceiver_v2", _none_config())
    traj_supernode_v2_cfg_raw = _get(cfg, "traj_supernode_perceiver_v2", _none_config())

    policy_cfg = PolicyConfig(
        d_model=int(policy_cfg_raw.d_model),
        n_heads=int(policy_cfg_raw.n_heads),
        denoiser_layers=int(policy_cfg_raw.denoiser_layers),
        denoiser_mlp_mult=int(policy_cfg_raw.denoiser_mlp_mult),
        dropout=float(policy_cfg_raw.dropout),
        grad_checkpoint_dit=as_bool(_get(policy_cfg_raw, "grad_checkpoint_dit", False)),
        context_attention_mode=str(_get(policy_cfg_raw, "context_attention_mode", "single")),
        attention_backend=str(_get(policy_cfg_raw, "attention_backend", "manual")),
        num_train_timesteps=int(policy_cfg_raw.num_train_timesteps),
        beta_start=float(_get(policy_cfg_raw, "beta_start", 1e-4)),
        beta_end=float(_get(policy_cfg_raw, "beta_end", 2e-2)),
        beta_schedule=str(_get(policy_cfg_raw, "beta_schedule", "squaredcos_cap_v2")),
        prediction_type=str(_get(policy_cfg_raw, "prediction_type", "v_prediction")),
        set_alpha_to_one=as_bool(_get(policy_cfg_raw, "set_alpha_to_one", True)),
        steps_offset=int(_get(policy_cfg_raw, "steps_offset", 0)),
        num_inference_steps=(
            int(_get(policy_cfg_raw, "num_inference_steps", None))
            if _get(policy_cfg_raw, "num_inference_steps", None) is not None
            else None
        ),
    )

    perceiver_cfg = PerceiverDemoQueryEncoderConfig(
        d_model=int(perceiver_cfg_raw.d_model),
        n_heads=int(perceiver_cfg_raw.n_heads),
        m_frame_tokens=int(perceiver_cfg_raw.m_frame_tokens),
        frame_tokenizer_layers=int(perceiver_cfg_raw.frame_tokenizer_layers),
        M_demo_latents=int(perceiver_cfg_raw.M_demo_latents),
        demo_perceiver_layers=int(perceiver_cfg_raw.demo_perceiver_layers),
        mask_hash_buckets=int(perceiver_cfg_raw.mask_hash_buckets),
        use_mask_id=as_bool(_get(perceiver_cfg_raw, "use_mask_id", True)),
        role_embed_max_K=int(perceiver_cfg_raw.role_embed_max_K),
        role_embed_max_L=int(perceiver_cfg_raw.role_embed_max_L),
        role_embed_max_Tobs=int(perceiver_cfg_raw.role_embed_max_Tobs),
        rgb_alpha_init=float(_get(perceiver_cfg_raw, "rgb_alpha_init", 1.0)),
        dropout=float(perceiver_cfg_raw.dropout),
        attention_backend=str(_get(perceiver_cfg_raw, "attention_backend", policy_cfg.attention_backend)),
        ignore_demos=as_bool(_get(perceiver_cfg_raw, "ignore_demos", False)),
        compress_demo_latents=as_bool(_get(perceiver_cfg_raw, "compress_demo_latents", True)),
        checkpoint_demo_memory=as_bool(_get(perceiver_cfg_raw, "checkpoint_demo_memory", False)),
        checkpoint_build_demo_memory=as_bool(_get(perceiver_cfg_raw, "checkpoint_build_demo_memory", False)),
        checkpoint_frame_tokenizer=as_bool(_get(perceiver_cfg_raw, "checkpoint_frame_tokenizer", False)),
        tokenize_frames_chunked=as_bool(_get(perceiver_cfg_raw, "tokenize_frames_chunked", False)),
        chunk_frames=int(_get(perceiver_cfg_raw, "chunk_frames", 32)),
    )

    conv3d_cfg = Conv3dDemoQueryEncoderConfig(
        d_model=int(_get(conv3d_cfg_raw, "d_model", policy_cfg.d_model)),
        n_heads=int(_get(conv3d_cfg_raw, "n_heads", policy_cfg.n_heads)),
        m_frame_tokens=int(_get(conv3d_cfg_raw, "m_frame_tokens", 64)),
        max_voxels=int(_get(conv3d_cfg_raw, "max_voxels", 4096)),
        voxel_size=float(_get(conv3d_cfg_raw, "voxel_size", 0.01)),
        use_learned_topk=as_bool(_get(conv3d_cfg_raw, "use_learned_topk", True)),
        n_mix_layers=int(_get(conv3d_cfg_raw, "n_mix_layers", 2)),
        M_demo_latents=int(_get(conv3d_cfg_raw, "M_demo_latents", 256)),
        demo_perceiver_layers=int(_get(conv3d_cfg_raw, "demo_perceiver_layers", 3)),
        mask_hash_buckets=int(_get(conv3d_cfg_raw, "mask_hash_buckets", 2048)),
        use_mask_id=as_bool(_get(conv3d_cfg_raw, "use_mask_id", True)),
        role_embed_max_K=int(_get(conv3d_cfg_raw, "role_embed_max_K", 32)),
        role_embed_max_L=int(_get(conv3d_cfg_raw, "role_embed_max_L", 64)),
        role_embed_max_Tobs=int(_get(conv3d_cfg_raw, "role_embed_max_Tobs", 16)),
        rgb_alpha_init=float(_get(conv3d_cfg_raw, "rgb_alpha_init", 1.0)),
        dropout=float(_get(conv3d_cfg_raw, "dropout", 0.0)),
        attention_backend=str(_get(conv3d_cfg_raw, "attention_backend", policy_cfg.attention_backend)),
        ignore_demos=as_bool(_get(conv3d_cfg_raw, "ignore_demos", False)),
    )

    traj_cfg = TrajPerceiverConfig(
        d_model=int(_get(traj_cfg_raw, "d_model", policy_cfg.d_model)),
        n_heads=int(_get(traj_cfg_raw, "n_heads", policy_cfg.n_heads)),
        dropout=float(_get(traj_cfg_raw, "dropout", 0.0)),
        m_frame_tokens=int(_get(traj_cfg_raw, "m_frame_tokens", 64)),
        frame_tokenizer_layers=int(_get(traj_cfg_raw, "frame_tokenizer_layers", 2)),
        M_demo_latents=int(_get(traj_cfg_raw, "M_demo_latents", 256)),
        demo_perceiver_layers=int(_get(traj_cfg_raw, "demo_perceiver_layers", 3)),
        mask_hash_buckets=int(_get(traj_cfg_raw, "mask_hash_buckets", 2048)),
        use_mask_id=as_bool(_get(traj_cfg_raw, "use_mask_id", True)),
        role_embed_max_K=int(_get(traj_cfg_raw, "role_embed_max_K", 32)),
        role_embed_max_L=int(_get(traj_cfg_raw, "role_embed_max_L", 64)),
        role_embed_max_Tobs=int(_get(traj_cfg_raw, "role_embed_max_Tobs", 16)),
        rgb_alpha_init=float(_get(traj_cfg_raw, "rgb_alpha_init", 1.0)),
        attention_backend=str(_get(traj_cfg_raw, "attention_backend", policy_cfg.attention_backend)),
        ignore_demos=as_bool(_get(traj_cfg_raw, "ignore_demos", False)),
        compress_demo_latents=as_bool(_get(traj_cfg_raw, "compress_demo_latents", True)),
        checkpoint_demo_memory=as_bool(_get(traj_cfg_raw, "checkpoint_demo_memory", False)),
        checkpoint_build_demo_memory=as_bool(_get(traj_cfg_raw, "checkpoint_build_demo_memory", False)),
        checkpoint_frame_tokenizer=as_bool(_get(traj_cfg_raw, "checkpoint_frame_tokenizer", False)),
        tokenize_frames_chunked=as_bool(_get(traj_cfg_raw, "tokenize_frames_chunked", False)),
        chunk_frames=int(_get(traj_cfg_raw, "chunk_frames", 32)),
        m_traj_tokens=int(_get(traj_cfg_raw, "m_traj_tokens", 16)),
        traj_perceiver_layers=int(_get(traj_cfg_raw, "traj_perceiver_layers", _get(traj_cfg_raw, "n_layers", 2))),
        traj_dim=int(_get(traj_cfg_raw, "traj_dim", 8)),
        use_demo_id_embed=as_bool(_get(traj_cfg_raw, "use_demo_id_embed", True)),
        include_traj_tokens=as_bool(_get(traj_cfg_raw, "include_traj_tokens", True)),
        use_cond_state_as_traj_fallback=as_bool(_get(traj_cfg_raw, "use_cond_state_as_traj_fallback", True)),
    )

    traj_conv3d_cfg = TrajConv3DConfig(
        d_model=int(_get(traj_conv3d_cfg_raw, "d_model", policy_cfg.d_model)),
        n_heads=int(_get(traj_conv3d_cfg_raw, "n_heads", policy_cfg.n_heads)),
        dropout=float(_get(traj_conv3d_cfg_raw, "dropout", 0.0)),
        m_frame_tokens=int(_get(traj_conv3d_cfg_raw, "m_frame_tokens", 64)),
        n_mix_layers=int(_get(traj_conv3d_cfg_raw, "n_mix_layers", 2)),
        max_voxels=int(_get(traj_conv3d_cfg_raw, "max_voxels", 4096)),
        voxel_size=float(_get(traj_conv3d_cfg_raw, "voxel_size", 0.01)),
        use_learned_topk=as_bool(_get(traj_conv3d_cfg_raw, "use_learned_topk", True)),
        M_demo_latents=int(_get(traj_conv3d_cfg_raw, "M_demo_latents", 256)),
        demo_perceiver_layers=int(_get(traj_conv3d_cfg_raw, "demo_perceiver_layers", 3)),
        mask_hash_buckets=int(_get(traj_conv3d_cfg_raw, "mask_hash_buckets", 2048)),
        use_mask_id=as_bool(_get(traj_conv3d_cfg_raw, "use_mask_id", True)),
        role_embed_max_K=int(_get(traj_conv3d_cfg_raw, "role_embed_max_K", 32)),
        role_embed_max_L=int(_get(traj_conv3d_cfg_raw, "role_embed_max_L", 64)),
        role_embed_max_Tobs=int(_get(traj_conv3d_cfg_raw, "role_embed_max_Tobs", 16)),
        rgb_alpha_init=float(_get(traj_conv3d_cfg_raw, "rgb_alpha_init", 1.0)),
        attention_backend=str(_get(traj_conv3d_cfg_raw, "attention_backend", policy_cfg.attention_backend)),
        ignore_demos=as_bool(_get(traj_conv3d_cfg_raw, "ignore_demos", False)),
        m_traj_tokens=int(_get(traj_conv3d_cfg_raw, "m_traj_tokens", 16)),
        traj_perceiver_layers=int(_get(traj_conv3d_cfg_raw, "traj_perceiver_layers", 2)),
        traj_dim=int(_get(traj_conv3d_cfg_raw, "traj_dim", 8)),
        use_demo_id_embed=as_bool(_get(traj_conv3d_cfg_raw, "use_demo_id_embed", True)),
        include_traj_tokens=as_bool(_get(traj_conv3d_cfg_raw, "include_traj_tokens", True)),
        use_cond_state_as_traj_fallback=as_bool(_get(traj_conv3d_cfg_raw, "use_cond_state_as_traj_fallback", True)),
    )

    v2_base = PerceiverDemoQueryEncoderV2Config()
    base_v2_heads = int(_get(perceiver_v2_cfg_raw, "n_heads", policy_cfg.n_heads))
    perceiver_v2_cfg = PerceiverDemoQueryEncoderV2Config(
        d_model=int(_get(perceiver_v2_cfg_raw, "d_model", policy_cfg.d_model)),
        n_heads=base_v2_heads,
        dropout=float(_get(perceiver_v2_cfg_raw, "dropout", 0.0)),
        attention_backend=str(_get(perceiver_v2_cfg_raw, "attention_backend", policy_cfg.attention_backend)),
        demo_m_frame_tokens=int(_get(perceiver_v2_cfg_raw, "demo_m_frame_tokens", _get(perceiver_v2_cfg_raw, "m_frame_tokens", v2_base.demo_m_frame_tokens))),
        demo_frame_tokenizer_layers=int(_get(perceiver_v2_cfg_raw, "demo_frame_tokenizer_layers", _get(perceiver_v2_cfg_raw, "frame_tokenizer_layers", v2_base.demo_frame_tokenizer_layers))),
        demo_n_heads=int(_get(perceiver_v2_cfg_raw, "demo_n_heads", base_v2_heads)),
        query_m_frame_tokens=int(_get(perceiver_v2_cfg_raw, "query_m_frame_tokens", _get(perceiver_v2_cfg_raw, "m_frame_tokens", v2_base.query_m_frame_tokens))),
        query_frame_tokenizer_layers=int(_get(perceiver_v2_cfg_raw, "query_frame_tokenizer_layers", _get(perceiver_v2_cfg_raw, "frame_tokenizer_layers", v2_base.query_frame_tokenizer_layers))),
        query_n_heads=int(_get(perceiver_v2_cfg_raw, "query_n_heads", base_v2_heads)),
        M_demo_latents=int(_get(perceiver_v2_cfg_raw, "M_demo_latents", v2_base.M_demo_latents)),
        demo_perceiver_layers=int(_get(perceiver_v2_cfg_raw, "demo_perceiver_layers", v2_base.demo_perceiver_layers)),
        mask_hash_buckets=int(_get(perceiver_v2_cfg_raw, "mask_hash_buckets", v2_base.mask_hash_buckets)),
        use_mask_id=as_bool(_get(perceiver_v2_cfg_raw, "use_mask_id", True)),
        role_embed_max_K=int(_get(perceiver_v2_cfg_raw, "role_embed_max_K", v2_base.role_embed_max_K)),
        role_embed_max_L=int(_get(perceiver_v2_cfg_raw, "role_embed_max_L", v2_base.role_embed_max_L)),
        role_embed_max_Tobs=int(_get(perceiver_v2_cfg_raw, "role_embed_max_Tobs", v2_base.role_embed_max_Tobs)),
        ignore_demos=as_bool(_get(perceiver_v2_cfg_raw, "ignore_demos", False)),
        compress_demo_latents=as_bool(_get(perceiver_v2_cfg_raw, "compress_demo_latents", True)),
        demo_rgb_alpha_init=float(_get(perceiver_v2_cfg_raw, "demo_rgb_alpha_init", _get(perceiver_v2_cfg_raw, "rgb_alpha_init", v2_base.demo_rgb_alpha_init))),
        query_rgb_alpha_init=float(_get(perceiver_v2_cfg_raw, "query_rgb_alpha_init", _get(perceiver_v2_cfg_raw, "rgb_alpha_init", v2_base.query_rgb_alpha_init))),
        use_gripper_point_features=as_bool(_get(perceiver_v2_cfg_raw, "use_gripper_point_features", False)),
        gripper_xyz_state_start=int(_get(perceiver_v2_cfg_raw, "gripper_xyz_state_start", 0)),
        gripper_alpha_init=float(_get(perceiver_v2_cfg_raw, "gripper_alpha_init", 1.0)),
        demo_post_self_attn_layers=int(_get(perceiver_v2_cfg_raw, "demo_post_self_attn_layers", 0)),
        query_post_self_attn_layers=int(_get(perceiver_v2_cfg_raw, "query_post_self_attn_layers", 0)),
        post_self_attn_mlp_mult=int(_get(perceiver_v2_cfg_raw, "post_self_attn_mlp_mult", 4)),
        checkpoint_demo_memory=as_bool(_get(perceiver_v2_cfg_raw, "checkpoint_demo_memory", False)),
        checkpoint_build_demo_memory=as_bool(_get(perceiver_v2_cfg_raw, "checkpoint_build_demo_memory", False)),
        checkpoint_frame_tokenizer=as_bool(_get(perceiver_v2_cfg_raw, "checkpoint_frame_tokenizer", False)),
        tokenize_frames_chunked=as_bool(_get(perceiver_v2_cfg_raw, "tokenize_frames_chunked", False)),
        chunk_frames=int(_get(perceiver_v2_cfg_raw, "chunk_frames", 32)),
    )

    traj_v2_base = TrajPerceiverV2Config()
    base_traj_v2_heads = int(_get(traj_v2_cfg_raw, "n_heads", policy_cfg.n_heads))
    traj_v2_cfg = TrajPerceiverV2Config(
        d_model=int(_get(traj_v2_cfg_raw, "d_model", policy_cfg.d_model)),
        n_heads=base_traj_v2_heads,
        dropout=float(_get(traj_v2_cfg_raw, "dropout", 0.0)),
        attention_backend=str(_get(traj_v2_cfg_raw, "attention_backend", policy_cfg.attention_backend)),
        demo_m_frame_tokens=int(_get(traj_v2_cfg_raw, "demo_m_frame_tokens", _get(traj_v2_cfg_raw, "m_frame_tokens", traj_v2_base.demo_m_frame_tokens))),
        demo_frame_tokenizer_layers=int(_get(traj_v2_cfg_raw, "demo_frame_tokenizer_layers", _get(traj_v2_cfg_raw, "frame_tokenizer_layers", traj_v2_base.demo_frame_tokenizer_layers))),
        demo_n_heads=int(_get(traj_v2_cfg_raw, "demo_n_heads", base_traj_v2_heads)),
        query_m_frame_tokens=int(_get(traj_v2_cfg_raw, "query_m_frame_tokens", _get(traj_v2_cfg_raw, "m_frame_tokens", traj_v2_base.query_m_frame_tokens))),
        query_frame_tokenizer_layers=int(_get(traj_v2_cfg_raw, "query_frame_tokenizer_layers", _get(traj_v2_cfg_raw, "frame_tokenizer_layers", traj_v2_base.query_frame_tokenizer_layers))),
        query_n_heads=int(_get(traj_v2_cfg_raw, "query_n_heads", base_traj_v2_heads)),
        M_demo_latents=int(_get(traj_v2_cfg_raw, "M_demo_latents", traj_v2_base.M_demo_latents)),
        demo_perceiver_layers=int(_get(traj_v2_cfg_raw, "demo_perceiver_layers", traj_v2_base.demo_perceiver_layers)),
        mask_hash_buckets=int(_get(traj_v2_cfg_raw, "mask_hash_buckets", traj_v2_base.mask_hash_buckets)),
        use_mask_id=as_bool(_get(traj_v2_cfg_raw, "use_mask_id", True)),
        role_embed_max_K=int(_get(traj_v2_cfg_raw, "role_embed_max_K", traj_v2_base.role_embed_max_K)),
        role_embed_max_L=int(_get(traj_v2_cfg_raw, "role_embed_max_L", traj_v2_base.role_embed_max_L)),
        role_embed_max_Tobs=int(_get(traj_v2_cfg_raw, "role_embed_max_Tobs", traj_v2_base.role_embed_max_Tobs)),
        ignore_demos=as_bool(_get(traj_v2_cfg_raw, "ignore_demos", False)),
        compress_demo_latents=as_bool(_get(traj_v2_cfg_raw, "compress_demo_latents", True)),
        demo_rgb_alpha_init=float(_get(traj_v2_cfg_raw, "demo_rgb_alpha_init", _get(traj_v2_cfg_raw, "rgb_alpha_init", traj_v2_base.demo_rgb_alpha_init))),
        query_rgb_alpha_init=float(_get(traj_v2_cfg_raw, "query_rgb_alpha_init", _get(traj_v2_cfg_raw, "rgb_alpha_init", traj_v2_base.query_rgb_alpha_init))),
        use_gripper_point_features=as_bool(_get(traj_v2_cfg_raw, "use_gripper_point_features", False)),
        gripper_xyz_state_start=int(_get(traj_v2_cfg_raw, "gripper_xyz_state_start", 0)),
        gripper_alpha_init=float(_get(traj_v2_cfg_raw, "gripper_alpha_init", 1.0)),
        demo_post_self_attn_layers=int(_get(traj_v2_cfg_raw, "demo_post_self_attn_layers", 0)),
        query_post_self_attn_layers=int(_get(traj_v2_cfg_raw, "query_post_self_attn_layers", 0)),
        post_self_attn_mlp_mult=int(_get(traj_v2_cfg_raw, "post_self_attn_mlp_mult", 4)),
        checkpoint_demo_memory=as_bool(_get(traj_v2_cfg_raw, "checkpoint_demo_memory", False)),
        checkpoint_build_demo_memory=as_bool(_get(traj_v2_cfg_raw, "checkpoint_build_demo_memory", False)),
        checkpoint_frame_tokenizer=as_bool(_get(traj_v2_cfg_raw, "checkpoint_frame_tokenizer", False)),
        tokenize_frames_chunked=as_bool(_get(traj_v2_cfg_raw, "tokenize_frames_chunked", False)),
        chunk_frames=int(_get(traj_v2_cfg_raw, "chunk_frames", 32)),
        m_traj_tokens=int(_get(traj_v2_cfg_raw, "m_traj_tokens", traj_v2_base.m_traj_tokens)),
        traj_perceiver_layers=int(_get(traj_v2_cfg_raw, "traj_perceiver_layers", _get(traj_v2_cfg_raw, "n_layers", traj_v2_base.traj_perceiver_layers))),
        traj_dim=int(_get(traj_v2_cfg_raw, "traj_dim", traj_v2_base.traj_dim)),
        use_demo_id_embed=as_bool(_get(traj_v2_cfg_raw, "use_demo_id_embed", True)),
        include_traj_tokens=as_bool(_get(traj_v2_cfg_raw, "include_traj_tokens", True)),
        use_cond_state_as_traj_fallback=as_bool(_get(traj_v2_cfg_raw, "use_cond_state_as_traj_fallback", True)),
    )

    perceiver_supernode_v2_cfg = _config_dataclass_from_raw(
        PerceiverDemoQuerySupernodeEncoderV2Config,
        perceiver_supernode_v2_cfg_raw,
        as_bool=as_bool,
        fallback_values={
            "d_model": policy_cfg.d_model,
            "n_heads": policy_cfg.n_heads,
            "demo_n_heads": policy_cfg.n_heads,
            "query_n_heads": policy_cfg.n_heads,
            "attention_backend": policy_cfg.attention_backend,
        },
    )
    traj_supernode_v2_cfg = _config_dataclass_from_raw(
        TrajSupernodePerceiverV2Config,
        traj_supernode_v2_cfg_raw,
        as_bool=as_bool,
        fallback_values={
            "d_model": policy_cfg.d_model,
            "n_heads": policy_cfg.n_heads,
            "demo_n_heads": policy_cfg.n_heads,
            "query_n_heads": policy_cfg.n_heads,
            "attention_backend": policy_cfg.attention_backend,
        },
    )

    return PolicyBuilderConfig(
        policy=policy_cfg,
        encoder_name=str(cfg.encoder_name),
        conv3d_demo_query=conv3d_cfg,
        perceiver_demo_query=perceiver_cfg,
        perceiver_demo_query_v2=perceiver_v2_cfg,
        perceiver_demo_query_supernode_v2=perceiver_supernode_v2_cfg,
        traj_conv3d=traj_conv3d_cfg,
        traj_perceiver=traj_cfg,
        traj_perceiver_v2=traj_v2_cfg,
        traj_supernode_perceiver_v2=traj_supernode_v2_cfg,
    )
