from __future__ import annotations

from typing import Iterable, List, Sequence

import torch.nn as nn


def _is_diffusion_policy(model: nn.Module) -> bool:
    return hasattr(model, "denoiser") and hasattr(model, "noise_scheduler")


def _is_direct_regression_policy(model: nn.Module) -> bool:
    return hasattr(model, "decoder") and hasattr(model, "forward_actions")


def _denoiser_ada_prefixes(block_idx: int) -> List[str]:
    return [
        f"denoiser.{block_idx}.adaln1.to_scale_shift.",
        f"denoiser.{block_idx}.adaln2.to_scale_shift.",
        f"denoiser.{block_idx}.adaln3.to_scale_shift.",
        f"denoiser.{block_idx}.adaln_q.to_scale_shift.",
        f"denoiser.{block_idx}.adaln_s.to_scale_shift.",
    ]


def _direct_decoder_ada_prefixes(block_idx: int) -> List[str]:
    return [
        f"decoder.{block_idx}.adaln1.to_scale_shift.",
        f"decoder.{block_idx}.adaln2.to_scale_shift.",
        f"decoder.{block_idx}.adaln3.to_scale_shift.",
        f"decoder.{block_idx}.adaln_q.to_scale_shift.",
        f"decoder.{block_idx}.adaln_s.to_scale_shift.",
    ]


def get_fast_param_names(
    model: nn.Module,
    last_frac: float = 0.25,
    include_decoder_mlp: bool = True,
    include_ada: bool = True,
    include_final_norm: bool = True,
    include_input_projections: bool = False,
    include_output_head: bool = False,
    include_diffusion_conditioning: bool = False,
) -> List[str]:
    del include_final_norm  # Neither diffusion Policy nor DirectRegressionPolicy has a final norm block.

    if _is_diffusion_policy(model):
        block_container_name = "denoiser"
        ada_prefixes_fn = _denoiser_ada_prefixes
        mlp_prefix_fmt = "denoiser.{block_idx}.mlp."
        extra_prefixes: List[str] = []
        if include_input_projections:
            extra_prefixes.append("action_in.")
        if include_output_head:
            extra_prefixes.append("action_out.")
        if include_diffusion_conditioning:
            extra_prefixes.append("t_mlp.")
        forbidden_substrings = (
            ".self_attn.",
            ".cross_attn.",
            ".cross_attn_q.",
            ".cross_attn_s.",
            "context_encoder.",
        )
    elif _is_direct_regression_policy(model):
        block_container_name = "decoder"
        ada_prefixes_fn = _direct_decoder_ada_prefixes
        mlp_prefix_fmt = "decoder.{block_idx}.mlp."
        extra_prefixes = []
        if include_input_projections:
            extra_prefixes.extend(["action_queries", "action_slot_embed"])
        if include_output_head:
            extra_prefixes.append("action_out.")
        forbidden_substrings = (
            ".self_attn.",
            ".cross_attn.",
            ".cross_attn_q.",
            ".cross_attn_s.",
            "context_encoder.",
            "context_conditioner.",
        )
    else:
        raise AttributeError(
            "Unsupported model type for fast-parameter selection. "
            "Expected diffusion Policy or DirectRegressionPolicy."
        )

    if not hasattr(model, block_container_name):
        raise AttributeError(f"Model has no attribute {block_container_name!r}.")
    blocks = getattr(model, block_container_name)
    n_blocks = len(blocks)
    if n_blocks <= 0:
        raise ValueError(f"Model {block_container_name} is empty.")

    if last_frac <= 0.0:
        num_blocks = 1
    else:
        num_blocks = max(1, int(round(float(n_blocks) * float(last_frac))))
        num_blocks = min(num_blocks, n_blocks)
    start_idx = n_blocks - num_blocks

    prefixes: List[str] = []
    for block_idx in range(start_idx, n_blocks):
        if include_decoder_mlp:
            prefixes.append(mlp_prefix_fmt.format(block_idx=block_idx))
        if include_ada:
            prefixes.extend(ada_prefixes_fn(block_idx))
    prefixes.extend(extra_prefixes)

    all_param_names = [name for name, _ in model.named_parameters()]
    fast_names = [name for name in all_param_names if any(name.startswith(prefix) for prefix in prefixes)]

    bad_names = [name for name in fast_names if any(substr in name for substr in forbidden_substrings)]
    if bad_names:
        raise RuntimeError(
            "Fast-parameter selection included forbidden parameters. "
            f"Examples: {bad_names[:5]}"
        )
    if not fast_names:
        raise RuntimeError(f"No fast parameters were selected from the {block_container_name}.")
    return sorted(fast_names)


def get_outer_param_names(
    model: nn.Module,
    *,
    train_encoder: bool = False,
    train_decoder: bool = True,
    train_input_projections: bool = True,
    train_output_head: bool = True,
    train_diffusion_conditioning: bool = True,
) -> List[str]:
    prefixes: List[str] = []
    exact_names: List[str] = []
    if train_encoder:
        prefixes.append("context_encoder.")
    if _is_diffusion_policy(model):
        if train_decoder:
            prefixes.append("denoiser.")
        if train_input_projections:
            prefixes.append("action_in.")
        if train_output_head:
            prefixes.append("action_out.")
        if train_diffusion_conditioning:
            prefixes.append("t_mlp.")
    elif _is_direct_regression_policy(model):
        if train_decoder:
            prefixes.extend(["decoder.", "context_conditioner."])
        if train_decoder or train_input_projections:
            exact_names.extend(["action_queries", "action_slot_embed"])
        if train_output_head:
            prefixes.append("action_out.")
    else:
        raise AttributeError(
            "Unsupported model type for outer-parameter selection. "
            "Expected diffusion Policy or DirectRegressionPolicy."
        )

    outer_names = []
    exact_name_set = set(exact_names)
    for name, _ in model.named_parameters():
        if name in exact_name_set or any(name.startswith(prefix) for prefix in prefixes):
            outer_names.append(name)
    if not outer_names:
        raise RuntimeError("No outer-trainable parameters were selected.")
    return sorted(outer_names)


def set_outer_trainable_params(model: nn.Module, outer_names: Sequence[str]) -> None:
    outer_name_set = set(outer_names)
    for name, param in model.named_parameters():
        param.requires_grad_(name in outer_name_set)


def count_params_by_name(model: nn.Module, names: Sequence[str]) -> int:
    name_set = set(names)
    return sum(param.numel() for name, param in model.named_parameters() if name in name_set)


def prefix_param_names(names: Iterable[str], prefix: str = "policy.") -> List[str]:
    return [f"{prefix}{name}" for name in names]
