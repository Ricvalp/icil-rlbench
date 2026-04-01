from __future__ import annotations

from typing import Iterable, List, Sequence

import torch.nn as nn

from icil.models.policies.policy import Policy


def _denoiser_ada_prefixes(block_idx: int) -> List[str]:
    return [
        f"denoiser.{block_idx}.adaln1.to_scale_shift.",
        f"denoiser.{block_idx}.adaln2.to_scale_shift.",
        f"denoiser.{block_idx}.adaln3.to_scale_shift.",
        f"denoiser.{block_idx}.adaln_q.to_scale_shift.",
        f"denoiser.{block_idx}.adaln_s.to_scale_shift.",
    ]


def get_fast_param_names(
    model: Policy,
    last_frac: float = 0.25,
    include_ada: bool = True,
    include_final_norm: bool = True,
) -> List[str]:
    del include_final_norm  # Current Policy has no final norm block.

    if not hasattr(model, "denoiser"):
        raise AttributeError("Policy has no attribute 'denoiser'.")

    n_blocks = len(model.denoiser)
    if n_blocks <= 0:
        raise ValueError("Policy denoiser is empty.")

    if last_frac <= 0.0:
        num_blocks = 1
    else:
        num_blocks = max(1, int(round(float(n_blocks) * float(last_frac))))
        num_blocks = min(num_blocks, n_blocks)
    start_idx = n_blocks - num_blocks

    prefixes: List[str] = []
    for block_idx in range(start_idx, n_blocks):
        prefixes.append(f"denoiser.{block_idx}.mlp.")
        if include_ada:
            prefixes.extend(_denoiser_ada_prefixes(block_idx))

    all_param_names = [name for name, _ in model.named_parameters()]
    fast_names = [name for name in all_param_names if any(name.startswith(prefix) for prefix in prefixes)]

    forbidden_substrings = (
        ".self_attn.",
        ".cross_attn.",
        ".cross_attn_q.",
        ".cross_attn_s.",
        "context_encoder.",
        "action_in.",
        "action_out.",
        "t_mlp.",
    )
    bad_names = [name for name in fast_names if any(substr in name for substr in forbidden_substrings)]
    if bad_names:
        raise RuntimeError(
            "Fast-parameter selection included forbidden parameters. "
            f"Examples: {bad_names[:5]}"
        )
    if not fast_names:
        raise RuntimeError("No fast parameters were selected from the denoiser.")
    return sorted(fast_names)


def get_outer_param_names(
    model: Policy,
    *,
    train_encoder: bool = False,
    train_decoder: bool = True,
    train_input_projections: bool = True,
    train_output_head: bool = True,
    train_diffusion_conditioning: bool = True,
) -> List[str]:
    prefixes: List[str] = []
    if train_encoder:
        prefixes.append("context_encoder.")
    if train_decoder:
        prefixes.append("denoiser.")
    if train_input_projections:
        prefixes.append("action_in.")
    if train_output_head:
        prefixes.append("action_out.")
    if train_diffusion_conditioning:
        prefixes.append("t_mlp.")

    outer_names = [
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(prefix) for prefix in prefixes)
    ]
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
