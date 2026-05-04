from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryMemoryMetaConfig:
    inner_steps: int = 1
    inner_lr: float = 1e-4
    inner_lr_mode: str = 'fixed'
    outer_lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_queries_per_step: int = 1
    num_inner_batches: int = 0
    num_query_loss_samples: int = 1
    holdout_index: int = -1
    first_order: bool = True
    reuse_diffusion_noise: bool = False
    grad_accum_steps: int = 1
    inner_loss_mode: str = 'read'
    memory_layer_norm_after_update: bool = False
    use_read_improvement_margin: bool = False
    read_improvement_margin: float = 0.0
    read_improvement_margin_weight: float = 0.0
    log_output_delta: bool = False
    training_mode_metrics_only: bool = False
    use_wrong_support_margin: bool = False
    wrong_support_margin: float = 0.0
    wrong_support_margin_weight: float = 0.0
    wrong_support_strategy: str = 'random_different_task'
    use_memory_contrast: bool = False
    memory_contrast_weight: float = 0.0
    memory_contrast_temperature: float = 0.1
    memory_contrast_on_delta: bool = True
