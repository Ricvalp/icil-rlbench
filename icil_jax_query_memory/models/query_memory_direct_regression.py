from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .direct_decoder import DirectDecoderConfig, DirectDecoderCore
from .object_centric_state import ObjectCentricStateTokenizer, ObjectCentricStateTokenizerConfig
from .simple_query_point_encoder import SimpleQueryPointEncoder, SimpleQueryPointEncoderConfig
from .support_encoder_memory import SupportEncoderConfig, SupportEncoderMemory


@dataclass(frozen=True)
class QueryMemoryDirectRegressionConfig:
    state_dim: int
    action_dim: int
    query_encoder: SimpleQueryPointEncoderConfig
    decoder: DirectDecoderConfig
    query_tokenizer_name: str = 'simple_query_point_encoder'
    object_tokenizer: Optional[ObjectCentricStateTokenizerConfig] = None
    support_encoder: Optional[SupportEncoderConfig] = None
    memory_initialization_mode: str = 'base_only'
    query_goal_visible: bool = True
    support_goal_visible: bool = True


class QueryMemoryDirectRegressionModel(nn.Module):
    cfg: QueryMemoryDirectRegressionConfig

    def _memory_mode(self) -> str:
        mode = str(getattr(self.cfg, 'memory_initialization_mode', 'base_only')).strip().lower()
        aliases = {
            'learned': 'base_only',
            'learned_base': 'base_only',
            'none': 'base_only',
            'encoder': 'pure_encoder',
            'encoder_only': 'pure_encoder',
            'encoder_plus_base': 'additive',
        }
        return aliases.get(mode, mode)

    def _encode_context(
        self,
        *,
        query_xyz: Optional[jnp.ndarray],
        query_state: jnp.ndarray,
        query_valid: Optional[jnp.ndarray],
        query_rgb: Optional[jnp.ndarray],
        query_mask_id: Optional[jnp.ndarray],
        goal_visible: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        name = str(getattr(self.cfg, 'query_tokenizer_name', 'simple_query_point_encoder'))
        if name == 'simple_query_point_encoder':
            if query_xyz is None:
                raise ValueError('simple_query_point_encoder requires query_xyz.')
            encoder = SimpleQueryPointEncoder(cfg=self.cfg.query_encoder, state_dim=int(self.cfg.state_dim), name='context_encoder')
            return encoder(
                query_xyz=query_xyz,
                query_state=query_state,
                query_valid=query_valid,
                query_rgb=query_rgb,
                query_mask_id=query_mask_id,
            )
        if name == 'object_centric_state':
            cfg = self.cfg.object_tokenizer
            if cfg is None:
                raise ValueError('query_tokenizer_name="object_centric_state" requires cfg.object_tokenizer.')
            encoder = ObjectCentricStateTokenizer(cfg=cfg, state_dim=int(self.cfg.state_dim), name='object_context_encoder')
            return encoder(query_state=query_state, goal_visible=bool(goal_visible))
        raise ValueError(
            "query_tokenizer_name must be one of: 'simple_query_point_encoder', 'object_centric_state'. "
            f'Got {name!r}.'
        )

    def _support_memory(
        self,
        *,
        support_query_state: jnp.ndarray,
        support_target_action: jnp.ndarray,
        support_demo_id: Optional[jnp.ndarray],
        support_chunk_start: Optional[jnp.ndarray],
        train: bool,
    ) -> jnp.ndarray:
        if self.cfg.support_encoder is None:
            raise ValueError('Support-encoder memory requires cfg.support_encoder.')
        if self.cfg.object_tokenizer is None:
            raise ValueError('Support-encoder memory requires cfg.object_tokenizer.')
        return SupportEncoderMemory(
            cfg=self.cfg.support_encoder,
            state_tokenizer_cfg=self.cfg.object_tokenizer,
            state_dim=int(self.cfg.state_dim),
            action_dim=int(self.cfg.action_dim),
            name='support_memory_encoder',
        )(
            support_state=support_query_state,
            support_target_action=support_target_action,
            support_demo_id=support_demo_id,
            support_chunk_start=support_chunk_start,
            train=train,
        )

    @nn.compact
    def initial_memory_from_support(
        self,
        *,
        support_query_state: Optional[jnp.ndarray] = None,
        support_target_action: Optional[jnp.ndarray] = None,
        support_demo_id: Optional[jnp.ndarray] = None,
        support_chunk_start: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        mem_init = self.param(
            'memory_token_init',
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.decoder.memory_num_tokens), int(self.cfg.decoder.d_model)),
            self.cfg.decoder.param_dtype,
        ).astype(self.cfg.decoder.dtype)
        mode = self._memory_mode()
        if mode == 'base_only':
            return mem_init
        if support_query_state is None or support_target_action is None:
            raise ValueError(f'memory_initialization_mode={mode!r} requires support_query_state and support_target_action.')
        support_memory = self._support_memory(
            support_query_state=support_query_state,
            support_target_action=support_target_action,
            support_demo_id=support_demo_id,
            support_chunk_start=support_chunk_start,
            train=train,
        ).astype(self.cfg.decoder.dtype)
        if mode == 'pure_encoder':
            return support_memory
        if mode == 'additive':
            return mem_init + support_memory
        raise ValueError("memory_initialization_mode must be one of: 'base_only', 'pure_encoder', 'additive'.")

    @nn.compact
    def __call__(
        self,
        *,
        query_xyz: Optional[jnp.ndarray] = None,
        query_state: Optional[jnp.ndarray] = None,
        query_valid: Optional[jnp.ndarray] = None,
        query_rgb: Optional[jnp.ndarray] = None,
        query_mask_id: Optional[jnp.ndarray] = None,
        memory_tokens: Optional[jnp.ndarray] = None,
        support_query_state: Optional[jnp.ndarray] = None,
        support_target_action: Optional[jnp.ndarray] = None,
        support_demo_id: Optional[jnp.ndarray] = None,
        support_chunk_start: Optional[jnp.ndarray] = None,
        mode: str = 'read',
        write_demo_id: Optional[jnp.ndarray] = None,
        write_chunk_start: Optional[jnp.ndarray] = None,
        train: bool = False,
    ) -> jnp.ndarray:
        mode = str(mode).lower()
        if mode not in ('read', 'write', 'goal'):
            raise ValueError(f"mode must be one of: 'read', 'write', 'goal'. Got {mode!r}.")

        if mode in ('read', 'goal'):
            if query_xyz is None or query_state is None:
                raise ValueError(f'{mode.upper()} mode requires query_xyz and query_state.')
            query_tokens, query_mask = self._encode_context(
                query_xyz=query_xyz,
                query_state=query_state,
                query_valid=query_valid,
                query_rgb=query_rgb,
                query_mask_id=query_mask_id,
                goal_visible=bool(self.cfg.query_goal_visible),
            )
            batch_size = int(query_xyz.shape[0])
        else:
            if bool(self.cfg.decoder.write_use_support_obs) and query_xyz is not None and query_state is not None:
                query_tokens, query_mask = self._encode_context(
                    query_xyz=query_xyz,
                    query_state=query_state,
                    query_valid=query_valid,
                    query_rgb=query_rgb,
                    query_mask_id=query_mask_id,
                    goal_visible=bool(self.cfg.support_goal_visible),
                )
                batch_size = int(query_xyz.shape[0])
            else:
                query_tokens = None
                query_mask = None
                batch_size = -1
            if batch_size > 0:
                pass
            elif memory_tokens is not None and memory_tokens.ndim == 3:
                batch_size = int(memory_tokens.shape[0])
            elif write_demo_id is not None:
                arr = jnp.asarray(write_demo_id)
                batch_size = int(arr.shape[0]) if arr.ndim > 0 else 1
            elif write_chunk_start is not None:
                arr = jnp.asarray(write_chunk_start)
                batch_size = int(arr.shape[0]) if arr.ndim > 0 else 1
            else:
                batch_size = 1

        mem_init = self.param(
            'memory_token_init',
            nn.initializers.normal(stddev=0.02),
            (int(self.cfg.decoder.memory_num_tokens), int(self.cfg.decoder.d_model)),
            self.cfg.decoder.param_dtype,
        )
        if memory_tokens is None and support_query_state is not None and support_target_action is not None:
            mode_for_init = self._memory_mode()
            if mode_for_init == 'base_only':
                initial_memory = mem_init.astype(self.cfg.decoder.dtype)
            else:
                support_memory = self._support_memory(
                    support_query_state=support_query_state,
                    support_target_action=support_target_action,
                    support_demo_id=support_demo_id,
                    support_chunk_start=support_chunk_start,
                    train=train,
                ).astype(self.cfg.decoder.dtype)
                if mode_for_init == 'pure_encoder':
                    initial_memory = support_memory
                elif mode_for_init == 'additive':
                    initial_memory = mem_init.astype(self.cfg.decoder.dtype) + support_memory
                else:
                    raise ValueError("memory_initialization_mode must be one of: 'base_only', 'pure_encoder', 'additive'.")
            if initial_memory.ndim == 2:
                support_tokens = jnp.broadcast_to(
                    initial_memory.astype(self.cfg.decoder.dtype)[None, :, :],
                    (batch_size,) + tuple(initial_memory.shape),
                )
            else:
                support_tokens = initial_memory.astype(self.cfg.decoder.dtype)
        elif memory_tokens is None:
            support_tokens = jnp.broadcast_to(mem_init.astype(self.cfg.decoder.dtype)[None, :, :], (batch_size,) + mem_init.shape)
        else:
            if memory_tokens.ndim == 2:
                support_tokens = jnp.broadcast_to(memory_tokens.astype(self.cfg.decoder.dtype)[None, :, :], (batch_size,) + tuple(memory_tokens.shape))
            elif memory_tokens.ndim == 3:
                support_tokens = memory_tokens.astype(self.cfg.decoder.dtype)
            else:
                raise ValueError(f'memory_tokens must have ndim 2 or 3, got {memory_tokens.ndim}.')
        support_mask = jnp.ones(support_tokens.shape[:2], dtype=jnp.bool_)
        decoder = DirectDecoderCore(cfg=self.cfg.decoder, action_dim=int(self.cfg.action_dim), name='decoder')
        return decoder(
            query_tokens=query_tokens,
            query_mask=query_mask,
            support_tokens=support_tokens,
            support_mask=support_mask,
            mode=mode,
            write_demo_id=write_demo_id,
            write_chunk_start=write_chunk_start,
            train=train,
        )
