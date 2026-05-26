from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn.functional as F

from dragdiff_repro.config import DragConfig


@dataclass
class ReferenceAttentionController:
    """Controls K/V sharing between reference and edited latent denoising.

    Official MutualSelfAttentionControl applies to self-attention layers 10-15
    (up_blocks[2] and up_blocks[3]) from start_step onwards. We replicate this
    by enabling processors only for those blocks and gating on step index.
    """
    mode: str = "read"
    kv_bank: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    current_step: int = 0
    start_step: int = 0

    def set_step(self, step: int) -> None:
        self.current_step = step

    @property
    def active(self) -> bool:
        return self.current_step >= self.start_step


class ReferenceKVProcessor:
    def __init__(self, name: str, controller: ReferenceAttentionController, enabled: bool) -> None:
        self.name = name
        self.controller = controller
        self.enabled = enabled

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        should_swap = self.enabled and not is_cross_attention and self.controller.active
        if should_swap and self.controller.mode == "write":
            self.controller.kv_bank[self.name] = (key.detach(), value.detach())
        elif should_swap and self.controller.mode == "read" and self.name in self.controller.kv_bank:
            key, value = self.controller.kv_bank[self.name]

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def _map_self_attn_layers(unet) -> dict[str, int]:
    """Map each self-attention processor name to its global self-attention layer index.

    SD1.5 UNet has 16 self-attention layers total:
    - down_blocks: 6 layers (indices 0-5)
    - mid_block: 1 layer (index 6)
    - up_blocks: 9 layers (indices 7-15)
      - up_blocks[1]: 3 layers (7-9)
      - up_blocks[2]: 3 layers (10-12)
      - up_blocks[3]: 3 layers (13-15)

    Official default: start_layer=10 enables layers 10-15 = up_blocks[2] + up_blocks[3].
    """
    layer_map: dict[str, int] = {}
    self_attn_count = 0

    for name in unet.attn_processors:
        if "attn1" in name:
            layer_map[name] = self_attn_count
            self_attn_count += 1

    return layer_map


@contextmanager
def reference_latent_control(
    unet,
    config: DragConfig | None = None,
) -> Iterator[ReferenceAttentionController]:
    """Install self-attention K/V swap processors on UNet.

    Official applies to self-attention layers [start_layer, 16) at steps >= start_step.
    Default start_layer=10 covers up_blocks[2] and up_blocks[3] self-attention.
    """
    start_layer = 10 if config is None else config.ref_attn_start_layer
    start_step = 0 if config is None else config.ref_attn_start_step

    original_processors = unet.attn_processors
    controller = ReferenceAttentionController(start_step=start_step)

    layer_map = _map_self_attn_layers(unet)
    processors = {}

    for name in original_processors:
        layer_idx = layer_map.get(name, -1)
        enabled = layer_idx >= start_layer
        processors[name] = ReferenceKVProcessor(name=name, controller=controller, enabled=enabled)

    unet.set_attn_processor(processors)
    try:
        yield controller
    finally:
        unet.set_attn_processor(original_processors)
