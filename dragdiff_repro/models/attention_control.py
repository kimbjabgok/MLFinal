from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch
import torch.nn.functional as F

@dataclass
class ReferenceAttentionController:
    mode: str = "read"
    kv_bank: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


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

        should_swap = self.enabled and not is_cross_attention
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


@contextmanager
def reference_latent_control(unet) -> Iterator[ReferenceAttentionController]:
    """Install self-attention K/V swap processors for UNet upsampling blocks."""

    original_processors = unet.attn_processors
    controller = ReferenceAttentionController()
    processors = {}

    for name in original_processors:
        enabled = "up_blocks.2" in name and "attn1" in name
        processors[name] = ReferenceKVProcessor(name=name, controller=controller, enabled=enabled)

    unet.set_attn_processor(processors)
    try:
        yield controller
    finally:
        unet.set_attn_processor(original_processors)
