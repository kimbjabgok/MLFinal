from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Mode = Literal["generated", "real"]


@dataclass
class DragConfig:
    height: int = 384
    width: int = 384
    num_ddim_steps: int = 50
    # For real images: number of DDIM inversion steps from clean to noisy.
    # Paper default = 35. Higher = more noise = more editing flexibility but less identity.
    inversion_steps: int = 35
    # For generated images: fraction of total steps to use for editing.
    # Official default = 0.7 → 35 out of 50 steps for denoising from cached latent.
    inversion_strength: float = 0.7
    lora_rank: int = 8
    lora_batch_size: int = 2
    lora_steps: int = 60
    lora_lr: float = 5e-4
    drag_steps: int = 80
    latent_lr: float = 0.01
    lambda_mask: float = 0.1
    r1: int = 1
    r2: int = 3
    point_stop_threshold: float = 2.0
    # Official: up_blocks[2] output = all_intermediate_features[3].
    # Our hook indexes up_blocks directly, so block_index=2 is correct.
    unet_feature_block_index: int = 2
    auto_mask_radius: int = 48
    guidance_scale_real: float = 1.0
    guidance_scale_generated: float = 7.5
    # Reference-latent-control: start from this self-attention layer index (0-15).
    # Official default: start_layer=10 → applies to layers 10-15 (up_blocks 2 & 3).
    ref_attn_start_layer: int = 10
    # Reference-latent-control: start from this denoising step index.
    ref_attn_start_step: int = 0
    use_xformers: bool = True
    vae_tiling: bool = False
    cpu_offload: bool = False
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 42
    output_dir: str = "outputs"

    @property
    def feature_supervision_size(self) -> int:
        """Official uses int(0.5 * image_height). T4-safe default."""
        return self.height // 2

    @property
    def generated_cache_index(self) -> int:
        """Index into scheduler.timesteps where we cache the latent during generation.
        Official: n_inference_step - n_actual_inference_step.
        With inversion_strength=0.7 and 50 steps: 50 - 35 = 15."""
        n_actual = round(self.num_ddim_steps * self.inversion_strength)
        return self.num_ddim_steps - n_actual

    @property
    def generated_denoise_steps(self) -> int:
        """Number of actual denoising steps for generated mode editing."""
        return round(self.num_ddim_steps * self.inversion_strength)


@dataclass
class EditRequest:
    mode: Mode
    prompt: str
    image: object | None = None
    mask: object | None = None
    handle_points: list[tuple[int, int]] = field(default_factory=list)
    target_points: list[tuple[int, int]] = field(default_factory=list)
    config: DragConfig = field(default_factory=DragConfig)
    # Pre-cached latent for generated mode (avoids regeneration)
    cached_latent_zt: object | None = None
