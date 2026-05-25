from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Mode = Literal["generated", "real"]


@dataclass
class DragConfig:
    height: int = 384
    width: int = 384
    num_ddim_steps: int = 50
    target_timestep_index: int = 35
    lora_rank: int = 8
    lora_batch_size: int = 2
    lora_steps: int = 40
    lora_lr: float = 5e-4
    drag_steps: int = 30
    latent_lr: float = 0.03
    lambda_mask: float = 0.1
    r1: int = 4
    r2: int = 3
    point_stop_threshold: float = 0.5
    guidance_scale_real: float = 1.0
    guidance_scale_generated: float = 7.5
    use_xformers: bool = True
    vae_tiling: bool = False
    cpu_offload: bool = False
    model_id: str = "runwayml/stable-diffusion-v1-5"
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 42
    output_dir: str = "outputs"


@dataclass
class EditRequest:
    mode: Mode
    prompt: str
    image: object | None = None
    mask: object | None = None
    handle_points: list[tuple[int, int]] = field(default_factory=list)
    target_points: list[tuple[int, int]] = field(default_factory=list)
    config: DragConfig = field(default_factory=DragConfig)
