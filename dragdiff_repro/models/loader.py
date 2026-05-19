from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from dragdiff_repro.config import DragConfig


@dataclass
class ModelBundle:
    pipe: StableDiffusionPipeline
    device: torch.device
    dtype: torch.dtype

    @property
    def vae(self):
        return self.pipe.vae

    @property
    def unet(self):
        return self.pipe.unet

    @property
    def tokenizer(self):
        return self.pipe.tokenizer

    @property
    def text_encoder(self):
        return self.pipe.text_encoder

    @property
    def scheduler(self):
        return self.pipe.scheduler


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_model_bundle(config: DragConfig) -> ModelBundle:
    dtype = resolve_dtype(config.dtype)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    scheduler = DDIMScheduler.from_pretrained(config.model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe.scheduler.set_timesteps(config.num_ddim_steps, device=device)
    pipe.enable_vae_slicing()

    if config.use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pipe.enable_attention_slicing()
    else:
        pipe.enable_attention_slicing()

    if config.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    pipe.set_progress_bar_config(disable=False)
    return ModelBundle(pipe=pipe, device=device, dtype=dtype)


def encode_prompt(bundle: ModelBundle, prompt: str) -> torch.Tensor:
    tokenizer = bundle.tokenizer
    tokens = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(bundle.device)
    with torch.no_grad():
        embeds = bundle.text_encoder(tokens)[0]
    return embeds.to(dtype=bundle.dtype).detach()


def encode_empty_prompt(bundle: ModelBundle) -> torch.Tensor:
    return encode_prompt(bundle, "")
