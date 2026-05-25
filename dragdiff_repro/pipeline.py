from __future__ import annotations

import torch
from PIL import Image

from dragdiff_repro.config import DragConfig, EditRequest
from dragdiff_repro.methods.ddim_inversion import (
    ddim_invert,
    image_to_latent,
    latent_to_image,
    predict_noise,
)
from dragdiff_repro.methods.latent_optimization import optimize_latent
from dragdiff_repro.methods.lora_finetune import finetune_lora
from dragdiff_repro.models.attention_control import reference_latent_control
from dragdiff_repro.models.loader import ModelBundle
from dragdiff_repro.utils.image import tensor_to_pil


@torch.no_grad()
def generate_image_and_latent(
    bundle: ModelBundle,
    prompt: str,
    config: DragConfig,
) -> tuple[Image.Image, torch.Tensor]:
    generator = torch.Generator(device=bundle.device).manual_seed(config.seed)
    shape = (
        1,
        bundle.unet.config.in_channels,
        config.height // 8,
        config.width // 8,
    )
    latents = torch.randn(shape, generator=generator, device=bundle.device, dtype=bundle.dtype)
    latents = latents * bundle.scheduler.init_noise_sigma
    cached = None

    for index, timestep in enumerate(bundle.scheduler.timesteps):
        noise_pred = predict_noise(
            bundle,
            latents,
            timestep,
            prompt,
            config.guidance_scale_generated,
        )
        latents = bundle.scheduler.step(noise_pred, timestep, latents).prev_sample
        if index == config.target_timestep_index:
            cached = latents.detach().clone()

    image_tensor = latent_to_image(bundle, latents)
    image = tensor_to_pil(image_tensor)
    return image, cached if cached is not None else latents.detach().clone()


@torch.no_grad()
def denoise_from_timestep(
    bundle: ModelBundle,
    latents: torch.Tensor,
    reference_latents: torch.Tensor,
    prompt: str,
    start_index: int,
    guidance_scale: float,
) -> Image.Image:
    current = torch.nan_to_num(latents.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
    reference_current = torch.nan_to_num(reference_latents.detach(), nan=0.0, posinf=1.0, neginf=-1.0)
    timesteps = bundle.scheduler.timesteps[start_index:]

    with reference_latent_control(bundle.unet) as controller:
        for timestep in timesteps:
            controller.mode = "write"
            controller.kv_bank.clear()
            reference_noise = predict_noise(bundle, reference_current, timestep, prompt, guidance_scale)
            reference_current = bundle.scheduler.step(reference_noise, timestep, reference_current).prev_sample
            reference_current = torch.nan_to_num(reference_current, nan=0.0, posinf=1.0, neginf=-1.0)

            controller.mode = "read"
            noise_pred = predict_noise(bundle, current, timestep, prompt, guidance_scale)
            current = bundle.scheduler.step(noise_pred, timestep, current).prev_sample
            current = torch.nan_to_num(current, nan=0.0, posinf=1.0, neginf=-1.0)

    image_tensor = latent_to_image(bundle, current)
    return tensor_to_pil(image_tensor)


def run_dragdiffusion(bundle: ModelBundle, request: EditRequest) -> dict:
    config = request.config

    if request.mode == "generated":
        generated_image, latent_zt = generate_image_and_latent(bundle, request.prompt, config)
        original_latent_zt = latent_zt.detach().clone()
        source_image = generated_image
        guidance_scale = config.guidance_scale_generated
        denoise_start_index = config.target_timestep_index
    else:
        if request.image is None:
            raise ValueError("Real image mode requires an image tensor.")

        latent_z0 = image_to_latent(bundle, request.image)
        finetune_lora(bundle, latent_z0, request.prompt, config)
        latent_zt, _, denoise_start_index = ddim_invert(
            bundle,
            latent_z0,
            request.prompt,
            config.target_timestep_index,
            guidance_scale=config.guidance_scale_real,
        )
        original_latent_zt = latent_zt.detach().clone()
        source_image = tensor_to_pil(request.image)
        guidance_scale = config.guidance_scale_real

    optimized_latent, log = optimize_latent(
        bundle=bundle,
        latent_zt=latent_zt,
        original_latent_zt=original_latent_zt,
        mask=request.mask,
        prompt=request.prompt,
        handle_points=request.handle_points,
        target_points=request.target_points,
        config=config,
    )

    edited_image = denoise_from_timestep(
        bundle=bundle,
        latents=optimized_latent,
        reference_latents=original_latent_zt,
        prompt=request.prompt,
        start_index=denoise_start_index,
        guidance_scale=guidance_scale,
    )

    return {
        "source_image": source_image,
        "edited_image": edited_image,
        "tracked_points": log.point_history[-1] if log.point_history else request.handle_points,
        "logs": log.to_dict(),
    }
