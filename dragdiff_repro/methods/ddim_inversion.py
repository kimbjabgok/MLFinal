from __future__ import annotations

import torch

from dragdiff_repro.models.loader import ModelBundle, encode_empty_prompt, encode_prompt


@torch.no_grad()
def image_to_latent(bundle: ModelBundle, image_tensor: torch.Tensor) -> torch.Tensor:
    posterior = bundle.vae.encode(image_tensor).latent_dist
    return posterior.sample() * bundle.vae.config.scaling_factor


@torch.no_grad()
def latent_to_image(bundle: ModelBundle, latent: torch.Tensor):
    image = bundle.vae.decode(latent / bundle.vae.config.scaling_factor).sample
    return image


@torch.no_grad()
def predict_noise(
    bundle: ModelBundle,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt: str,
    guidance_scale: float,
) -> torch.Tensor:
    cond = encode_prompt(bundle, prompt)
    if guidance_scale == 1.0:
        return bundle.unet(latents, timestep, encoder_hidden_states=cond).sample

    uncond = encode_empty_prompt(bundle)
    latent_in = torch.cat([latents, latents], dim=0)
    embeds = torch.cat([uncond, cond], dim=0)
    noise_uncond, noise_cond = bundle.unet(latent_in, timestep, encoder_hidden_states=embeds).sample.chunk(2)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


@torch.no_grad()
def ddim_invert(
    bundle: ModelBundle,
    latent_z0: torch.Tensor,
    prompt: str,
    target_timestep_index: int,
    guidance_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
 

    scheduler = bundle.scheduler
    denoise_timesteps = list(scheduler.timesteps)
    timesteps = list(reversed(denoise_timesteps))
    latents = latent_z0
    target = min(max(target_timestep_index, 0), len(timesteps) - 1)
    ref_prev = latent_z0

    for index, timestep in enumerate(timesteps[: target + 1]):
        noise_pred = predict_noise(bundle, latents, timestep, prompt, guidance_scale)
        alpha_prod_t = scheduler.alphas_cumprod[timestep]
        if index + 1 < len(timesteps):
            next_t = timesteps[index + 1]
            alpha_prod_next = scheduler.alphas_cumprod[next_t]
        else:
            alpha_prod_next = scheduler.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        pred_original = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
        direction = (1 - alpha_prod_next).sqrt() * noise_pred
        ref_prev = latents
        latents = alpha_prod_next.sqrt() * pred_original + direction

    denoise_start_index = len(denoise_timesteps) - 1 - target
    return latents.detach(), ref_prev.detach(), denoise_start_index
