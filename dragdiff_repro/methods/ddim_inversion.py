from __future__ import annotations

import torch

from dragdiff_repro.models.loader import ModelBundle, encode_empty_prompt, encode_prompt


@torch.no_grad()
def image_to_latent(bundle: ModelBundle, image_tensor: torch.Tensor) -> torch.Tensor:
    posterior = bundle.vae.encode(image_tensor).latent_dist
    return posterior.mean * bundle.vae.config.scaling_factor


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
    prompt_embeds: torch.Tensor | None = None,
    empty_prompt_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    cond = prompt_embeds if prompt_embeds is not None else encode_prompt(bundle, prompt)
    if guidance_scale == 1.0:
        return bundle.unet(latents, timestep, encoder_hidden_states=cond).sample

    uncond = empty_prompt_embeds if empty_prompt_embeds is not None else encode_empty_prompt(bundle)
    latent_in = torch.cat([latents, latents], dim=0)
    embeds = torch.cat([uncond, cond], dim=0)
    noise_uncond, noise_cond = bundle.unet(latent_in, timestep, encoder_hidden_states=embeds).sample.chunk(2)
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


def inv_step(
    bundle: ModelBundle,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scheduler = bundle.scheduler
    next_step = int(timestep)
    step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    prev_step = min(next_step - step_ratio, scheduler.config.num_train_timesteps - 1)
    alpha_prod_t = scheduler.alphas_cumprod[prev_step] if prev_step >= 0 else scheduler.final_alpha_cumprod
    alpha_prod_t_next = scheduler.alphas_cumprod[next_step]
    beta_prod_t = 1 - alpha_prod_t
    pred_original = (latents - beta_prod_t.sqrt() * noise_pred) / alpha_prod_t.sqrt()
    direction = (1 - alpha_prod_t_next).sqrt() * noise_pred
    latents_next = alpha_prod_t_next.sqrt() * pred_original + direction
    return latents_next, pred_original


@torch.no_grad()
def ddim_invert(
    bundle: ModelBundle,
    latent_z0: torch.Tensor,
    prompt: str,
    target_timestep_index: int,
    guidance_scale: float = 1.0,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int] | tuple[torch.Tensor, torch.Tensor, int, list[torch.Tensor]]:
 

    scheduler = bundle.scheduler
    denoise_timesteps = list(scheduler.timesteps)
    timesteps = list(reversed(denoise_timesteps))
    latents = latent_z0
    target = min(max(target_timestep_index, 0), len(timesteps) - 1)
    ref_prev = latent_z0
    prompt_embeds = encode_prompt(bundle, prompt)
    empty_prompt_embeds = encode_empty_prompt(bundle) if guidance_scale != 1.0 else None
    intermediates = [latents.detach().cpu()] if return_intermediates else None

    for index, timestep in enumerate(timesteps[: target + 1]):
        noise_pred = predict_noise(
            bundle,
            latents,
            timestep,
            prompt,
            guidance_scale,
            prompt_embeds=prompt_embeds,
            empty_prompt_embeds=empty_prompt_embeds,
        )
        ref_prev = latents
        latents, _ = inv_step(bundle, noise_pred, timestep, latents)
        if intermediates is not None:
            intermediates.append(latents.detach().cpu())

    denoise_start_index = len(denoise_timesteps) - 1 - target
    if intermediates is not None:
        return latents.detach(), ref_prev.detach(), denoise_start_index, intermediates
    return latents.detach(), ref_prev.detach(), denoise_start_index
