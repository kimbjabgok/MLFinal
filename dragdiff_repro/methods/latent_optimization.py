from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dragdiff_repro.config import DragConfig
from dragdiff_repro.models.feature_hooks import capture_up_block_feature
from dragdiff_repro.models.loader import ModelBundle, encode_prompt, encode_empty_prompt
from dragdiff_repro.methods.point_tracking import (
    all_points_reached,
    nearest_neighbor_track,
    sample_feature,
)
from dragdiff_repro.utils.logging import RunLog


def _freeze_unet(unet) -> None:
    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    unet.eval()


def _unet_noise_and_feature(
    bundle: ModelBundle,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    text_embeds: torch.Tensor,
    feature_size: int,
    block_index: int,
    guidance_scale: float = 1.0,
    empty_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run UNet forward pass and extract feature maps from the specified up block.

    For generated mode with guidance_scale > 1.0, we handle CFG-aware feature
    extraction following the official implementation's strategy 3:
    combine unconditional and conditional features with guidance-weighted coefficients.

    T4 note: CFG doubles the batch through the UNet. For 384x384 this fits in ~15GB.
    """
    latents_in = latents.to(device=bundle.device, dtype=bundle.dtype)

    if guidance_scale > 1.0 and empty_embeds is not None:
        latents_in = torch.cat([latents_in, latents_in], dim=0)
        combined_embeds = torch.cat([empty_embeds, text_embeds], dim=0)
    else:
        combined_embeds = text_embeds

    with capture_up_block_feature(bundle.unet, block_index=block_index) as capture:
        noise_pred = bundle.unet(latents_in, timestep, encoder_hidden_states=combined_embeds).sample

    if capture.feature is None:
        raise RuntimeError("UNet feature hook did not capture an activation.")

    feature = capture.feature

    if guidance_scale > 1.0 and empty_embeds is not None:
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Official strategy 3: weighted combination of uncond/cond features
        coef = guidance_scale / (2 * guidance_scale - 1.0)
        feat_uncond, feat_cond = feature.chunk(2, dim=0)
        feature = (1 - coef) * feat_uncond + coef * feat_cond
        feature = feature.unsqueeze(0) if feature.dim() == 3 else feature

    target_size = (feature_size, feature_size)
    if feature.shape[-2:] != target_size:
        feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)

    return noise_pred, feature


def _interpolate_feature_patch(
    feature: torch.Tensor,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    step_x: float,
    step_y: float,
) -> torch.Tensor:
    """Sample a feature patch shifted by (step_x, step_y) using grid_sample.

    Public point convention is (x, y). PyTorch grid_sample expects normalized
    coordinates as (x, y), while feature indexing is feature[:, :, y, x].
    """
    _, _, h, w = feature.shape
    ys = torch.arange(y_min, y_max, device=feature.device, dtype=torch.float32) + step_y
    xs = torch.arange(x_min, x_max, device=feature.device, dtype=torch.float32) + step_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    norm_x = (2.0 * grid_x / max(w - 1, 1)) - 1.0
    norm_y = (2.0 * grid_y / max(h - 1, 1)) - 1.0
    grid = torch.stack((norm_x, norm_y), dim=-1).unsqueeze(0)
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _motion_supervision_loss(
    feature: torch.Tensor,
    handle_points: list[tuple[int, int]],
    target_points: list[tuple[int, int]],
    radius: int,
) -> torch.Tensor:
    """Matching official drag_utils.py motion supervision loss (Eq. 3 in paper).

    handle_points and target_points use (x, y). We convert to tensor indexing
    as (row=y, col=x) whenever slicing feature[:, :, row, col].

    Official: f0_patch = F1[:,:,r1:r2,c1:c2].detach()  (source, stopped gradient)
              f1_patch = interpolate_feature_patch(F1, r1+di[0], ...)  (shifted, has gradient)
              loss += (2*r_m+1)**2 * F.l1_loss(f0_patch, f1_patch)
    """
    _, _, h, w = feature.shape
    loss = feature.new_tensor(0.0)

    for handle, target in zip(handle_points, target_points):
        hx, hy = handle
        tx, ty = target
        dx = tx - hx
        dy = ty - hy
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
        step_x = dx / norm
        step_y = dy / norm
        if norm < 2.0:
            continue

        x_min = max(0, int(hx) - radius)
        x_max = min(w, int(hx) + radius + 1)
        y_min = max(0, int(hy) - radius)
        y_max = min(h, int(hy) + radius + 1)
        if x_min >= x_max or y_min >= y_max:
            continue

        src_patch = feature[:, :, y_min:y_max, x_min:x_max].detach()
        dst_patch = _interpolate_feature_patch(feature, y_min, y_max, x_min, x_max, step_x, step_y)
        loss = loss + ((2 * radius + 1) ** 2) * F.l1_loss(src_patch, dst_patch)

    return loss


def _one_step_denoise(
    bundle: ModelBundle,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Compute one DDIM denoising step to get z_{t-1} from z_t.

    Paper Eq. 3 preservation term: compare z_{t-1} from current vs original.
    Official: x_prev, _ = model.step(unet_output, t, init_code)
    """
    return bundle.scheduler.step(noise_pred, timestep, latent).prev_sample


def optimize_latent(
    bundle: ModelBundle,
    latent_zt: torch.Tensor,
    original_latent_zt: torch.Tensor,
    mask: torch.Tensor,
    prompt: str,
    handle_points: list[tuple[int, int]],
    target_points: list[tuple[int, int]],
    config: DragConfig,
    timestep_index: int | None = None,
    guidance_scale: float = 1.0,
) -> tuple[torch.Tensor, RunLog]:
    _freeze_unet(bundle.unet)

    active_timestep_index = timestep_index if timestep_index is not None else config.generated_cache_index
    timestep = bundle.scheduler.timesteps[active_timestep_index].to(bundle.device)
    text_embeds = encode_prompt(bundle, prompt)
    empty_embeds = encode_empty_prompt(bundle) if guidance_scale > 1.0 else None
    feature_size = config.feature_supervision_size
    feature_block_index = config.unet_feature_block_index

    optimized = latent_zt.detach().clone().float().requires_grad_(True)
    original_latent_zt = original_latent_zt.detach().float()

    # Mask: interpolate to latent spatial size for preservation loss (official approach)
    interp_mask = F.interpolate(
        mask.to(device=optimized.device, dtype=optimized.dtype),
        (optimized.shape[2], optimized.shape[3]),
        mode="nearest",
    )
    using_mask = interp_mask.sum() != 0.0

    optimizer = torch.optim.Adam([optimized], lr=config.latent_lr)
    use_amp = bundle.device.type == "cuda" and bundle.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log = RunLog()

    # Compute reference features and one-step denoised reference (x_prev_0)
    with torch.no_grad():
        ref_noise, ref_feature = _unet_noise_and_feature(
            bundle, original_latent_zt, timestep, text_embeds,
            feature_size, feature_block_index,
            guidance_scale=guidance_scale, empty_embeds=empty_embeds,
        )
        x_prev_0 = _one_step_denoise(bundle, ref_noise, timestep, original_latent_zt).detach()
        ref_feature = ref_feature.detach()
        ref_vectors = sample_feature(ref_feature, handle_points).detach()

    current_points = list(handle_points)

    for step_idx in tqdm(range(config.drag_steps), desc="Latent optimization"):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            noise_pred, feature = _unet_noise_and_feature(
                bundle, optimized, timestep, text_embeds,
                feature_size, feature_block_index,
                guidance_scale=guidance_scale, empty_embeds=empty_embeds,
            )

            # Point tracking (official: done before computing loss, skipped at step 0)
            if step_idx != 0:
                with torch.no_grad():
                    current_points = nearest_neighbor_track(
                        feature.detach(),
                        ref_vectors,
                        current_points,
                        config.r2,
                    )

            if all_points_reached(current_points, target_points, config.point_stop_threshold):
                break

            motion_loss = _motion_supervision_loss(feature, current_points, target_points, config.r1)

            # Preservation loss: compare one-step denoised current vs reference
            # Official: loss += lam * ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
            if using_mask:
                x_prev_updated = _one_step_denoise(bundle, noise_pred, timestep, optimized)
                preserve_loss = ((x_prev_updated - x_prev_0) * (1.0 - interp_mask)).abs().sum()
                loss = motion_loss + config.lambda_mask * preserve_loss
            else:
                loss = motion_loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([optimized], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            optimized.copy_(torch.nan_to_num(optimized, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-10.0, 10.0))
            log.loss_history.append(float(loss.detach().cpu()))
            log.point_history.append(list(current_points))

    optimized = optimized.detach().to(dtype=latent_zt.dtype)
    torch.cuda.empty_cache()
    return optimized, log
