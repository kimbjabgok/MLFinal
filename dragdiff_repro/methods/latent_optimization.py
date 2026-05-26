from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dragdiff_repro.config import DragConfig
from dragdiff_repro.models.feature_hooks import capture_up_block_feature
from dragdiff_repro.models.loader import ModelBundle, encode_prompt
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
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = latents.to(device=bundle.device, dtype=bundle.dtype)
    with capture_up_block_feature(bundle.unet, block_index=block_index) as capture:
        noise_pred = bundle.unet(latents, timestep, encoder_hidden_states=text_embeds).sample
    if capture.feature is None:
        raise RuntimeError("UNet feature hook did not capture an activation.")
    feature = capture.feature
    target_size = (feature_size, feature_size)
    if feature.shape[-2:] != target_size:
        feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
    return noise_pred, feature


def _unet_feature(
    bundle: ModelBundle,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    text_embeds: torch.Tensor,
    feature_size: int,
    block_index: int,
) -> torch.Tensor:
    _, feature = _unet_noise_and_feature(bundle, latents, timestep, text_embeds, feature_size, block_index)
    return feature


def _interpolate_feature_patch(
    feature: torch.Tensor,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    step_x: float,
    step_y: float,
) -> torch.Tensor:
    """Sample a feature patch shifted by (step_x, step_y).

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
    """Official-style patch motion supervision.

    handle_points and target_points use (x, y). We convert to tensor indexing
    as (row=y, col=x) whenever slicing feature[:, :, row, col].
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
) -> tuple[torch.Tensor, RunLog]:
    _freeze_unet(bundle.unet)
    # Generated mode optimizes at config.target_timestep_index. Real mode must
    # optimize at the DDIM inversion latent's actual denoise_start_index.
    active_timestep_index = config.target_timestep_index if timestep_index is None else timestep_index
    timestep = bundle.scheduler.timesteps[active_timestep_index].to(bundle.device)
    text_embeds = encode_prompt(bundle, prompt)
    feature_size = config.feature_supervision_size
    feature_block_index = config.unet_feature_block_index

    optimized = latent_zt.detach().clone().float().requires_grad_(True)
    original_latent_zt = original_latent_zt.detach().float()
    mask = mask.to(device=optimized.device, dtype=optimized.dtype)
    optimizer = torch.optim.Adam([optimized], lr=config.latent_lr)
    use_amp = bundle.device.type == "cuda" and bundle.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log = RunLog()

    with torch.no_grad():
        ref_noise, ref_feature = _unet_noise_and_feature(
            bundle,
            original_latent_zt,
            timestep,
            text_embeds,
            feature_size,
            feature_block_index,
        )
        original_prev = bundle.scheduler.step(ref_noise, timestep, original_latent_zt).prev_sample.detach()
        ref_feature = ref_feature.detach()
        ref_vectors = sample_feature(ref_feature, handle_points).detach()

    current_points = list(handle_points)

    for step_idx in tqdm(range(config.drag_steps), desc="Latent optimization"):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            noise_pred, feature = _unet_noise_and_feature(
                bundle,
                optimized,
                timestep,
                text_embeds,
                feature_size,
                feature_block_index,
            )

            if step_idx != 0:
                with torch.no_grad():
                    current_points = nearest_neighbor_track(
                        feature.detach(),
                        ref_vectors,
                        current_points,
                        config.r2,
                    )

            motion_loss = _motion_supervision_loss(feature, current_points, target_points, config.r1)
            optimized_prev = bundle.scheduler.step(noise_pred, timestep, optimized).prev_sample
            preserve_loss = torch.abs((optimized_prev - original_prev) * (1.0 - mask)).sum()
            loss = motion_loss + config.lambda_mask * preserve_loss

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

        if all_points_reached(current_points, target_points, config.point_stop_threshold):
            break

    optimized = optimized.detach().to(dtype=latent_zt.dtype)
    torch.cuda.empty_cache()
    return optimized, log
