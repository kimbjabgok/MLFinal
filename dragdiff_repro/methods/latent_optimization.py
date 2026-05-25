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


def _unet_feature(
    bundle: ModelBundle,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    text_embeds: torch.Tensor,
) -> torch.Tensor:
    with capture_up_block_feature(bundle.unet, block_index=2) as capture:
        _ = bundle.unet(latents, timestep, encoder_hidden_states=text_embeds).sample
    if capture.feature is None:
        raise RuntimeError("UNet feature hook did not capture an activation.")
    feature = capture.feature
    if feature.shape[-2:] != latents.shape[-2:]:
        feature = F.interpolate(feature, size=latents.shape[-2:], mode="bilinear", align_corners=False)
    return feature


def _motion_supervision_loss(
    feature: torch.Tensor,
    handle_points: list[tuple[int, int]],
    target_points: list[tuple[int, int]],
    radius: int,
) -> torch.Tensor:
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

        for yy in range(max(0, hy - radius), min(h, hy + radius + 1)):
            for xx in range(max(0, hx - radius), min(w, hx + radius + 1)):
                src = sample_feature(feature, [(xx, yy)])[0].detach()
                dst = sample_feature(feature, [(xx + step_x, yy + step_y)])[0]
                loss = loss + torch.abs(dst - src).mean()

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
) -> tuple[torch.Tensor, RunLog]:
    _freeze_unet(bundle.unet)
    timestep = bundle.scheduler.timesteps[config.target_timestep_index].to(bundle.device)
    text_embeds = encode_prompt(bundle, prompt)

    optimized = latent_zt.detach().clone().float().requires_grad_(True)
    original_latent_zt = original_latent_zt.detach().float()
    mask = mask.to(device=optimized.device, dtype=optimized.dtype)
    optimizer = torch.optim.Adam([optimized], lr=config.latent_lr)
    use_amp = bundle.device.type == "cuda" and bundle.dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    log = RunLog()

    with torch.no_grad():
        ref_feature = _unet_feature(bundle, original_latent_zt, timestep, text_embeds).detach()
        ref_vectors = sample_feature(ref_feature, handle_points).detach()

    current_points = list(handle_points)

    for step_idx in tqdm(range(config.drag_steps), desc="Latent optimization"):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            feature = _unet_feature(bundle, optimized, timestep, text_embeds)

            if step_idx != 0:
                with torch.no_grad():
                    current_points = nearest_neighbor_track(feature.detach(), ref_vectors, current_points, config.r2)

            motion_loss = _motion_supervision_loss(feature, current_points, target_points, config.r1)

            preserve_loss = torch.abs((optimized - original_latent_zt) * (1.0 - mask)).mean()
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
