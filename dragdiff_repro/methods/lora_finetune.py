from __future__ import annotations

import importlib
from importlib import metadata

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from dragdiff_repro.config import DragConfig
from dragdiff_repro.models.loader import ModelBundle, encode_prompt


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for part in value.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if digits:
            parts.append(int(digits))
    return tuple(parts)


def disable_incompatible_torchao_dispatch() -> None:
    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    if _version_tuple(torchao_version) >= (0, 16, 0):
        return

    for module_name in ("peft.import_utils", "peft.tuners.lora.torchao"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, "is_torchao_available"):
            module.is_torchao_available = lambda: False


def attach_lora_to_unet(bundle: ModelBundle, config: DragConfig) -> None:
    if hasattr(bundle.unet, "peft_config") and bundle.unet.peft_config:
        return

    disable_incompatible_torchao_dispatch()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )

    bundle.pipe.unet = get_peft_model(bundle.unet, lora_config)
    bundle.pipe.unet.train()


def reset_lora_on_unet(bundle: ModelBundle) -> None:
    """Remove any existing LoRA adapter and restore base UNet weights.

    Called before each real-image Run Editing to prevent LoRA accumulation
    across repeated runs. This ensures each run starts from the same base model.
    """
    unet = bundle.unet
    if not (hasattr(unet, "peft_config") and unet.peft_config):
        return

    if hasattr(unet, "unload"):
        bundle.pipe.unet = unet.unload()
        bundle.pipe.unet.to(device=bundle.device, dtype=bundle.dtype)
        bundle.pipe.unet.eval()
        torch.cuda.empty_cache()
        return

    raise RuntimeError("Existing LoRA adapter is attached, but this PEFT version cannot unload it.")


def finetune_lora(
    bundle: ModelBundle,
    image_latent: torch.Tensor,
    prompt: str,
    config: DragConfig,
) -> None:
    """Identity-preserving LoRA fine-tuning (Paper Section 3.2).

    Official: LoRA rank 16, batch 4, 80 steps, lr 5e-4, AdamW.
    T4 downgrade: rank 8, batch 2, 60 steps (configurable).

    Seed-based generator ensures reproducibility: same input image + seed
    produces the same LoRA weights across runs.
    """
    reset_lora_on_unet(bundle)
    attach_lora_to_unet(bundle, config)
    unet = bundle.unet
    unet.train()

    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    for name, parameter in unet.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=config.lora_lr,
    )
    text_embeds = encode_prompt(bundle, prompt)
    batch_size = max(1, int(config.lora_batch_size))
    train_scheduler = DDPMScheduler.from_config(bundle.scheduler.config)
    image_latent_batch = image_latent.repeat(batch_size, 1, 1, 1)
    text_embeds = text_embeds.repeat(batch_size, 1, 1)

    # Seed-based generator for reproducible noise sampling
    generator = torch.Generator(device=bundle.device).manual_seed(config.seed)

    for _ in tqdm(range(config.lora_steps), desc="LoRA fine-tuning"):
        timestep = torch.randint(
            0,
            train_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=bundle.device,
            generator=generator,
        ).long()

        noise = torch.randn(
            image_latent_batch.shape,
            device=bundle.device,
            dtype=image_latent_batch.dtype,
            generator=generator,
        )
        noisy = train_scheduler.add_noise(image_latent_batch, noise, timestep)

        pred = unet(noisy, timestep, encoder_hidden_states=text_embeds).sample

        loss = F.mse_loss(pred.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    unet.eval()
    torch.cuda.empty_cache()
