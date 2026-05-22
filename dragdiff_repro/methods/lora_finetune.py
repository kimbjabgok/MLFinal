from __future__ import annotations

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from dragdiff_repro.config import DragConfig
from dragdiff_repro.models.loader import ModelBundle, encode_prompt


def attach_lora_to_unet(bundle: ModelBundle, config: DragConfig) -> None:
    if hasattr(bundle.unet, "peft_config") and bundle.unet.peft_config:
        return

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    bundle.pipe.unet = get_peft_model(bundle.unet, lora_config)
    bundle.pipe.unet.train()


def finetune_lora(
    bundle: ModelBundle,
    image_latent: torch.Tensor,
    prompt: str,
    config: DragConfig,
) -> None:
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

    for _ in tqdm(range(config.lora_steps), desc="LoRA fine-tuning"):
        timestep = torch.randint(
            0,
            train_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=bundle.device,
        ).long()
        noise = torch.randn_like(image_latent_batch)
        noisy = train_scheduler.add_noise(image_latent_batch, noise, timestep)

        pred = unet(noisy, timestep, encoder_hidden_states=text_embeds).sample
        loss = F.mse_loss(pred.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    unet.eval()
    torch.cuda.empty_cache()

