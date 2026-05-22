from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from dragdiff_repro.config import DragConfig
from dragdiff_repro.methods.ddim_inversion import (
    ddim_invert,
    image_to_latent,
    latent_to_image,
    predict_noise,
)
from dragdiff_repro.methods.lora_finetune import finetune_lora
from dragdiff_repro.models.loader import encode_empty_prompt, encode_prompt, load_model_bundle
from dragdiff_repro.utils.image import pil_to_rgb, pil_to_tensor, tensor_to_pil


@torch.no_grad()
def reconstruct_from_timestep(
    bundle,
    latents: torch.Tensor,
    prompt: str,
    start_index: int,
    guidance_scale: float,
) -> Image.Image:
    current = latents.detach()
    prompt_embeds = encode_prompt(bundle, prompt)
    empty_prompt_embeds = encode_empty_prompt(bundle) if guidance_scale != 1.0 else None

    for timestep in bundle.scheduler.timesteps[start_index:]:
        noise_pred = predict_noise(
            bundle,
            current,
            timestep,
            prompt,
            guidance_scale,
            prompt_embeds=prompt_embeds,
            empty_prompt_embeds=empty_prompt_embeds,
        )
        current = bundle.scheduler.step(noise_pred, timestep, current).prev_sample

    return tensor_to_pil(latent_to_image(bundle, current))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check DDIM inversion reconstruction quality.")
    parser.add_argument("--image", required=True, help="Path to the real image.")
    parser.add_argument("--prompt", required=True, help="Prompt describing the image.")
    parser.add_argument("--output-dir", default="outputs/inversion_debug")
    parser.add_argument("--model-id", default=DragConfig.model_id)
    parser.add_argument("--height", type=int, default=DragConfig.height)
    parser.add_argument("--width", type=int, default=DragConfig.width)
    parser.add_argument("--num-ddim-steps", type=int, default=DragConfig.num_ddim_steps)
    parser.add_argument("--target-timestep-index", type=int, default=DragConfig.target_timestep_index)
    parser.add_argument("--guidance-scale", type=float, default=DragConfig.guidance_scale_real)
    parser.add_argument("--skip-lora", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DragConfig(
        model_id=args.model_id,
        height=args.height,
        width=args.width,
        num_ddim_steps=args.num_ddim_steps,
        target_timestep_index=args.target_timestep_index,
        guidance_scale_real=args.guidance_scale,
    )
    bundle = load_model_bundle(config)

    source = pil_to_rgb(Image.open(args.image), (config.width, config.height))
    image_tensor = pil_to_tensor(source, str(bundle.device), bundle.dtype)
    latent_z0 = image_to_latent(bundle, image_tensor)

    if not args.skip_lora:
        finetune_lora(bundle, latent_z0, args.prompt, config)

    latent_zt, _, denoise_start_index, intermediates = ddim_invert(
        bundle,
        latent_z0,
        args.prompt,
        config.target_timestep_index,
        guidance_scale=config.guidance_scale_real,
        return_intermediates=True,
    )
    reconstruction = reconstruct_from_timestep(
        bundle,
        latent_zt,
        args.prompt,
        denoise_start_index,
        config.guidance_scale_real,
    )

    source.save(output_dir / "source.png")
    reconstruction.save(output_dir / "reconstruction.png")
    print(f"Saved source and reconstruction to {output_dir}")
    print(f"Stored {len(intermediates)} intermediate latents on CPU during inversion.")


if __name__ == "__main__":
    main()
