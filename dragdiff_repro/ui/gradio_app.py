from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch

from dragdiff_repro.config import DragConfig, EditRequest
from dragdiff_repro.models.loader import ModelBundle, load_model_bundle
from dragdiff_repro.pipeline import run_dragdiffusion
from dragdiff_repro.utils.image import parse_points, pil_to_rgb, pil_to_tensor, prepare_mask


_MODEL_BUNDLE: ModelBundle | None = None


def _get_model(config: DragConfig) -> ModelBundle:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = load_model_bundle(config)
    return _MODEL_BUNDLE


def _save_result(result: dict, config: DragConfig) -> str:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "edited_image.png"
    log_path = out_dir / "run_log.json"
    result["edited_image"].save(image_path)

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tracked_points": result["tracked_points"],
                "logs": result["logs"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return f"Saved: {image_path} / {log_path}"


def _run(
    mode: str,
    image,
    mask,
    prompt: str,
    handle_points_text: str,
    target_points_text: str,
    height: int,
    lora_steps: int,
    drag_steps: int,
    seed: int,
):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    config = DragConfig(
        height=int(height),
        width=int(height),
        lora_steps=int(lora_steps),
        drag_steps=int(drag_steps),
        seed=int(seed),
        cpu_offload=False,
    )
    bundle = _get_model(config)
    latent_hw = (config.height // 8, config.width // 8)

    handles = parse_points(handle_points_text)
    targets = parse_points(target_points_text)
    if len(handles) == 0 or len(handles) != len(targets):
        raise gr.Error("Please enter the same number of handle and target points. Example: 180,220;250,220")

    image_tensor = None
    source_pil = None
    if mode == "real":
        if image is None:
            raise gr.Error("Real Image mode requires an input image.")
        source_pil = pil_to_rgb(image, (config.width, config.height))
        image_tensor = pil_to_tensor(source_pil, str(bundle.device), bundle.dtype)

    mask_tensor = prepare_mask(mask, latent_hw, str(bundle.device))

    request = EditRequest(
        mode="real" if mode == "real" else "generated",
        image=image_tensor,
        mask=mask_tensor,
        prompt=prompt,
        handle_points=handles,
        target_points=targets,
        config=config,
    )

    result = run_dragdiffusion(bundle, request)
    save_message = _save_result(result, config)
    torch.cuda.empty_cache()

    return result.get("source_image", source_pil), result["edited_image"], save_message


def build_demo() -> gr.Blocks:
    css = """
    .work-title {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0.4rem 0 0.8rem;
    }
    .compact-note {
        color: #a8b3c7;
        font-size: 0.92rem;
        margin-top: -0.2rem;
    }
    .primary-run button {
        min-height: 48px;
        font-weight: 700;
    }
    """

    with gr.Blocks(title="DragDiffusion Reproduction", css=css) as demo:
        gr.Markdown("# DragDiffusion Reproduction")
        gr.Markdown(
            "Colab T4 friendly UI for point-based editing. "
            "Points are entered as original image pixel coordinates: `x,y; x,y`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='work-title'>Input</div>")
                mode = gr.Radio(
                    choices=[("Generated Image", "generated"), ("Real Image", "real")],
                    value="generated",
                    label="Mode",
                )
                image = gr.Image(type="pil", label="Real image upload")
                mask = gr.Image(type="pil", label="Mask PNG (white = editable)")
                prompt = gr.Textbox(value="a photo of a cat", label="Prompt")

            with gr.Column(scale=1):
                gr.Markdown("<div class='work-title'>Points & Settings</div>")
                handle_points = gr.Textbox(value="180,220", label="Handle points")
                target_points = gr.Textbox(value="250,220", label="Target points")
                gr.Markdown("<div class='compact-note'>Multiple points: `180,220;210,240`</div>")

                with gr.Row():
                    height = gr.Dropdown([384, 512], value=384, label="Resolution")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    lora_steps = gr.Slider(0, 80, value=50, step=1, label="LoRA steps")
                    drag_steps = gr.Slider(1, 80, value=50, step=1, label="Drag steps")

                run_btn = gr.Button("Run Editing", variant="primary", elem_classes=["primary-run"])

            with gr.Column(scale=1):
                gr.Markdown("<div class='work-title'>Result</div>")
                source = gr.Image(type="pil", label="Source / Generated")
                edited = gr.Image(type="pil", label="Edited result")
                saved = gr.Textbox(label="Save status")

        run_btn.click(
            _run,
            inputs=[
                mode,
                image,
                mask,
                prompt,
                handle_points,
                target_points,
                height,
                lora_steps,
                drag_steps,
                seed,
            ],
            outputs=[source, edited, saved],
        )

    return demo
