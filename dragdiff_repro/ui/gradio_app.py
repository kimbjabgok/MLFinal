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
        raise gr.Error("Prompt를 입력하세요.")

    config = DragConfig(
        height=int(height),
        width=int(height),
        lora_steps=int(lora_steps),
        drag_steps=int(drag_steps),
        seed=int(seed),
    )
    bundle = _get_model(config)
    latent_hw = (config.height // 8, config.width // 8)

    handles = parse_points(handle_points_text)
    targets = parse_points(target_points_text)
    if len(handles) == 0 or len(handles) != len(targets):
        raise gr.Error("handle/target 좌표 개수를 맞춰 입력하세요. 예: 180,220;250,220")

    image_tensor = None
    source_pil = None
    if mode == "real":
        if image is None:
            raise gr.Error("Real Image 모드에서는 입력 이미지가 필요합니다.")
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

    return result.get("source_image", source_pil), result["edited_image"], json.dumps(result["logs"], indent=2), save_message


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="DragDiffusion Reproduction") as demo:
        gr.Markdown("# DragDiffusion Reproduction")
        gr.Markdown("Colab T4 기준 간단 UI입니다. 좌표는 원본 이미지 픽셀 기준 `x,y; x,y` 형식으로 입력합니다.")

        with gr.Row():
            with gr.Column():
                mode = gr.Radio(
                    choices=[("Generated Image", "generated"), ("Real Image", "real")],
                    value="generated",
                    label="Mode",
                )
                image = gr.Image(type="pil", label="Input image (Real mode)")
                mask = gr.Image(type="pil", label="Mask PNG (white = editable, optional)")
                prompt = gr.Textbox(value="a photo of a cat", label="Prompt")
                handle_points = gr.Textbox(value="180,220", label="Handle points")
                target_points = gr.Textbox(value="250,220", label="Target points")

                with gr.Row():
                    height = gr.Dropdown([384, 512], value=384, label="Resolution")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    lora_steps = gr.Slider(0, 80, value=50, step=1, label="LoRA steps")
                    drag_steps = gr.Slider(1, 80, value=50, step=1, label="Drag steps")

                run_btn = gr.Button("Run", variant="primary")

            with gr.Column():
                source = gr.Image(type="pil", label="Source / Generated")
                edited = gr.Image(type="pil", label="Edited result")
                logs = gr.Code(label="Logs", language="json")
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
            outputs=[source, edited, logs, saved],
        )

    return demo

