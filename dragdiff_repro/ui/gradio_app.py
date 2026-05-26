from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw

from dragdiff_repro.config import DragConfig, EditRequest
from dragdiff_repro.models.loader import ModelBundle, load_model_bundle
from dragdiff_repro.pipeline import run_dragdiffusion
from dragdiff_repro.utils.image import pil_to_rgb, pil_to_tensor, prepare_drag_mask


_MODEL_BUNDLE: ModelBundle | None = None
_DEFAULT_HANDLES = [(180, 220)]
_DEFAULT_TARGETS = [(250, 220)]


def _get_model(config: DragConfig) -> ModelBundle:
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        _MODEL_BUNDLE = load_model_bundle(config)
    return _MODEL_BUNDLE


def _save_result(result: dict, config: DragConfig) -> str:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = out_dir / "edited_image.png"
    reconstruction_path = out_dir / "reconstruction.png"
    log_path = out_dir / "run_log.json"
    result["edited_image"].save(image_path)
    if result.get("reconstruction_image") is not None:
        result["reconstruction_image"].save(reconstruction_path)

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tracked_points": result["tracked_points"],
                "logs": result["logs"],
                "debug": result.get("debug", {}),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if result.get("reconstruction_image") is not None:
        return f"Saved: {image_path} / {reconstruction_path} / {log_path}"
    return f"Saved: {image_path} / {log_path}"


def _draw_point(draw: ImageDraw.ImageDraw, point: tuple[int, int], color: str) -> None:
    x, y = point
    radius = 10
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)


def _draw_overlay(
    image: Image.Image | None,
    handles: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    pending: tuple[int, int] | None,
) -> Image.Image | None:
    if image is None:
        return None

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for handle, target in zip(handles, targets):
        _draw_point(draw, handle, "red")
        _draw_point(draw, target, "blue")
        draw.line((handle[0], handle[1], target[0], target[1]), fill="white", width=6)
        draw.line((handle[0], handle[1], target[0], target[1]), fill="#4f7cff", width=3)

    if pending is not None:
        _draw_point(draw, pending, "red")

    return canvas


def _format_points(
    handles: list[tuple[int, int]],
    targets: list[tuple[int, int]],
    pending: tuple[int, int] | None,
) -> str:
    lines = []
    for index, (handle, target) in enumerate(zip(handles, targets), start=1):
        lines.append(f"{index}. handle {handle} -> target {target}")
    if pending is not None:
        lines.append(f"Pending handle: {pending}. Click a target point.")
    if not lines:
        return "Click once for a handle point, then click once for its target point."
    return "\n".join(lines)


def _sync_image(image):
    if image is None:
        return None, [], [], None, "Upload an image, then click handle and target points."
    return image, [], [], None, "Click once for a handle point, then click once for its target point."


def _select_point(image, handles, targets, pending, evt: gr.SelectData):
    if image is None:
        raise gr.Error("Upload an image first.")

    x, y = evt.index
    point = (int(x), int(y))
    handles = list(handles or [])
    targets = list(targets or [])

    if pending is None:
        pending = point
    else:
        handles.append(tuple(pending))
        targets.append(point)
        pending = None

    overlay = _draw_overlay(image, handles, targets, pending)
    return overlay, handles, targets, pending, _format_points(handles, targets, pending)


def _undo_point(image, handles, targets, pending):
    handles = list(handles or [])
    targets = list(targets or [])

    if pending is not None:
        pending = None
    elif handles and targets:
        handles.pop()
        targets.pop()

    overlay = _draw_overlay(image, handles, targets, pending)
    return overlay, handles, targets, pending, _format_points(handles, targets, pending)


def _clear_points(image):
    return image, [], [], None, "Click once for a handle point, then click once for its target point."


def _pixel_to_feature_points(
    points: list[tuple[int, int]],
    source_size: tuple[int, int],
    config: DragConfig,
) -> list[tuple[int, int]]:
    """Convert UI pixel points to feature-map (x, y) coordinates.

    The optimizer keeps public points as (x, y). Feature tensors are still
    indexed internally as feature[:, :, y, x].
    """

    src_w, src_h = source_size
    feature_size = config.feature_supervision_size
    points_out = []
    for x, y in points:
        scaled_x = x * feature_size / max(src_w, 1)
        scaled_y = y * feature_size / max(src_h, 1)
        fx = min(max(int(round(scaled_x)), 0), feature_size - 1)
        fy = min(max(int(round(scaled_y)), 0), feature_size - 1)
        points_out.append((fx, fy))
    return points_out


def _run(
    mode: str,
    image,
    prompt: str,
    handles_state,
    targets_state,
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

    pixel_handles = list(handles_state or [])
    pixel_targets = list(targets_state or [])
    if not pixel_handles and mode == "generated":
        pixel_handles = list(_DEFAULT_HANDLES)
        pixel_targets = list(_DEFAULT_TARGETS)

    if len(pixel_handles) == 0 or len(pixel_handles) != len(pixel_targets):
        raise gr.Error("Click handle and target points on the image first.")

    image_tensor = None
    source_pil = None
    source_size = (config.width, config.height)
    if mode == "real":
        if image is None:
            raise gr.Error("Real Image mode requires an input image.")
        source_size = image.size
        source_pil = pil_to_rgb(image, (config.width, config.height))
        image_tensor = pil_to_tensor(source_pil, str(bundle.device), bundle.dtype)

    handles = _pixel_to_feature_points(pixel_handles, source_size, config)
    targets = _pixel_to_feature_points(pixel_targets, source_size, config)
    mask_tensor = prepare_drag_mask(
        pixel_handles,
        pixel_targets,
        source_size=source_size,
        image_size=(config.width, config.height),
        latent_hw=latent_hw,
        radius_px=config.auto_mask_radius,
        device=str(bundle.device),
    )

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
            "Upload an image, click a handle point, then click its target point. "
            "The selected drag is drawn over the image."
        )

        handles_state = gr.State([])
        targets_state = gr.State([])
        pending_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("<div class='work-title'>Input</div>")
                mode = gr.Radio(
                    choices=[("Generated Image", "generated"), ("Real Image", "real")],
                    value="real",
                    label="Mode",
                )
                image = gr.Image(
                    type="pil",
                    label="Image upload",
                    sources=["upload"],
                    interactive=True,
                )
                prompt = gr.Textbox(value="a photo of a cat", label="Prompt")

                with gr.Row():
                    height = gr.Dropdown([384, 512], value=384, label="Resolution")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                with gr.Row():
                    lora_steps = gr.Slider(0, 80, value=40, step=1, label="LoRA steps")
                    drag_steps = gr.Slider(1, 80, value=30, step=1, label="Drag steps")

            with gr.Column(scale=1):
                gr.Markdown("<div class='work-title'>User Edit</div>")
                edit_view = gr.Image(
                    type="pil",
                    label="Click handle, then target",
                    sources=["upload"],
                    interactive=True,
                )
                point_readout = gr.Textbox(
                    label="Selected drags",
                    value="Upload an image, then click handle and target points.",
                    lines=5,
                    interactive=False,
                )
                with gr.Row():
                    undo_btn = gr.Button("Undo point")
                    clear_btn = gr.Button("Clear points")

            with gr.Column(scale=1):
                gr.Markdown("<div class='work-title'>Result</div>")
                source = gr.Image(type="pil", label="Source / Generated", interactive=False)
                edited = gr.Image(type="pil", label="Edited result", interactive=False)
                run_btn = gr.Button("Run Editing", variant="primary", elem_classes=["primary-run"])
                saved = gr.Textbox(label="Save status", interactive=False)

        image.change(
            _sync_image,
            inputs=[image],
            outputs=[edit_view, handles_state, targets_state, pending_state, point_readout],
        )
        edit_view.select(
            _select_point,
            inputs=[image, handles_state, targets_state, pending_state],
            outputs=[edit_view, handles_state, targets_state, pending_state, point_readout],
        )
        undo_btn.click(
            _undo_point,
            inputs=[image, handles_state, targets_state, pending_state],
            outputs=[edit_view, handles_state, targets_state, pending_state, point_readout],
        )
        clear_btn.click(
            _clear_points,
            inputs=[image],
            outputs=[edit_view, handles_state, targets_state, pending_state, point_readout],
        )
        run_btn.click(
            _run,
            inputs=[
                mode,
                image,
                prompt,
                handles_state,
                targets_state,
                height,
                lora_steps,
                drag_steps,
                seed,
            ],
            outputs=[source, edited, saved],
        )

    return demo
