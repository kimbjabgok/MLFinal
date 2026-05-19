from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def pil_to_rgb(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def pil_to_tensor(image: Image.Image, device: str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device=device, dtype=dtype)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = torch.nan_to_num(tensor.detach().float().cpu(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0
    arr = tensor[0].permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def prepare_mask(mask_image: Image.Image | None, latent_hw: tuple[int, int], device: str) -> torch.Tensor:
    h, w = latent_hw
    if mask_image is None:
        return torch.ones((1, 1, h, w), device=device)

    mask = mask_image.convert("L").resize((w, h), Image.Resampling.NEAREST)
    arr = (np.asarray(mask) > 127).astype("float32")
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device=device)


def parse_points(text: str, scale: int = 8) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    if not text.strip():
        return points

    for item in text.replace("\n", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        x_raw, y_raw = item.split(",")
        points.append((int(round(float(x_raw) / scale)), int(round(float(y_raw) / scale))))
    return points
