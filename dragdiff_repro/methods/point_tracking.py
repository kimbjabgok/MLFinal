from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_feature(feature: torch.Tensor, points: list[tuple[int, int]]) -> torch.Tensor:
    """Bilinear sample feature vectors at integer latent-space points."""

    _, _, h, w = feature.shape
    coords = []
    for x, y in points:
        nx = (2.0 * x / max(w - 1, 1)) - 1.0
        ny = (2.0 * y / max(h - 1, 1)) - 1.0
        coords.append([nx, ny])
    grid = torch.tensor(coords, device=feature.device, dtype=feature.dtype).view(1, len(points), 1, 2)
    sampled = F.grid_sample(feature, grid, mode="bilinear", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)


def nearest_neighbor_track(
    feature: torch.Tensor,
    reference_vectors: torch.Tensor,
    current_points: list[tuple[int, int]],
    radius: int,
    target_points: list[tuple[int, int]] | None = None,
    target_weight: float = 0.05,
) -> list[tuple[int, int]]:
    _, _, h, w = feature.shape
    next_points: list[tuple[int, int]] = []

    for index, (x, y) in enumerate(current_points):
        ref = reference_vectors[index]
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)

        patch = feature[0, :, y_min:y_max, x_min:x_max]
        distances = torch.abs(patch - ref[:, None, None]).mean(dim=0)
        if target_points is not None:
            target_x, target_y = target_points[index]
            ys = torch.arange(y_min, y_max, device=feature.device, dtype=distances.dtype)[:, None]
            xs = torch.arange(x_min, x_max, device=feature.device, dtype=distances.dtype)[None, :]
            target_distance = torch.sqrt((xs - target_x) ** 2 + (ys - target_y) ** 2)
            distances = distances + target_weight * target_distance
        best_index = int(torch.argmin(distances).item())
        patch_width = x_max - x_min
        best_y, best_x = divmod(best_index, patch_width)

        next_points.append((x_min + best_x, y_min + best_y))

    return next_points


def all_points_reached(
    handle_points: list[tuple[int, int]],
    target_points: list[tuple[int, int]],
    threshold: float,
) -> bool:
    for handle, target in zip(handle_points, target_points):
        dx = handle[0] - target[0]
        dy = handle[1] - target[1]
        if (dx * dx + dy * dy) ** 0.5 > threshold:
            return False
    return True

