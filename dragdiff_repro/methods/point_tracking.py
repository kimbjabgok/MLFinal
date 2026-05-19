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
) -> list[tuple[int, int]]:
    _, _, h, w = feature.shape
    next_points: list[tuple[int, int]] = []

    for index, (x, y) in enumerate(current_points):
        best_point = (x, y)
        best_dist = None
        ref = reference_vectors[index]

        for yy in range(max(0, y - radius), min(h, y + radius + 1)):
            for xx in range(max(0, x - radius), min(w, x + radius + 1)):
                vec = feature[0, :, yy, xx]
                dist = torch.abs(vec - ref).mean()
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_point = (xx, yy)

        next_points.append(best_point)

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

