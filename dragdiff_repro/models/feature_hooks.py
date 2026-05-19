from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass
class FeatureCapture:
    feature: torch.Tensor | None = None


def _pick_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError("Cannot find tensor output from hooked UNet block.")


@contextmanager
def capture_up_block_feature(unet, block_index: int = 2) -> Iterator[FeatureCapture]:
    capture = FeatureCapture()

    def hook(_module, _inputs, output):
        capture.feature = _pick_tensor(output)

    handle = unet.up_blocks[block_index].register_forward_hook(hook)
    try:
        yield capture
    finally:
        handle.remove()

