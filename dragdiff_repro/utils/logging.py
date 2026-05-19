from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunLog:
    loss_history: list[float] = field(default_factory=list)
    point_history: list[list[tuple[int, int]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "loss_history": self.loss_history,
            "point_history": self.point_history,
        }
