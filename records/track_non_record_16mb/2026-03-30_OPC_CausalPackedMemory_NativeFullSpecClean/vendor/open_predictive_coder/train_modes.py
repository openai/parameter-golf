from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrainStateMode = Literal["detached", "through_state"]


@dataclass(frozen=True)
class TrainModeConfig:
    state_mode: TrainStateMode = "detached"
    slow_update_stride: int = 1
    rollout_checkpoints: tuple[int, ...] = ()
    rollout_checkpoint_stride: int | None = None
    include_final_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.state_mode not in {"detached", "through_state"}:
            raise ValueError("state_mode must be 'detached' or 'through_state'")
        if self.slow_update_stride < 1:
            raise ValueError("slow_update_stride must be >= 1")
        if self.rollout_checkpoint_stride is not None and self.rollout_checkpoint_stride < 1:
            raise ValueError("rollout_checkpoint_stride must be >= 1")
        if any(step < 1 for step in self.rollout_checkpoints):
            raise ValueError("rollout_checkpoints must contain positive step indices")

    @property
    def uses_detached_state(self) -> bool:
        return self.state_mode == "detached"

    @property
    def uses_through_state(self) -> bool:
        return self.state_mode == "through_state"

    @property
    def uses_sparse_slow_updates(self) -> bool:
        return self.slow_update_stride > 1

    def should_update_slow(self, step_index: int) -> bool:
        if step_index < 0:
            raise ValueError("step_index must be >= 0")
        return (step_index + 1) % self.slow_update_stride == 0

    def resolve_rollout_checkpoints(self, total_steps: int) -> tuple[int, ...]:
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if any(step > total_steps for step in self.rollout_checkpoints):
            raise ValueError("rollout_checkpoints must lie within the rollout length")

        checkpoints = set(self.rollout_checkpoints)
        if self.rollout_checkpoint_stride is not None:
            checkpoints.update(range(self.rollout_checkpoint_stride, total_steps + 1, self.rollout_checkpoint_stride))
        if self.include_final_checkpoint:
            checkpoints.add(total_steps)

        return tuple(sorted(step for step in checkpoints if 1 <= step <= total_steps))


__all__ = ["TrainModeConfig", "TrainStateMode"]
