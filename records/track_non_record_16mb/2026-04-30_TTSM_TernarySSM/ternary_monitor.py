"""
TernaryMonitor: Adaptive LR controller for ternary QAT training.

Ported from GeoCog's AdaptiveLRController (geocog/adaptive_lr.py).

GeoCog watches coherence_H (directional consistency of latent movement) to
detect training states. We adapt this for ternary QAT by watching:
  - flip_rate: fraction of weights that changed ternary state per step
  - loss_velocity: smoothed slope of training loss

State mapping:
  GeoCog coherence_H → Ternary flip_rate
  GROUPTHINK (high coh + low acc) → FROZEN (low flip + stagnant loss)
  NOISY (low coh) → CHURNING (high flip + stagnant loss)
  LEARNING (healthy coh range) → LEARNING (decreasing loss)
  CRYSTALLIZING → PLATEAU (low flip + slow improvement)
  CONVERGED → not mapped (we don't have accuracy for LM training)

The key insight: flip_rate is to ternary weights what coherence_H is to
z-trajectory direction — both measure "is the model making meaningful
progress or stuck?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
from torch import Tensor


class TernaryState(Enum):
    """Detected training state based on ternary monitoring signals."""
    FROZEN = "frozen"       # Weights stuck, loss not improving → spike LR
    CHURNING = "churning"   # Weights thrashing, loss not improving → reduce LR
    LEARNING = "learning"   # Loss improving → maintain LR
    PLATEAU = "plateau"     # Slow/no improvement, moderate flip → nudge LR up


@dataclass
class TernaryMonitorConfig:
    """Configuration for the ternary monitor."""

    # Flip rate thresholds
    frozen_threshold: float = 0.001     # Below this = frozen
    churning_threshold: float = 0.05    # Above this = churning

    # Loss velocity threshold (negative = improving)
    improving_threshold: float = -1e-4  # Below this = loss is improving

    # LR adjustment factors (applied to current LR factor)
    frozen_factor: float = 2.0          # Spike LR to shake free (geocog: GROUPTHINK)
    churning_factor: float = 0.5        # Halve LR to stabilize (geocog: NOISY)
    learning_decay: float = 0.9         # Decay factor toward 1.0 when healthy
    plateau_factor: float = 1.2         # Gentle nudge up

    # Smoothing
    window: int = 50                    # Steps to average over
    hysteresis_steps: int = 2           # Steps state must be stable before acting

    # LR bounds
    min_lr_factor: float = 0.1          # Don't go below base_lr * this
    max_lr_factor: float = 5.0          # Don't go above base_lr * this

    # Gradient perturbation on frozen state
    perturb_on_frozen: bool = True
    perturb_magnitude: float = 0.3      # Noise fraction added to gradients

    # Logging
    log_every: int = 100                # Log monitor state every N steps


class TernaryMonitor:
    """
    Monitors ternary weight dynamics and adjusts learning rate accordingly.

    Usage:
        monitor = TernaryMonitor(config, base_lr=0.02)

        for step in training:
            loss = train_step()

            # Snapshot ternary weights periodically
            if step % monitor_every == 0:
                state, lr_factor = monitor.step(model, loss.item())
                # Apply lr_factor to optimizer groups
    """

    def __init__(
        self,
        config: Optional[TernaryMonitorConfig] = None,
        base_lr: float = 0.02,
    ):
        self.config = config or TernaryMonitorConfig()
        self.base_lr = base_lr

        # History buffers
        self.loss_history: list[float] = []
        self.flip_rate_history: list[float] = []
        self.state_history: list[TernaryState] = []
        self.lr_factor_history: list[float] = []

        # Weight snapshots
        self._prev_weights: Optional[dict[str, Tensor]] = None

        # State tracking
        self.current_state = TernaryState.LEARNING
        self.steps_in_state = 0
        self.total_steps = 0
        self.current_lr_factor = 1.0

    def _extract_ternary_weights(self, model: torch.nn.Module) -> dict[str, Tensor]:
        """Extract quantized ternary weights from all TernaryLinear modules."""
        weights = {}
        for name, module in model.named_modules():
            if hasattr(module, 'get_ternary_weights'):
                ternary, _ = module.get_ternary_weights()
                weights[name] = ternary
        return weights

    def _compute_flip_rate(
        self, prev: dict[str, Tensor], curr: dict[str, Tensor]
    ) -> float:
        """Compute fraction of weights that changed ternary state."""
        total = 0
        flipped = 0
        for name in prev:
            if name in curr:
                p, c = prev[name], curr[name]
                total += p.numel()
                flipped += (p != c).sum().item()
        return flipped / max(total, 1)

    def _compute_loss_velocity(self) -> float:
        """Compute smoothed loss velocity (negative = improving)."""
        w = self.config.window
        if len(self.loss_history) < w:
            return -1.0  # Assume learning early on
        return (self.loss_history[-1] - self.loss_history[-w]) / w

    def _detect_state(self, flip_rate: float, loss_vel: float) -> TernaryState:
        """Detect training state from signals."""
        cfg = self.config

        # Average flip rate for stability
        recent_flips = self.flip_rate_history[-cfg.window:]
        avg_flip = sum(recent_flips) / max(len(recent_flips), 1)

        is_improving = loss_vel < cfg.improving_threshold

        if avg_flip < cfg.frozen_threshold and not is_improving:
            return TernaryState.FROZEN
        elif avg_flip > cfg.churning_threshold and not is_improving:
            return TernaryState.CHURNING
        elif is_improving:
            return TernaryState.LEARNING
        else:
            return TernaryState.PLATEAU

    def step(
        self,
        model: torch.nn.Module,
        loss_value: float,
    ) -> tuple[TernaryState, float]:
        """
        Update monitor with current model state and loss.

        Args:
            model: the model (must contain TernaryLinear modules)
            loss_value: current training loss

        Returns:
            state: detected training state
            lr_factor: multiplier to apply to learning rate
        """
        # Extract current ternary weights
        curr_weights = self._extract_ternary_weights(model)

        # Compute flip rate
        if self._prev_weights is not None:
            flip_rate = self._compute_flip_rate(self._prev_weights, curr_weights)
        else:
            flip_rate = 1.0  # First step: everything is "new"
        self._prev_weights = {k: v.clone() for k, v in curr_weights.items()}

        # Update histories
        self.loss_history.append(loss_value)
        self.flip_rate_history.append(flip_rate)

        # Compute loss velocity
        loss_vel = self._compute_loss_velocity()

        # Detect state
        new_state = self._detect_state(flip_rate, loss_vel)

        # Track state transitions (hysteresis)
        if new_state != self.current_state:
            self.steps_in_state = 0
        else:
            self.steps_in_state += 1
        self.current_state = new_state

        # Compute LR factor
        cfg = self.config
        if self.steps_in_state < cfg.hysteresis_steps:
            # Not stable yet — hold current factor
            lr_factor = self.current_lr_factor
        elif new_state == TernaryState.LEARNING:
            # Decay factor toward 1.0: factor = 1.0 + (factor - 1.0) * decay
            # After a FROZEN spike (factor=2.0), this gradually returns to baseline
            lr_factor = 1.0 + (self.current_lr_factor - 1.0) * cfg.learning_decay
        else:
            factor_map = {
                TernaryState.FROZEN: cfg.frozen_factor,
                TernaryState.CHURNING: cfg.churning_factor,
                TernaryState.PLATEAU: cfg.plateau_factor,
            }
            raw_factor = self.current_lr_factor * factor_map[new_state]
            lr_factor = max(cfg.min_lr_factor, min(cfg.max_lr_factor, raw_factor))

        self.current_lr_factor = lr_factor
        self.state_history.append(new_state)
        self.lr_factor_history.append(lr_factor)
        self.total_steps += 1

        return new_state, lr_factor

    def should_perturb(self) -> bool:
        """Check if we should add noise to gradients (frozen escape)."""
        return (
            self.config.perturb_on_frozen
            and self.current_state == TernaryState.FROZEN
            and self.steps_in_state >= self.config.hysteresis_steps
        )

    def get_summary(self) -> dict:
        """Get summary statistics for logging."""
        state_counts: dict[str, int] = {}
        for s in self.state_history:
            state_counts[s.value] = state_counts.get(s.value, 0) + 1

        return {
            "total_steps": self.total_steps,
            "final_state": self.current_state.value,
            "final_lr_factor": self.current_lr_factor,
            "state_counts": state_counts,
            "avg_flip_rate": (
                sum(self.flip_rate_history) / max(len(self.flip_rate_history), 1)
            ),
            "final_loss_velocity": self._compute_loss_velocity(),
            "frozen_episodes": state_counts.get("frozen", 0),
            "churning_episodes": state_counts.get("churning", 0),
        }

    def format_status(self) -> str:
        """Format current status for logging."""
        flip = self.flip_rate_history[-1] if self.flip_rate_history else 0.0
        loss_vel = self._compute_loss_velocity()
        return (
            f"monitor:{self.current_state.value} "
            f"flip_rate:{flip:.6f} loss_vel:{loss_vel:.6f} "
            f"lr_factor:{self.current_lr_factor:.3f} "
            f"steps_in_state:{self.steps_in_state}"
        )
