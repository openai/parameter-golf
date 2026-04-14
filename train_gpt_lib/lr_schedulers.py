"""Learning rate schedulers.

Each scheduler is a function with the signature::

    fn(step, total_steps, warmdown_iters, elapsed_ms, max_wallclock_ms) -> float

Returns a *multiplier* in [0, 1] that is applied to every optimizer group's
``base_lr`` at each training step.

When ``max_wallclock_ms`` is not None the training is capped by wall-clock
time, so schedulers that care about "how far along are we" must use time
rather than step count.  ``_progress()`` is a helper that abstracts that.
"""
from __future__ import annotations

import math
from typing import Callable

# (step, total_steps, warmdown_iters, elapsed_ms, max_wallclock_ms) -> lr_mul
LrSchedulerFn = Callable[[int, int, int, float, "float | None"], float]


def _progress(step: int, total_steps: int, elapsed_ms: float, max_wallclock_ms: "float | None") -> float:
    """Training progress in [0, 1]: either by step count or by wall-clock."""
    if max_wallclock_ms is not None and max_wallclock_ms > 0:
        return min(elapsed_ms / max_wallclock_ms, 1.0)
    return min(step / max(total_steps, 1), 1.0)


def _warmdown_t(
    step: int,
    total_steps: int,
    warmdown_iters: int,
    elapsed_ms: float,
    max_wallclock_ms: "float | None",
) -> "float | None":
    """Fraction through the warmdown window [0, 1], or None if not yet in warmdown."""
    if warmdown_iters <= 0:
        return None
    if max_wallclock_ms is None:
        warmdown_start = max(total_steps - warmdown_iters, 0)
        if step < warmdown_start:
            return None
        return min((step - warmdown_start) / max(warmdown_iters, 1), 1.0)
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    if remaining_ms > warmdown_ms:
        return None
    return 1.0 - remaining_ms / max(warmdown_ms, 1e-9)


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------

def trapezoid(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Flat then **linear** decay to 0 (current default). Wallclock-aware."""
    t = _warmdown_t(step, total_steps, warmdown_iters, elapsed_ms, max_wallclock_ms)
    if t is None:
        return 1.0
    return max(1.0 - t, 0.0)


def trapezoid_cosine(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Flat then **cosine** decay to 0. Same shape as trapezoid but smooth tail."""
    t = _warmdown_t(step, total_steps, warmdown_iters, elapsed_ms, max_wallclock_ms)
    if t is None:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * t))


def trapezoid_cosine_min10(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Flat then cosine decay to **10 % of peak** (never fully zeros out)."""
    min_frac = 0.1
    t = _warmdown_t(step, total_steps, warmdown_iters, elapsed_ms, max_wallclock_ms)
    if t is None:
        return 1.0
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_frac + (1.0 - min_frac) * cosine_val


def cosine(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Full cosine annealing from 1 → 0 over the entire training run."""
    p = _progress(step, total_steps, elapsed_ms, max_wallclock_ms)
    return 0.5 * (1.0 + math.cos(math.pi * p))


def cosine_min10(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Full cosine annealing from 1 → 0.1 (10 % floor) over entire training."""
    min_frac = 0.1
    p = _progress(step, total_steps, elapsed_ms, max_wallclock_ms)
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * p))


def linear(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Linear decay from 1 → 0 starting from step 0."""
    p = _progress(step, total_steps, elapsed_ms, max_wallclock_ms)
    return max(1.0 - p, 0.0)


def constant(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """No decay — constant LR throughout training (ablation baseline)."""
    return 1.0


def rsqrt(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Reciprocal-sqrt decay: lr ∝ 1/√step (normalized so step=1 → 1.0).

    Classic T5/transformer schedule shape without warmup phase.
    Decays quickly early on, then slowly — opposite of trapezoid.
    """
    if max_wallclock_ms is not None and max_wallclock_ms > 0:
        p = min(elapsed_ms / max_wallclock_ms, 1.0)
        # Map to a pseudo-step in [1, total_steps]
        pseudo_step = max(p * max(total_steps, 1), 1.0)
        return 1.0 / math.sqrt(pseudo_step)
    return 1.0 / math.sqrt(max(step, 1))


def cosine_warmup(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Linear warmup for ``warmdown_iters`` steps, then cosine annealing to 0.

    Set ``WARMDOWN_ITERS`` to the desired warmup length (e.g. 600).
    Warmup phase always uses step count, cosine phase is wallclock-aware.
    """
    warmup = max(warmdown_iters, 1)
    if step < warmup:
        return step / warmup
    p = _progress(step - warmup, max(total_steps - warmup, 1), elapsed_ms, max_wallclock_ms)
    return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))


def rsqrt_warmup(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Linear warmup for ``warmdown_iters`` steps, then 1/√t decay.

    Normalized so lr=1.0 at the end of warmup.  Set ``WARMDOWN_ITERS`` to the
    desired warmup length (e.g. 600).  Unlike other schedulers, this one always
    uses step count for the warmup phase (not wall-clock time).
    """
    warmup = max(warmdown_iters, 1)
    if step < warmup:
        return step / warmup
    # 1/sqrt normalized so step==warmup → 1.0, then decays
    return math.sqrt(warmup) / math.sqrt(step)


def cosine_warmup_10pct(
    step: int, total_steps: int, warmdown_iters: int, elapsed_ms: float, max_wallclock_ms: "float | None"
) -> float:
    """Linear warmup for 10 % of training, then cosine decay 1 → 0.

    Cosine phase begins immediately after warmup ends.
    Wallclock-aware when ``max_wallclock_ms`` is set.
    """
    if max_wallclock_ms is not None and max_wallclock_ms > 0:
        warmup_ms = 0.1 * max_wallclock_ms
        if elapsed_ms < warmup_ms:
            return elapsed_ms / max(warmup_ms, 1e-9)
        decay_ms = max_wallclock_ms - warmup_ms
        t = min((elapsed_ms - warmup_ms) / max(decay_ms, 1e-9), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    warmup = max(total_steps // 10, 1)
    if step < warmup:
        return step / warmup
    speed = 3
    t = min(((step - warmup) / max(total_steps - warmup, 1)) * speed, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * 4*t))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, LrSchedulerFn] = {
    "trapezoid":              trapezoid,
    "trapezoid_cosine":       trapezoid_cosine,
    "trapezoid_cosine_min10": trapezoid_cosine_min10,
    "cosine":                 cosine,
    "cosine_min10":           cosine_min10,
    "linear":                 linear,
    "constant":               constant,
    "rsqrt":                  rsqrt,
    "rsqrt_warmup":           rsqrt_warmup,
    "cosine_warmup":          cosine_warmup,
    "cosine_warmup_10pct":    cosine_warmup_10pct,
}


def get_scheduler(name: str) -> LrSchedulerFn:
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown lr_schedule={name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]
