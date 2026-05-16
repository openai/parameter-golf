/// WSD (Warmup-Stable-Decay) learning rate schedule.
///
/// Three phases:
/// 1. Warmup:   steps 0 → 20      linear 0 → peak_lr
/// 2. Stable:   steps 20 → 5500   constant peak_lr
/// 3. Warmdown: steps 5500 → 9000 linear peak_lr → 0
///
/// The warmdown is linear (not cosine), decaying all the way to zero.
/// Extended warmdown (33-39% of training) is critical for competition performance.

/// Compute LR for any parameter group at a given step.
pub fn wsd_lr(step: usize, base_lr: f32, warmup: usize, total: usize, warmdown: usize) -> f32 {
    if step < warmup {
        base_lr * (step as f32 / warmup as f32)
    } else if step < total {
        base_lr * lr_scale(step, warmup, total, warmdown)
    } else {
        0.0
    }
}

/// Compute WSD LR with a floor applied after warmup.
pub fn wsd_lr_with_floor(
    step: usize,
    base_lr: f32,
    warmup: usize,
    total: usize,
    warmdown: usize,
    min_scale: f32,
) -> f32 {
    if step < warmup {
        base_lr * (step as f32 / warmup as f32)
    } else {
        base_lr * lr_scale_with_floor(step, warmup, total, warmdown, min_scale)
    }
}

/// LR scale factor (0.0 to 1.0) at a given step.
pub fn lr_scale(step: usize, warmup: usize, total: usize, warmdown: usize) -> f32 {
    let warmdown_start = total.saturating_sub(warmdown);
    if step < warmup {
        step as f32 / warmup as f32
    } else if step < warmdown_start {
        1.0
    } else if step < total {
        ((total - step) as f32 / warmdown as f32).max(0.0)
    } else {
        0.0
    }
}

/// LR scale with a nonzero floor after warmup. This matches late-frontier
/// MIN_LR schedules while preserving a true zero-start warmup.
pub fn lr_scale_with_floor(
    step: usize,
    warmup: usize,
    total: usize,
    warmdown: usize,
    min_scale: f32,
) -> f32 {
    if step < warmup {
        lr_scale(step, warmup, total, warmdown)
    } else {
        lr_scale(step, warmup, total, warmdown).max(min_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wsd_schedule() {
        let lr = 0.025;
        let warmup = 20;
        let total = 9000;
        let warmdown = 3500;

        // Warmup
        assert_eq!(wsd_lr(0, lr, warmup, total, warmdown), 0.0);
        assert!((wsd_lr(10, lr, warmup, total, warmdown) - 0.0125).abs() < 1e-6);
        assert_eq!(wsd_lr(20, lr, warmup, total, warmdown), lr);

        // Stable
        assert_eq!(wsd_lr(100, lr, warmup, total, warmdown), lr);
        assert_eq!(wsd_lr(5499, lr, warmup, total, warmdown), lr);

        // Warmdown
        assert_eq!(wsd_lr(5500, lr, warmup, total, warmdown), lr);
        assert_eq!(wsd_lr(9000, lr, warmup, total, warmdown), 0.0);

        // Scale
        assert_eq!(lr_scale(0, warmup, total, warmdown), 0.0);
        assert_eq!(lr_scale(100, warmup, total, warmdown), 1.0);
        assert_eq!(lr_scale(9000, warmup, total, warmdown), 0.0);

        assert_eq!(lr_scale_with_floor(0, warmup, total, warmdown, 0.1), 0.0);
        assert_eq!(lr_scale_with_floor(100, warmup, total, warmdown, 0.1), 1.0);
        assert_eq!(lr_scale_with_floor(9000, warmup, total, warmdown, 0.1), 0.1);
    }
}
