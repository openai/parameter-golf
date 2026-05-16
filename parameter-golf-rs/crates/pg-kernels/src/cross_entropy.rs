/// Fused cross-entropy loss with logit softcap.
///
/// Computes: loss = -log(softmax(softcap(logits))[target])
/// where softcap(x) = cap * tanh(x / cap)
///
/// The fused implementation avoids materializing the full softmax output.
/// With vocab=1024 this saves modest memory but eliminates a global memory
/// round-trip.

/// Forward: compute per-token cross-entropy loss with softcap.
/// logits: [num_tokens, vocab_size]
/// targets: [num_tokens] (token IDs)
/// Returns per-token losses.
pub fn cross_entropy_forward(
    logits: &[f32],
    targets: &[u32],
    losses: &mut [f32],
    vocab_size: usize,
    softcap: f32,
) {
    cross_entropy_forward_asym(logits, targets, losses, vocab_size, softcap, softcap);
}

pub fn cross_entropy_forward_asym(
    logits: &[f32],
    targets: &[u32],
    losses: &mut [f32],
    vocab_size: usize,
    softcap_pos: f32,
    softcap_neg: f32,
) {
    let num_tokens = targets.len();
    let cap = |v: f32| {
        let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
        if softcap > 0.0 {
            softcap * (v / softcap).tanh()
        } else {
            v
        }
    };

    for t in 0..num_tokens {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];
        let target = targets[t] as usize;

        // Apply softcap: cap * tanh(x / cap)
        // Find max for numerical stability (after softcap)
        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            let capped = cap(v);
            if capped > max_val {
                max_val = capped;
            }
        }

        // Compute log-sum-exp
        let mut sum_exp = 0.0f32;
        for &v in row {
            let capped = cap(v);
            sum_exp += (capped - max_val).exp();
        }
        let log_sum_exp = max_val + sum_exp.ln();

        // Loss = -log(softmax[target]) = -(capped_target - log_sum_exp)
        let capped_target = cap(row[target]);
        losses[t] = log_sum_exp - capped_target;
    }
}

/// Backward: compute gradient w.r.t. logits.
/// grad_logits: [num_tokens, vocab_size]
/// grad_loss: scalar multiplier (typically 1/num_tokens for mean reduction)
pub fn cross_entropy_backward(
    logits: &[f32],
    targets: &[u32],
    grad_logits: &mut [f32],
    vocab_size: usize,
    softcap: f32,
    grad_loss: f32,
) {
    cross_entropy_backward_asym(
        logits,
        targets,
        grad_logits,
        vocab_size,
        softcap,
        softcap,
        grad_loss,
    );
}

pub fn cross_entropy_backward_asym(
    logits: &[f32],
    targets: &[u32],
    grad_logits: &mut [f32],
    vocab_size: usize,
    softcap_pos: f32,
    softcap_neg: f32,
    grad_loss: f32,
) {
    let num_tokens = targets.len();
    let mut exps = vec![0.0f32; vocab_size]; // pre-allocated, reused across tokens
    let cap = |v: f32| {
        let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
        if softcap > 0.0 {
            softcap * (v / softcap).tanh()
        } else {
            v
        }
    };
    let cap_deriv = |v: f32| {
        let softcap = if v >= 0.0 { softcap_pos } else { softcap_neg };
        if softcap > 0.0 {
            let t_val = (v / softcap).tanh();
            1.0 - t_val * t_val
        } else {
            1.0
        }
    };

    for t in 0..num_tokens {
        let offset = t * vocab_size;
        let row = &logits[offset..offset + vocab_size];
        let target = targets[t] as usize;

        // Compute softmax of capped logits
        let mut max_val = f32::NEG_INFINITY;
        for &v in row {
            let capped = cap(v);
            if capped > max_val {
                max_val = capped;
            }
        }

        let mut sum_exp = 0.0f32;
        for (i, &v) in row.iter().enumerate() {
            let capped = cap(v);
            exps[i] = (capped - max_val).exp();
            sum_exp += exps[i];
        }

        // grad_logits = grad_loss * (softmax - one_hot) * d_softcap/d_logits
        for i in 0..vocab_size {
            let prob = exps[i] / sum_exp;
            let one_hot = if i == target { 1.0 } else { 0.0 };
            // d_softcap/d_x = 1 - tanh²(x/cap)
            grad_logits[offset + i] = grad_loss * (prob - one_hot) * cap_deriv(row[i]);
        }
    }
}

/// Compute mean loss from per-token losses.
pub fn mean_loss(losses: &[f32]) -> f32 {
    if losses.is_empty() {
        return 0.0;
    }
    losses.iter().sum::<f32>() / losses.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_basic() {
        // vocab=4, one token, target=2
        // logits that strongly predict class 2
        let logits = vec![0.0f32, 0.0, 10.0, 0.0];
        let targets = vec![2u32];
        let mut losses = vec![0.0f32];

        cross_entropy_forward(&logits, &targets, &mut losses, 4, 30.0);
        // With softcap=30, tanh(10/30) ≈ 0.321, capped ≈ 9.64
        // Should be low loss since target has highest logit
        assert!(losses[0] < 1.0, "loss too high: {}", losses[0]);
        assert!(losses[0] > 0.0, "loss should be positive");
    }

    #[test]
    fn test_cross_entropy_uniform() {
        // All logits equal → loss = ln(vocab_size)
        let vocab = 1024;
        let logits = vec![0.0f32; vocab];
        let targets = vec![0u32];
        let mut losses = vec![0.0f32];

        cross_entropy_forward(&logits, &targets, &mut losses, vocab, 30.0);
        let expected = (vocab as f32).ln();
        assert!(
            (losses[0] - expected).abs() < 0.01,
            "expected ~{}, got {}",
            expected,
            losses[0]
        );
    }

    #[test]
    fn test_cross_entropy_gradient() {
        // Numerical gradient check
        let vocab = 8;
        let logits = vec![1.0f32, -0.5, 2.0, 0.3, -1.0, 0.5, 1.5, -0.2];
        let targets = vec![3u32];
        let softcap = 30.0;

        let mut grad = vec![0.0f32; vocab];
        cross_entropy_backward(&logits, &targets, &mut grad, vocab, softcap, 1.0);

        // Numerical gradient
        let eps = 1e-4;
        for i in 0..vocab {
            let mut logits_plus = logits.clone();
            let mut logits_minus = logits.clone();
            logits_plus[i] += eps;
            logits_minus[i] -= eps;

            let mut loss_plus = vec![0.0f32];
            let mut loss_minus = vec![0.0f32];
            cross_entropy_forward(&logits_plus, &targets, &mut loss_plus, vocab, softcap);
            cross_entropy_forward(&logits_minus, &targets, &mut loss_minus, vocab, softcap);

            let numerical = (loss_plus[0] - loss_minus[0]) / (2.0 * eps);
            let tol = 2e-2 * grad[i].abs().max(numerical.abs()).max(1e-3);
            assert!(
                (grad[i] - numerical).abs() < tol,
                "grad mismatch at {}: analytical={}, numerical={}, diff={}",
                i,
                grad[i],
                numerical,
                (grad[i] - numerical).abs()
            );
        }
    }

    #[test]
    fn test_asymmetric_softcap_changes_negative_logits() {
        let vocab = 4;
        let logits = vec![-40.0f32, -2.0, 1.0, 4.0];
        let targets = vec![0u32];
        let mut symmetric = vec![0.0f32];
        let mut asymmetric = vec![0.0f32];

        cross_entropy_forward(&logits, &targets, &mut symmetric, vocab, 30.0);
        cross_entropy_forward_asym(&logits, &targets, &mut asymmetric, vocab, 30.0, 34.0);

        assert!(
            (symmetric[0] - asymmetric[0]).abs() > 1e-3,
            "asymmetric negative softcap should affect loss"
        );
    }
}
