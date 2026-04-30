/// SmearGate — gated residual mixing with previous layer's output.
///
/// output = (1 - σ(g)) * x + σ(g) * x_prev
///
/// where g is a learned scalar gate per dimension.
/// Present in SOTA code but NOT validated in ablation tables.
/// Implement for completeness; may be disabled if validation shows <0.001 BPB.

/// Forward: output = (1 - sigmoid(gate)) * x + sigmoid(gate) * x_prev
/// gate: [dim], x: [tokens, dim], x_prev: [tokens, dim], output: [tokens, dim]
pub fn smear_gate_forward(
    x: &[f32],
    x_prev: &[f32],
    gate: &[f32],
    output: &mut [f32],
    tokens: usize,
    dim: usize,
) {
    for t in 0..tokens {
        for d in 0..dim {
            let idx = t * dim + d;
            let sig = sigmoid(gate[d]);
            output[idx] = (1.0 - sig) * x[idx] + sig * x_prev[idx];
        }
    }
}

/// Boundary-aware SmearGate used by leaderboard-safe packed document streams.
///
/// When the current token is the configured boundary/BOS token, the previous
/// token contribution is masked out. This prevents the final token of one
/// packed document from leaking into the BOS state of the next document.
pub fn smear_gate_forward_boundary(
    x: &[f32],
    input_ids: &[u32],
    gate: &[f32],
    output: &mut [f32],
    tokens: usize,
    dim: usize,
    boundary_token_id: u32,
) {
    for t in 0..tokens {
        let use_prev = t > 0 && input_ids.get(t).copied() != Some(boundary_token_id);
        for d in 0..dim {
            let idx = t * dim + d;
            let sig = sigmoid(gate[d]);
            let x_prev = if use_prev { x[idx - dim] } else { 0.0 };
            output[idx] = (1.0 - sig) * x[idx] + sig * x_prev;
        }
    }
}

/// Backward for SmearGate.
/// Returns gradients for x, x_prev, and gate.
pub fn smear_gate_backward(
    x: &[f32],
    x_prev: &[f32],
    gate: &[f32],
    grad_output: &[f32],
    grad_x: &mut [f32],
    grad_x_prev: &mut [f32],
    grad_gate: &mut [f32], // [dim], accumulated over tokens
    tokens: usize,
    dim: usize,
) {
    // Zero grad_gate first (accumulated over tokens)
    grad_gate.iter_mut().for_each(|g| *g = 0.0);

    for t in 0..tokens {
        for d in 0..dim {
            let idx = t * dim + d;
            let sig = sigmoid(gate[d]);
            let go = grad_output[idx];

            // d/dx = (1 - sig) * go
            grad_x[idx] = (1.0 - sig) * go;
            // d/dx_prev = sig * go
            grad_x_prev[idx] = sig * go;
            // d/dgate = go * (x_prev - x) * sig * (1 - sig)
            grad_gate[d] += go * (x_prev[idx] - x[idx]) * sig * (1.0 - sig);
        }
    }
}

/// Boundary-aware SmearGate backward matching `smear_gate_forward_boundary`.
pub fn smear_gate_backward_boundary(
    x: &[f32],
    input_ids: &[u32],
    gate: &[f32],
    grad_output: &[f32],
    grad_x: &mut [f32],
    grad_x_prev: &mut [f32],
    grad_gate: &mut [f32],
    tokens: usize,
    dim: usize,
    boundary_token_id: u32,
) {
    grad_gate.iter_mut().for_each(|g| *g = 0.0);

    for t in 0..tokens {
        let use_prev = t > 0 && input_ids.get(t).copied() != Some(boundary_token_id);
        for d in 0..dim {
            let idx = t * dim + d;
            let sig = sigmoid(gate[d]);
            let go = grad_output[idx];
            let x_prev = if use_prev { x[idx - dim] } else { 0.0 };

            grad_x[idx] = (1.0 - sig) * go;
            grad_x_prev[idx] = if use_prev { sig * go } else { 0.0 };
            grad_gate[d] += go * (x_prev - x[idx]) * sig * (1.0 - sig);
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smear_gate_zero_gate() {
        // gate = -large → sigmoid ≈ 0 → output ≈ x
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let x_prev = vec![5.0, 6.0, 7.0, 8.0];
        let gate = vec![-20.0, -20.0]; // sigmoid ≈ 0
        let mut out = vec![0.0; 4];
        smear_gate_forward(&x, &x_prev, &gate, &mut out, 2, 2);
        for i in 0..4 {
            assert!((out[i] - x[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_smear_gate_full_gate() {
        // gate = +large → sigmoid ≈ 1 → output ≈ x_prev
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let x_prev = vec![5.0, 6.0, 7.0, 8.0];
        let gate = vec![20.0, 20.0];
        let mut out = vec![0.0; 4];
        smear_gate_forward(&x, &x_prev, &gate, &mut out, 2, 2);
        for i in 0..4 {
            assert!((out[i] - x_prev[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_smear_gate_gradient() {
        let dim = 3;
        let tokens = 2;
        let x = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let x_prev = vec![0.5, 0.3, -0.1, -0.2, 0.4, 0.1];
        let gate = vec![0.5, -0.3, 1.0];
        let grad_out = vec![1.0; tokens * dim];

        let mut grad_x = vec![0.0; tokens * dim];
        let mut grad_x_prev = vec![0.0; tokens * dim];
        let mut grad_gate = vec![0.0; dim];
        smear_gate_backward(
            &x,
            &x_prev,
            &gate,
            &grad_out,
            &mut grad_x,
            &mut grad_x_prev,
            &mut grad_gate,
            tokens,
            dim,
        );

        // Numerical check for gate gradient
        let eps = 1e-4;
        for d in 0..dim {
            let mut gate_p = gate.clone();
            let mut gate_m = gate.clone();
            gate_p[d] += eps;
            gate_m[d] -= eps;

            let mut out_p = vec![0.0; tokens * dim];
            let mut out_m = vec![0.0; tokens * dim];
            smear_gate_forward(&x, &x_prev, &gate_p, &mut out_p, tokens, dim);
            smear_gate_forward(&x, &x_prev, &gate_m, &mut out_m, tokens, dim);

            let loss_p: f32 = grad_out
                .iter()
                .zip(out_p.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let loss_m: f32 = grad_out
                .iter()
                .zip(out_m.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            let numerical = (loss_p - loss_m) / (2.0 * eps);

            assert!(
                (grad_gate[d] - numerical).abs() < 1e-3,
                "gate grad mismatch at {}: analytical={}, numerical={}",
                d,
                grad_gate[d],
                numerical
            );
        }
    }

    #[test]
    fn test_smear_gate_boundary_masks_previous_document() {
        let dim = 2;
        let tokens = 5;
        let boundary = 1u32;
        let ids = vec![boundary, 7, 8, boundary, 9];
        let gate = vec![20.0, 20.0];
        let mut x = vec![
            1.0, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0, //
            9.0, 10.0,
        ];
        let mut out_a = vec![0.0; tokens * dim];
        smear_gate_forward_boundary(&x, &ids, &gate, &mut out_a, tokens, dim, boundary);

        x[dim] = 500.0;
        x[dim + 1] = 600.0;
        let mut out_b = vec![0.0; tokens * dim];
        smear_gate_forward_boundary(&x, &ids, &gate, &mut out_b, tokens, dim, boundary);

        let doc_b_bos = 3 * dim;
        assert_eq!(
            &out_a[doc_b_bos..doc_b_bos + dim],
            &out_b[doc_b_bos..doc_b_bos + dim]
        );
        assert_ne!(&out_a[2 * dim..3 * dim], &out_b[2 * dim..3 * dim]);
    }
}
