/// RMSNorm: x / sqrt(mean(x²) + eps)
///
/// Fused with LN scale factor: output *= 1/sqrt(layer_idx + 1)
///
/// CPU reference implementation for testing.
/// GPU implementation via CubeCL when cuda feature is enabled.

/// Forward: y = rms_norm(x) * scale_factor (CPU reference)
pub fn rms_norm_forward_cpu(
    x: &[f32],
    output: &mut [f32],
    dim: usize,
    ln_scale_factor: f32,
    eps: f32,
) {
    let num_rows = x.len() / dim;
    for row in 0..num_rows {
        let start = row * dim;
        let end = start + dim;
        let row_data = &x[start..end];

        // Compute RMS
        let sq_sum: f32 = row_data.iter().map(|&v| v * v).sum();
        let rms = (sq_sum / dim as f32 + eps).sqrt();

        // Normalize and scale
        for (i, &v) in row_data.iter().enumerate() {
            output[start + i] = (v / rms) * ln_scale_factor;
        }
    }
}

/// Backward: compute gradient w.r.t. input x (CPU reference).
pub fn rms_norm_backward_cpu(
    x: &[f32],
    grad_output: &[f32],
    grad_input: &mut [f32],
    dim: usize,
    ln_scale_factor: f32,
    eps: f32,
) {
    let num_rows = x.len() / dim;
    for row in 0..num_rows {
        let start = row * dim;
        let end = start + dim;
        let x_row = &x[start..end];
        let dy_row = &grad_output[start..end];

        let sq_sum: f32 = x_row.iter().map(|&v| v * v).sum();
        let rms = (sq_sum / dim as f32 + eps).sqrt();
        let inv_rms = ln_scale_factor / rms;

        // dot(x, dy) / (rms^2 * dim)
        let x_dot_dy: f32 = x_row.iter().zip(dy_row.iter()).map(|(&a, &b)| a * b).sum();
        let coeff = x_dot_dy / (rms * rms * dim as f32);

        for i in 0..dim {
            grad_input[start + i] = inv_rms * (dy_row[i] - x_row[i] * coeff);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_forward() {
        let dim = 4;
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 8];

        rms_norm_forward_cpu(&x, &mut output, dim, 1.0, 1e-5);

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms0 = (30.0f32 / 4.0).sqrt();
        assert!((output[0] - 1.0 / rms0).abs() < 1e-5);
        assert!((output[1] - 2.0 / rms0).abs() < 1e-5);

        // Test with LN scale
        let mut scaled_output = vec![0.0f32; 8];
        let scale = 1.0 / (3.0f32).sqrt(); // layer_idx = 2
        rms_norm_forward_cpu(&x, &mut scaled_output, dim, scale, 1e-5);
        assert!((scaled_output[0] - output[0] * scale).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm_backward() {
        // Numerical gradient check
        let dim = 4;
        let x = vec![1.0f32, -0.5, 2.0, -1.5];
        let dy = vec![0.3f32, -0.1, 0.5, 0.2];
        let mut dx = vec![0.0f32; dim];

        rms_norm_backward_cpu(&x, &dy, &mut dx, dim, 1.0, 1e-5);

        // Verify via finite differences
        let eps_fd = 1e-4;
        for i in 0..dim {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps_fd;
            x_minus[i] -= eps_fd;

            let mut out_plus = vec![0.0f32; dim];
            let mut out_minus = vec![0.0f32; dim];
            rms_norm_forward_cpu(&x_plus, &mut out_plus, dim, 1.0, 1e-5);
            rms_norm_forward_cpu(&x_minus, &mut out_minus, dim, 1.0, 1e-5);

            let numerical_grad: f32 = (0..dim)
                .map(|j| dy[j] * (out_plus[j] - out_minus[j]) / (2.0 * eps_fd))
                .sum();

            assert!(
                (dx[i] - numerical_grad).abs() < 1e-3,
                "gradient mismatch at {}: analytical={}, numerical={}",
                i,
                dx[i],
                numerical_grad
            );
        }
    }
}
