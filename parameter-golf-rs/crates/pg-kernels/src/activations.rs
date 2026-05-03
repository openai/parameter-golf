/// LeakyReLU(0.5)² activation — the key activation from the SOTA submission.
///
/// Forward: y = leaky_relu(x, 0.5)²
/// This preserves negative gradient flow through the MLP while maintaining
/// the relu² inductive bias (non-negative outputs). -0.003 BPB improvement.
///
/// Backward: dy/dx = 2 * leaky_relu(x, 0.5) * d_leaky_relu(x, 0.5)
/// where d_leaky_relu(x, 0.5) = 1.0 if x >= 0, else 0.5

const NEGATIVE_SLOPE: f32 = 0.5;

/// Forward: y = leaky_relu(x, 0.5)²
pub fn leaky_relu_sq_forward(x: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(x.iter()) {
        let lr = if v >= 0.0 { v } else { NEGATIVE_SLOPE * v };
        *o = lr * lr;
    }
}

/// Backward: grad_input = grad_output * 2 * leaky_relu(x, 0.5) * d_leaky_relu(x, 0.5)
pub fn leaky_relu_sq_backward(x: &[f32], grad_output: &[f32], grad_input: &mut [f32]) {
    for i in 0..x.len() {
        let v = x[i];
        let lr = if v >= 0.0 { v } else { NEGATIVE_SLOPE * v };
        let d_lr = if v >= 0.0 { 1.0 } else { NEGATIVE_SLOPE };
        grad_input[i] = grad_output[i] * 2.0 * lr * d_lr;
    }
}

/// Logit softcap: cap * tanh(x / cap)
/// Applied before softmax to prevent logit explosion.
pub fn softcap_forward(x: &[f32], output: &mut [f32], cap: f32) {
    let inv_cap = 1.0 / cap;
    for (o, &v) in output.iter_mut().zip(x.iter()) {
        *o = cap * (v * inv_cap).tanh();
    }
}

/// Softcap backward: grad_input = grad_output * (1 - tanh²(x/cap))
pub fn softcap_backward(x: &[f32], grad_output: &[f32], grad_input: &mut [f32], cap: f32) {
    let inv_cap = 1.0 / cap;
    for i in 0..x.len() {
        let t = (x[i] * inv_cap).tanh();
        grad_input[i] = grad_output[i] * (1.0 - t * t);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leaky_relu_sq_forward() {
        let x = vec![2.0f32, -1.0, 0.0, 3.0, -2.0];
        let mut y = vec![0.0; 5];
        leaky_relu_sq_forward(&x, &mut y);

        assert!((y[0] - 4.0).abs() < 1e-6); // 2² = 4
        assert!((y[1] - 0.25).abs() < 1e-6); // (0.5 * -1)² = 0.25
        assert!((y[2] - 0.0).abs() < 1e-6); // 0² = 0
        assert!((y[3] - 9.0).abs() < 1e-6); // 3² = 9
        assert!((y[4] - 1.0).abs() < 1e-6); // (0.5 * -2)² = 1
    }

    #[test]
    fn test_leaky_relu_sq_backward() {
        let x = vec![2.0f32, -1.0, 0.5, -0.5];
        let dy = vec![1.0; 4];
        let mut dx = vec![0.0; 4];
        leaky_relu_sq_backward(&x, &dy, &mut dx);

        // x=2: 2 * 2.0 * 1.0 = 4.0
        assert!((dx[0] - 4.0).abs() < 1e-6);
        // x=-1: 2 * (-0.5) * 0.5 = -0.5
        assert!((dx[1] - (-0.5)).abs() < 1e-6);
        // x=0.5: 2 * 0.5 * 1.0 = 1.0
        assert!((dx[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softcap() {
        let x = vec![0.0f32, 15.0, -15.0, 100.0];
        let mut y = vec![0.0; 4];
        softcap_forward(&x, &mut y, 30.0);

        assert!((y[0] - 0.0).abs() < 1e-5); // tanh(0) = 0
        assert!(y[1] > 0.0 && y[1] < 30.0); // bounded
        assert!(y[2] < 0.0 && y[2] > -30.0); // bounded
        assert!(y[3] < 30.0); // saturates toward ±30
    }
}
