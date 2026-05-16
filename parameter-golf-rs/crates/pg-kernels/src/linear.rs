/// CPU reference linear (matmul) operations for forward and backward.
///
/// Forward:  Y = X @ W^T + bias  (optional bias)
/// Backward: dX = dY @ W, dW = dY^T @ X, dbias = sum(dY, dim=0)
///
/// Shapes: X [M, K], W [N, K], Y [M, N]
/// These are reference implementations; GPU uses cuBLASLt.

/// Forward: Y[m,n] = sum_k X[m,k] * W[n,k]
/// X: [M, K], W: [N, K] (row-major), Y: [M, N]
pub fn linear_forward(x: &[f32], w: &[f32], y: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += x[i * k + l] * w[j * k + l];
            }
            y[i * n + j] = sum;
        }
    }
}

/// Forward with bias: Y[m,n] = sum_k X[m,k] * W[n,k] + bias[n]
pub fn linear_forward_bias(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    y: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    linear_forward(x, w, y, m, n, k);
    for i in 0..m {
        for j in 0..n {
            y[i * n + j] += bias[j];
        }
    }
}

/// Backward w.r.t. input: dX[m,k] = sum_n dY[m,n] * W[n,k]
/// dY: [M, N], W: [N, K], dX: [M, K]
pub fn linear_backward_input(dy: &[f32], w: &[f32], dx: &mut [f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for l in 0..k {
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += dy[i * n + j] * w[j * k + l];
            }
            dx[i * k + l] = sum;
        }
    }
}

/// Backward w.r.t. weight: dW[n,k] = sum_m dY[m,n] * X[m,k]
/// dY: [M, N], X: [M, K], dW: [N, K]
pub fn linear_backward_weight(dy: &[f32], x: &[f32], dw: &mut [f32], m: usize, n: usize, k: usize) {
    for j in 0..n {
        for l in 0..k {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += dy[i * n + j] * x[i * k + l];
            }
            dw[j * k + l] = sum;
        }
    }
}

/// Backward w.r.t. bias: dbias[n] = sum_m dY[m,n]
pub fn linear_backward_bias(dy: &[f32], dbias: &mut [f32], m: usize, n: usize) {
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..m {
            sum += dy[i * n + j];
        }
        dbias[j] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        // X=[2,3], W=[2,3] → Y=[2,2]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]; // W[0]=[1,0,1], W[1]=[0,1,0]
        let mut y = vec![0.0; 4];
        linear_forward(&x, &w, &mut y, 2, 2, 3);

        // Y[0,0] = 1*1 + 2*0 + 3*1 = 4
        // Y[0,1] = 1*0 + 2*1 + 3*0 = 2
        // Y[1,0] = 4*1 + 5*0 + 6*1 = 10
        // Y[1,1] = 4*0 + 5*1 + 6*0 = 5
        assert!((y[0] - 4.0).abs() < 1e-6);
        assert!((y[1] - 2.0).abs() < 1e-6);
        assert!((y[2] - 10.0).abs() < 1e-6);
        assert!((y[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_gradient_numerical() {
        let m = 3;
        let n = 2;
        let k = 4;
        let x = vec![
            0.1, -0.3, 0.5, 0.2, -0.1, 0.4, -0.2, 0.3, 0.6, -0.5, 0.1, 0.7,
        ];
        let w = vec![0.3, -0.1, 0.4, 0.2, -0.2, 0.5, -0.3, 0.1];
        let dy = vec![1.0, 0.5, -0.3, 0.8, 0.2, -0.1];

        // Analytical gradients
        let mut dx = vec![0.0; m * k];
        let mut dw = vec![0.0; n * k];
        linear_backward_input(&dy, &w, &mut dx, m, n, k);
        linear_backward_weight(&dy, &x, &mut dw, m, n, k);

        // Numerical gradient for dx
        let eps = 1e-4;
        for idx in 0..m * k {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[idx] += eps;
            x_m[idx] -= eps;

            let mut y_p = vec![0.0; m * n];
            let mut y_m = vec![0.0; m * n];
            linear_forward(&x_p, &w, &mut y_p, m, n, k);
            linear_forward(&x_m, &w, &mut y_m, m, n, k);

            // loss = sum(dy * y)
            let loss_p: f32 = dy.iter().zip(y_p.iter()).map(|(&a, &b)| a * b).sum();
            let loss_m: f32 = dy.iter().zip(y_m.iter()).map(|(&a, &b)| a * b).sum();
            let numerical = (loss_p - loss_m) / (2.0 * eps);

            assert!(
                (dx[idx] - numerical).abs() < 1e-3,
                "dx mismatch at {}: analytical={}, numerical={}",
                idx,
                dx[idx],
                numerical
            );
        }
    }
}
