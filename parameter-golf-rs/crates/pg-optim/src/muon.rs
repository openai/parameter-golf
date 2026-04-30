/// Parallel Muon optimizer — Newton-Schulz 5 orthogonalized momentum.
///
/// CPU reference implementation. GPU version will use cuBLASLt strided batched GEMM
/// for Newton-Schulz and NCCL for reduce-scatter/all-gather.
///
/// Algorithm per bank:
///   1. buf = momentum * buf + grad
///   2. update = grad + momentum * buf  (Nesterov)
///   3. update = NS5(update)  (orthogonalize)
///   4. param -= lr * scale * update + lr * wd * param

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MuonNsProfile {
    Simple,
    Quintic,
    PolarExpress,
}

impl MuonNsProfile {
    fn from_env() -> Self {
        let raw = std::env::var("PG_MUON_NS_PROFILE")
            .or_else(|_| std::env::var("PG_GPU_MUON_NS_PROFILE"))
            .unwrap_or_else(|_| "polar_express".to_string());
        match raw.to_ascii_lowercase().as_str() {
            "simple" | "legacy" | "ns5" => Self::Simple,
            "quintic" | "modded_nanogpt" => Self::Quintic,
            "polar" | "polar_express" | "polarns" | "polar_ns" => Self::PolarExpress,
            _ => Self::PolarExpress,
        }
    }

    fn coeff(self, step: usize) -> (f32, f32, f32) {
        match self {
            Self::Simple => SIMPLE_NS[0],
            Self::Quintic => QUINTIC_NS[step % QUINTIC_NS.len()],
            Self::PolarExpress => {
                let idx = step.min(POLAR_EXPRESS_NS.len() - 1);
                POLAR_EXPRESS_NS[idx]
            }
        }
    }
}

const SIMPLE_NS: [(f32, f32, f32); 1] = [(3.4445, -4.7750, 2.0315)];

const QUINTIC_NS: [(f32, f32, f32); 5] = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
];

const POLAR_EXPRESS_NS: [(f32, f32, f32); 8] = [
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
];

/// Newton-Schulz 5-step orthogonalization.
/// G: [B, M, N] or [M, N] — finds the closest orthogonal matrix.
/// Returns X with X^T X ≈ I (or X X^T ≈ I if transposed).
pub fn newton_schulz5(
    g: &[f32],
    shape: &[usize],
    steps: usize,
    x: &mut [f32],
    a_buf: &mut [f32],
    b_buf: &mut [f32],
    aa: &mut [f32],
    new_x: &mut [f32],
    result: &mut [f32],
) {
    let profile = MuonNsProfile::from_env();

    // Handle 2D or 3D input
    let (batch, m, n) = match shape.len() {
        2 => (1, shape[0], shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => panic!("NS5 requires 2D or 3D input"),
    };

    let transposed = m > n;
    let (rows, cols) = if transposed { (n, m) } else { (m, n) };

    // X = G (optionally transposed) / ||G||_F
    for bi in 0..batch {
        let g_off = bi * m * n;
        let x_off = bi * rows * cols;

        // Copy (with transpose if needed)
        for r in 0..rows {
            for c_idx in 0..cols {
                if transposed {
                    x[x_off + r * cols + c_idx] = g[g_off + c_idx * n + r];
                } else {
                    x[x_off + r * cols + c_idx] = g[g_off + r * n + c_idx];
                }
            }
        }

        // Normalize by Frobenius norm
        let norm: f32 = x[x_off..x_off + rows * cols]
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt()
            + 1e-7;
        let inv_norm = 1.0 / norm;
        for v in x[x_off..x_off + rows * cols].iter_mut() {
            *v *= inv_norm;
        }
    }

    for ns_step in 0..steps {
        let (a, b, c) = profile.coeff(ns_step);
        for bi in 0..batch {
            let x_off = bi * rows * cols;
            let a_off = bi * rows * rows;

            // A = X @ X^T  [rows, rows]
            for i in 0..rows {
                for j in 0..rows {
                    let mut sum = 0.0f32;
                    for k in 0..cols {
                        sum += x[x_off + i * cols + k] * x[x_off + j * cols + k];
                    }
                    a_buf[a_off + i * rows + j] = sum;
                }
            }

            // AA = A @ A  [rows, rows]
            for i in 0..rows {
                for j in 0..rows {
                    let mut sum = 0.0f32;
                    for k in 0..rows {
                        sum += a_buf[a_off + i * rows + k] * a_buf[a_off + k * rows + j];
                    }
                    aa[a_off + i * rows + j] = sum;
                }
            }

            // B = b*A + c*AA
            for i in 0..rows * rows {
                b_buf[a_off + i] = b * a_buf[a_off + i] + c * aa[a_off + i];
            }

            // X_new = a*X + B @ X
            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = a * x[x_off + i * cols + j];
                    for k in 0..rows {
                        sum += b_buf[a_off + i * rows + k] * x[x_off + k * cols + j];
                    }
                    new_x[x_off + i * cols + j] = sum;
                }
            }
        }
        x.copy_from_slice(&new_x);
    }

    // Transpose back if needed
    if transposed {
        for bi in 0..batch {
            let x_off = bi * rows * cols;
            let r_off = bi * m * n;
            for r in 0..m {
                for c_idx in 0..n {
                    result[r_off + r * n + c_idx] = x[x_off + c_idx * cols + r];
                }
            }
        }
    } else {
        result[..batch * m * n].copy_from_slice(&x[..batch * m * n]);
    }
}

/// Per-bank state for Muon optimizer.
pub struct MuonBankState {
    pub momentum_buffer: Vec<f32>, // same shape as parameter
    pub scale: f32,                // max(1, M/N)^0.5
    // Pre-allocated scratch for NS5 (avoid per-step allocs)
    pub ns_x: Vec<f32>,      // [B*rows*cols]
    pub ns_a: Vec<f32>,      // [B*rows*rows]
    pub ns_aa: Vec<f32>,     // [B*rows*rows]
    pub ns_b: Vec<f32>,      // [B*rows*rows]
    pub ns_new_x: Vec<f32>,  // [B*rows*cols]
    pub ns_update: Vec<f32>, // [B*M*N] — nesterov update scratch
    pub ns_result: Vec<f32>, // [B*M*N] — result scratch
}

/// Muon optimizer (CPU single-GPU reference).
pub struct Muon {
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    pub weight_decay: f32,
    pub ns_steps: usize,
    pub bank_states: Vec<MuonBankState>,
}

impl Muon {
    pub fn new(
        lr: f32,
        momentum: f32,
        ns_steps: usize,
        nesterov: bool,
        weight_decay: f32,
        bank_shapes: &[[usize; 3]],
    ) -> Self {
        let bank_states = bank_shapes
            .iter()
            .map(|shape| {
                let (batch, m, n) = (shape[0], shape[1], shape[2]);
                let numel = batch * m * n;
                let scale = (m as f32 / n as f32).max(1.0).sqrt();
                let transposed = m > n;
                let (rows, cols) = if transposed { (n, m) } else { (m, n) };
                MuonBankState {
                    momentum_buffer: vec![0.0; numel],
                    scale,
                    ns_x: vec![0.0; batch * rows * cols],
                    ns_a: vec![0.0; batch * rows * rows],
                    ns_aa: vec![0.0; batch * rows * rows],
                    ns_b: vec![0.0; batch * rows * rows],
                    ns_new_x: vec![0.0; batch * rows * cols],
                    ns_update: vec![0.0; numel],
                    ns_result: vec![0.0; numel],
                }
            })
            .collect();

        Self {
            lr,
            momentum,
            nesterov,
            weight_decay,
            ns_steps,
            bank_states,
        }
    }

    /// Step for a single bank parameter.
    /// param: [B, M, N] flat, grad: [B, M, N] flat
    pub fn step_bank(
        &mut self,
        bank_idx: usize,
        param: &mut [f32],
        grad: &[f32],
        shape: &[usize; 3],
    ) {
        let state = &mut self.bank_states[bank_idx];

        // 1. Momentum: buf = momentum * buf + grad
        for i in 0..grad.len() {
            state.momentum_buffer[i] = self.momentum * state.momentum_buffer[i] + grad[i];
        }

        // 2. Nesterov: update = grad + momentum * buf (into pre-allocated scratch)
        if self.nesterov {
            for i in 0..grad.len() {
                state.ns_update[i] = grad[i] + self.momentum * state.momentum_buffer[i];
            }
        } else {
            state.ns_update[..grad.len()].copy_from_slice(&state.momentum_buffer[..grad.len()]);
        }
        // 3. Newton-Schulz orthogonalization
        let param_len = param.len();
        newton_schulz5(
            &state.ns_update[..param_len],
            &[shape[0], shape[1], shape[2]],
            self.ns_steps,
            &mut state.ns_x,
            &mut state.ns_a,
            &mut state.ns_b,
            &mut state.ns_aa,
            &mut state.ns_new_x,
            &mut state.ns_result,
        );

        // 4. Weight decay + update
        let lr_scale = self.lr * state.scale;
        if self.weight_decay > 0.0 {
            let decay = 1.0 - self.lr * self.weight_decay;
            for i in 0..param.len() {
                param[i] = decay * param[i] - lr_scale * state.ns_result[i];
            }
        } else {
            for i in 0..param.len() {
                param[i] -= lr_scale * state.ns_result[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_schulz_produces_orthogonal() {
        // Near-orthogonal 4x4 matrix (typical of normalized gradients)
        // Start from perturbed identity
        let mut g = vec![0.0f32; 16];
        for i in 0..4 {
            g[i * 4 + i] = 1.0;
            // Small perturbation
            g[i * 4 + ((i + 1) % 4)] = 0.1;
        }
        let mut x = vec![0.0; 16];
        let mut a_buf = vec![0.0; 16];
        let mut b_buf = vec![0.0; 16];
        let mut aa = vec![0.0; 16];
        let mut new_x = vec![0.0; 16];
        let mut result = vec![0.0; 16];
        newton_schulz5(
            &g,
            &[4, 4],
            10,
            &mut x,
            &mut a_buf,
            &mut b_buf,
            &mut aa,
            &mut new_x,
            &mut result,
        );

        // Check that X^T X is approximately identity
        // NS5 pushes singular values toward uniformity
        for i in 0..4 {
            for j in 0..4 {
                let mut dot = 0.0f32;
                for k in 0..4 {
                    dot += result[k * 4 + i] * result[k * 4 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 0.5,
                    "XtX[{},{}]={}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }

        // Key property: output should have similar magnitude across columns
        let mut col_norms = vec![0.0f32; 4];
        for j in 0..4 {
            for i in 0..4 {
                col_norms[j] += result[i * 4 + j] * result[i * 4 + j];
            }
            col_norms[j] = col_norms[j].sqrt();
        }
        let max_norm = col_norms.iter().cloned().fold(0.0f32, f32::max);
        let min_norm = col_norms.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            max_norm / min_norm < 2.0,
            "column norms should be similar: {:?}",
            col_norms
        );
    }

    #[test]
    fn test_newton_schulz_batched() {
        // [2, 4, 4] batch of near-identity matrices
        let mut g = vec![0.0f32; 2 * 4 * 4];
        for b in 0..2 {
            for i in 0..4 {
                g[b * 16 + i * 4 + i] = 1.0 + 0.1 * b as f32;
                // Small off-diagonal perturbation
                if i + 1 < 4 {
                    g[b * 16 + i * 4 + i + 1] = 0.05;
                }
            }
        }
        let mut x = vec![0.0; 32];
        let mut a_buf = vec![0.0; 32];
        let mut b_buf = vec![0.0; 32];
        let mut aa = vec![0.0; 32];
        let mut new_x = vec![0.0; 32];
        let mut result = vec![0.0; 32];
        newton_schulz5(
            &g,
            &[2, 4, 4],
            10,
            &mut x,
            &mut a_buf,
            &mut b_buf,
            &mut aa,
            &mut new_x,
            &mut result,
        );
        assert_eq!(result.len(), 32);

        // Diagonal elements should be close to ±1
        for b in 0..2 {
            for i in 0..4 {
                let diag = result[b * 16 + i * 4 + i];
                assert!(
                    diag.abs() > 0.5,
                    "batch {} diag {} = {} (should be near ±1)",
                    b,
                    i,
                    diag
                );
            }
        }
    }

    #[test]
    fn test_muon_step() {
        let shape = [2, 3, 3];
        let mut muon = Muon::new(0.01, 0.9, 5, true, 0.0, &[shape]);

        let mut param = vec![0.1; 18];
        let grad = vec![0.01; 18];

        let param_before: Vec<f32> = param.clone();
        muon.step_bank(0, &mut param, &grad, &shape);

        // Parameters should change
        let changed = param
            .iter()
            .zip(param_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "params should have changed after step");
    }
}
