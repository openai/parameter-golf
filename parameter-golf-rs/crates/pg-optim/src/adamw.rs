/// AdamW optimizer for scalar/embedding parameters.
///
/// Used for: token embeddings, bigram params, skip weights, per-layer scalars,
/// VE params, q_gain, smear_gate, attn_scale, mlp_scale, resid_mix.
///
/// The Muon optimizer handles the 4 large parameter banks.

/// Per-parameter AdamW state.
pub struct AdamWState {
    pub m: Vec<f32>, // first moment
    pub v: Vec<f32>, // second moment
    pub step: usize,
}

impl AdamWState {
    pub fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
            step: 0,
        }
    }
}

/// AdamW optimizer.
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
        }
    }

    /// Step for a single parameter group.
    pub fn step(&self, param: &mut [f32], grad: &[f32], state: &mut AdamWState) {
        state.step += 1;

        // Bias correction
        let bc1 = 1.0 - self.beta1.powi(state.step as i32);
        let bc2 = 1.0 - self.beta2.powi(state.step as i32);

        for i in 0..param.len() {
            // Update moments
            state.m[i] = self.beta1 * state.m[i] + (1.0 - self.beta1) * grad[i];
            state.v[i] = self.beta2 * state.v[i] + (1.0 - self.beta2) * grad[i] * grad[i];

            // Bias-corrected moments
            let m_hat = state.m[i] / bc1;
            let v_hat = state.v[i] / bc2;

            // Weight decay (decoupled)
            if self.weight_decay > 0.0 {
                param[i] *= 1.0 - self.lr * self.weight_decay;
            }

            // Update
            param[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adamw_converges() {
        // Minimize f(x) = x^2 → grad = 2x
        let mut param = vec![5.0f32];
        let mut state = AdamWState::new(1);
        let adam = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.0);

        for _ in 0..200 {
            let grad = vec![2.0 * param[0]];
            adam.step(&mut param, &grad, &mut state);
        }

        assert!(
            param[0].abs() < 0.01,
            "should converge to 0, got {}",
            param[0]
        );
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut param = vec![1.0f32];
        let mut state = AdamWState::new(1);
        let adam = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);

        // Zero gradient — only weight decay should shrink param
        let grad = vec![0.0f32];
        adam.step(&mut param, &grad, &mut state);

        assert!(param[0] < 1.0, "weight decay should shrink param");
        assert!(param[0] > 0.9, "shouldn't shrink too much in one step");
    }
}
