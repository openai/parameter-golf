/// Exponential Moving Average of model parameters.
///
/// EMA(0.997) updated every step during training.
/// Final inference uses EMA weights, not the last training weights.
///
/// update: ema_param = decay * ema_param + (1 - decay) * param

/// EMA state: shadow copy of all parameters.
pub struct Ema {
    pub decay: f32,
    pub shadow: Vec<f32>,
    pub initialized: bool,
}

impl Ema {
    pub fn new(decay: f32, num_params: usize) -> Self {
        Self {
            decay,
            shadow: vec![0.0; num_params],
            initialized: false,
        }
    }

    /// Update EMA shadow with current parameters.
    /// On first call, copies params directly (no decay).
    pub fn update(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.shadow.len());
        if !self.initialized {
            self.shadow.copy_from_slice(params);
            self.initialized = true;
            return;
        }
        let one_minus_decay = 1.0 - self.decay;
        for i in 0..self.shadow.len() {
            self.shadow[i] = self.decay * self.shadow[i] + one_minus_decay * params[i];
        }
    }

    /// Copy EMA shadow into destination buffer.
    pub fn copy_to(&self, dest: &mut [f32]) {
        dest.copy_from_slice(&self.shadow);
    }
}

/// Stochastic Weight Averaging — averages checkpoints during warmdown.
pub struct Swa {
    pub sum: Vec<f32>,
    pub count: usize,
}

impl Swa {
    pub fn new(num_params: usize) -> Self {
        Self {
            sum: vec![0.0; num_params],
            count: 0,
        }
    }

    /// Accumulate a checkpoint into the running average.
    pub fn accumulate(&mut self, params: &[f32]) {
        assert_eq!(params.len(), self.sum.len());
        for i in 0..self.sum.len() {
            self.sum[i] += params[i];
        }
        self.count += 1;
    }

    /// Compute the averaged weights.
    pub fn average(&self) -> Vec<f32> {
        if self.count == 0 {
            return self.sum.clone();
        }
        let inv = 1.0 / self.count as f32;
        self.sum.iter().map(|&v| v * inv).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ema_initialization() {
        let mut ema = Ema::new(0.997, 4);
        let params = vec![1.0, 2.0, 3.0, 4.0];
        ema.update(&params);
        assert_eq!(ema.shadow, params);
    }

    #[test]
    fn test_ema_decay() {
        let mut ema = Ema::new(0.9, 2);
        ema.update(&[10.0, 20.0]); // init
        ema.update(&[0.0, 0.0]); // second update

        // shadow = 0.9 * [10, 20] + 0.1 * [0, 0] = [9, 18]
        assert!((ema.shadow[0] - 9.0).abs() < 1e-5);
        assert!((ema.shadow[1] - 18.0).abs() < 1e-5);
    }

    #[test]
    fn test_swa() {
        let mut swa = Swa::new(3);
        swa.accumulate(&[1.0, 2.0, 3.0]);
        swa.accumulate(&[3.0, 4.0, 5.0]);

        let avg = swa.average();
        assert!((avg[0] - 2.0).abs() < 1e-6);
        assert!((avg[1] - 3.0).abs() < 1e-6);
        assert!((avg[2] - 4.0).abs() < 1e-6);
    }
}
