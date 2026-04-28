pub struct EmaTracker {
    pub alpha: f64,
    pub value: Option<f64>,
}

impl EmaTracker {
    pub fn new(alpha: f64) -> Self {
        Self { alpha, value: None }
    }

    pub fn update(&mut self, x: f64) -> f64 {
        self.value = Some(match self.value {
            Some(v) => v * (1.0 - self.alpha) + x * self.alpha,
            None => x,
        });
        self.value.unwrap()
    }
}
