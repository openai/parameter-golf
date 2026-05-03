/// Complete model and training configuration.
///
/// All values match the SOTA submission (1.1194 BPB):
/// records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/
#[derive(Debug, Clone)]
pub struct ModelConfig {
    // Architecture
    pub vocab_size: usize,
    pub num_layers: usize,
    pub model_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub mlp_mult: f32,
    pub mlp_dim: usize,

    // Positional encoding
    pub rope_base: f32,
    pub rope_dims: usize, // partial RoPE: 16 of 64 dims

    // Attention modifications
    pub xsa_last_n: usize, // XSA on last N layers
    pub logit_softcap: f32,
    pub qk_gain_init: f32,
    pub recurrence_enabled: bool,
    pub recurrence_start_layer: usize,
    pub recurrence_repeat_layers: usize,
    pub parallel_residual: bool,
    pub attn_out_gate_enabled: bool,
    pub attn_out_gate_width: usize,
    pub sparse_attn_gate_enabled: bool,
    pub sparse_attn_gate_width: usize,
    pub sparse_attn_gate_scale: f32,
    pub smear_gate_boundary_token_id: Option<u32>,

    // Value Residual Learning
    pub vrl_enabled: bool,
    pub ve_enabled: bool,
    pub ve_dim: usize,
    pub ve_layers: Vec<usize>,

    // BigramHash
    pub bigram_vocab_size: usize,
    pub bigram_dim: usize,

    // LN scale
    pub ln_scale: bool,

    // Embeddings
    pub tie_embeddings: bool,
    pub tied_embed_init_std: f32,

    // Sequence
    pub train_seq_len: usize,
    pub eval_seq_len: usize,
}

impl ModelConfig {
    /// The exact SOTA competition config (1.1194 BPB record).
    pub fn sota() -> Self {
        let model_dim = 512;
        let num_heads = 8;
        let head_dim = model_dim / num_heads; // 64
        let mlp_mult = 3.0;
        let mlp_dim = (mlp_mult * model_dim as f32) as usize; // 1536
        let num_kv_heads = 4;

        Self {
            vocab_size: 1024,
            num_layers: 11,
            model_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            mlp_mult,
            mlp_dim,

            rope_base: 10000.0,
            rope_dims: 16, // partial RoPE: 16/64

            xsa_last_n: 4,
            logit_softcap: 30.0,
            qk_gain_init: 1.5,
            recurrence_enabled: false,
            recurrence_start_layer: 0,
            recurrence_repeat_layers: 0,
            parallel_residual: false,
            attn_out_gate_enabled: false,
            attn_out_gate_width: 24,
            sparse_attn_gate_enabled: false,
            sparse_attn_gate_width: 12,
            sparse_attn_gate_scale: 1.0,
            smear_gate_boundary_token_id: Some(1),

            vrl_enabled: false, // value_residual flag (separate from VE)
            ve_enabled: true,
            ve_dim: 128,
            ve_layers: vec![9, 10],

            bigram_vocab_size: 1536,
            bigram_dim: 128,

            ln_scale: true,

            tie_embeddings: true,
            tied_embed_init_std: 0.005,

            train_seq_len: 2048,
            eval_seq_len: 2048,
        }
    }

    /// KV dimension (for GQA: num_kv_heads * head_dim).
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn is_recurrent_layer(&self, layer: usize) -> bool {
        self.recurrence_enabled
            && layer >= self.recurrence_start_layer
            && layer < self.recurrence_start_layer + self.recurrence_repeat_layers
    }

    /// Number of encoder layers (first half, for U-Net).
    pub fn num_encoder_layers(&self) -> usize {
        self.num_layers / 2
    }

    /// Number of decoder layers (second half, for U-Net).
    pub fn num_decoder_layers(&self) -> usize {
        self.num_layers - self.num_encoder_layers()
    }

    /// Number of skip connection weights (min of encoder, decoder).
    pub fn num_skip_weights(&self) -> usize {
        self.num_encoder_layers().min(self.num_decoder_layers())
    }

    /// Parameter bank shapes.
    pub fn qo_bank_shape(&self) -> [usize; 3] {
        [2 * self.num_layers, self.model_dim, self.model_dim]
    }

    pub fn kv_bank_shape(&self) -> [usize; 3] {
        [2 * self.num_layers, self.kv_dim(), self.model_dim]
    }

    pub fn mlp_up_bank_shape(&self) -> [usize; 3] {
        [self.num_layers, self.mlp_dim, self.model_dim]
    }

    pub fn mlp_down_bank_shape(&self) -> [usize; 3] {
        [self.num_layers, self.model_dim, self.mlp_dim]
    }

    /// Estimate total parameter count.
    pub fn param_count(&self) -> usize {
        let n = self.num_layers;
        let d = self.model_dim;
        let kv = self.kv_dim();
        let mlp = self.mlp_dim;

        let embeddings = self.vocab_size * d; // tied
        let qo_bank = 2 * n * d * d;
        let kv_bank = 2 * n * kv * d;
        let mlp_up = n * mlp * d;
        let mlp_down = n * d * mlp;
        let bigram = self.bigram_vocab_size * self.bigram_dim + self.bigram_dim * d;
        let ve = if self.ve_enabled {
            self.vocab_size * self.ve_dim + self.ve_dim * kv
        } else {
            0
        };
        // Per-layer scalars: attn_scale, mlp_scale, resid_mix, q_gain, etc.
        let per_layer_scalars = n * (d + d + 2 * d + self.num_heads); // approximate
        let attn_out_gate = if self.attn_out_gate_enabled {
            n * self.num_heads * (self.attn_out_gate_width + 1)
        } else {
            0
        };
        let sparse_attn_gate = if self.sparse_attn_gate_enabled {
            n * self.num_heads * self.sparse_attn_gate_width
        } else {
            0
        };

        embeddings
            + qo_bank
            + kv_bank
            + mlp_up
            + mlp_down
            + bigram
            + ve
            + per_layer_scalars
            + attn_out_gate
            + sparse_attn_gate
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::sota()
    }
}

/// Training hyperparameters matching the SOTA submission.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    // Learning rates
    pub matrix_lr: f32,
    pub scalar_lr: f32,
    pub embed_lr: f32,
    pub tied_embed_lr: f32,
    pub head_lr: f32,

    // Muon optimizer
    pub muon_momentum: f32,
    pub muon_momentum_warmup_start: f32,
    pub muon_momentum_warmup_steps: usize,
    pub muon_wd: f32,
    pub newton_schulz_steps: usize,

    // AdamW (for scalar/embed params)
    pub adam_beta1: f32,
    pub adam_beta2: f32,
    pub adam_eps: f32,
    pub adam_wd: f32,

    // Schedule
    pub warmup_steps: usize,
    pub warmdown_iters: usize,
    pub total_iterations: usize,
    pub min_lr_scale: f32,
    pub max_wallclock_seconds: f32,

    // Batch
    pub train_batch_tokens: usize,
    pub grad_clip_norm: f32,

    // EMA
    pub ema_decay: f32,

    // SWA (only during warmdown)
    pub swa_enabled: bool,
    pub swa_every: usize,

    // Late QAT
    pub late_qat_threshold: f32, // activate when LR scale < this

    // TTT
    pub ttt_enabled: bool,
    pub ttt_lr: f32,
    pub ttt_epochs: usize,
    pub ttt_chunk_tokens: usize,
    pub ttt_freeze_blocks: usize,
    pub ttt_momentum: f32,
    pub ttt_batch_seqs: usize,
    pub ttt_grad_clip: f32,

    // Eval
    pub eval_stride: usize,
}

impl TrainConfig {
    /// The exact SOTA competition config.
    pub fn sota() -> Self {
        Self {
            matrix_lr: 0.025,
            scalar_lr: 0.025,
            embed_lr: 0.6,
            tied_embed_lr: 0.035,
            head_lr: 0.008,

            muon_momentum: 0.99,
            muon_momentum_warmup_start: 0.92,
            muon_momentum_warmup_steps: 1500,
            muon_wd: 0.04,
            newton_schulz_steps: 5,

            adam_beta1: 0.9,
            adam_beta2: 0.95,
            adam_eps: 1e-8,
            adam_wd: 0.04,

            warmup_steps: 20,
            warmdown_iters: 3500,
            total_iterations: 9000,
            min_lr_scale: 0.0,
            max_wallclock_seconds: 600.0,

            train_batch_tokens: 786_432,
            grad_clip_norm: 0.3,

            ema_decay: 0.997,

            swa_enabled: true,
            swa_every: 50,

            late_qat_threshold: 0.15,

            ttt_enabled: true,
            ttt_lr: 0.002,
            ttt_epochs: 3,
            ttt_chunk_tokens: 32768,
            ttt_freeze_blocks: 0,
            ttt_momentum: 0.9,
            ttt_batch_seqs: 32,
            ttt_grad_clip: 1.0,

            eval_stride: 64,
        }
    }

    /// Compute WSD learning rate at a given step.
    pub fn wsd_lr(&self, step: usize) -> f32 {
        let warmdown_start = self.total_iterations.saturating_sub(self.warmdown_iters);
        let scale = if step < self.warmup_steps {
            // Linear warmup
            return self.matrix_lr * (step as f32 / self.warmup_steps as f32);
        } else if step < warmdown_start {
            // Stable phase
            1.0
        } else {
            // Linear warmdown to the configured floor.
            let remaining = self.total_iterations - step;
            (remaining as f32 / self.warmdown_iters as f32).max(0.0)
        };
        self.matrix_lr * scale.max(self.min_lr_scale)
    }

    /// LR scale factor at a given step (0.0 to 1.0).
    pub fn lr_scale(&self, step: usize) -> f32 {
        self.wsd_lr(step) / self.matrix_lr
    }

    /// Current Muon momentum with warmup.
    pub fn muon_momentum_at(&self, step: usize) -> f32 {
        if step >= self.muon_momentum_warmup_steps {
            self.muon_momentum
        } else {
            let t = step as f32 / self.muon_momentum_warmup_steps as f32;
            self.muon_momentum_warmup_start
                + t * (self.muon_momentum - self.muon_momentum_warmup_start)
        }
    }

    /// Whether SWA should collect at this step (only during warmdown).
    pub fn should_swa(&self, step: usize) -> bool {
        if !self.swa_enabled {
            return false;
        }
        let warmdown_start = self.total_iterations.saturating_sub(self.warmdown_iters);
        step >= warmdown_start && step % self.swa_every == 0
    }

    /// Whether late QAT should be active at this step.
    /// Only activates during warmdown (not during warmup when scale is also low).
    pub fn qat_active(&self, step: usize) -> bool {
        let warmdown_start = self.total_iterations.saturating_sub(self.warmdown_iters);
        step >= warmdown_start && self.lr_scale(step) < self.late_qat_threshold
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self::sota()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_count() {
        let config = ModelConfig::sota();
        let count = config.param_count();
        // The SOTA model has ~27-35M params depending on exact counting
        // Key components: qo_bank(11.5M) + kv_bank(5.7M) + mlp_up(8.6M) + mlp_down(8.6M)
        // + embeddings(0.5M) + bigram(0.3M) + VE(0.2M) + scalars
        assert!(count > 20_000_000, "too few params: {}", count);
        assert!(count < 40_000_000, "too many params: {}", count);
        eprintln!("SOTA param count: {}", count);
    }

    #[test]
    fn test_wsd_schedule() {
        let config = TrainConfig::sota();
        // Warmup
        assert_eq!(config.wsd_lr(0), 0.0);
        assert!((config.wsd_lr(10) - 0.0125).abs() < 1e-6);
        // Stable
        assert_eq!(config.wsd_lr(100), 0.025);
        assert_eq!(config.wsd_lr(5000), 0.025);
        // Warmdown start
        let wd_start = 9000 - 3500; // 5500
        assert_eq!(config.wsd_lr(wd_start), 0.025);
        // Warmdown end
        assert_eq!(config.wsd_lr(9000), 0.0);
        // Mid warmdown
        let mid = wd_start + 1750;
        assert!((config.wsd_lr(mid) - 0.0125).abs() < 0.001);

        let mut floored = config.clone();
        floored.min_lr_scale = 0.10;
        assert_eq!(floored.wsd_lr(0), 0.0);
        assert!((floored.wsd_lr(9000) - floored.matrix_lr * 0.10).abs() < 1e-6);
    }

    #[test]
    fn test_muon_momentum_warmup() {
        let config = TrainConfig::sota();
        assert_eq!(config.muon_momentum_at(0), 0.92);
        assert_eq!(config.muon_momentum_at(1500), 0.99);
        assert_eq!(config.muon_momentum_at(3000), 0.99);
        // Midpoint
        let mid = config.muon_momentum_at(750);
        assert!((mid - 0.955).abs() < 0.001);
    }

    #[test]
    fn test_qat_activation() {
        let config = TrainConfig::sota();
        assert!(!config.qat_active(0));
        assert!(!config.qat_active(5000));
        // QAT activates when lr_scale < 0.15
        // warmdown: 5500 to 9000, scale goes 1.0 → 0.0
        // scale = 0.15 when step = 9000 - 0.15 * 3500 = 9000 - 525 = 8475
        assert!(!config.qat_active(8400));
        assert!(config.qat_active(8500));
        assert!(config.qat_active(8999));
    }

    #[test]
    fn test_swa_warmdown_only() {
        let config = TrainConfig::sota();
        // SWA should NOT activate during stable phase
        assert!(!config.should_swa(50));
        assert!(!config.should_swa(5000));
        // SWA SHOULD activate during warmdown (step >= 5500, every 50 steps)
        assert!(config.should_swa(5500));
        assert!(!config.should_swa(5501));
        assert!(config.should_swa(5550));
    }

    #[test]
    fn test_bank_shapes() {
        let config = ModelConfig::sota();
        assert_eq!(config.qo_bank_shape(), [22, 512, 512]);
        assert_eq!(config.kv_bank_shape(), [22, 256, 512]);
        assert_eq!(config.mlp_up_bank_shape(), [11, 1536, 512]);
        assert_eq!(config.mlp_down_bank_shape(), [11, 512, 1536]);
    }
}
