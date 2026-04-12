from __future__ import annotations

import os
import uuid


class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    iterations: int = int(os.environ.get("ITERATIONS", 4_000))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 50))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 200))
    val_at_start: bool = bool(int(os.environ.get("VAL_AT_START", "1")))
    val_at_end: bool = bool(int(os.environ.get("VAL_AT_END", "1")))
    sample_every: int = int(os.environ.get("SAMPLE_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 65_536))
    val_batch_tokens: int = int(os.environ.get("VAL_BATCH_TOKENS", 65_536))
    val_max_tokens: int = int(os.environ.get("VAL_MAX_TOKENS", 0))
    val_metric: str = os.environ.get("VAL_METRIC", "both")
    val_timestep_samples: int = int(os.environ.get("VAL_TIMESTEP_SAMPLES", 1))
    val_corruption_samples: int = int(os.environ.get("VAL_CORRUPTION_SAMPLES", 4))
    val_seed: int = int(os.environ.get("VAL_SEED", 12345))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 4))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 256))
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 5))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 0.0))

    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 6))
    model_dim: int = int(os.environ.get("MODEL_DIM", 256))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.02))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    learning_rate: float = float(os.environ.get("LEARNING_RATE", 3e-4))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.0))
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 1.0))

    num_diffusion_steps: int = int(os.environ.get("NUM_DIFFUSION_STEPS", 32))
    mask_schedule: str = os.environ.get("MASK_SCHEDULE", "cosine")
    min_mask_rate: float = float(os.environ.get("MIN_MASK_RATE", 0.0))
    max_mask_rate: float = float(os.environ.get("MAX_MASK_RATE", 1.0))
    train_timestep_sampling: str = os.environ.get("TRAIN_TIMESTEP_SAMPLING", "random")
    loss_reweighting: str = os.environ.get("LOSS_REWEIGHTING", "none")
    loss_reweighting_eps: float = float(os.environ.get("LOSS_REWEIGHTING_EPS", 1e-3))
    parameterization: str = os.environ.get("PARAMETERIZATION", "x0")
    self_conditioning: bool = bool(int(os.environ.get("SELF_CONDITIONING", "0")))
    mask_token_id: int = int(os.environ.get("MASK_TOKEN_ID", -1))
    sample_temperature: float = float(os.environ.get("SAMPLE_TEMPERATURE", 1.0))
    sample_prompt: str = os.environ.get("SAMPLE_PROMPT", "")
    sample_num_steps: int = int(os.environ.get("SAMPLE_NUM_STEPS", 0))
    sample_num_steps_list_raw: str = os.environ.get("SAMPLE_NUM_STEPS_LIST", "")
    save_best_checkpoint: bool = bool(int(os.environ.get("SAVE_BEST_CHECKPOINT", "0")))
    best_checkpoint_metric: str = os.environ.get("BEST_CHECKPOINT_METRIC", "val_bpb")
    init_checkpoint: str = os.environ.get("INIT_CHECKPOINT", "").strip()
    early_stop_patience: int = int(os.environ.get("EARLY_STOP_PATIENCE", "0"))
    early_stop_metric: str = os.environ.get(
        "EARLY_STOP_METRIC",
        os.environ.get("BEST_CHECKPOINT_METRIC", "val_bpb"),
    )
    early_stop_min_delta: float = float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.0"))

    train_shards: int = int(os.environ.get("TRAIN_SHARDS", 0))
    synthetic_data: bool = bool(int(os.environ.get("SYNTHETIC_DATA", "0")))
    synthetic_vocab_size: int = int(os.environ.get("SYNTHETIC_VOCAB_SIZE", 32))
    synthetic_pattern_len: int = int(os.environ.get("SYNTHETIC_PATTERN_LEN", 8))
    synthetic_train_tokens: int = int(os.environ.get("SYNTHETIC_TRAIN_TOKENS", 131_072))
    synthetic_val_tokens: int = int(os.environ.get("SYNTHETIC_VAL_TOKENS", 16_384))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    @property
    def sample_steps(self) -> int:
        return self.sample_num_steps if self.sample_num_steps > 0 else self.num_diffusion_steps

    @property
    def sample_steps_list(self) -> list[int]:
        if self.sample_num_steps_list_raw.strip():
            values = [int(part.strip()) for part in self.sample_num_steps_list_raw.split(",") if part.strip()]
        elif self.sample_num_steps > 0:
            values = [self.sample_num_steps]
        else:
            base = self.num_diffusion_steps
            values = [max(1, base // 4), max(1, base // 2), base]
        unique = sorted({min(max(1, int(v)), self.num_diffusion_steps) for v in values})
        return unique
