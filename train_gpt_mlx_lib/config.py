from __future__ import annotations

import os
import uuid

import mlx.core as mx

# MLX kernels are happiest when activations run in a reduced compute dtype, but
# a number of parameters are deliberately stored elsewhere (often fp32) and cast
# on demand. The rest of the code treats this as the default activation dtype
# rather than as a blanket "every tensor in the program should use bfloat16"
# directive.
COMPUTE_DTYPE = mx.bfloat16


class Hyperparameters:
    # This is intentionally a lightweight environment-backed container rather
    # than a dataclass or argparse tree. The training script is currently used
    # as a hacker-friendly entrypoint where "set env vars and run" is the main
    # interface, and class attributes make those defaults easy to scan in one
    # place. A future cleanup could turn this into an explicitly validated
    # config object, but the current shape keeps import-time overhead and
    # boilerplate low.
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    # `train_batch_tokens` is the effective optimizer batch. We split that batch
    # into `grad_accum_steps` logical microbatches so the optimizer still sees
    # the same total number of tokens while peak memory is kept small enough for
    # Apple unified-memory laptops. This script uses token counts instead of
    # batch-size counts because sequence length is fixed and tokens are the more
    # relevant budget for both memory and throughput.
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    # This is one of the more MLX-specific knobs in the file: the goal is to
    # keep the model's mathematical batch shape stable while slicing execution
    # into units that do not cause large lazy graphs or activation spikes.
    # A future refactor could infer this budget from device memory and model
    # size, but for now it remains an explicit manual override.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    # In other words: we are explicitly paying some synchronization overhead to
    # stop the runtime from hoarding too much deferred work. This is a pragmatic
    # compromise for small-memory Macs, not an ideal steady-state design.
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    # Warmdown is wall-clock-aware because this script is often run against a
    # fixed time budget. That means "last N iterations" is not always the right
    # notion of a decay tail if compile overhead, validation, or system load
    # changes the realized step rate.
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model (defaults match the current baseline setup).
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        # This derived budget is what each accumulation pass is allowed to read
        # before its gradients are averaged into the optimizer step. Keeping it
        # derived rather than separately configurable avoids one more place where
        # the script could become internally inconsistent.
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        # The learning-rate multiplier only implements the warmdown tail; the
        # base learning rates live with the optimizer groups. When a wall-clock
        # budget is active, we estimate how much time the remaining warmdown
        # window would consume at the current average step duration and scale the
        # LR accordingly. This heuristic is intentionally simple: it preserves
        # the "decay near the end of the run" intent without introducing a more
        # stateful scheduler abstraction into a relatively small script.
        #
        # Possible improvement: move this into a dedicated scheduler object with
        # clearer semantics for pure step-based runs, wall-clock-constrained
        # runs, and mixed schedules.
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
# These name patterns are a deliberately blunt instrument for identifying
# learned scalars / control-like tensors that should not be treated like large
# matrix weights by the optimizer or quantizer. String matching is simple and
# works well enough for this compact codebase, but it is also one of the
# clearer places where a richer parameter-tagging system would make the code
# less fragile as the architecture evolves.

INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
# Quantization reuses the same "control tensor" idea because many of these
# small, numerically sensitive tensors are poor candidates for aggressive int8
# compression. The duplication via env vars is intentional for flexibility, but
# an explicit typed config surface would make the relationship between optimizer
# groups and quantization exclusions easier to reason about.

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}
