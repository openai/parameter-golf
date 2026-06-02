"""
config.py — Central place for all tunable variables.

Mirrors the Hyperparameters class in train_gpt.py but with explicit comments
so you can tweak and experiment without hunting through the training code.
Copy these values into env vars when launching:

    SEED=42 NUM_LAYERS=11 MODEL_DIM=512 torchrun --standalone --nproc_per_node=8 train_gpt.py

Or set them here and export in a shell script / RunPod launch template.
"""

# ─────────────────────────────────────────────
# PATHS  (usually left at defaults)
# ─────────────────────────────────────────────
DATA_PATH          = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH     = "./data/tokenizers/fineweb_1024_bpe.model"
RUN_ID             = None          # None → auto-generated UUID

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED               = 1337          # Change for multi-seed statistical tests (e.g. 42, 314, 999)

# ─────────────────────────────────────────────
# TRAINING SCHEDULE
# ─────────────────────────────────────────────
ITERATIONS         = 20_000        # Total training steps (wall-clock cap kicks in first)
WARMUP_STEPS       = 20            # Linear LR warm-up from 0
WARMDOWN_ITERS     = 1200          # Cosine decay at end; top runs use 3500–4000
MAX_WALLCLOCK_SECONDS = 600.0      # Hard stop at 600 s (10 min) for leaderboard compliance

# ─────────────────────────────────────────────
# BATCH / SEQUENCE
# ─────────────────────────────────────────────
TRAIN_BATCH_TOKENS = 524_288       # Global tokens per step (= B × T × world_size)
TRAIN_SEQ_LEN      = 1024          # Context window; top runs use 2048–4096
VAL_BATCH_SIZE     = 524_288       # Tokens evaluated per validation pass
VAL_LOSS_EVERY     = 1000          # Steps between validation; set 0 to skip periodic
TRAIN_LOG_EVERY    = 200           # Steps between train-loss prints

# ─────────────────────────────────────────────
# MODEL SHAPE — most impactful on artifact size
# ─────────────────────────────────────────────
VOCAB_SIZE         = 1024          # Must match the tokenizer; custom tokenizers change this
NUM_LAYERS         = 9             # Baseline 9; top runs use 10–11
MODEL_DIM          = 512           # Hidden width; changing this ripples everywhere
NUM_HEADS          = 8             # Query heads
NUM_KV_HEADS       = 4             # Key/Value heads (GQA ratio = NUM_HEADS / NUM_KV_HEADS)
MLP_MULT           = 2             # MLP hidden = MODEL_DIM × MLP_MULT; top runs use 3 (1536)
TIE_EMBEDDINGS     = True          # Tying input & output embeddings saves parameters
ROPE_BASE          = 10_000.0      # RoPE θ; YaRN / partial-RoPE submissions change this
LOGIT_SOFTCAP      = 30.0          # Tanh soft-cap on logits (Gemma-style)
QK_GAIN_INIT       = 1.5           # Initial QK scale gain

# ─────────────────────────────────────────────
# OPTIMIZER — Muon for matrices, Adam for scalars
# ─────────────────────────────────────────────
MATRIX_LR                  = 0.04   # Muon LR for weight matrices
SCALAR_LR                  = 0.04   # Adam LR for scalar params
EMBED_LR                   = 0.6    # Adam LR for untied embeddings
HEAD_LR                    = 0.008  # Adam LR for the output head (if not tied)
TIED_EMBED_LR              = 0.05   # Adam LR for tied embeddings
TIED_EMBED_INIT_STD        = 0.005  # Std-dev for tied embedding init
MUON_MOMENTUM              = 0.95   # Muon momentum (top runs: 0.99)
MUON_BACKEND_STEPS         = 5      # Newton-Schulz iterations inside Muon
MUON_MOMENTUM_WARMUP_START = 0.85   # Momentum ramp start value
MUON_MOMENTUM_WARMUP_STEPS = 500    # Steps to ramp momentum to MUON_MOMENTUM
BETA1                      = 0.9    # Adam β₁
BETA2                      = 0.95   # Adam β₂
ADAM_EPS                   = 1e-8   # Adam ε
GRAD_CLIP_NORM             = 0.0    # 0 = disabled; some runs use 1.0

# ─────────────────────────────────────────────
# QUANTIZATION  (post-training, artifact size)
# ─────────────────────────────────────────────
# Default scheme: int8 per-row + zlib.  Top runs switch to int6 GPTQ + lzma/zstd.
INT8_CLIP_PERCENTILE            = 99.99984  # How aggressively to clip before quantizing
INT8_KEEP_FLOAT_MAX_NUMEL       = 65_536    # Tensors ≤ this size kept in fp16, not int8
TARGET_MB                       = 16.0      # Artifact budget in MB (decimal, not MiB)

# ─────────────────────────────────────────────
# ADVANCED TECHNIQUES (used in top records)
# ─────────────────────────────────────────────
# Uncomment / set to non-None to activate in your custom train_gpt.py fork:

# --- BigramHash embedding augmentation ---
BIGRAM_VOCAB_SIZE   = None   # e.g. 3072  — number of bigram buckets
BIGRAM_DIM          = None   # e.g. 112   — bigram embedding width

# --- Sliding-window evaluation ---
SLIDING_WINDOW_STRIDE = None  # e.g. 64  — stride for shifted eval passes

# --- Exponential / Stochastic Weight Averaging ---
EMA_DECAY           = None   # e.g. 0.997  — EMA coefficient
SWA_EVERY           = None   # e.g. 50     — SWA snapshot interval

# --- QAT (Quantization-Aware Training) ---
QAT_START_SCALE     = None   # e.g. 0.15  — LR scale threshold to start STE

# --- Partial RoPE ---
ROPE_PARTIAL_DIMS   = None   # e.g. 16    — how many head-dims get RoPE (rest unrotated)

# --- Test-Time Training ---
TTT_ENABLED         = False
TTT_LORA_RANK       = None   # e.g. 4

# --- XSA (Cross-Sequence Attention) ---
XSA_LAYERS          = None   # e.g. "all" or [7, 8, 9, 10]  — layers that use XSA

# --- GPTQ quantization calibration ---
GPTQ_CALIB_SEQS     = None   # e.g. 64   — sequences for Hessian collection
GPTQ_CALIB_TEMP     = None   # e.g. 0.8  — AR generation temperature

# ─────────────────────────────────────────────
# EXPORT / SHELL HELPER
# ─────────────────────────────────────────────
def to_env_string(**overrides) -> str:
    """
    Build an env-var prefix for torchrun from the variables above.
    Usage:
        print(to_env_string(SEED=42, NUM_LAYERS=11, WARMDOWN_ITERS=4000))
    """
    defaults = {
        "SEED": SEED, "NUM_LAYERS": NUM_LAYERS, "MODEL_DIM": MODEL_DIM,
        "NUM_HEADS": NUM_HEADS, "NUM_KV_HEADS": NUM_KV_HEADS,
        "MLP_MULT": MLP_MULT, "TRAIN_SEQ_LEN": TRAIN_SEQ_LEN,
        "WARMDOWN_ITERS": WARMDOWN_ITERS, "MATRIX_LR": MATRIX_LR,
        "MUON_MOMENTUM": MUON_MOMENTUM, "TARGET_MB": TARGET_MB,
    }
    defaults.update(overrides)
    return " ".join(f"{k}={v}" for k, v in defaults.items() if v is not None)


if __name__ == "__main__":
    print(to_env_string())
