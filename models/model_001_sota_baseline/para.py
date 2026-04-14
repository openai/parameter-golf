#!/usr/bin/env python3
"""
model_001_sota_baseline/para.py

Central parameter file for Model #001: SOTA Baseline replica.
Start here to understand what changed vs baseline. Edit values to experiment.

Usage:
    python para.py                    # Print the torchrun command
    python para.py --seed 314         # Print single-seed command
    python para.py --launch           # Actually run training (all 3 seeds)
    python para.py --launch --seed 314 --mode cuda  # Run single seed on CUDA

The script auto-logs all runs to docs/training-log.md with the parameters used.
"""

import argparse
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# MODEL METADATA
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ID = "model_001"
MODEL_NAME = "SOTA Baseline (AR Self-Gen GPTQ + XSA + BigramHash)"
DESCRIPTION = "Replica of 2026-03-25 SOTA submission (1.1147 BPB mean). 11L, GPTQ, XSA-all, BigramHash 3072x112, Muon, EMA, SWA, sliding window eval."
BASE_SUBMISSION = "openai/parameter-golf#1019 (2026-03-25)"

# ─────────────────────────────────────────────────────────────────────────────
# PATHS (usually left at defaults)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "./data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"

# ─────────────────────────────────────────────────────────────────────────────
# CORE MODEL SHAPE
# ─────────────────────────────────────────────────────────────────────────────
NUM_LAYERS = 11              # Depth of transformer (SOTA: 11)
MODEL_DIM = 512              # Hidden dimension
NUM_HEADS = 8                # Query heads (vocab-independent)
NUM_KV_HEADS = 4             # Key/Value heads (GQA: NUM_HEADS / NUM_KV_HEADS = 2:1)
MLP_MULT = 3.0               # MLP width multiplier (MLP hidden = MODEL_DIM * MLP_MULT)
VOCAB_SIZE = 1024            # Tokenizer vocab (must match data)
TIE_EMBEDDINGS = True        # Tie input/output embeddings (saves params)
TRAIN_SEQ_LEN = 2048         # Context length during training
EVAL_SEQ_LEN = 2048          # Context length during eval
ROPE_BASE = 10000.0          # RoPE θ base
ROPE_DIMS = 16               # Number of dimensions that get RoPE (partial, not full)
LOGIT_SOFTCAP = 30.0         # Tanh softcap on logits (Gemma-style)
QK_GAIN_INIT = 1.5           # Initial QK scale gain

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
ITERATIONS = 20_000          # Total training steps
WARMUP_STEPS = 20            # Linear warmup from 0 → LR
WARMDOWN_ITERS = 4000        # Cosine decay at end (SOTA: 4000; baseline: 1200)
MAX_WALLCLOCK_SECONDS = 600.0  # Hard stop at 10 minutes for leaderboard
TRAIN_BATCH_TOKENS = 786_432  # Global tokens/step (B × T × world_size)
VAL_BATCH_SIZE = 524_288     # Tokens evaluated per validation pass
VAL_LOSS_EVERY = 4000        # Steps between validation checks
TRAIN_LOG_EVERY = 500        # Steps between train loss prints

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER — Muon for matrices, Adam for scalars
# ─────────────────────────────────────────────────────────────────────────────
MATRIX_LR = 0.025            # Muon learning rate for weight matrices
SCALAR_LR = 0.025            # Adam learning rate for scalar params
EMBED_LR = 0.6               # Adam LR for untied embeddings
HEAD_LR = 0.008              # Adam LR for output head
TIED_EMBED_LR = 0.035        # Adam LR for tied embeddings
TIED_EMBED_INIT_STD = 0.005  # Std-dev for tied embedding init
MUON_MOMENTUM = 0.99         # Muon momentum (SOTA: 0.99; baseline: 0.95)
MUON_BACKEND_STEPS = 5       # Newton-Schulz iterations inside Muon
MUON_MOMENTUM_WARMUP_START = 0.92  # Momentum ramp start
MUON_MOMENTUM_WARMUP_STEPS = 1500  # Steps to reach target momentum
BETA1 = 0.9                  # Adam β₁
BETA2 = 0.95                 # Adam β₂
MUON_BETA2 = 0.95            # Muon β₂
ADAM_EPS = 1e-8              # Adam ε
GRAD_CLIP_NORM = 0.3         # Gradient clipping norm (0 = disabled)
MUON_WD = 0.04               # Muon weight decay
ADAM_WD = 0.04               # Adam weight decay

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED TECHNIQUES (all in SOTA)
# ─────────────────────────────────────────────────────────────────────────────

# BigramHash — hash consecutive token pairs into embedding table
BIGRAM_VOCAB_SIZE = 3072     # Number of bigram buckets (SOTA: 3072)
BIGRAM_DIM = 112             # Bigram embedding dimension (SOTA: 112)

# Sliding-window evaluation — shift eval window at stride
EVAL_STRIDE = 64             # Stride for eval window shifts (+0.019 BPB)

# Exponential Moving Average — shadow weights updated via exponential smoothing
# (Not directly a parameter in train_gpt.py, but mentioned in submission; implicit in SOTA)

# Stochastic Weight Averaging — periodic checkpoint averaging
SWA_ENABLED = True           # Enable SWA
SWA_EVERY = 50               # Take SWA snapshot every N steps

# XSA (Cross-Sequence Attention) — cross-position information mixing
XSA_LAST_N = 11              # Apply XSA to last N layers (SOTA: 11 = all; PR #549: 4)

# Late QAT — Quantization-Aware Training in late training phase
QAT_ENABLED = False          # Don't enable for this SOTA config
LATE_QAT_THRESHOLD = 0.15    # LR scale below which STE activates

# Other flags
TRIGRAM = False              # TrigramHash (risky, not in SOTA)
LN_SCALE = True              # Scale layernorm by 1/√(layer+1)
VE_ENABLED = True            # Value Embedding (VE128)
VE_DIM = 128                 # VE dimension
VE_LAYERS = "9,10"           # VE applied to layers 9–10
GATED_ATTENTION = False      # Gated attention (not in SOTA)
VALUE_RESIDUAL = False       # Value residual with sigmoid gates (risky)
DTG_ENABLED = False          # Directional Token Gradients
MTP_NUM_HEADS = 0            # Multi-token prediction heads (0 = disabled)
MTP_LOSS_WEIGHT = 0.2        # MTP loss weight if enabled
LAWA_ENABLED = False         # Layer-Adaptive Weight Averaging
LAWA_K = 10                  # LAWA lookback
LAWA_FREQ = 100              # LAWA frequency

# ─────────────────────────────────────────────────────────────────────────────
# QUANTIZATION (post-training)
# ─────────────────────────────────────────────────────────────────────────────
GPTQ_CALIB_BATCHES = 256     # Number of batches for Hessian collection
GPTQ_BLOCK_SIZE = 128        # Block size for GPTQ
TARGET_MB = 15.9             # Artifact size target in MB (decimal, not MiB)
INT8_CLIP_PERCENTILE = 99.99984  # Clipping percentile for int8 quant
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536  # Tensors ≤ this size stay float

# ─────────────────────────────────────────────────────────────────────────────
# MULTI-SEED VALIDATION (Welch t-test requires ≥3 independent runs)
# ─────────────────────────────────────────────────────────────────────────────
SEEDS = [314, 42, 999]       # Three seeds for statistical significance

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLE EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def to_env_vars(seed=None) -> dict:
    """
    Convert all parameters to a dict suitable for os.environ.
    If seed is provided, override SEED. Otherwise, SEED is not set.
    """
    env = {
        # Paths
        "DATA_PATH": str(DATA_PATH),
        "TOKENIZER_PATH": str(TOKENIZER_PATH),

        # Reproducibility
        "SEED": str(seed) if seed is not None else str(SEEDS[0]),

        # Training schedule
        "ITERATIONS": str(ITERATIONS),
        "WARMUP_STEPS": str(WARMUP_STEPS),
        "WARMDOWN_ITERS": str(WARMDOWN_ITERS),
        "MAX_WALLCLOCK_SECONDS": str(MAX_WALLCLOCK_SECONDS),
        "TRAIN_BATCH_TOKENS": str(TRAIN_BATCH_TOKENS),
        "VAL_BATCH_SIZE": str(VAL_BATCH_SIZE),
        "VAL_LOSS_EVERY": str(VAL_LOSS_EVERY),
        "TRAIN_LOG_EVERY": str(TRAIN_LOG_EVERY),

        # Model shape
        "NUM_LAYERS": str(NUM_LAYERS),
        "MODEL_DIM": str(MODEL_DIM),
        "NUM_HEADS": str(NUM_HEADS),
        "NUM_KV_HEADS": str(NUM_KV_HEADS),
        "MLP_MULT": str(MLP_MULT),
        "VOCAB_SIZE": str(VOCAB_SIZE),
        "TIE_EMBEDDINGS": str(int(TIE_EMBEDDINGS)),
        "TRAIN_SEQ_LEN": str(TRAIN_SEQ_LEN),
        "EVAL_SEQ_LEN": str(EVAL_SEQ_LEN),
        "ROPE_BASE": str(ROPE_BASE),
        "ROPE_DIMS": str(ROPE_DIMS),
        "LOGIT_SOFTCAP": str(LOGIT_SOFTCAP),
        "QK_GAIN_INIT": str(QK_GAIN_INIT),

        # Optimizer
        "MATRIX_LR": str(MATRIX_LR),
        "SCALAR_LR": str(SCALAR_LR),
        "EMBED_LR": str(EMBED_LR),
        "HEAD_LR": str(HEAD_LR),
        "TIED_EMBED_LR": str(TIED_EMBED_LR),
        "TIED_EMBED_INIT_STD": str(TIED_EMBED_INIT_STD),
        "MUON_MOMENTUM": str(MUON_MOMENTUM),
        "MUON_BACKEND_STEPS": str(MUON_BACKEND_STEPS),
        "MUON_MOMENTUM_WARMUP_START": str(MUON_MOMENTUM_WARMUP_START),
        "MUON_MOMENTUM_WARMUP_STEPS": str(MUON_MOMENTUM_WARMUP_STEPS),
        "BETA1": str(BETA1),
        "BETA2": str(BETA2),
        "MUON_BETA2": str(MUON_BETA2),
        "ADAM_EPS": str(ADAM_EPS),
        "GRAD_CLIP_NORM": str(GRAD_CLIP_NORM),
        "MUON_WD": str(MUON_WD),
        "ADAM_WD": str(ADAM_WD),

        # Advanced techniques
        "BIGRAM_VOCAB_SIZE": str(BIGRAM_VOCAB_SIZE),
        "BIGRAM_DIM": str(BIGRAM_DIM),
        "EVAL_STRIDE": str(EVAL_STRIDE),
        "SWA_ENABLED": str(int(SWA_ENABLED)),
        "SWA_EVERY": str(SWA_EVERY),
        "XSA_LAST_N": str(XSA_LAST_N),
        "QAT_ENABLED": str(int(QAT_ENABLED)),
        "LATE_QAT_THRESHOLD": str(LATE_QAT_THRESHOLD),
        "TRIGRAM": str(int(TRIGRAM)),
        "LN_SCALE": str(int(LN_SCALE)),
        "VE_ENABLED": str(int(VE_ENABLED)),
        "VE_DIM": str(VE_DIM),
        "VE_LAYERS": str(VE_LAYERS),
        "GATED_ATTENTION": str(int(GATED_ATTENTION)),
        "VALUE_RESIDUAL": str(int(VALUE_RESIDUAL)),
        "DTG_ENABLED": str(int(DTG_ENABLED)),
        "MTP_NUM_HEADS": str(MTP_NUM_HEADS),
        "MTP_LOSS_WEIGHT": str(MTP_LOSS_WEIGHT),
        "LAWA_ENABLED": str(int(LAWA_ENABLED)),
        "LAWA_K": str(LAWA_K),
        "LAWA_FREQ": str(LAWA_FREQ),

        # Quantization
        "GPTQ_CALIB_BATCHES": str(GPTQ_CALIB_BATCHES),
        "GPTQ_BLOCK_SIZE": str(GPTQ_BLOCK_SIZE),
        "TARGET_MB": str(TARGET_MB),
        "INT8_CLIP_PERCENTILE": str(INT8_CLIP_PERCENTILE),
        "INT8_KEEP_FLOAT_MAX_NUMEL": str(INT8_KEEP_FLOAT_MAX_NUMEL),
    }
    return env

def build_launch_cmd(seed, _mode="cuda") -> str:
    """
    Build the torchrun command with environment variables.
    """
    env_vars = to_env_vars(seed)
    env_prefix = " ".join(f"{k}={v}" for k, v in env_vars.items())

    # Currently only CUDA/torchrun is supported for submission
    cmd = f"{env_prefix} torchrun --standalone --nproc_per_node=8 train_gpt.py"
    return cmd

def get_params_diff_vs_sota() -> dict:
    """
    Compare current para.py values against SOTA defaults.
    Returns only the parameters that differ (for logging).
    """
    sota_defaults = {
        "NUM_LAYERS": 11,
        "MODEL_DIM": 512,
        "NUM_HEADS": 8,
        "NUM_KV_HEADS": 4,
        "MLP_MULT": 3.0,
        "WARMDOWN_ITERS": 4000,
        "BIGRAM_VOCAB_SIZE": 3072,
        "BIGRAM_DIM": 112,
        "XSA_LAST_N": 11,
        "SWA_ENABLED": True,
        "SWA_EVERY": 50,
        "MUON_MOMENTUM": 0.99,
    }

    current = {
        "NUM_LAYERS": NUM_LAYERS,
        "MODEL_DIM": MODEL_DIM,
        "NUM_HEADS": NUM_HEADS,
        "NUM_KV_HEADS": NUM_KV_HEADS,
        "MLP_MULT": MLP_MULT,
        "WARMDOWN_ITERS": WARMDOWN_ITERS,
        "BIGRAM_VOCAB_SIZE": BIGRAM_VOCAB_SIZE,
        "BIGRAM_DIM": BIGRAM_DIM,
        "XSA_LAST_N": XSA_LAST_N,
        "SWA_ENABLED": SWA_ENABLED,
        "SWA_EVERY": SWA_EVERY,
        "MUON_MOMENTUM": MUON_MOMENTUM,
    }

    diff = {k: v for k, v in current.items() if sota_defaults.get(k) != v}
    return diff if diff else {"_note": "SOTA baseline (no diff)"}

def log_run_start(run_id, seed, mode="cuda") -> None:
    """
    Append a START entry to docs/training-log.md.
    Called before training begins.
    """
    log_file = Path(__file__).parent.parent.parent / "docs" / "training-log.md"

    diff = get_params_diff_vs_sota()
    params_str = ", ".join(f"{k}={v}" for k, v in diff.items())

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"| {run_id} | {MODEL_ID} | {seed} | {timestamp} | (running...) | — | — | — | {params_str} |\n"

    with open(log_file, "a") as f:
        f.write(entry)

def log_run_end(run_id, _seed, val_bpb, artifact_bytes=None, steps=None, _notes="") -> None:
    """
    Helper for recording run completion (simplified version for now).
    Full implementation would parse train_gpt.py output automatically.
    """
    # For now, the user manually fills in BPB after running
    print(f"\n✅ Run {run_id} complete.")
    print(f"   Add results to docs/training-log.md (BPB: {val_bpb}, Artifact: {artifact_bytes}, Steps: {steps})")

def print_launch_cmd(seed=None, mode="cuda") -> None:
    """
    Print the launch command(s) to stdout.
    """
    seeds = [seed] if seed is not None else SEEDS

    print(f"\n{'='*80}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  ID:    {MODEL_ID}")
    print(f"{'='*80}\n")

    for s in seeds:
        cmd = build_launch_cmd(s, mode)
        print(f"Seed {s}:")
        print(f"  {cmd}\n")

    print(f"{'='*80}")
    print(f"Copy-paste one of the commands above, then:")
    print(f"  1. Edit docs/training-log.md manually when run completes")
    print(f"  2. Record: val_bpb, artifact_bytes, steps, date")
    print(f"{'='*80}\n")

def launch(seed=None, _mode="cuda", dry_run=False) -> None:
    """
    Helper for launching training (prints commands for manual execution).
    For actual runs on 8×H100, the user copies and runs the command manually.
    """
    if dry_run:
        print_launch_cmd(seed, _mode)
        return

    # Print the command(s) for user to copy
    print_launch_cmd(seed, _mode)
    print("\n⚠️  This script prints the launch command. Run it manually on your compute cluster.")
    print("   When training completes, manually update docs/training-log.md with results.\n")

# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Golf Model #001 launcher")
    parser.add_argument("--launch", action="store_true", help="Launch training (prints command)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run (default: all 3)")
    parser.add_argument("--mode", choices=["cuda", "mlx"], default="cuda", help="Hardware mode")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")

    args = parser.parse_args()

    # Default: just print the launch command(s)
    if not args.launch and not args.dry_run:
        print_launch_cmd(args.seed, args.mode)
    elif args.dry_run:
        print_launch_cmd(args.seed, args.mode)
    else:
        launch(args.seed, args.mode, dry_run=False)
