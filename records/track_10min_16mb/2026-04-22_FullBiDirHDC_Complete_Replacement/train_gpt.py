"""FullBiDirHDC Parameter Golf Submission — Main Entry Point.

Complete replacement of the hash-addressed NMF + DirectionalSemanticVec pipeline
with the FullBiDirHDC joint manifold engine from the ARC-AGI-3 submission.

Each independent leaderboard run uses a SINGLE seed (for statistical variance
across the 3 required independent runs). The seed is set via the SEED env var.

Usage (8×H100 SXM, leaderboard — standard contest invocation):
    # Run 1 (seed 42):
    RUN_ID=bidi_hdc N_WORDS=512 SEED=42 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    # Run 2 (seed 7):
    RUN_ID=bidi_hdc N_WORDS=512 SEED=7 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    # Run 3 (seed 1337):
    RUN_ID=bidi_hdc N_WORDS=512 SEED=1337 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

Convenience: run all 3 seeds sequentially (leaderboard verification):
    for seed in 42 7 1337; do
      RUN_ID=bidi_hdc N_WORDS=512 SEED=$seed \\
      DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
      TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
      VOCAB_SIZE=1024 \\
      torchrun --standalone --nproc_per_node=8 train_gpt.py
    done

Usage (single GPU, smoke test):
    N_WORDS=16 SEED=42 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    python train_gpt.py

Environment variables:
    N_WORDS              : HV width in uint64 words (default 512 → 32,768 bits)
    SEED                 : Single random seed for this run (default 42)
    MAX_WALLCLOCK_SECONDS: Training time cap in seconds (default 600)
    DATA_PATH            : Path to fineweb10B_sp1024/ directory
    TOKENIZER_PATH       : Path to fineweb_1024_bpe.model
    VOCAB_SIZE           : Vocabulary size (default 1024)
    RUN_ID               : Run identifier for artifact naming (default "bidi_hdc")

Each run auto-generates:
    bidi_hdc_seed{SEED}_{TIMESTAMP}.bdhgz  — compressed model artifact
    train_{TIMESTAMP}.log                   — training log (via shell redirect)
    submission.json                         — updated with actual val_bpb, val_loss, etc.

BPB formula (identical to reference train_gpt.py):
    BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
        = bits_per_token × tokens_per_byte
"""

import argparse
import datetime
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Add the record folder to sys.path so local imports work
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Distributed init (same pattern as existing submissions)
# ─────────────────────────────────────────────────────────────────────────────

def _init_distributed():
    """Initialise torch.distributed if LOCAL_RANK is set (torchrun)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return 0, 1  # single-process mode

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def _dist_rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def _dist_is_main() -> bool:
    return _dist_rank() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Token loading (same as existing submissions)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokens(data_path: str, split: str, max_shards: int = 80) -> np.ndarray:
    """Load tokenised FineWeb shards from data_path.

    Args:
        data_path  : Path to fineweb10B_sp1024/ directory
        split      : "train" or "val"
        max_shards : Maximum number of training shards to load

    Returns:
        (N,) uint16 array of token IDs
    """
    pattern = os.path.join(data_path, f"fineweb_{split}_*.bin")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No {split} shards found at {pattern}")

    if split == "train":
        shard_files = shard_files[:max_shards]

    # Each shard has a 1024-byte header: 256 × int32
    # (magic=0x134D888, version, n_tokens, …).  Skip it before reading tokens.
    _HEADER_BYTES = 256 * 4   # 1024 bytes = 512 uint16 words

    all_tokens = []
    for shard_path in shard_files:
        tokens = np.fromfile(shard_path, dtype=np.uint16)[_HEADER_BYTES // 2:]
        all_tokens.append(tokens)
        if _dist_is_main():
            total = sum(len(t) for t in all_tokens)
            print(f"[TokenLoad] Loaded {total:,} tokens from {os.path.basename(shard_path)}")

    return np.concatenate(all_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_bidi_hdc(args):
    """Main FullBiDirHDC training and evaluation pipeline.

    Each independent run uses a SINGLE seed for statistical variance.
    The 3 required leaderboard runs use seeds 42, 7, and 1337 respectively.
    """
    from _bidi_train import (
        train_bidi_model,
        bidi_bpb,
        save_bidi_artifact,
        check_artifact_size,
        ARTIFACT_LIMIT,
    )

    t_global_start = time.time()
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # ── Configuration ────────────────────────────────────────────────────────
    n_words    = int(os.environ.get("N_WORDS",    args.n_words))
    # Single seed per run (for statistical variance across 3 independent runs)
    seed       = int(os.environ.get("SEED", args.seed))
    max_secs   = int(os.environ.get("MAX_WALLCLOCK_SECONDS", args.max_wallclock_seconds))
    vocab_size = int(os.environ.get("VOCAB_SIZE", args.vocab_size))
    run_id     = os.environ.get("RUN_ID", args.run_id)
    data_path  = os.environ.get("DATA_PATH", args.data_path)
    tok_path   = os.environ.get("TOKENIZER_PATH", args.tokenizer_path)

    if _dist_is_main():
        # ── GPU status report ─────────────────────────────────────────────
        try:
            from _gpu import gpu_available, _get_device
            _gpu_on = gpu_available()
            _dev    = _get_device() if _gpu_on else "cpu"
        except Exception:
            _gpu_on = False
            _dev    = "cpu"
        print(f"\n{'='*60}")
        print(f"[BiDirHDC] FullBiDirHDC Parameter Golf Submission")
        print(f"[BiDirHDC] GPU acceleration: {'ENABLED (' + str(_dev) + ')' if _gpu_on else 'DISABLED (CPU fallback)'}")
        print(f"[BiDirHDC] n_words={n_words} ({n_words*64:,} bits per HV)")
        print(f"[BiDirHDC] seed={seed}  (single seed — use different seeds for 3 independent runs)")
        print(f"[BiDirHDC] vocab_size={vocab_size}")
        print(f"[BiDirHDC] max_wallclock={max_secs}s")
        print(f"[BiDirHDC] timestamp={timestamp}")
        print(f"{'='*60}\n")

    # ── Load tokeniser ────────────────────────────────────────────────────────
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    if _dist_is_main():
        print(f"[BiDirHDC] Tokeniser loaded: {tok_path} (vocab={sp.GetPieceSize()})")

    # ── Load training tokens ──────────────────────────────────────────────────
    if _dist_is_main():
        print(f"[BiDirHDC] Loading training tokens from {data_path}...")
    train_tokens = _load_tokens(data_path, split="train")
    if _dist_is_main():
        print(f"[BiDirHDC] Training tokens: {len(train_tokens):,}")

    # ── Training ──────────────────────────────────────────────────────────────
    # Reserve 75s for eval + artifact save.
    # With eigen optimisations, train_bidi_model() completes in ~5–8s,
    # leaving ~547s for SpiralDSV build. We use a per-seed budget of 30s
    # so the engine trains quickly, then the remaining time goes to SpiralDSV.
    train_budget = max(60.0, max_secs - 75.0)

    engine = train_bidi_model(
        tokens        = train_tokens,
        vocab_size    = vocab_size,
        n_words       = n_words,
        seeds         = [seed],   # single seed per run for statistical variance
        time_budget_s = train_budget,
        verbose       = _dist_is_main(),
    )

    # Non-main ranks exit after training
    if not _dist_is_main():
        return

    # ── Optional SpiralDSV build ──────────────────────────────────────────────
    # spiral_budget = remaining time after training, minus 45s eval reserve.
    # With eigen optimisations, elapsed_train ≈ 5–8s, so spiral_budget ≈ 547s.
    # The SpiralDSV build is already inside the 600s budget — it is Phase 3
    # of the pipeline, not bonus time.
    spiral_dsv = None
    elapsed_train = time.time() - t_global_start
    spiral_budget = max(0.0, max_secs - elapsed_train - 45.0)

    if spiral_budget > 5.0:
        try:
            from _spiral_dsv_lm import SpiralDSVLanguageModel
            print(f"\n[BiDirHDC] Building SpiralDSV bilateral tables "
                  f"(budget={spiral_budget:.0f}s, elapsed_so_far={elapsed_train:.1f}s)...")
            spiral_dsv = SpiralDSVLanguageModel(
                vocab_size = vocab_size,
                n_words    = n_words,
                seed       = seed,
            )
            # ctx_len=4 completes in <0.1s with EigenSpiralBuilder.
            # The full spiral_budget is passed so the time-budget guard inside
            # build_from_tokens() can use remaining time for deeper lags if desired.
            spiral_dsv.build_from_tokens(
                tokens        = train_tokens,
                ctx_len       = 4,
                time_budget_s = spiral_budget,
                verbose       = True,
            )
        except Exception as e:
            print(f"[BiDirHDC] SpiralDSV build failed ({e}) — skipping")
            spiral_dsv = None

    # ── Save artifact ─────────────────────────────────────────────────────────
    artifact_path = os.path.join(_THIS_DIR, f"{run_id}_seed{seed}_{timestamp}.bdhgz")
    compressed_size = save_bidi_artifact(
        engine     = engine,
        path       = artifact_path,
        spiral_dsv = spiral_dsv,
        verbose    = True,
    )

    # ── Artifact size check ───────────────────────────────────────────────────
    code_bytes = os.path.getsize(os.path.abspath(__file__))
    # Also count helper module sizes (include _gpu.py)
    for helper in ["_bidi_hdc_engine.py", "_bidi_train.py", "_spiral_dsv_lm.py", "_gpu.py"]:
        helper_path = os.path.join(_THIS_DIR, helper)
        if os.path.exists(helper_path):
            code_bytes += os.path.getsize(helper_path)

    total_artifact, passes = check_artifact_size(artifact_path, code_bytes)

    # ── Load validation tokens ────────────────────────────────────────────────
    print(f"\n[BiDirHDC] Running BPB evaluation on validation set...")
    val_tokens = _load_tokens(data_path, split="val")
    print(f"[BiDirHDC] Validation tokens: {len(val_tokens):,}")

    # ── BPB Evaluation ────────────────────────────────────────────────────────
    bpb, val_loss = bidi_bpb(
        val_tokens         = val_tokens,
        engine             = engine,
        sp_model           = sp,
        spiral_dsv         = spiral_dsv,
        chunk_size         = 4096,
        spiral_blend_alpha = 0.3 if spiral_dsv is not None else 0.0,
        verbose            = True,
    )

    # ── Final results (same format as existing submissions) ───────────────────
    elapsed = time.time() - t_global_start
    print(f"\n[TensorCore] FINAL RESULTS")
    print(f"BPB: {bpb:.4f}  |  Val Loss: {val_loss:.4f}  |  Time: {elapsed:.1f}s")
    print(f"Code size: {code_bytes:,} bytes  |  Total artifact: {total_artifact:,} bytes")
    print(f"Artifact size check: {'PASS' if passes else 'FAIL'} "
          f"(limit: {ARTIFACT_LIMIT:,} bytes)")

    # ── Auto-generate submission.json ─────────────────────────────────────────
    # Each run overwrites submission.json with the latest results.
    # The leaderboard submission uses the best result across 3 seeds.
    submission = {
        "track": "10min_16mb",
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "name": "FullBiDirHDC Complete Replacement — Bilateral Joint Manifold Engine",
        "author": "Ashley Klimpel",
        "github_id": "viasky657",
        "val_loss": float(val_loss),
        "val_bpb": float(bpb),
        "artifact_bytes": int(total_artifact),
        "code_bytes": int(code_bytes),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "n_words": n_words,
        "hv_bits": n_words * 64,
        "seed": seed,
        "elapsed_s": float(elapsed),
        "timestamp": timestamp,
        "artifact_path": os.path.basename(artifact_path),
        "artifact_size_check": "PASS" if passes else "FAIL",
    }
    submission_path = os.path.join(_THIS_DIR, "submission.json")
    with open(submission_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"[BiDirHDC] submission.json written: {submission_path}")

    return bpb, val_loss, total_artifact, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="FullBiDirHDC Parameter Golf Submission"
    )
    # All configuration is driven by environment variables (contest standard).
    # Command-line args are optional overrides for convenience.
    parser.add_argument(
        "--data_path",
        default=os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"),
        help="Path to fineweb10B_sp1024/ directory"
    )
    parser.add_argument(
        "--tokenizer_path",
        default=os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"),
        help="Path to SentencePiece model file"
    )
    parser.add_argument(
        "--n_words",
        type=int,
        default=int(os.environ.get("N_WORDS", 512)),
        help="HV width in uint64 words (default 512 → 32,768 bits)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Single random seed for this run (default 42). Use different seeds for 3 independent runs."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=int(os.environ.get("VOCAB_SIZE", 1024)),
        help="Vocabulary size (default 1024)"
    )
    parser.add_argument(
        "--max_wallclock_seconds",
        type=int,
        default=int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600)),
        help="Maximum wallclock time in seconds (default 600)"
    )
    parser.add_argument(
        "--run_id",
        default=os.environ.get("RUN_ID", "bidi_hdc"),
        help="Run identifier for artifact naming (default 'bidi_hdc')"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialise distributed (no-op if not using torchrun)
    rank, world_size = _init_distributed()

    args = _parse_args()

    # Run the FullBiDirHDC pipeline unconditionally — this is the sole pipeline
    # for this submission. No flag required; matches contest standard invocation:
    #   torchrun --standalone --nproc_per_node=8 train_gpt.py
    _run_bidi_hdc(args)
