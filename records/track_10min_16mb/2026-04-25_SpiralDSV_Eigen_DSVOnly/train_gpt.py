"""SpiralDSV + Eigen — DSV-Only Parameter Golf Submission.

Replaces the entire 2026-04-07 NMF + DSV pipeline with a DSV-only system:
  - Removes all NMF phases (0–5, 8–9): no embed, no W_out, no Hadamard codebook
  - Keeps only Phase 6: EigenTrainer.build_bilateral_from_tokens()
    with GoldenAxisShift per-lag codebook rotation + PMI centering
  - Budget reallocation: 16 MB freed from NMF → 8 MB sem_fwd + 8 MB sem_bwd
  - n_words=1024 → 65,536 bits per token (64× increase from 1,024 bits)
  - Artifact: HGZ3 format, LZMA9 compressed (~2–4 MB on disk)

Usage (8×H100 SXM, leaderboard — standard contest invocation):
    # Run 1 (seed 42):
    RUN_ID=spiral_dsv N_WORDS=1024 SEED=42 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    # Run 2 (seed 7):
    RUN_ID=spiral_dsv N_WORDS=1024 SEED=7 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

    # Run 3 (seed 1337):
    RUN_ID=spiral_dsv N_WORDS=1024 SEED=1337 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

Usage (single GPU, smoke test with reduced n_words):
    N_WORDS=128 SEED=42 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    python train_gpt.py

Environment variables:
    N_WORDS              : HV width in uint64 words (default 1024 → 65,536 bits)
                           Use 128 for RTX 4090, 16 for CPU smoke test
    SEED                 : Single random seed for this run (default 42)
    MAX_WALLCLOCK_SECONDS: Training time cap in seconds (default 600)
    DATA_PATH            : Path to fineweb10B_sp1024/ directory
    TOKENIZER_PATH       : Path to fineweb_1024_bpe.model
    VOCAB_SIZE           : Vocabulary size (default 1024)
    RUN_ID               : Run identifier for artifact naming (default "spiral_dsv")
    W_COHERENCE          : Coherence gating weight (default 0.3, 0.0 = disabled)
    CTX_LEN              : Context depth / number of lags (default 4)

Each run auto-generates:
    spiral_dsv_seed{SEED}_{TIMESTAMP}.hgz  — compressed model artifact
    train_{TIMESTAMP}.log                   — training log (via shell redirect)
    submission.json                         — updated with actual val_bpb, val_loss

BPB formula (identical to reference train_gpt.py):
    BPB = Σ(-log₂ p_correct) / Σ(utf8_bytes(token))
        = bits_per_token × tokens_per_byte
"""

import argparse
import datetime
import glob
import json
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


def _dist_world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _dist_is_main() -> bool:
    return _dist_rank() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Token loading (same as existing submissions)
# ─────────────────────────────────────────────────────────────────────────────

def _load_tokens(
    data_path: str,
    split: str,
    max_shards: int = 80,
    rank: int = 0,
    world_size: int = 1,
) -> np.ndarray:
    """Load tokenised FineWeb shards from data_path.

    For the training split in a distributed run each rank receives only its
    interleaved slice of shard files (rank, rank+world_size, rank+2*world_size,
    …).  This reduces per-rank peak memory by world_size× while guaranteeing
    that the union of all ranks' shards covers the full corpus — exactly as if
    every rank had loaded everything.  Histograms built from these partial
    corpora are summed via all_reduce in EigenTrainer.build_bilateral_from_tokens(),
    so the final statistics are identical to the single-rank baseline.

    The validation split is always loaded in full (only rank 0 uses it).

    Args:
        data_path  : Path to fineweb10B_sp1024/ directory
        split      : "train" or "val"
        max_shards : Maximum number of training shards to consider
        rank       : This rank's index (0-based)
        world_size : Total number of ranks (1 = single-process)

    Returns:
        (N,) uint16 array of token IDs
    """
    pattern = os.path.join(data_path, f"fineweb_{split}_*.bin")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No {split} shards found at {pattern}")

    if split == "train":
        shard_files = shard_files[:max_shards]
        # Distribute shards across ranks — each rank loads 1/world_size of the
        # shards.  The interleaved stride keeps the per-rank corpus roughly
        # uniform in size and representative of the full data distribution.
        if world_size > 1:
            shard_files = shard_files[rank::world_size]

    # Each shard has a 1024-byte header: 256 × int32
    # (magic=0x134D888, version, n_tokens, …).  Skip it before reading tokens.
    _HEADER_BYTES = 256 * 4   # 1024 bytes = 512 uint16 words

    all_tokens = []
    for shard_path in shard_files:
        tokens = np.fromfile(shard_path, dtype=np.uint16)[_HEADER_BYTES // 2:]
        all_tokens.append(tokens)
        if _dist_is_main():
            total = sum(len(t) for t in all_tokens)
            print(f"[TokenLoad] Loaded {total:,} tokens from "
                  f"{os.path.basename(shard_path)}")

    return np.concatenate(all_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Main DSV-only pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_spiral_dsv(args):
    """Main SpiralDSV + Eigen DSV-only training and evaluation pipeline.

    Pipeline (Phase 6 only — all NMF phases removed):
      1. Load 500M training tokens (sharded per rank)
      2. Build SpiralDSV bilateral tables via EigenTrainer (all ranks)
         - GoldenAxisShift per-lag codebook rotation (lags 1..ctx_len)
         - PMI centering (encodes anti-correlations)
         - GPU-accelerated dual HGEMM matmul
         - Distributed all-reduce across all ranks
      3. Non-zero ranks exit after all-reduce
      4. Rank 0: save artifact (HGZ3 format, LZMA9 compressed)
      5. Rank 0: load val tokens, compute BPB
      6. Rank 0: print audit block + write submission.json
    """
    from _semantic_layer import (
        build_spiral_dsv,
        eval_spiral_dsv_bpb,
        save_spiral_dsv_artifact,
        check_artifact_size,
        build_token_byte_arrays,
        ARTIFACT_LIMIT,
        N_WORDS_H100,
        N_WORDS_4090,
        N_WORDS_TEST,
    )

    t_global_start = time.time()
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # ── Configuration ────────────────────────────────────────────────────────
    n_words    = int(os.environ.get("N_WORDS",    args.n_words))
    seed       = int(os.environ.get("SEED",       args.seed))
    max_secs   = int(os.environ.get("MAX_WALLCLOCK_SECONDS", args.max_wallclock_seconds))
    vocab_size = int(os.environ.get("VOCAB_SIZE", args.vocab_size))
    run_id     = os.environ.get("RUN_ID",         args.run_id)
    data_path  = os.environ.get("DATA_PATH",      args.data_path)
    tok_path   = os.environ.get("TOKENIZER_PATH", args.tokenizer_path)
    w_coh      = float(os.environ.get("W_COHERENCE", args.w_coherence))
    ctx_len    = int(os.environ.get("CTX_LEN",    args.ctx_len))

    rank       = _dist_rank()
    world_size = _dist_world_size()

    if _dist_is_main():
        # ── GPU status report ─────────────────────────────────────────────
        try:
            from _gpu import gpu_available, _get_device
            _gpu_on = gpu_available()
            _dev    = _get_device() if _gpu_on else "cpu"
        except Exception:
            _gpu_on = False
            _dev    = "cpu"

        n_bits = n_words * 64
        sem_mb = 2 * vocab_size * n_words * 8 / 1_000_000

        print(f"\n{'='*65}")
        print(f"[SpiralDSV] DSV-Only Parameter Golf Submission")
        print(f"[SpiralDSV] GPU: {'ENABLED (' + str(_dev) + ')' if _gpu_on else 'DISABLED (CPU fallback)'}")
        print(f"[SpiralDSV] n_words={n_words} ({n_bits:,} bits per HV)")
        print(f"[SpiralDSV] sem_fwd+sem_bwd budget: {sem_mb:.1f} MB")
        print(f"[SpiralDSV] seed={seed}  ctx_len={ctx_len}  vocab_size={vocab_size}")
        print(f"[SpiralDSV] max_wallclock={max_secs}s  W_COHERENCE={w_coh}")
        print(f"[SpiralDSV] world_size={world_size}  timestamp={timestamp}")
        print(f"{'='*65}\n")

    # ── Load tokeniser ────────────────────────────────────────────────────────
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    if _dist_is_main():
        print(f"[SpiralDSV] Tokeniser loaded: {tok_path} "
              f"(vocab={sp.GetPieceSize()})")

    # ── Load training tokens (sharded per rank) ───────────────────────────────
    if _dist_is_main():
        print(f"[SpiralDSV] Loading training tokens from {data_path}...")
        print(f"[SpiralDSV] (Distributed shard loading: each of {world_size} "
              f"rank(s) loads ~1/{world_size} of shards)")
    train_tokens = _load_tokens(
        data_path,
        split="train",
        rank=rank,
        world_size=world_size,
    )
    if _dist_is_main():
        print(f"[SpiralDSV] Training tokens (this rank): {len(train_tokens):,}")

    # ── Phase 6: Build bilateral DSV tables (all ranks participate) ───────────
    # Reserve 60s for eval + artifact save.
    build_budget = max(60.0, max_secs - 60.0)

    model = build_spiral_dsv(
        tokens          = train_tokens,
        vocab_size      = vocab_size,
        n_words         = n_words,
        ctx_len         = ctx_len,
        seed            = seed,
        time_budget_s   = build_budget,
        dist_rank       = rank,
        dist_world_size = world_size,
        verbose         = _dist_is_main(),
    )

    # Free per-rank training tokens — no longer needed
    del train_tokens

    # Non-zero ranks: histograms contributed via all-reduce — exit now
    if not _dist_is_main():
        return

    # ── Rank 0 only from here ─────────────────────────────────────────────────

    if not getattr(model, '_built', False):
        print(f"[SpiralDSV] ERROR: model not built — aborting")
        return

    # ── Save artifact ─────────────────────────────────────────────────────────
    artifact_path = os.path.join(
        _THIS_DIR, f"{run_id}_seed{seed}_{timestamp}.hgz"
    )
    compressed_size = save_spiral_dsv_artifact(
        model   = model,
        path    = artifact_path,
        verbose = True,
    )

    # ── Artifact size check ───────────────────────────────────────────────────
    code_bytes = 0
    for helper in [
        "train_gpt.py",
        "_semantic_layer.py",
        "_spiral_dsv_lm.py",
        "_eigen_convergence.py",
        "_gpu.py",
    ]:
        helper_path = os.path.join(_THIS_DIR, helper)
        if os.path.exists(helper_path):
            code_bytes += os.path.getsize(helper_path)

    total_artifact, passes = check_artifact_size(
        artifact_path, code_bytes, verbose=True
    )

    # ── Load validation tokens ────────────────────────────────────────────────
    print(f"\n[SpiralDSV] Loading validation tokens from {data_path}...")
    val_tokens = _load_tokens(data_path, split="val")
    print(f"[SpiralDSV] Validation tokens: {len(val_tokens):,}")

    # ── Build per-token byte arrays from tokeniser ────────────────────────────
    base_bytes, has_leading_space, is_boundary_token = build_token_byte_arrays(
        sp_model   = sp,
        vocab_size = vocab_size,
    )

    # ── BPB Evaluation ────────────────────────────────────────────────────────
    bpb, val_loss = eval_spiral_dsv_bpb(
        val_tokens        = val_tokens,
        model             = model,
        base_bytes        = base_bytes,
        has_leading_space = has_leading_space,
        is_boundary_token = is_boundary_token,
        batch_size        = 500_000,
        W_COHERENCE       = w_coh,
        verbose           = True,
    )

    # ── Final results (same format as existing submissions) ───────────────────
    elapsed = time.time() - t_global_start

    print(f"\n[HashGrad BPB audit]")
    print(f"  val_bpb  = {bpb:.6f}")
    print(f"  val_loss = {val_loss:.6f}")
    print(f"  elapsed  = {elapsed:.1f}s")
    print(f"  n_words  = {n_words}  ({n_words*64:,} bits)")
    print(f"  seed     = {seed}")
    print(f"  artifact = {os.path.basename(artifact_path)}")
    print(f"  artifact_bytes = {compressed_size:,}")
    print(f"  code_bytes     = {code_bytes:,}")
    print(f"  total_bytes    = {total_artifact:,}")
    print(f"  size_check     = {'PASS' if passes else 'FAIL'}")

    print(f"\n[TensorCore] FINAL RESULTS")
    print(f"BPB: {bpb:.4f}  |  Val Loss: {val_loss:.4f}  |  Time: {elapsed:.1f}s")
    print(f"Code size: {code_bytes:,} bytes  |  "
          f"Total artifact: {total_artifact:,} bytes")
    print(f"Artifact size check: {'PASS' if passes else 'FAIL'} "
          f"(limit: {ARTIFACT_LIMIT:,} bytes)")

    # ── Auto-generate submission.json ─────────────────────────────────────────
    submission = {
        "track": "10min_16mb",
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
        "name": "SpiralDSV + Eigen — DSV-Only (no NMF)",
        "author": "Ashley Klimpel",
        "github_id": "viasky657",
        "val_loss": float(val_loss),
        "val_bpb": float(bpb),
        "artifact_bytes": int(total_artifact),
        "code_bytes": int(code_bytes),
        "world_size": int(world_size),
        "n_words": n_words,
        "hv_bits": n_words * 64,
        "ctx_len": ctx_len,
        "seed": seed,
        "w_coherence": w_coh,
        "elapsed_s": float(elapsed),
        "timestamp": timestamp,
        "artifact_path": os.path.basename(artifact_path),
        "artifact_size_check": "PASS" if passes else "FAIL",
    }
    submission_path = os.path.join(_THIS_DIR, "submission.json")
    with open(submission_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"[SpiralDSV] submission.json written: {submission_path}")

    return bpb, val_loss, total_artifact, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SpiralDSV + Eigen DSV-Only Parameter Golf Submission"
    )
    parser.add_argument(
        "--data_path",
        default=os.environ.get(
            "DATA_PATH", "./data/datasets/fineweb10B_sp1024"
        ),
        help="Path to fineweb10B_sp1024/ directory",
    )
    parser.add_argument(
        "--tokenizer_path",
        default=os.environ.get(
            "TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
        ),
        help="Path to SentencePiece model file",
    )
    parser.add_argument(
        "--n_words",
        type=int,
        default=int(os.environ.get("N_WORDS", 1024)),
        help=(
            "HV width in uint64 words "
            "(default 1024 → 65,536 bits for H100; "
            "use 128 for RTX 4090, 16 for CPU smoke test)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", 42)),
        help="Single random seed for this run (default 42). "
             "Use different seeds for 3 independent runs.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=int(os.environ.get("VOCAB_SIZE", 1024)),
        help="Vocabulary size (default 1024)",
    )
    parser.add_argument(
        "--max_wallclock_seconds",
        type=int,
        default=int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600)),
        help="Maximum wallclock time in seconds (default 600)",
    )
    parser.add_argument(
        "--run_id",
        default=os.environ.get("RUN_ID", "spiral_dsv"),
        help="Run identifier for artifact naming (default 'spiral_dsv')",
    )
    parser.add_argument(
        "--w_coherence",
        type=float,
        default=float(os.environ.get("W_COHERENCE", 0.3)),
        help="Coherence gating weight (default 0.3, 0.0 = disabled). "
             "Biases predictions toward tokens coherent with the document topic.",
    )
    parser.add_argument(
        "--ctx_len",
        type=int,
        default=int(os.environ.get("CTX_LEN", 4)),
        help="Context depth / number of lags (default 4)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Initialise distributed (no-op if not using torchrun)
    rank, world_size = _init_distributed()

    args = _parse_args()

    # Run the DSV-only pipeline unconditionally.
    # Matches contest standard invocation:
    #   torchrun --standalone --nproc_per_node=8 train_gpt.py
    _run_spiral_dsv(args)
