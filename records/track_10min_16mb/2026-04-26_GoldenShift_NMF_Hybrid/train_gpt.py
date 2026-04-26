"""GoldenShift_NMF_Hybrid — Parameter Golf Submission.

Two-tier language model combining:
  Tier 1: NMF hash table  (G[p] rolling hash → softmax(embed[bucket] @ W_out))
          Provides properly calibrated probability distributions for seen contexts.
          Unlimited context depth via rolling hash.  Fires ~70-80% of eval positions.

  Tier 2: GoldenAxisShift DSV  (sem_fwd[prev_tok] → precomputed score table lookup)
          Provides a reliable semantic fallback for bucket misses and hash collisions.
          Architecture-native lag-subspace separation (lags 1..ctx_len).
          Fires ~20-30% of eval positions.

Budget (Option B — default):
  NMF:  TABLE_BITS=18, EMBED_DIM=16: 256K × 16 × 2 = 8 MB uncompressed
  DSV:  N_WORDS=1024:                 1024 × 1024 × 8 = 8 MB uncompressed
  Combined compressed (LZMA9): ~11 MB (fits ≤ 16 MB)

Usage (8×H100 SXM, all 3 contest seeds automatic):
    TABLE_BITS=18 EMBED_DIM=16 N_WORDS=1024 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

Usage (single seed):
    SEED=42 TABLE_BITS=18 EMBED_DIM=16 N_WORDS=1024 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

RTX 4090 smoke test (single GPU, reduced config):
    TABLE_BITS=16 EMBED_DIM=16 N_WORDS=128 SEED=42 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    python3 -u train_gpt.py

Environment variables:
    TABLE_BITS           : log₂ NMF hash table size (default 18 → 256K buckets)
    EMBED_DIM            : NMF embedding dimension (default 16)
    N_WORDS              : DSV HV width in uint64 words (default 1024 → 65,536 bits)
    SEED                 : Single seed (default: auto-run 42→7→1337 if unset)
    CTX_LEN              : GoldenAxisShift lags (default 4)
    USE_FREQ_WEIGHTS     : 1/freq DSV weighting (default 1)
    MAX_WALLCLOCK_SECONDS: Training time cap (default 600)
    DATA_PATH            : Path to fineweb10B_sp1024/ directory
    TOKENIZER_PATH       : Path to fineweb_1024_bpe.model
    VOCAB_SIZE           : Vocabulary size (default 1024)
    RUN_ID               : Artifact naming prefix (default 'gs_nmf_hyb')
    NPROC_PER_NODE       : GPUs per node for multi-seed orchestration (default 8)
    LAG_DEPTH            : Sliding-window n-gram depth for GoldenGram hash (default 8)
                           0 = use old absolute rolling hash (precompute_g_states)
                           8 = 8-gram translational-invariant GoldenGram hash (recommended)
                           Higher lag depth → richer context → lower BPB but slower g_states build
"""

import glob
import json
import os
import sys
import time
import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Add record directory to sys.path for local imports
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def _init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank < 0:
        return 0, 1
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size()

def _dist_rank():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0

def _dist_world_size():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1

def _is_main():
    return _dist_rank() == 0


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

_SHARD_HEADER_BYTES = 256 * 4   # 1024 bytes = 512 uint16 words

def _load_tokens(data_path, split, max_shards=80, rank=0, world_size=1):
    pattern = os.path.join(data_path, f"fineweb_{split}_*.bin")
    shard_files = sorted(glob.glob(pattern))
    if not shard_files:
        raise FileNotFoundError(f"No {split} shards found at {pattern}")

    if split == "train":
        shard_files = shard_files[:max_shards]
        if world_size > 1:
            shard_files = shard_files[rank::world_size]

    all_tokens = []
    for shard_path in shard_files:
        tokens = np.fromfile(shard_path, dtype=np.uint16)[_SHARD_HEADER_BYTES // 2:]
        all_tokens.append(tokens)
        if _is_main():
            total = sum(len(t) for t in all_tokens)
            print(f"[TokenLoad] Loaded {total:,} tokens from "
                  f"{os.path.basename(shard_path)}", flush=True)

    return np.concatenate(all_tokens)


# ---------------------------------------------------------------------------
# Main hybrid pipeline
# ---------------------------------------------------------------------------

def _run_hybrid(args):
    """Run the full NMF + GoldenAxisShift DSV training + evaluation pipeline.

    Phase ordering:
      1.  Load training tokens (distributed shard split)
      2.  Precompute G[p] rolling hash states (seed-independent, all ranks)
      3.  NMF Phases 0–5, 9  (distributed tabulation, rank-0 fit)
          — non-zero ranks exit after Phase 2 all-reduce
      4.  GoldenAxisShift DSV Phase 6  (rank 0 only)
      5.  Save HGZ4 artifact
      6.  Load val tokens + precompute g_states_val
      7.  eval_hybrid_bpb()  (2-tier eval waterfall)
      8.  Print audit + write submission.json
    """
    from _hash_layer import (precompute_g_states, precompute_golden_gram_states,
                             precompute_circular_golden_gram_states, build_nmf_table)
    from _semantic_layer import (
        build_spiral_dsv, save_hybrid_artifact, eval_hybrid_bpb,
        build_token_byte_arrays, check_artifact_size, ARTIFACT_LIMIT,
        N_WORDS_HYB, N_WORDS_4090, N_WORDS_TEST,
    )

    t_global_start = time.time()
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    # Config
    table_bits   = int(os.environ.get("TABLE_BITS",   args.table_bits))
    embed_dim    = int(os.environ.get("EMBED_DIM",    args.embed_dim))
    n_words      = int(os.environ.get("N_WORDS",      args.n_words))
    seed         = int(os.environ.get("SEED",         args.seed))
    max_secs     = int(os.environ.get("MAX_WALLCLOCK_SECONDS", args.max_wallclock_seconds))
    vocab_size   = int(os.environ.get("VOCAB_SIZE",   args.vocab_size))
    run_id       = os.environ.get("RUN_ID",           args.run_id)
    data_path    = os.environ.get("DATA_PATH",        args.data_path)
    tok_path     = os.environ.get("TOKENIZER_PATH",   args.tokenizer_path)
    ctx_len      = int(os.environ.get("CTX_LEN",          args.ctx_len))
    use_fw       = int(os.environ.get("USE_FREQ_WEIGHTS",  1)) != 0
    lag_depth    = int(os.environ.get("LAG_DEPTH",         args.lag_depth))
    circular_hash = int(os.environ.get("CIRCULAR_HASH",   getattr(args, "circular_hash", 0))) != 0

    rank       = _dist_rank()
    world_size = _dist_world_size()

    TABLE_SIZE = 1 << table_bits
    nmf_mb     = TABLE_SIZE * embed_dim * 2 / 1_000_000
    dsv_mb     = vocab_size * n_words  * 8 / 1_000_000

    if _is_main():
        try:
            from _gpu import gpu_available, _get_device
            _gpu_on = gpu_available()
            _dev    = _get_device() if _gpu_on else "cpu"
        except Exception:
            _gpu_on, _dev = False, "cpu"

        print(f"\n{'='*65}")
        print(f"[Hybrid] GoldenShift_NMF_Hybrid Submission")
        print(f"[Hybrid] GPU: {'ENABLED (' + str(_dev) + ')' if _gpu_on else 'DISABLED (CPU)'}")
        print(f"[Hybrid] NMF: TABLE_BITS={table_bits} ({TABLE_SIZE:,} buckets)  "
              f"EMBED_DIM={embed_dim}  ({nmf_mb:.1f} MB uncompressed)")
        print(f"[Hybrid] DSV: n_words={n_words} ({n_words*64:,} bits)  "
              f"({dsv_mb:.1f} MB uncompressed)")
        if lag_depth > 0:
            hash_mode = (f"CircularGoldenGram lag={lag_depth} phi_offset=39"
                         if circular_hash else f"GoldenGram lag={lag_depth}")
        else:
            hash_mode = "AbsoluteRolling"
        print(f"[Hybrid] Hash: {hash_mode}")
        print(f"[Hybrid] seed={seed}  ctx_len={ctx_len}  vocab_size={vocab_size}")
        print(f"[Hybrid] use_freq_weights={use_fw}")
        print(f"[Hybrid] max_wallclock={max_secs}s  world_size={world_size}")
        print(f"[Hybrid] timestamp={timestamp}")
        print(f"{'='*65}\n")

    # Tokeniser
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    if _is_main():
        print(f"[Hybrid] Tokeniser loaded: vocab={sp.GetPieceSize()}", flush=True)

    base_bytes, has_leading_space, is_boundary_token = build_token_byte_arrays(
        sp, vocab_size
    )

    # 1. Load training tokens
    if _is_main():
        print(f"[Hybrid] Loading training tokens from {data_path}...", flush=True)
    train_tokens = _load_tokens(
        data_path, split="train", rank=rank, world_size=world_size,
    )
    if _is_main():
        print(f"[Hybrid] Training tokens (this rank): {len(train_tokens):,}", flush=True)

    # 2. Precompute g_states (or GoldenGram n-gram hash, seed-independent)
    if _is_main():
        if lag_depth > 0 and circular_hash:
            print(f"[Hybrid] Computing CircularGoldenGram g_states "
                  f"(lag_depth={lag_depth}, phi_offset=39)...", flush=True)
        elif lag_depth > 0:
            print(f"[Hybrid] Computing GoldenGram g_states (lag_depth={lag_depth})...",
                  flush=True)
        else:
            print(f"[Hybrid] Computing absolute rolling hash g_states...", flush=True)
    t_g = time.time()
    if lag_depth > 0 and circular_hash:
        g_states_train = precompute_circular_golden_gram_states(
            train_tokens, lag_depth=lag_depth)
    elif lag_depth > 0:
        g_states_train = precompute_golden_gram_states(train_tokens, lag_depth=lag_depth)
    else:
        g_states_train = precompute_g_states(train_tokens)
    if _is_main():
        print(f"[Hybrid] g_states done in {time.time()-t_g:.1f}s  "
              f"({g_states_train.nbytes/1e9:.2f} GB)", flush=True)

    # 3. NMF pipeline (Phases 0-5, 9) — budget ~30% of wallclock
    nmf_budget = max(30.0, max_secs * 0.30)
    if _is_main():
        print(f"\n[Hybrid] === Phase 3: NMF Hash Table (budget={nmf_budget:.0f}s) ===",
              flush=True)
    embed, W_out, fingerprint = build_nmf_table(
        tokens          = train_tokens,
        g_states        = g_states_train,
        seed            = seed,
        table_bits      = table_bits,
        embed_dim       = embed_dim,
        vocab_size      = vocab_size,
        nmf_max_iter    = 1,
        dist_rank       = rank,
        dist_world_size = world_size,
        time_budget_s   = nmf_budget,
        verbose         = _is_main(),
    )

    # Non-zero ranks exit after Phase 2 all-reduce
    if rank != 0:
        return

    # 4. GoldenAxisShift DSV Phase 6 — rank 0 only
    dsv_budget = max(60.0, max_secs - nmf_budget - 60.0)
    print(f"\n[Hybrid] === Phase 6: GoldenAxisShift DSV (budget={dsv_budget:.0f}s) ===",
          flush=True)
    model = build_spiral_dsv(
        tokens           = train_tokens,
        vocab_size       = vocab_size,
        n_words          = n_words,
        ctx_len          = ctx_len,
        seed             = seed,
        time_budget_s    = dsv_budget,
        dist_rank        = 0,
        dist_world_size  = 1,
        use_freq_weights = use_fw,
        verbose          = True,
    )
    del train_tokens, g_states_train

    if not getattr(model, '_built', False):
        print(f"[Hybrid] ERROR: DSV model not built — aborting", flush=True)
        return

    # 5. Save HGZ4 artifact
    artifact_name = f"{run_id}_seed{seed}_{timestamp}.hgz"
    artifact_path = os.path.join(_THIS_DIR, artifact_name)

    # Count code bytes for size check
    code_bytes = sum(
        os.path.getsize(os.path.join(_THIS_DIR, f))
        for f in os.listdir(_THIS_DIR)
        if f.endswith('.py')
    )

    artifact_bytes = save_hybrid_artifact(
        model=model, embed=embed, W_out=W_out, fingerprint=fingerprint,
        nmf_seed=seed, table_bits=table_bits,
        path=artifact_path, verbose=True,
    )

    total_bytes, passes = check_artifact_size(artifact_path, code_bytes)

    # 6. Load validation tokens + g_states
    print(f"\n[Hybrid] Loading validation tokens...", flush=True)
    val_tokens = _load_tokens(data_path, split="val")
    print(f"[Hybrid] Validation tokens: {len(val_tokens):,}", flush=True)

    if lag_depth > 0 and circular_hash:
        print(f"[Hybrid] Computing val CircularGoldenGram g_states "
              f"(lag_depth={lag_depth}, phi_offset=39)...", flush=True)
    elif lag_depth > 0:
        print(f"[Hybrid] Computing val GoldenGram g_states (lag_depth={lag_depth})...",
              flush=True)
    else:
        print(f"[Hybrid] Computing val absolute rolling hash...", flush=True)
    t_g = time.time()
    if lag_depth > 0 and circular_hash:
        g_states_val = precompute_circular_golden_gram_states(
            val_tokens, lag_depth=lag_depth)
    elif lag_depth > 0:
        g_states_val = precompute_golden_gram_states(val_tokens, lag_depth=lag_depth)
    else:
        g_states_val = precompute_g_states(val_tokens)
    print(f"[Hybrid] Val g_states done in {time.time()-t_g:.1f}s", flush=True)

    # 7. 2-tier BPB evaluation
    bpb, val_loss = eval_hybrid_bpb(
        val_tokens        = val_tokens,
        g_states_val      = g_states_val,
        model             = model,
        embed             = embed,
        W_out             = W_out,
        fingerprint       = fingerprint,
        nmf_seed          = seed,
        table_bits        = table_bits,
        base_bytes        = base_bytes,
        has_leading_space = has_leading_space,
        is_boundary_token = is_boundary_token,
        verbose           = True,
    )

    elapsed = time.time() - t_global_start

    # 8. Audit block + submission.json
    print(f"\n[Hybrid BPB audit]")
    print(f"  val_bpb  = {bpb:.6f}")
    print(f"  val_loss = {val_loss:.6f}")
    print(f"  elapsed  = {elapsed:.1f}s")
    print(f"  TABLE_BITS = {table_bits}  EMBED_DIM = {embed_dim}  n_words = {n_words}")
    print(f"  seed     = {seed}")
    print(f"  artifact = {artifact_name}")
    print(f"  artifact_bytes = {artifact_bytes:,}")
    print(f"  code_bytes     = {code_bytes:,}")
    print(f"  total_bytes    = {total_bytes:,}")
    print(f"  size_check     = {'PASS' if passes else 'FAIL'}")

    print(f"\n[Hybrid] FINAL RESULTS")
    print(f"BPB: {bpb:.4f}  |  Val Loss: {val_loss:.4f}  |  Time: {elapsed:.1f}s")
    print(f"Code size: {code_bytes:,} bytes  |  Total artifact: {total_bytes:,} bytes")
    print(f"Artifact size check: {'PASS' if passes else 'FAIL'}")

    submission = {
        "track":            "10min_16mb",
        "date":             datetime.date.today().isoformat(),
        "name":             "GoldenShift_NMF_Hybrid (NMF Tier1 + GoldenAxisShift DSV Tier2)",
        "author":           "Ashley Klimpel",
        "github_id":        "viasky657",
        "val_loss":         float(val_loss),
        "val_bpb":          float(bpb),
        "artifact_bytes":   total_bytes,
        "code_bytes":       code_bytes,
        "world_size":       world_size,
        "table_bits":       table_bits,
        "embed_dim":        embed_dim,
        "n_words":          n_words,
        "ctx_len":          ctx_len,
        "seed":             seed,
        "use_freq_weights": use_fw,
        "elapsed_s":        elapsed,
        "timestamp":        timestamp,
        "artifact_path":    artifact_name,
        "artifact_size_check": "PASS" if passes else "FAIL",
    }

    sub_path = os.path.join(_THIS_DIR, "submission.json")
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"[Hybrid] submission.json written: {sub_path}", flush=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="GoldenShift_NMF_Hybrid")
    parser.add_argument("--table_bits", type=int,
                        default=int(os.environ.get("TABLE_BITS", 18)),
                        help="log₂ of NMF hash table size (18 → 256K buckets)")
    parser.add_argument("--embed_dim", type=int,
                        default=int(os.environ.get("EMBED_DIM", 16)),
                        help="NMF embedding dimension per bucket")
    parser.add_argument("--n_words", type=int,
                        default=int(os.environ.get("N_WORDS", 1024)),
                        help="DSV HV width in uint64 words (1024 → 65,536 bits; "
                             "128 for RTX 4090, 16 for CPU)")
    parser.add_argument("--seed", type=int,
                        default=int(os.environ.get("SEED", 42)),
                        help="Single random seed (default 42)")
    parser.add_argument("--vocab_size", type=int,
                        default=int(os.environ.get("VOCAB_SIZE", 1024)))
    parser.add_argument("--max_wallclock_seconds", type=int,
                        default=int(os.environ.get("MAX_WALLCLOCK_SECONDS", 600)))
    parser.add_argument("--run_id", default=os.environ.get("RUN_ID", "gs_nmf_hyb"))
    parser.add_argument("--ctx_len", type=int,
                        default=int(os.environ.get("CTX_LEN", 4)))
    parser.add_argument("--lag_depth", type=int,
                        default=int(os.environ.get("LAG_DEPTH", 8)),
                        help="GoldenGram sliding-window depth (0=old absolute hash, 8=8-gram)")
    parser.add_argument("--circular_hash", type=int,
                        default=int(os.environ.get("CIRCULAR_HASH", 0)),
                        help="1=use CircularGoldenGram hash (DSV-aligned rotation geometry), "
                             "0=standard GoldenGram (default)")
    parser.add_argument("--data_path",
                        default=os.environ.get("DATA_PATH",
                                               "./data/datasets/fineweb10B_sp1024"))
    parser.add_argument("--tokenizer_path",
                        default=os.environ.get("TOKENIZER_PATH",
                                               "./data/tokenizers/fineweb_1024_bpe.model"))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Multi-seed sequential orchestration
# ---------------------------------------------------------------------------

_CONTEST_SEEDS = [42, 7, 1337]


def _run_all_seeds():
    """Launch torchrun for each contest seed sequentially.

    Called when SEED is not set in the environment AND we are not already
    a torchrun worker (LOCAL_RANK not set).
    """
    import subprocess
    nproc  = int(os.environ.get("NPROC_PER_NODE", 8))
    script = os.path.abspath(__file__)
    base_env = os.environ.copy()
    results = []

    for seed in _CONTEST_SEEDS:
        print(f"\n{'#'*65}")
        print(f"# Multi-seed orchestrator: starting seed={seed}")
        print(f"{'#'*65}\n")

        child_env         = base_env.copy()
        child_env["SEED"] = str(seed)

        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", script]
        print(f"[Orchestrator] Running: {' '.join(cmd)}", flush=True)

        ret    = subprocess.run(cmd, env=child_env)
        status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
        results.append((seed, status))
        print(f"\n[Orchestrator] seed={seed} finished: {status}", flush=True)

    print(f"\n{'='*65}")
    print(f"[Orchestrator] All seeds complete:")
    for seed, status in results:
        print(f"  seed={seed:>5}  ->  {status}")
    print(f"{'='*65}\n")

    if any("FAILED" in s for _, s in results):
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _seed_explicit   = "SEED" in os.environ
    _is_torchrun     = "LOCAL_RANK" in os.environ

    if not _seed_explicit and not _is_torchrun:
        _run_all_seeds()
        sys.exit(0)

    rank, world_size = _init_distributed()
    args = _parse_args()
    _run_hybrid(args)
