
import concurrent.futures
import glob
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------------------
# SentencePiece LUT builder (official competition formula)
# ---------------------------------------------------------------------------

def build_sentencepiece_luts(
    sp, vocab_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build byte-count and leading-space lookup tables.

    is_boundary_token is initialised ALL-TRUE (official standard), then set
    False for every real (non-control/unknown/unused) token.  This matches
    the reference train_gpt.py exactly.
    """
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading_space = np.zeros((table_size,), dtype=bool)
    is_boundary_token = np.ones((table_size,), dtype=bool)   # all-True init

    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token[token_id] = False
        if sp.is_byte(token_id):
            base_bytes[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):   # leading space marker
            has_leading_space[token_id] = True
            piece = piece[1:]
        base_bytes[token_id] = len(piece.encode("utf-8"))

    return base_bytes, has_leading_space, is_boundary_token

# ---------------------------------------------------------------------------
# Token shard loading
# ---------------------------------------------------------------------------

_SHARD_HEADER_SIZE = 256
_SHARD_MAGIC = 20240520


def _read_shard_header(filepath: str) -> int:
    with open(filepath, "rb") as f:
        hdr = f.read(16)
    magic = struct.unpack('<I', hdr[:4])[0]
    if magic != _SHARD_MAGIC:
        raise ValueError(f"Invalid magic number in {filepath}")
    token_count = struct.unpack('<Q', hdr[8:16])[0]
    return token_count


def _mmap_copy_shard(filepath: str, dst: np.ndarray, dst_offset: int, count: int) -> None:
    mm = np.memmap(filepath, dtype=np.uint16, mode='r',
                   offset=_SHARD_HEADER_SIZE, shape=(count,))
    dst[dst_offset:dst_offset + count] = mm
    del mm


def fast_load_token_shards(
    shard_files: List[str],
    max_tokens: int,
    label: str = "Loading",
    num_workers: int = 8,
) -> np.ndarray:
    plan = []
    total_planned = 0
    for shard_file in shard_files:
        if total_planned >= max_tokens:
            break
        shard_count = _read_shard_header(shard_file)
        take = min(shard_count, max_tokens - total_planned)
        plan.append((shard_file, total_planned, take))
        total_planned += take

    if total_planned == 0:
        return np.empty(0, dtype=np.uint16)

    tokens = np.empty(total_planned, dtype=np.uint16)
    print(f"[{label}] Pre-allocated {total_planned:,} token buffer "
          f"({total_planned * 2 / (1024**3):.2f} GiB)")

    def _worker(entry):
        filepath, dst_offset, count = entry
        _mmap_copy_shard(filepath, tokens, dst_offset, count)
        return dst_offset + count, Path(filepath).name

    effective_workers = min(num_workers, len(plan))
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
        for loaded_up_to, name in pool.map(_worker, plan):
            print(f"[{label}] Loaded {loaded_up_to:,} tokens from {name}")

    return tokens

# ---------------------------------------------------------------------------
# Distributed init
# ---------------------------------------------------------------------------

def _init_distributed() -> tuple:
    import torch
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        return 0, 1
    import torch.distributed as dist
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

# ---------------------------------------------------------------------------
# Repo root / logging helpers
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = here
    for _ in range(6):
        if (os.path.isdir(os.path.join(candidate, "data")) and
                os.path.isfile(os.path.join(candidate, "README.md"))):
            return candidate
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    return here


def _setup_tee_logging(log_path: str):
    import io

    class _Tee(io.TextIOWrapper):
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log = log_file

        def __getattr__(self, name):
            return getattr(self._stream, name)

        def write(self, data):
            self._stream.write(data)
            self._stream.flush()
            try:
                self._log.write(data)
                self._log.flush()
            except Exception:
                pass
            return len(data)

        def flush(self):
            self._stream.flush()
            try:
                self._log.flush()
            except Exception:
                pass

    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    return log_file

# ---------------------------------------------------------------------------
# Main hash-grad pipeline
# ---------------------------------------------------------------------------

def _run_hash_grad_single(args) -> int:
    from datetime import datetime, timezone

    rank, world_size = _init_distributed()
    is_main = (rank == 0)

    t_start = time.time()

    TABLE_BITS   = int(os.environ.get("TABLE_BITS", "19"))
    EMBED_DIM    = int(os.environ.get("EMBED_DIM",  "16"))
    hg_seeds_env = os.environ.get("HG_SEEDS", str(getattr(args, "seed", 42)))
    HG_SEEDS     = [int(s.strip()) for s in hg_seeds_env.split(",") if s.strip()]

    data_path      = args.data_path
    tokenizer_path = args.tokenizer_path
    max_seconds    = float(getattr(args, "max_time", 600.0))

    if is_main:
        print(f"\n{'='*60}")
        print(f"[HashGrad] Stripped DSV-only Pipeline")
        print(f"[HashGrad] world_size={world_size}, rank={rank}")
        print(f"[HashGrad] TABLE_BITS={TABLE_BITS}, EMBED_DIM={EMBED_DIM}")
        print(f"[HashGrad] Seeds: {HG_SEEDS}")
        print(f"[HashGrad] Data: {data_path}")
        print(f"[HashGrad] Max time: {max_seconds}s")
        print(f"{'='*60}\n")

    try:
        from _hash_grad_train import (
            tabulate_bucket_frequencies_distributed,
            merge_seed_frequencies,
            hash_grad_bpb,
            save_hash_grad_artifact,
            precompute_g_states,
        )
    except ImportError as _ie:
        print(f"[HashGrad] ERROR: required module not found: {_ie}")
        return 1

    if is_main:
        print("[HashGrad] Loading training tokens...")
    _train_pattern = os.path.join(data_path, "fineweb_train_*.bin")
    _train_shards  = sorted(glob.glob(_train_pattern))
    if not _train_shards:
        _train_shards = sorted(glob.glob(os.path.join(data_path, "*.bin")))
        print(f"[HashGrad] WARNING: no fineweb_train_*.bin found; "
              f"falling back to *.bin glob ({len(_train_shards)} shards)")
    tokens = fast_load_token_shards(
        _train_shards, max_tokens=500_000_000, label="HashGrad"
    )
    vocab_size = int(os.environ.get("VOCAB_SIZE", "1024"))

    if is_main:
        print(f"[HashGrad] Precomputing G[p] states...")
    g_states = precompute_g_states(tokens)

    # --- Phase 2+3: tabulate fingerprint per seed, merge ---
    freq_list, count_list, fp_list = [], [], []
    for seed in HG_SEEDS:
        if is_main:
            print(f"\n[HashGrad] Phase 2 -- seed {seed}")
        f, c, fp = tabulate_bucket_frequencies_distributed(
            tokens=tokens, g_states=g_states, seed=seed,
            table_bits=TABLE_BITS, vocab_size=vocab_size,
            build_fingerprint=True, label=f"Seed{seed}",
        )
        freq_list.append(f)
        count_list.append(c)
        fp_list.append(fp)

    if is_main:
        print(f"\n[HashGrad] Phase 3 -- merging {len(HG_SEEDS)} seed fingerprints...")
    _, _, fingerprint = merge_seed_frequencies(
        freq_list=freq_list, count_list=count_list,
        fingerprint_list=fp_list,
    )
    del freq_list, count_list, fp_list

    # --- Phase 6: DSV sem_fwd + skip-bigram lags ---
    sem_fwd = None
    codebook = None
    skip_bigram_lags = None
    _dsv = None
    _hg_W = EMBED_DIM
    _hg_uint64c = vocab_size * _hg_W
    _p6_budget = max(30.0, max_seconds - 75.0)
    _p6_t0 = time.time()

    def _allgather_xor_u64(arr_u64):
        import torch
        import torch.distributed as _d
        _lr  = int(os.environ.get("LOCAL_RANK", rank))
        _dev = (torch.device(f"cuda:{_lr}") if torch.cuda.is_available()
                else torch.device("cpu"))
        _t   = torch.from_numpy(arr_u64.view(np.int64).copy()).to(_dev)
        _all = [torch.zeros_like(_t) for _ in range(world_size)]
        _d.all_gather(_all, _t)
        if is_main:
            _m = _all[0].cpu().numpy().view(np.uint64).copy()
            for _ri in range(1, world_size):
                _m ^= _all[_ri].cpu().numpy().view(np.uint64)
            return _m
        return arr_u64

    try:
        from _semantic_layer import DirectionalSemanticVec as _DSV_cls
        _PHI64 = np.uint64(0x9E3779B97F4A7C15)
        _MIX64 = np.uint64(0xBF58476D1CE4E5B9)
        _ids   = np.arange(vocab_size, dtype=np.uint64)
        codebook = np.empty((vocab_size, _hg_W), dtype=np.uint64)
        for _k in range(_hg_W):
            _h = _ids * _PHI64
            _h = (_h ^ (_h >> np.uint64(30))) * _MIX64
            _h ^= (_h >> np.uint64(27))
            _h  = _h * np.uint64(_k * 0x0101010101010101 + 1)
            codebook[:, _k] = _h

        _N_tok = len(tokens)
        if world_size > 1:
            _sh_s = rank * _N_tok // world_size
            _sh_e = (rank + 1) * _N_tok // world_size
            _tok_shard = tokens[_sh_s:_sh_e]
        else:
            _tok_shard = tokens

        _dsv_budget = _p6_budget * 0.45
        _dsv = _DSV_cls.build_from_tokens(
            _tok_shard, codebook, ctx_len=4,
            vocab_size=vocab_size, W=_hg_W, uint64_count=_hg_uint64c,
            time_budget_s=_dsv_budget,
            label=f"HashGrad-DSV-r{rank}" if world_size > 1 else "HashGrad-DSV",
            verbose=is_main,
        )

        if world_size > 1:
            _dsv.sem_fwd = _allgather_xor_u64(_dsv.sem_fwd)
            if is_main:
                print(f"[HashGrad-DSV dist] all-gather XOR done across {world_size} ranks")

        _elapsed = time.time() - _p6_t0
        _sb_budget = max(10.0, _p6_budget - _elapsed - 30.0)
        _dsv.build_skip_bigram_lags(
            _tok_shard, codebook, max_lag=5,
            time_budget_s=_sb_budget,
            label=f"HashGrad-SkipBigram-r{rank}" if world_size > 1 else "HashGrad-SkipBigram",
            verbose=is_main,
        )

        if world_size > 1 and hasattr(_dsv, 'sem_fwd_lag') and _dsv.sem_fwd_lag:
            for _lag in sorted(_dsv.sem_fwd_lag.keys()):
                _dsv.sem_fwd_lag[_lag] = _allgather_xor_u64(_dsv.sem_fwd_lag[_lag])
            if is_main:
                print(f"[HashGrad-SkipBigram dist] all-gather XOR done for "
                      f"lags {sorted(_dsv.sem_fwd_lag.keys())}")

    except Exception as _e6:
        if is_main:
            print(f"[HashGrad] Phase 6 failed ({_e6!r})")
        codebook = None

    if not is_main:
        try:
            import torch.distributed as _dist
            if _dist.is_available() and _dist.is_initialized():
                _dist.destroy_process_group()
        except Exception:
            pass
        return 0

    # --- Rank 0 only from here ---
    try:
        if _dsv is not None and codebook is not None:
            sem_fwd = _dsv.sem_fwd.reshape(vocab_size, _hg_W)
            print(f"[HashGrad Phase6] DSV sem_fwd={_hg_uint64c * 8 // 1024}KB")
            if hasattr(_dsv, 'sem_fwd_lag') and _dsv.sem_fwd_lag:
                skip_bigram_lags = [
                    _dsv.sem_fwd_lag[lag].reshape(vocab_size, _hg_W)
                    for lag in sorted(_dsv.sem_fwd_lag.keys())
                ]
                print(f"[HashGrad Phase6] Skip-bigram lags: {sorted(_dsv.sem_fwd_lag.keys())}")
    except Exception as _e6b:
        print(f"[HashGrad] Phase 6 reshape failed ({_e6b!r})")
        sem_fwd = skip_bigram_lags = None

    # --- Phase 10: save artifact ---
    script_dir    = os.path.dirname(os.path.abspath(__file__)) or "."
    artifact_path = os.path.join(script_dir, f"hdc_hashgrad_seed{HG_SEEDS[0]}.hgz")

    if sem_fwd is not None and fingerprint is not None:
        artifact_bytes = save_hash_grad_artifact(
            fingerprint=fingerprint,
            sem_fwd=sem_fwd,
            seed=HG_SEEDS[0],
            table_bits=TABLE_BITS,
            path=artifact_path,
            skip_bigram_lags=skip_bigram_lags,
        )
    else:
        print("[HashGrad] WARNING: sem_fwd or fingerprint missing -- artifact not saved")
        artifact_bytes = 0

    # --- Eval ---
    print("\n[HashGrad] Running BPB evaluation on validation set...")
    bpb, val_loss = float("inf"), float("inf")
    try:
        _val_pattern = os.path.join(data_path, "fineweb_val_*.bin")
        val_tokens = fast_load_token_shards(
            sorted(glob.glob(_val_pattern)), max_tokens=5_000_000, label="ValEval"
        )
        val_tokens = np.clip(val_tokens.astype(np.int32), 0, vocab_size - 1).astype(np.uint16)
        g_val = precompute_g_states(val_tokens)

        sp = spm.SentencePieceProcessor()
        sp.Load(tokenizer_path)
        base_bytes_arr, has_leading_space, is_boundary_token = build_sentencepiece_luts(sp, vocab_size)

        bpb, val_loss = hash_grad_bpb(
            val_tokens=val_tokens,
            g_states_val=g_val,
            seed=HG_SEEDS[0],
            table_bits=TABLE_BITS,
            base_bytes=base_bytes_arr,
            has_leading_space=has_leading_space,
            is_boundary_token=is_boundary_token,
            fingerprint_packed=fingerprint,
            sem_fwd=sem_fwd,
            codebook=codebook,
            skip_bigram_lags=skip_bigram_lags,
        )
    except Exception as _eval_e:
        import traceback
        traceback.print_exc()
        print(f"[HashGrad] Evaluation failed ({_eval_e}) -- reporting inf BPB")

    elapsed = time.time() - t_start

    script_path     = os.path.abspath(__file__)
    code_size_bytes = os.path.getsize(script_path)
    total_bytes     = code_size_bytes + artifact_bytes
    size_ok         = total_bytes <= 16_000_000

    print(f"\n{'='*60}")
    print(f"[TensorCore] FINAL RESULTS")
    print(f"BPB: {bpb:.4f}  |  Val Loss: {val_loss:.4f}  |  Time: {elapsed:.1f}s")
    print(f"Code size: {code_size_bytes:,} bytes  |  Total artifact: {total_bytes:,} bytes")
    print(f"Artifact size check: {'PASS' if size_ok else 'FAIL'} (limit: 16,000,000 bytes)")
    print(f"{'='*60}")

    from datetime import datetime, timezone
    submission = {
        "track": "10min_16mb",
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "name": getattr(args, "run_name", "HDC Hash-Grad 8xH100 Stripped"),
        "author": getattr(args, "author", ""),
        "github_id": getattr(args, "github_id", ""),
        "val_loss": float(val_loss),
        "val_bpb": float(bpb),
        "artifact_bytes": total_bytes,
        "code_bytes": code_size_bytes,
        "world_size": world_size,
        "table_bits": TABLE_BITS,
        "embed_dim": EMBED_DIM,
        "seeds": HG_SEEDS,
        "elapsed_s": round(elapsed, 1),
    }
    submission_path = os.path.join(script_dir, "submission.json")
    with open(submission_path, "w") as _sf:
        json.dump(submission, _sf, indent=2)
    print(f"[TensorCore] Submission saved -> {submission_path}")

    try:
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized():
            _dist.destroy_process_group()
    except Exception:
        pass

    return 0 if size_ok and bpb < float("inf") else 1

# ---------------------------------------------------------------------------
# Multi-seed sequential orchestration
# ---------------------------------------------------------------------------

_CONTEST_SEEDS = [42, 7, 1337]


def _run_all_seeds_sequential():
    """Launch torchrun for each contest seed sequentially.

    Called when HG_SEEDS is not set in the environment AND LOCAL_RANK is not
    set (i.e. we are the top-level Python process, not a torchrun worker).

    Each seed is run as a full independent torchrun invocation so that GPU
    memory is fully released between runs.
    """
    import subprocess

    nproc = int(os.environ.get("NPROC_PER_NODE", 8))
    script = os.path.abspath(__file__)

    base_env = os.environ.copy()

    results = []
    for seed in _CONTEST_SEEDS:
        print(f"\n{'#'*65}")
        print(f"# Multi-seed orchestrator: starting seed={seed}")
        print(f"{'#'*65}\n")

        child_env = base_env.copy()
        child_env["HG_SEEDS"] = str(seed)

        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            script,
        ]

        print(f"[Orchestrator] Running: {' '.join(cmd)}")
        print(f"[Orchestrator] HG_SEEDS={seed}\n")

        ret = subprocess.run(cmd, env=child_env)

        status = "OK" if ret.returncode == 0 else f"FAILED (exit {ret.returncode})"
        results.append((seed, status))
        print(f"\n[Orchestrator] seed={seed} finished: {status}")

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

def main():
    import argparse
    from datetime import datetime, timezone

    # -----------------------------------------------------------------------
    # Multi-seed orchestration:
    # If HG_SEEDS is not set AND we are not already a torchrun worker
    # (LOCAL_RANK not set), run all 3 contest seeds sequentially.
    # -----------------------------------------------------------------------
    _hg_seeds_explicit = "HG_SEEDS" in os.environ
    _is_torchrun_worker = "LOCAL_RANK" in os.environ

    if not _hg_seeds_explicit and not _is_torchrun_worker:
        _run_all_seeds_sequential()
        return 0

    _repo_root = _find_repo_root()
    _default_data      = os.path.join(_repo_root, "data", "datasets", "fineweb10B_sp1024")
    _default_tokenizer = os.path.join(_repo_root, "data", "tokenizers", "fineweb_1024_bpe.model")

    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    _log_path = os.path.join(_script_dir, f"train_{_ts}.log")
    _log_fh = _setup_tee_logging(_log_path)
    print(f"[HDC] Logging to {_log_path}")

    parser = argparse.ArgumentParser(
        description="HDC Hash-Grad Stripped DSV-only pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",      type=str, default=_default_data)
    parser.add_argument("--tokenizer_path", type=str, default=_default_tokenizer)
    parser.add_argument("--max_time",       type=float, default=600.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--author",         type=str,   default="Ashley Klimpel")
    parser.add_argument("--github_id",      type=str,   default="viasky657")
    parser.add_argument("--run_name",       type=str,   default="HDC Hash-Grad 8xH100 Stripped")

    args = parser.parse_args()

    return _run_hash_grad_single(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
