#!/usr/bin/env python3
"""Run 1: count boundary-token predecessors in fineweb val.

Empirically verifies the methodology.md section 4 claim that
is_boundary[x_prev] is identically zero on this val stream.

Uses the same val-loading convention as scripts/canonical_rescore.py:
256 int32 header, then uint16 tokens.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sentencepiece as spm

TOKENIZER_PATH = "/workspace/parameter-golf/data/tokenizers/fineweb_8192_bpe.model"
VAL_DATA_GLOB = "/workspace/parameter-golf/data/datasets/fineweb10B_sp8192/fineweb_val_*.bin"
OUTPUT_JSON = "/workspace/agent-pgolf/audit/empirical_validation/run1_boundary_mask_check.json"

HEADER_BYTES = 256 * 4  # 256 little-endian int32 fields = 1024 bytes


def load_val_shard(path):
    """Load one shard. Header is 256 int32; first 3 are magic/version/n_tokens."""
    header = np.fromfile(path, dtype="<i4", count=256)
    magic, version, n = int(header[0]), int(header[1]), int(header[2])
    if magic != 20240520:
        raise ValueError(f"Bad magic in {path}: {magic}")
    toks = np.fromfile(path, dtype="<u2", count=n, offset=HEADER_BYTES)
    return toks


def main():
    print(f"[run1] Loading tokenizer from {TOKENIZER_PATH}")
    sp = spm.SentencePieceProcessor()
    sp.Load(TOKENIZER_PATH)
    vocab_size = sp.GetPieceSize()
    print(f"[run1] vocab_size = {vocab_size}")

    print(f"[run1] Computing per-token boundary flags...")
    is_control = np.zeros(vocab_size, dtype=bool)
    is_unknown = np.zeros(vocab_size, dtype=bool)
    is_unused = np.zeros(vocab_size, dtype=bool)
    for tid in range(vocab_size):
        is_control[tid] = sp.IsControl(tid)
        is_unknown[tid] = sp.IsUnknown(tid)
        is_unused[tid] = sp.IsUnused(tid)

    vocab_n_control = int(is_control.sum())
    vocab_n_unknown = int(is_unknown.sum())
    vocab_n_unused = int(is_unused.sum())
    print(f"[run1] vocab: {vocab_n_control} control, {vocab_n_unknown} unknown, {vocab_n_unused} unused")

    val_files = sorted(Path("/workspace/parameter-golf/data/datasets/fineweb10B_sp8192").glob("fineweb_val_*.bin"))
    print(f"[run1] Loading {len(val_files)} val shard(s)")
    arrays = []
    for f in val_files:
        a = load_val_shard(f)
        print(f"[run1]   {f.name}: {a.shape[0]:,} tokens (max id={int(a.max())}, min id={int(a.min())})")
        arrays.append(a)
    val_tokens = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    total_tokens = int(val_tokens.shape[0])
    print(f"[run1] total val tokens: {total_tokens:,}")

    # Sanity: all token ids must be in [0, vocab_size)
    actual_max = int(val_tokens.max())
    actual_min = int(val_tokens.min())
    if actual_max >= vocab_size or actual_min < 0:
        raise ValueError(f"Token id out of vocab range: min={actual_min}, max={actual_max}, vocab={vocab_size}")

    predecessor_tokens = val_tokens[:-1]
    total_predecessor = int(predecessor_tokens.shape[0])
    print(f"[run1] predecessor tokens: {total_predecessor:,}")

    pred_n_control = int(is_control[predecessor_tokens].sum())
    pred_n_unknown = int(is_unknown[predecessor_tokens].sum())
    pred_n_unused = int(is_unused[predecessor_tokens].sum())
    pred_n_any = int((is_control | is_unknown | is_unused)[predecessor_tokens].sum())

    print(f"[run1] predecessor counts: control={pred_n_control}, unknown={pred_n_unknown}, unused={pred_n_unused}")
    print(f"[run1] predecessor_n_any_boundary = {pred_n_any}")
    print(f"[run1] boundary_mask_is_no_op = {pred_n_any == 0}")

    result = {
        "total_predecessor_tokens": total_predecessor,
        "vocab_size": vocab_size,
        "vocab_n_control": vocab_n_control,
        "vocab_n_unknown": vocab_n_unknown,
        "vocab_n_unused": vocab_n_unused,
        "predecessor_n_control": pred_n_control,
        "predecessor_n_unknown": pred_n_unknown,
        "predecessor_n_unused": pred_n_unused,
        "predecessor_n_any_boundary": pred_n_any,
        "boundary_mask_is_no_op": (pred_n_any == 0),
        "tokenizer_path": TOKENIZER_PATH,
        "val_data_glob": VAL_DATA_GLOB,
        "n_val_shards_loaded": len(val_files),
        "val_token_id_min": actual_min,
        "val_token_id_max": actual_max,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[run1] Wrote {OUTPUT_JSON}")
    print(f"[run1] DONE")


if __name__ == "__main__":
    main()
