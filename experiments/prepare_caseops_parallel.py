"""Parallel CaseOps tokenizer — same output as prepare_caseops_data.py but uses
multiprocessing to saturate all CPU cores.  Typical speedup: ~Nx on N cores.

Strategy
--------
1. Read docs_selected.jsonl once (main process), dispatch docs to a pool of
   workers via a shared Queue.
2. First VAL_DOCS docs go to a dedicated val worker (single-threaded, must be
   sequential to preserve doc order for reproducibility).
3. Remaining docs are distributed round-robin across NWORKERS train workers.
   Each worker writes its own numbered temp shards under <out_dir>/tmp_worker_<i>/.
4. After all workers finish, main process renumbers and moves shards into the
   final output directory.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import shutil
import sys

import numpy as np
import sentencepiece as spm

SHARD_MAGIC   = 20240520
SHARD_VERSION = 1
SHARD_TOKENS  = 10_000_000
BOS_ID        = 1
VAL_DOCS      = 10_000
DATASET_NAME  = "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"


def _write_shard(path: pathlib.Path, arr: np.ndarray) -> None:
    assert arr.dtype == np.uint16
    header = np.zeros(256, dtype=np.int32)
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = int(arr.size)
    with path.open("wb") as fh:
        fh.write(header.tobytes())
        fh.write(arr.tobytes())


def _val_worker(sp_path: str, docs: list[str], out_dir: pathlib.Path) -> None:
    """Process exactly VAL_DOCS docs and write val + val_bytes shards."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                           "records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611"))
    from lossless_caps import (
        LOSSLESS_CAPS_CASEOPS_V1,
        encode_lossless_caps_v2,
        surface_piece_original_byte_counts,
    )
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    val_tok: list[int] = []
    val_bytes: list[int] = []
    val_written = 0
    for i, text in enumerate(docs):
        transformed = encode_lossless_caps_v2(text)
        ids = [BOS_ID] + sp.encode(transformed, out_type=int)
        proto = sp.encode_as_immutable_proto(transformed)
        bc = surface_piece_original_byte_counts(
            (p.surface for p in proto.pieces),
            text_transform_name=LOSSLESS_CAPS_CASEOPS_V1,
        )
        val_tok.extend(ids)
        val_bytes.append(0)
        val_bytes.extend(int(b) for b in bc)
        while len(val_tok) >= SHARD_TOKENS:
            _write_shard(
                out_dir / f"fineweb_val_{val_written:06d}.bin",
                np.array(val_tok[:SHARD_TOKENS], dtype=np.uint16),
            )
            _write_shard(
                out_dir / f"fineweb_val_bytes_{val_written:06d}.bin",
                np.array(val_bytes[:SHARD_TOKENS], dtype=np.uint16),
            )
            val_tok   = val_tok[SHARD_TOKENS:]
            val_bytes = val_bytes[SHARD_TOKENS:]
            val_written += 1
        if (i + 1) % 1000 == 0:
            print(f"  val: {i+1}/{len(docs)} docs", flush=True)
    if val_tok:
        _write_shard(out_dir / f"fineweb_val_{val_written:06d}.bin",
                     np.array(val_tok, dtype=np.uint16))
        _write_shard(out_dir / f"fineweb_val_bytes_{val_written:06d}.bin",
                     np.array(val_bytes, dtype=np.uint16))
        val_written += 1
    print(f"  val done: {val_written} shard(s)", flush=True)


def _train_worker(worker_id: int, sp_path: str, docs: list[str],
                  tmp_dir: pathlib.Path) -> int:
    """Tokenize a batch of train docs; return number of shards written."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent /
                           "records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611"))
    from lossless_caps import encode_lossless_caps_v2
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    buf: list[int] = []
    written = 0
    report_every = max(1, len(docs) // 20)
    for i, text in enumerate(docs):
        transformed = encode_lossless_caps_v2(text)
        ids = [BOS_ID] + sp.encode(transformed, out_type=int)
        buf.extend(ids)
        while len(buf) >= SHARD_TOKENS:
            _write_shard(
                tmp_dir / f"shard_{written:06d}.bin",
                np.array(buf[:SHARD_TOKENS], dtype=np.uint16),
            )
            buf = buf[SHARD_TOKENS:]
            written += 1
        if (i + 1) % report_every == 0:
            print(f"  worker {worker_id}: {i+1}/{len(docs)} docs  shards={written}", flush=True)
    if buf:
        _write_shard(tmp_dir / f"shard_{written:06d}.bin",
                     np.array(buf, dtype=np.uint16))
        written += 1
    print(f"  worker {worker_id}: done  shards={written}", flush=True)
    return written


def _worker_entry(args):
    worker_id, sp_path, docs, tmp_dir = args
    return _train_worker(worker_id, sp_path, docs, pathlib.Path(tmp_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs",    required=True, type=pathlib.Path)
    ap.add_argument("--out",     required=True, type=pathlib.Path)
    ap.add_argument("--sp",      required=True, type=pathlib.Path)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--val-docs", type=int, default=VAL_DOCS)
    args = ap.parse_args()

    out_dir = args.out / "datasets" / DATASET_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_path = str(args.sp)
    nworkers = max(1, args.workers)
    val_docs_count = args.val_docs

    print(f"Reading docs from {args.docs} ...", flush=True)
    print(f"Using {nworkers} train workers + 1 val worker", flush=True)

    # Read all docs into memory (48 GB raw → only text strings, typically ~15-20 GB RAM).
    val_docs: list[str] = []
    worker_docs: list[list[str]] = [[] for _ in range(nworkers)]
    n_total = 0
    with args.docs.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj["text"] if isinstance(obj, dict) else obj
            if n_total < val_docs_count:
                val_docs.append(text)
            else:
                worker_docs[(n_total - val_docs_count) % nworkers].append(text)
            n_total += 1
            if n_total % 500_000 == 0:
                print(f"  read {n_total} docs", flush=True)

    print(f"Total docs: {n_total}  val={len(val_docs)}  "
          f"train={sum(len(d) for d in worker_docs)}", flush=True)

    # Val worker (in-process, quick).
    print("Generating val shard ...", flush=True)
    _val_worker(sp_path, val_docs, out_dir)

    # Train workers (parallel).
    tmp_dirs: list[pathlib.Path] = []
    pool_args = []
    for i, docs in enumerate(worker_docs):
        tmp = out_dir / f"_tmp_worker_{i}"
        tmp.mkdir(exist_ok=True)
        tmp_dirs.append(tmp)
        pool_args.append((i, sp_path, docs, str(tmp)))

    print(f"Starting {nworkers} parallel train workers ...", flush=True)
    with mp.Pool(processes=nworkers) as pool:
        shard_counts = pool.map(_worker_entry, pool_args)

    # Merge: renumber shards sequentially.
    print("Merging shards ...", flush=True)
    global_idx = 0
    for i, (tmp, count) in enumerate(zip(tmp_dirs, shard_counts)):
        for j in range(count):
            src = tmp / f"shard_{j:06d}.bin"
            dst = out_dir / f"fineweb_train_{global_idx:06d}.bin"
            shutil.move(str(src), str(dst))
            global_idx += 1
        tmp.rmdir()

    print(f"Done. train_shards={global_idx} in {out_dir}", flush=True)
    val_shards = len(list(out_dir.glob("fineweb_val_[0-9]*.bin")))
    print(f"val_shards={val_shards}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
