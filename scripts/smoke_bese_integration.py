#!/usr/bin/env python3
"""
Smoke test (no GPU required): tokenizer JSON round-trip, byte identity, shard header I/O.

Run from repo root:
  .venv/bin/python scripts/smoke_bese_integration.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "tokenizer"
sys.path.insert(0, str(TOK_DIR))

from bese_bpe_tokenizer import BESEBPETokenizer, train_bpe_merges  # noqa: E402


def test_shard_roundtrip(shard_path: Path) -> None:
    import numpy as np

    header = np.fromfile(shard_path, dtype="<i4", count=256)
    assert int(header[0]) == 20240520, "magic"
    assert int(header[1]) == 1, "version"
    n = int(header[2])
    header_bytes = 256 * np.dtype("<i4").itemsize
    tokens = np.fromfile(shard_path, dtype="<u2", count=n, offset=header_bytes)
    assert tokens.size == n, "token count"


def main() -> int:
    sample = ROOT / "data" / "sample_docs.jsonl"
    texts = []
    with sample.open(encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    merges = train_bpe_merges(texts * 50, num_merges=32, verbose=False)
    tok = BESEBPETokenizer(merges=merges)
    bpt = tok.get_bytes_per_token_lut()
    for t in texts:
        enc = tok.encode(t)
        assert sum(bpt[x] for x in enc) == len(t.encode("utf-8")), "BPB bytes"

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        json_path = tdir / "tok.json"
        tok.save(json_path)
        tok2 = BESEBPETokenizer.load(json_path)
        assert tok2.vocab_size == tok.vocab_size

        # export_shards dry run via subprocess
        out = tdir / "ds"
        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_shards.py"),
                "--input",
                str(sample),
                "--tokenizer",
                str(json_path),
                "--output-dir",
                str(out),
                "--val-docs",
                "2",
                "--shard-tokens",
                "500",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return r.returncode
        val_bin = out / "fineweb_val_0.bin"
        assert val_bin.is_file(), "val shard"
        test_shard_roundtrip(val_bin)
        train_bins = list(out.glob("fineweb_train_*.bin"))
        assert train_bins, "train shards"
        test_shard_roundtrip(train_bins[0])

    print("smoke_bese_integration: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
