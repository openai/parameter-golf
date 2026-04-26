"""Sprint 001 smoke harness.

Validates that numpy removal in train_gpt.py left load_data_shard and
build_sentencepiece_luts functional. No torchrun, no full training.

Run from the repo root:
    python smoke_sprint001.py

Exit code 0 on full pass; nonzero on any failure.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN_SHARD = REPO / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_train_000000.bin"
VAL_SHARD = REPO / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_val_000000.bin"
TOKENIZER = REPO / "data" / "tokenizers" / "fineweb_1024_bpe.model"


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def check_import() -> object:
    print("[1/4] importing train_gpt ...")
    t0 = time.perf_counter()
    sys.path.insert(0, str(REPO))
    import train_gpt  # noqa: E402
    elapsed = time.perf_counter() - t0
    _ok(f"imported in {elapsed:.2f}s")

    # The sprint deliverable: train_gpt itself exposes no numpy alias.
    # Scoring counts train_gpt.py bytes, not transitive deps' load behavior.
    if hasattr(train_gpt, "np") or hasattr(train_gpt, "numpy"):
        _fail("train_gpt exposes a numpy alias — Sprint 001 incomplete")
        sys.exit(2)
    _ok("train_gpt has no numpy alias attribute")

    # Informational: surface transitive numpy load if any. Not a failure.
    if "numpy" in sys.modules:
        print("  INFO  numpy is loaded transitively (via torch/HF deps), not by train_gpt — OK")
    else:
        _ok("numpy is absent from sys.modules entirely")
    return train_gpt


def check_load_data_shard(train_gpt) -> None:
    print("[2/4] load_data_shard on validation shard ...")
    if not VAL_SHARD.exists():
        _fail(f"missing {VAL_SHARD}")
        sys.exit(3)

    t0 = time.perf_counter()
    tokens = train_gpt.load_data_shard(VAL_SHARD)
    elapsed = time.perf_counter() - t0

    import torch
    if not isinstance(tokens, torch.Tensor):
        _fail(f"return type {type(tokens)!r} is not a torch.Tensor")
        sys.exit(3)
    if tokens.dtype != torch.uint16:
        _fail(f"dtype {tokens.dtype} != torch.uint16")
        sys.exit(3)
    if tokens.numel() == 0:
        _fail("empty token tensor")
        sys.exit(3)

    expected = (VAL_SHARD.stat().st_size - 256 * 4) // 2
    if tokens.numel() != expected:
        _fail(f"token count {tokens.numel()} != expected {expected}")
        sys.exit(3)

    _ok(f"{tokens.numel():,} uint16 tokens loaded in {elapsed:.2f}s")
    # uint16 has no min/max kernel in torch — cast for the diagnostic only.
    diag = tokens.to(torch.int32)
    _ok(f"min={int(diag.min())} max={int(diag.max())}")


def check_build_luts(train_gpt) -> None:
    print("[3/4] build_sentencepiece_luts ...")
    if not TOKENIZER.exists():
        _fail(f"missing {TOKENIZER}")
        sys.exit(4)

    import sentencepiece as spm
    import torch

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER))
    vocab_size = int(sp.vocab_size())
    device = torch.device("cpu")

    t0 = time.perf_counter()
    luts = train_gpt.build_sentencepiece_luts(sp, vocab_size, device)
    elapsed = time.perf_counter() - t0

    if not isinstance(luts, tuple) or len(luts) != 3:
        _fail(f"expected 3-tuple, got {type(luts).__name__} of len {len(luts) if hasattr(luts, '__len__') else 'n/a'}")
        sys.exit(4)
    for i, t in enumerate(luts):
        if not isinstance(t, torch.Tensor):
            _fail(f"LUT[{i}] type {type(t)!r} is not a torch.Tensor")
            sys.exit(4)
        if t.device.type != "cpu":
            _fail(f"LUT[{i}] on device {t.device}, expected cpu for this smoke")
            sys.exit(4)

    _ok(f"3 tensors built in {elapsed:.2f}s, vocab_size={vocab_size}")
    _ok(f"shapes: {[tuple(t.shape) for t in luts]}")
    _ok(f"dtypes: {[str(t.dtype) for t in luts]}")


def check_train_shard_size(train_gpt) -> None:
    print("[4/4] load_data_shard on full train shard ...")
    if not TRAIN_SHARD.exists():
        _fail(f"missing {TRAIN_SHARD}")
        sys.exit(5)

    t0 = time.perf_counter()
    tokens = train_gpt.load_data_shard(TRAIN_SHARD)
    elapsed = time.perf_counter() - t0
    expected = (TRAIN_SHARD.stat().st_size - 256 * 4) // 2

    if tokens.numel() != expected:
        _fail(f"train token count {tokens.numel()} != expected {expected}")
        sys.exit(5)
    _ok(f"{tokens.numel():,} tokens loaded in {elapsed:.2f}s")


def main() -> int:
    print("=" * 60)
    print("Sprint 001 smoke: numpy-free train_gpt.py verification")
    print("=" * 60)
    train_gpt = check_import()
    check_load_data_shard(train_gpt)
    check_build_luts(train_gpt)
    check_train_shard_size(train_gpt)
    print("=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
