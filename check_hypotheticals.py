"""Measure int8+zlib size of hypothetical exp variants without training them.

Strategy:
1. Load train_exp6 for the exp6 GPT + quantize pipeline.
2. For BigramHash addition: monkey-patch exp6's GPT with BigramHash from exp10,
   then build+randomize+measure the combined state_dict.
3. For exp9 wider/deeper: override Hyperparameters via env before import, then
   measure with the already-correct pipeline.
"""
from __future__ import annotations

import io
import os
import sys
import zlib
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent))

import check_sizes as _c  # noqa: E402  reuse helpers

LIMIT = 16 * 1024 * 1024


def measure_state_dict(state, quant_fn) -> tuple[int, int]:
    quant_obj, quant_stats = quant_fn(state)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    blob = zlib.compress(raw, level=9)
    return int(quant_stats["int8_payload_bytes"]), len(blob)


def fmtmb(n: int) -> str:
    return f"{n:>12,d} B ({n/1024/1024:6.3f} MB)"


def randomize(model: nn.Module) -> None:
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.randn(p.shape, generator=g, dtype=p.dtype)


def measure_exp6_plus_bigram(buckets: int, bh_dim: int) -> tuple[int, int, int]:
    """Build exp6 model + attach an exp10 BigramHash, measure combined."""
    # Fresh env, construct exp6 (9L+XSA+EMA).
    mod6 = _c.load_module("train_exp6")
    mod10 = _c.load_module("train_exp10")
    model = _c.build_model(mod6)  # exp6 GPT
    randomize(model)

    # Attach a BigramHash module alongside so its params flow into state_dict.
    bh = mod10.BigramHash(buckets, bh_dim, mod6.Hyperparameters.model_dim)
    randomize(bh)
    model.add_module("bigram_hash", bh)

    n = sum(p.numel() for p in model.parameters())
    state = model.state_dict()
    payload, z = measure_state_dict(state, mod6.quantize_state_dict_int8)
    return n, payload, z


def measure_exp6_vanilla() -> tuple[int, int, int]:
    mod6 = _c.load_module("train_exp6")
    model = _c.build_model(mod6)
    randomize(model)
    n = sum(p.numel() for p in model.parameters())
    state = model.state_dict()
    payload, z = measure_state_dict(state, mod6.quantize_state_dict_int8)
    return n, payload, z


def measure_exp6_with_layers(num_layers: int) -> tuple[int, int, int]:
    os.environ["NUM_LAYERS"] = str(num_layers)
    # Reload so Hyperparameters picks up new env.
    if "train_exp6" in sys.modules:
        del sys.modules["train_exp6"]
    n, pl, z = _c.measure("train_exp6")
    del os.environ["NUM_LAYERS"]
    return n, pl, z


def measure_exp9_wide(
    model_dim: int,
    num_unique: int,
    num_rounds: int,
    add_bigram: bool,
    bh_buckets: int = 4096,
    bh_dim: int = 128,
) -> tuple[int, int, int]:
    """exp9 with wider d, optional BigramHash. XSA/EMA add zero params so
    we don't need to actually implement them here — the artifact size is
    identical whether XSA/EMA are present or not.
    """
    os.environ["MODEL_DIM"] = str(model_dim)
    os.environ["NUM_UNIQUE_BLOCKS"] = str(num_unique)
    os.environ["NUM_ROUNDS"] = str(num_rounds)
    # model_dim must be divisible by num_heads (default 8) and kv_heads (4).
    # 512, 640, 704, 768 are all multiples of 8.
    if "train_exp9" in sys.modules:
        del sys.modules["train_exp9"]
    mod9 = _c.load_module("train_exp9")
    model = _c.build_model(mod9)
    randomize(model)
    if add_bigram:
        mod10 = _c.load_module("train_exp10")
        bh = mod10.BigramHash(bh_buckets, bh_dim, model_dim)
        randomize(bh)
        model.add_module("bigram_hash", bh)
    n = sum(p.numel() for p in model.parameters())
    state = model.state_dict()
    payload, z = measure_state_dict(state, mod9.quantize_state_dict_int8)
    for k in ("MODEL_DIM", "NUM_UNIQUE_BLOCKS", "NUM_ROUNDS"):
        os.environ.pop(k, None)
    return n, payload, z


def main() -> None:
    print(f"Budget: {LIMIT:,} B (16.000 MB)")
    print()

    print("=" * 72)
    print("exp6 family")
    print("=" * 72)

    n, pl, z = measure_exp6_vanilla()
    base_z = z
    print(f"  exp6 baseline (9L, no BH)        params={n:>12,}  size={fmtmb(z)}  margin={LIMIT-z:>+8,}  {'OK' if z<=LIMIT else 'OVER'}")

    # +1 layer
    n, pl, z = measure_exp6_with_layers(10)
    print(f"  exp6 + 1 layer (10L)              params={n:>12,}  size={fmtmb(z)}  margin={LIMIT-z:>+8,}  {'OK' if z<=LIMIT else 'OVER'}")

    # restore
    if "train_exp6" in sys.modules:
        del sys.modules["train_exp6"]

    # Various BigramHash sizes on top of exp6
    print()
    print("  exp6 + BigramHash variants:")
    for (b, d) in [(4096, 128), (2048, 128), (1024, 128), (1024, 64), (512, 128), (512, 64), (256, 128)]:
        n, pl, z = measure_exp6_plus_bigram(b, d)
        delta = z - base_z
        tag = "OK" if z <= LIMIT else "OVER"
        print(f"    BH({b:>4},{d:>3})  params={n:>12,}  size={fmtmb(z)}  Δ={delta:>+8,}  margin={LIMIT-z:>+9,}  {tag}")

    print()
    print("=" * 72)
    print("exp9 family (d=512 -> 768)")
    print("=" * 72)

    # Current baseline
    os.environ.pop("MODEL_DIM", None)
    if "train_exp9" in sys.modules:
        del sys.modules["train_exp9"]
    mod9 = _c.load_module("train_exp9")
    model = _c.build_model(mod9); randomize(model)
    n = sum(p.numel() for p in model.parameters()); state = model.state_dict()
    _, z = measure_state_dict(state, mod9.quantize_state_dict_int8)
    print(f"  exp9 baseline (d=512, 4x3, no BH)   params={n:>12,}  size={fmtmb(z)}  margin={LIMIT-z:>+8,}")

    configs = [
        ("d=768 4x3 no BH",                   768, 4, 3, False, 0, 0),
        ("d=768 4x3 + BH(4096,128)",          768, 4, 3, True,  4096, 128),
        ("d=768 4x3 + BH(2048, 64)",          768, 4, 3, True,  2048, 64),
        ("d=768 4x3 + BH(1024, 64)",          768, 4, 3, True,  1024, 64),
        ("d=768 3x3 + BH(4096,128)",          768, 3, 3, True,  4096, 128),
        ("d=768 3x4 + BH(4096,128)",          768, 3, 4, True,  4096, 128),
        ("d=704 4x3 + BH(4096,128)",          704, 4, 3, True,  4096, 128),
        ("d=640 4x3 + BH(4096,128)",          640, 4, 3, True,  4096, 128),
    ]
    for (label, d, u, r, bh, bb, bd) in configs:
        try:
            n, pl, z = measure_exp9_wide(d, u, r, bh, bb, bd)
        except Exception as e:
            print(f"  {label:<38}  ERROR: {type(e).__name__}: {e}")
            continue
        tag = "OK" if z <= LIMIT else "OVER"
        print(f"  {label:<38}  params={n:>12,}  size={fmtmb(z)}  margin={LIMIT-z:>+9,}  {tag}")


if __name__ == "__main__":
    main()
