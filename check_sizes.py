"""
check_sizes.py: dummy-init each train_expN.py model and measure int8+zlib size.

For every variant file we build the model on CPU using its own GPT class +
Hyperparameters, run the same quantize_state_dict_int8 + zlib pipeline as the
training script, and report the resulting on-disk size.

This is meant to catch budget regressions BEFORE launching a real run — the
16 MB challenge limit is on the int8+zlib artifact, so estimates based on
fp32 param counts can be misleading (zlib ratio depends on the tensor shapes).

Run:
  python check_sizes.py
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import sys
import zlib
from pathlib import Path

import torch

LIMIT_BYTES = 16 * 1024 * 1024

VARIANTS = [
    "train_exp6",
    "train_exp7",
    "train_exp8",
    "train_exp9",
    "train_exp10",
]


def load_module(name: str):
    """Import a train_expN.py file by absolute path so we don't depend on cwd."""
    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to spec_from_file_location for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(mod):
    """Construct a GPT instance from the module's Hyperparameters.

    Uses inspect to match GPT.__init__ parameter names against Hyperparameters
    attributes, so it works for variants with different signatures (exp9's
    num_unique_blocks/num_rounds, exp10's bigram_hash_*, exp6/7's xsa_last_n).
    """
    H = mod.Hyperparameters()
    sig = inspect.signature(mod.GPT.__init__).parameters
    kwargs = {}
    for name in sig:
        if name == "self":
            continue
        if hasattr(H, name):
            kwargs[name] = getattr(H, name)
        else:
            raise RuntimeError(
                f"GPT.__init__ requires '{name}' but Hyperparameters has no such attribute"
            )
    return mod.GPT(**kwargs)


def randomize_for_size_estimate(model: torch.nn.Module) -> None:
    """Fill every parameter with high-entropy random noise.

    Default model construction leaves zero-initialized projections (e.g.
    Block.proj, lm_head, BigramHash.proj) at exactly zero, which zlib
    compresses to almost nothing. The result is a wildly optimistic size
    estimate. To reflect what a *trained* model produces, we overwrite every
    parameter with N(0, 1) noise, which after int8 quantization yields a
    similar entropy profile to a real training run.
    """
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.randn(p.shape, generator=g, dtype=p.dtype)


def measure(mod_name: str) -> tuple[int, int, int]:
    """Return (param_count, int8_payload_bytes, zlib_compressed_bytes)."""
    mod = load_module(mod_name)
    model = build_model(mod)
    randomize_for_size_estimate(model)
    n_params = sum(int(p.numel()) for p in model.parameters())

    # Use the file's own quantize function so we measure exactly what the
    # training script would write.
    state = model.state_dict()
    quant_obj, quant_stats = mod.quantize_state_dict_int8(state)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    blob = zlib.compress(raw, level=9)
    return n_params, int(quant_stats["int8_payload_bytes"]), len(blob)


def fmt_bytes(n: int) -> str:
    return f"{n:>11,d} B  ({n / (1024 * 1024):6.2f} MB)"


def main() -> int:
    print(f"check_sizes.py — int8+zlib budget: {fmt_bytes(LIMIT_BYTES)}")
    print()
    header = f"{'variant':<14} {'params':>12}  {'int8 payload':>22}  {'int8+zlib (final)':>22}  status"
    print(header)
    print("-" * len(header))

    overall_ok = True
    for name in VARIANTS:
        try:
            n_params, payload_bytes, zlib_bytes = measure(name)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<14} ERROR: {type(e).__name__}: {e}")
            overall_ok = False
            continue

        status = "OK " if zlib_bytes <= LIMIT_BYTES else "OVER"
        if zlib_bytes > LIMIT_BYTES:
            overall_ok = False
        print(
            f"{name:<14} {n_params:>12,}  {fmt_bytes(payload_bytes)}  {fmt_bytes(zlib_bytes)}  {status}"
        )

    print()
    print("PASS" if overall_ok else "FAIL — at least one variant exceeds 16 MB")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
