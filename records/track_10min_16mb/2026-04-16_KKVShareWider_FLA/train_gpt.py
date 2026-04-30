#!/usr/bin/env python3
"""FLA / GatedDeltaNet entrypoint wrapper.

The actual training logic lives in `train_gdn_7k.py`. `evaluate.py` expects
`torchrun train_gpt.py`, so this wrapper preserves the standard repo entrypoint
while keeping the scored path in the records folder self-contained.
"""

import os
import sys
import traceback
from pathlib import Path

# These defaults keep the wrapper aligned with the intended SP8192 scored path.
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 8192))
DATA_PATH = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
ARCH_MODE = os.environ.get("ARCH_MODE", "K")
os.environ.setdefault("VOCAB_SIZE", str(VOCAB_SIZE))
os.environ.setdefault("DATA_PATH", DATA_PATH)
os.environ.setdefault("TOKENIZER_PATH", TOKENIZER_PATH)
os.environ.setdefault("ARCH_MODE", ARCH_MODE)
os.environ.setdefault("MAX_WALLCLOCK_SECONDS", "600")
os.environ.setdefault("VAL_LOSS_EVERY", "0")
os.environ.setdefault("EVAL_COMPILE_ENABLED", "0")
if ARCH_MODE in ("D", "G", "M"):
    os.environ.setdefault("XSA_EVAL", "1")


_VENDOR_DIR = Path(__file__).resolve().parent / ".fla_vendor"
_VENDOR_PKGS = [
    "triton==3.2.0",
    "flash-linear-attention==0.4.2",
    "fla-core==0.4.2",
    "transformers==5.5.4",
    "tokenizers==0.22.2",
    "safetensors==0.7.0",
]
if ARCH_MODE in ("F", "G"):
    _VENDOR_PKGS.extend(
        [
            "mamba-ssm==2.3.1",
            "causal-conv1d==1.6.1",
        ]
    )


def _ensure_vendor_on_path() -> None:
    p = str(_VENDOR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_fla_vendor_available() -> None:
    _ensure_vendor_on_path()
    try:
        if ARCH_MODE in ("F", "G"):
            from fla.layers.mamba2 import Mamba2  # noqa: F401
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401
            from causal_conv1d import causal_conv1d_fn  # noqa: F401
        else:
            from fla.layers.gated_deltanet import GatedDeltaNet  # noqa: F401
        print("wrapper: local vendored FLA imports already work", flush=True)
        return
    except Exception:
        vendor_pkgs = ", ".join(_VENDOR_PKGS)
        raise RuntimeError(
            "wrapper: required FLA deps are missing from the local environment. "
            f"Expected vendored packages under {_VENDOR_DIR}. "
            f"Install them before evaluation (e.g. via launcher/requirements), packages: {vendor_pkgs}"
        )


def main():
    _ensure_fla_vendor_available()
    print("wrapper: importing train_gdn_7k", flush=True)
    try:
        from train_gdn_7k import main as train_main
    except Exception:
        traceback.print_exc()
        raise
    print("wrapper: import ok, entering train_main", flush=True)
    train_main()


if __name__ == "__main__":
    main()
