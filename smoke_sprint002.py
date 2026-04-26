"""Sprint 002 smoke harness.

Validates that the ablation harness toggles in train_gpt.py resolve to the
expected branches without launching training. No CUDA, no torchrun.

Toggles covered:
  A1  QUANTIZE_WEIGHTS={int8|none}        (artifact path + dequant branch)
  A2  NUM_KV_HEADS                        (Hyperparameters field)
  A3  SDPA_BACKEND={flash|math|cudnn|mem_efficient}
  A4  OPTIMIZER={muon|adamw}
  A5  QUANT_SCHEME={per_row|per_tensor}   (exercises quantize_float_tensor directly)
  A6  TIE_EMBEDDINGS                      (Hyperparameters field)

What this smoke does NOT do:
  - launch CUDA training (those toggles are exercised end-to-end on the burst H100)
  - validate numerical equivalence between roundtrips

Run from the repo root:
    python smoke_sprint002.py

Exit code 0 on full pass; nonzero on any failure.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
TRAIN_GPT_SRC = REPO / "train_gpt.py"


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)


def _reload_train_gpt():
    """Reload train_gpt so module-level imports re-evaluate. Hyperparameters
    reads env vars at *class definition* time, so a stale import would lock in
    whatever environment the previous reload saw."""
    sys.path.insert(0, str(REPO))
    if "train_gpt" in sys.modules:
        return importlib.reload(sys.modules["train_gpt"])
    return importlib.import_module("train_gpt")


def check_source_tokens() -> None:
    """Cheap sanity check that the four new toggles still appear in source."""
    print("[1/6] source-level toggle presence ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    expected = [
        ('os.environ.get("QUANTIZE_WEIGHTS"', "A1 QUANTIZE_WEIGHTS"),
        ('os.environ.get("QUANT_SCHEME"', "A5 QUANT_SCHEME"),
        ('os.environ.get("SDPA_BACKEND"', "A3 SDPA_BACKEND"),
        ('os.environ.get("OPTIMIZER"', "A4 OPTIMIZER"),
    ]
    for needle, label in expected:
        if needle not in src:
            _fail(f"missing {label} env-var read in train_gpt.py")
            sys.exit(2)
        _ok(f"{label} env-var read present")


def check_hyperparameters_a2_a6() -> None:
    """A2 (NUM_KV_HEADS) and A6 (TIE_EMBEDDINGS) flip Hyperparameters fields directly."""
    print("[2/6] Hyperparameters env-var resolution (A2, A6) ...")

    # Default.
    for k in ("NUM_KV_HEADS", "TIE_EMBEDDINGS", "NUM_HEADS"):
        os.environ.pop(k, None)
    train_gpt = _reload_train_gpt()
    h = train_gpt.Hyperparameters()
    if h.num_kv_heads != 4 or h.num_heads != 8 or h.tie_embeddings is not True:
        _fail(f"defaults wrong: kv={h.num_kv_heads} heads={h.num_heads} tie={h.tie_embeddings}")
        sys.exit(3)
    _ok(f"defaults: num_heads=8, num_kv_heads=4, tie_embeddings=True")

    # A2: NUM_KV_HEADS=8 (MHA, equal to num_heads).
    os.environ["NUM_KV_HEADS"] = "8"
    train_gpt = _reload_train_gpt()
    h = train_gpt.Hyperparameters()
    if h.num_kv_heads != 8:
        _fail(f"A2 NUM_KV_HEADS=8 not picked up, got {h.num_kv_heads}")
        sys.exit(3)
    _ok("A2: NUM_KV_HEADS=8 -> Hyperparameters.num_kv_heads = 8 (MHA)")
    os.environ.pop("NUM_KV_HEADS", None)

    # A6: TIE_EMBEDDINGS=0.
    os.environ["TIE_EMBEDDINGS"] = "0"
    train_gpt = _reload_train_gpt()
    h = train_gpt.Hyperparameters()
    if h.tie_embeddings is not False:
        _fail(f"A6 TIE_EMBEDDINGS=0 not picked up, got {h.tie_embeddings}")
        sys.exit(3)
    _ok("A6: TIE_EMBEDDINGS=0 -> Hyperparameters.tie_embeddings = False")
    os.environ.pop("TIE_EMBEDDINGS", None)


def check_quant_scheme_a5() -> None:
    """A5: exercise quantize_float_tensor directly under both schemes."""
    print("[3/6] QUANT_SCHEME branch in quantize_float_tensor (A5) ...")
    import torch

    os.environ.pop("QUANT_SCHEME", None)
    train_gpt = _reload_train_gpt()

    t = torch.randn(8, 16)  # 2D matrix

    # per_row default
    q_row, s_row = train_gpt.quantize_float_tensor(t)
    if q_row.dtype != torch.int8 or q_row.shape != t.shape:
        _fail(f"per_row q shape/dtype: {q_row.shape} {q_row.dtype}")
        sys.exit(4)
    if s_row.ndim != 1 or s_row.numel() != t.shape[0]:
        _fail(f"per_row scale should be 1D of len {t.shape[0]}, got ndim={s_row.ndim} numel={s_row.numel()}")
        sys.exit(4)
    _ok(f"per_row default: q={tuple(q_row.shape)}/int8, scale={tuple(s_row.shape)} ({s_row.dtype})")

    # per_tensor branch
    os.environ["QUANT_SCHEME"] = "per_tensor"
    q_tens, s_tens = train_gpt.quantize_float_tensor(t)
    if q_tens.dtype != torch.int8 or q_tens.shape != t.shape:
        _fail(f"per_tensor q shape/dtype: {q_tens.shape} {q_tens.dtype}")
        sys.exit(4)
    if s_tens.ndim != 0:
        _fail(f"per_tensor scale should be 0-dim scalar, got ndim={s_tens.ndim}")
        sys.exit(4)
    _ok(f"per_tensor: q={tuple(q_tens.shape)}/int8, scale=scalar ({s_tens.dtype})")

    # Bad value rejected
    os.environ["QUANT_SCHEME"] = "bogus"
    try:
        train_gpt.quantize_float_tensor(t)
    except ValueError as e:
        if "QUANT_SCHEME" not in str(e):
            _fail(f"unexpected ValueError text: {e}")
            sys.exit(4)
        _ok("bogus QUANT_SCHEME rejected with ValueError")
    else:
        _fail("bogus QUANT_SCHEME silently accepted")
        sys.exit(4)
    os.environ.pop("QUANT_SCHEME", None)


def check_main_inline_validators() -> None:
    """A1 (QUANTIZE_WEIGHTS), A3 (SDPA_BACKEND), A4 (OPTIMIZER) are validated
    inline in main(). We can't run main without CUDA, but we can scan the source
    to confirm the fail-fast `not in {...}` guards exist for each."""
    print("[4/6] inline validator presence for A1, A3, A4 ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    patterns = [
        (r'QUANTIZE_WEIGHTS.*?{[^}]*"int8"[^}]*"none"[^}]*}', "A1 QUANTIZE_WEIGHTS={int8,none}"),
        (r'SDPA_BACKEND.*?{[^}]*"flash"[^}]*"math"[^}]*}', "A3 SDPA_BACKEND={flash,math,...}"),
        (r'OPTIMIZER.*?{[^}]*"muon"[^}]*"adamw"[^}]*}', "A4 OPTIMIZER={muon,adamw}"),
    ]
    for pattern, label in patterns:
        if not re.search(pattern, src, flags=re.DOTALL):
            _fail(f"missing inline guard for {label}")
            sys.exit(5)
        _ok(f"{label} guard present")


def check_artifact_path_branch() -> None:
    """A1 changes the artifact filename. Confirm both candidates appear in source."""
    print("[5/6] artifact path branch (A1) ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    if "final_model.int8.ptz" not in src:
        _fail("int8 artifact path missing")
        sys.exit(6)
    if "final_model.raw.ptz" not in src:
        _fail("raw artifact path missing")
        sys.exit(6)
    _ok("both artifact paths reachable in source")


def check_momentum_warmup_gated() -> None:
    """A4: Muon momentum warmup ramp must be gated on optimizer_choice == 'muon'."""
    print("[6/6] Muon momentum-warmup ramp gated on optimizer_choice (A4) ...")
    src = TRAIN_GPT_SRC.read_text(encoding="utf-8")
    pattern = r'if optimizer_choice == "muon":\s*\n\s*frac = min\(step / args\.muon_momentum_warmup_steps'
    if not re.search(pattern, src):
        _fail("momentum warmup ramp is not gated; AdamW path would dirty param groups")
        sys.exit(7)
    _ok("momentum warmup ramp gated correctly")


def main() -> int:
    print("=" * 60)
    print("Sprint 002 smoke: ablation harness toggle verification")
    print("=" * 60)
    t0 = time.perf_counter()
    check_source_tokens()
    check_hyperparameters_a2_a6()
    check_quant_scheme_a5()
    check_main_inline_validators()
    check_artifact_path_branch()
    check_momentum_warmup_gated()
    print("=" * 60)
    print(f"ALL CHECKS PASSED in {time.perf_counter() - t0:.2f}s")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
