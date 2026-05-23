#!/usr/bin/env python3
"""Verify the goldens generator and bpb_compare.py agree.

Runs `per_byte_bits_diffusion` (with the *same* deterministic masks the
goldens generator used) and `per_byte_bits_ar` from bpb_compare.py and
asserts the per-byte BPB arrays are byte-identical to the goldens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "text_diffusion"))
sys.path.insert(0, str(REPO_ROOT / "visualization" / "scripts"))

import sentencepiece as spm
from bpb_compare import per_byte_bits_ar, per_byte_bits_diffusion  # noqa: E402

# Re-use loaders from generate_goldens
from generate_goldens import (  # noqa: E402
    GOLDENS_DIR, TOKENIZER_PATH, build_ar_model, build_diffusion_model, load_masks,
)


def check_mdlm(artifact_path: str) -> None:
    text = (GOLDENS_DIR / "transformer_abstract.txt").read_text(encoding="utf-8")
    text_bytes = text.encode("utf-8")
    n_bytes = len(text_bytes)

    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    token_ids = sp.encode_as_ids(text)
    n_tokens = len(token_ids)
    print(f"text: {n_bytes} bytes, {n_tokens} tokens")

    t_values, masks, eps = load_masks()
    K = t_values.shape[0]
    print(f"masks: K={K}, eps={eps}")

    model, label = build_diffusion_model(artifact_path)
    print(f"model: {label}")

    seq_len = max(1024, n_tokens)
    per_byte_compare = per_byte_bits_diffusion(
        model, token_ids, sp, n_bytes,
        seq_len=seq_len, K=K, eps=eps,
        t_values=t_values, masks=masks,
    )
    bpb_compare = float(np.mean(per_byte_compare))

    # Goldens
    g = json.loads((GOLDENS_DIR / "mdlm_per_byte_bpb.json").read_text())
    per_byte_golden = g["per_byte"]
    bpb_golden = g["bpb"]

    a = np.asarray(per_byte_compare, dtype=np.float64)
    b = np.asarray(per_byte_golden, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    diff = np.abs(a - b)
    print()
    print(f"per-byte max abs diff:   {diff.max():.3e}")
    print(f"per-byte mean abs diff:  {diff.mean():.3e}")
    print(f"BPB (bpb_compare path):  {bpb_compare:.6f}")
    print(f"BPB (goldens path):      {bpb_golden:.6f}")
    print(f"BPB diff:                {abs(bpb_compare - bpb_golden):.3e}")

    # Tighter bar: float64 reductions of the same float32 sums should agree to ~1e-12
    assert diff.max() < 1e-9, "per-byte BPB drift exceeds float64 reduction noise"
    print("✓ MDLM goldens match bpb_compare.per_byte_bits_diffusion exactly\n")


def check_ar(artifact_path: str) -> None:
    text = (GOLDENS_DIR / "transformer_abstract.txt").read_text(encoding="utf-8")
    text_bytes = text.encode("utf-8")
    n_bytes = len(text_bytes)
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    token_ids = sp.encode_as_ids(text)

    model, label = build_ar_model(artifact_path)
    print(f"== AR check ==")
    print(f"model: {label}")
    seq_len = max(1024, len(token_ids))
    per_byte_compare = per_byte_bits_ar(model, token_ids, sp, n_bytes, seq_len=seq_len)
    bpb_compare = float(np.mean(per_byte_compare))

    g = json.loads((GOLDENS_DIR / "ar_per_byte_bpb.json").read_text())
    per_byte_golden = g["per_byte"]
    bpb_golden = g["bpb"]

    a = np.asarray(per_byte_compare, dtype=np.float64)
    b = np.asarray(per_byte_golden, dtype=np.float64)
    diff = np.abs(a - b)
    print(f"per-byte max abs diff:   {diff.max():.3e}")
    print(f"BPB (bpb_compare path):  {bpb_compare:.6f}")
    print(f"BPB (goldens path):      {bpb_golden:.6f}")
    print(f"BPB diff:                {abs(bpb_compare - bpb_golden):.3e}")
    assert diff.max() < 1e-9, "AR per-byte BPB drift exceeds float64 reduction noise"
    print("✓ AR goldens match bpb_compare.per_byte_bits_ar exactly\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mdlm-artifact",
                   default="artifacts/mdlm-submission-seed42-19M.ptz")
    p.add_argument("--ar-artifact",
                   default="artifacts/ar-baseline-9L-512d.ptz")
    args = p.parse_args()
    if args.mdlm_artifact:
        check_mdlm(args.mdlm_artifact)
    if args.ar_artifact:
        check_ar(args.ar_artifact)
