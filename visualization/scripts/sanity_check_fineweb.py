#!/usr/bin/env python3
"""Sanity-check that AR + MDLM eval on a FineWeb val slice reproduces the
training-time BPB (≈1.23 for AR, ≈1.48 for MDLM).

Bypasses re-tokenization: feeds the raw token IDs from the val shard directly
to the model, then computes bits/byte using the SP-decoded byte count.

Usage:
  .venv/bin/python visualization/scripts/sanity_check_fineweb.py \\
      --ar-artifact artifacts/ar-baseline-9L-512d.ptz \\
      --mdlm-artifact artifacts/mdlm-submission-seed42-19M.ptz \\
      --n-tokens 2000 --K 64
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "text_diffusion"))
sys.path.insert(0, str(REPO_ROOT / "visualization" / "scripts"))

import sentencepiece as spm
from generate_goldens import build_ar_model, build_diffusion_model, TOKENIZER_PATH

VAL_SHARD = REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_val_000000.bin"


def load_val_tokens(n_tokens: int, skip: int = 0) -> tuple[list[int], int]:
    """Read `n_tokens` from val shard starting at offset `skip`.

    Returns (token_ids, n_decoded_utf8_bytes).
    """
    header_bytes = 256 * np.dtype("<i4").itemsize
    sample = np.fromfile(VAL_SHARD, dtype="<u2",
                         count=n_tokens, offset=header_bytes + skip * 2)
    token_ids = sample.astype(np.int64).tolist()
    sp = spm.SentencePieceProcessor(model_file=str(TOKENIZER_PATH))
    text = sp.decode_ids(token_ids)
    return token_ids, len(text.encode("utf-8")), text


@torch.inference_mode()
def ar_bpb(model, token_ids: list[int], n_bytes: int) -> float:
    """Per-byte AR BPB, mirror of bpb_compare.per_byte_bits_ar's recipe."""
    n = len(token_ids)
    ids = torch.tensor(token_ids[:n - 1], dtype=torch.int64).unsqueeze(0)
    logits = model.forward_logits(ids)  # [1, n-1, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)
    targets = torch.tensor(token_ids[1:], dtype=torch.int64)
    nll = -log_probs[torch.arange(n - 1), targets]  # nats
    total_nats = float(nll.sum().item()) + math.log(1024)  # +log(V) for pos 0
    total_bits = total_nats / math.log(2)
    return total_bits / n_bytes


@torch.inference_mode()
def mdlm_bpb(model, token_ids: list[int], n_bytes: int, K: int, eps: float,
             seed: int) -> float:
    """Per-byte MDLM NELBO, mirror of bpb_compare.per_byte_bits_diffusion."""
    n = len(token_ids)
    ids = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0)
    g = torch.Generator().manual_seed(seed)
    t_values = torch.linspace(eps, 1.0, K)
    scale = (1 - eps) / K
    total_nats = 0.0
    for t in t_values:
        tv = float(t.item())
        mask = (torch.rand(1, n, generator=g) < tv)
        if tv < 0.01:
            mask[0, 0] = True
        logits = model.forward_logits(ids, mask)  # [1, n, V]
        log_probs = F.log_softmax(logits[0].float(), dim=-1)
        nll = -log_probs.gather(-1, ids[0].unsqueeze(-1)).squeeze(-1)  # [n]
        weight = scale / tv
        # Sum over masked positions only
        total_nats += weight * float(nll[mask[0]].sum().item())
    total_bits = total_nats / math.log(2)
    return total_bits / n_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ar-artifact", default="artifacts/ar-baseline-9L-512d.ptz")
    p.add_argument("--mdlm-artifact",
                   default="artifacts/mdlm-submission-seed42-19M.ptz")
    p.add_argument("--n-tokens", type=int, default=2000)
    p.add_argument("--skip-tokens", type=int, default=1,
                   help="Skip first N tokens; default 1 to skip the <s> BOS.")
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--save-text",
                   default="visualization/goldens/fineweb_val_sample.txt",
                   help="Save the decoded UTF-8 text to this path.")
    args = p.parse_args()

    print(f"loading {args.n_tokens} tokens from {VAL_SHARD.name} (skip={args.skip_tokens})")
    token_ids, n_bytes, text = load_val_tokens(args.n_tokens, args.skip_tokens)
    print(f"  tokens: {len(token_ids)}")
    print(f"  decoded bytes: {n_bytes}")
    print(f"  bytes/token: {n_bytes/len(token_ids):.3f}")

    if args.save_text:
        out = Path(args.save_text)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"  saved decoded text to {out}")

    print(f"\nReference BPB on FineWeb val (training-time):")
    print(f"  AR baseline:   1.2344  (final_int8_zlib_roundtrip_exact)")
    print(f"  MDLM seed42:   1.4786  (final_fp8_zlib_roundtrip_exact, K=256)")

    print(f"\n--- AR ---")
    ar_model, ar_label = build_ar_model(args.ar_artifact)
    print(f"  model: {ar_label}")
    ar_result = ar_bpb(ar_model, token_ids, n_bytes)
    print(f"  AR BPB on this slice: {ar_result:.4f}")

    print(f"\n--- MDLM (K={args.K}, seed={args.seed}) ---")
    mdlm_model, mdlm_label = build_diffusion_model(args.mdlm_artifact)
    print(f"  model: {mdlm_label}")
    mdlm_result = mdlm_bpb(mdlm_model, token_ids, n_bytes, args.K, args.eps, args.seed)
    print(f"  MDLM BPB on this slice: {mdlm_result:.4f}")

    print(f"\n--- Summary ---")
    print(f"  AR on this slice / training:    {ar_result:.4f} / 1.2344  "
          f"(diff: {ar_result - 1.2344:+.4f})")
    print(f"  MDLM on this slice / training:  {mdlm_result:.4f} / 1.4786  "
          f"(diff: {mdlm_result - 1.4786:+.4f})")


if __name__ == "__main__":
    main()
