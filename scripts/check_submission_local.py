#!/usr/bin/env python3
"""
Load a competition train_gpt.py submission as a module (without running main) and
smoke-test on CPU or Apple MPS: syntax, import, tiny forward pass, int6 export roundtrip.

This does NOT produce an official leaderboard BPB: the submission script is built for
CUDA + distributed + full validation. On a Mac you use this to catch breakage early;
for a real score you still need a GPU pod (see README).

Usage:
  python3 scripts/check_submission_local.py \\
    records/track_10min_16mb/2026-03-20_SOTA_TTT_RoPE50K_EMA_Curriculum/train_gpt.py

Optional env (before smaller/faster smoke model):
  LOCAL_SMOKE_LAYERS=4 LOCAL_SMOKE_DIM=256 python3 scripts/check_submission_local.py PATH
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Local CPU/MPS smoke test for a train_gpt.py submission.")
    parser.add_argument("train_gpt", type=Path, help="Path to train_gpt.py")
    parser.add_argument("--seq", type=int, default=64, help="Sequence length for forward smoke")
    args = parser.parse_args()
    path = args.train_gpt.resolve()
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    src = path.read_text(encoding="utf-8")
    try:
        ast.parse(src)
    except SyntaxError as e:
        print(f"error: syntax: {e}", file=sys.stderr)
        return 1

    # Smaller architecture for laptop RAM/time (class Hyperparameters reads env at import).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUM_LAYERS", os.environ.get("LOCAL_SMOKE_LAYERS", "4"))
    os.environ.setdefault("MODEL_DIM", os.environ.get("LOCAL_SMOKE_DIM", "256"))
    os.environ.setdefault("NUM_HEADS", "8")
    os.environ.setdefault("NUM_KV_HEADS", "4")
    os.environ.setdefault("MLP_MULT", "2")
    os.environ.setdefault("BIGRAM_VOCAB_SIZE", "512")
    os.environ.setdefault("BIGRAM_DIM", "64")
    os.environ.setdefault("VOCAB_SIZE", "1024")

    spec = importlib.util.spec_from_file_location("_pg_submission_smoke", path)
    if spec is None or spec.loader is None:
        print("error: could not create module spec", file=sys.stderr)
        return 1
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"error: import failed: {e}", file=sys.stderr)
        return 1

    if not hasattr(mod, "GPT") or not hasattr(mod, "mixed_quantize_int6") or not hasattr(mod, "dequantize_mixed_int6"):
        print("error: module missing GPT / mixed_quantize_int6 / dequantize_mixed_int6", file=sys.stderr)
        return 1

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        print("note: CUDA visible; using cuda:0 for smoke (set CUDA_VISIBLE_DEVICES= to force CPU)")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("device: mps (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("device: cpu")

    hp = mod.Hyperparameters()
    model = mod.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=int(hp.mlp_mult),
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim,
    )
    # Full training uses bf16 on CUDA; CPU/MPS smoke is more reliable in fp32.
    train_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=train_dtype)
    for m in model.modules():
        if isinstance(m, mod.CastedLinear):
            m.float()
    mod.restore_low_dim_params_to_fp32(model)

    bsz, seq = 2, min(args.seq, 128)
    x = torch.randint(0, hp.vocab_size, (bsz, seq), device=device, dtype=torch.int64)
    y = torch.randint(0, hp.vocab_size, (bsz, seq), device=device, dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss = float(model(x, y).item())
            logits = model.forward_logits(x)
    print(f"forward_ok loss={loss:.4f} logits_shape={tuple(logits.shape)}")

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    qres, qmeta = mod.mixed_quantize_int6(sd, {"mlp", "attn"})
    deq = mod.dequantize_mixed_int6(qres, qmeta, sd)
    model.load_state_dict(deq, strict=True)
    model.to(device=device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            loss2 = float(model(x, y).item())
    print(f"quant_roundtrip_ok loss_after={loss2:.4f} delta={abs(loss2 - loss):.6f}")

    buf = io.BytesIO()
    torch.save({"w": qres, "m": qmeta}, buf)
    raw_n = len(buf.getvalue())
    try:
        import zstandard

        zblob = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
        comp = "zstd-22"
    except ImportError:
        import zlib

        zblob = zlib.compress(buf.getvalue(), 9)
        comp = "zlib-9"
    code_n = len(src.encode("utf-8"))
    print(f"artifact_smoke {comp}_compressed_weights={len(zblob)} code_bytes={code_n} raw_torch_save={raw_n}")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
