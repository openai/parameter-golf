#!/usr/bin/env python3
"""
Quick test: verify eval_val gives the same BPB as training reported.
Compares uncompiled vs compiled model to find the discrepancy source.

Usage:
    python test_eval_match.py
"""
import io
import zlib

import sentencepiece as spm
import torch

from train_gpt import (
    CastedLinear,
    GPT,
    Hyperparameters,
    build_sentencepiece_luts,
    dequantize_state_dict_int8,
    eval_val,
    load_validation_tokens,
    restore_low_dim_params_to_fp32,
)


def main():
    device = torch.device("cuda")
    args = Hyperparameters()

    # Create model — match training init exactly
    model = GPT(
        vocab_size=args.vocab_size,
        num_unique_blocks=args.num_unique_blocks,
        num_loops=args.num_loops,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)

    # Load int8+zlib checkpoint
    print("Loading checkpoint...")
    with open("final_model.int8.ptz", "rb") as f:
        blob = f.read()
    state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
    model.to(device)
    print(f"model loaded, params={sum(p.numel() for p in model.parameters()):,}")

    # Load tokenizer and val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    print(f"val_tokens: {val_tokens.numel():,}")

    # Test 0: raw checkpoint (no quantization)
    import os
    if os.path.exists("final_model.pt"):
        print("\n=== Test 0: raw checkpoint (final_model.pt) ===")
        raw_state = torch.load("final_model.pt", map_location="cpu")
        model.load_state_dict(raw_state, strict=True)
        model.to(device)
        vl0, vb0 = eval_val(args, model, 0, 1, device, 8, val_tokens, *luts)
        print(f"val_loss={vl0:.4f} val_bpb={vb0:.4f}")
    else:
        print("\nfinal_model.pt not found, skipping raw test")
        vb0 = 0.0

    # Test 1: int8+zlib checkpoint
    print("\n=== Test 1: int8+zlib checkpoint ===")
    with open("final_model.int8.ptz", "rb") as f:
        blob = f.read()
    state = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(state), strict=True)
    model.to(device)
    vl, vb = eval_val(args, model, 0, 1, device, 8, val_tokens, *luts)
    print(f"val_loss={vl:.4f} val_bpb={vb:.4f}")

    # Summary
    print("\n=== COMPARISON ===")
    print(f"Training reported:    val_bpb=1.4177")
    if vb0 > 0:
        print(f"Raw checkpoint:       val_bpb={vb0:.4f}")
    print(f"Int8+zlib checkpoint: val_bpb={vb:.4f}")


if __name__ == "__main__":
    main()
