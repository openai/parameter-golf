"""
Post-training evaluation script for exp14.
Tests sliding window eval (stride=64) on both fp32 and int6 roundtrip models.
"""
from __future__ import annotations
import io
import math
import os
import sys
import time

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# Import model + quantization from exp14
sys.path.insert(0, "/mnt/c/dev/parameter-golf/experiments/r3")
from exp15_qkgain5_mlp4x import (
    Hyperparameters,
    GPT,
    quantize_state_dict_mixed,
    dequantize_state_dict_mixed,
    _decompress_auto,
    load_validation_tokens,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_val_data(args):
    """Load validation tokens and build byte LUTs."""
    import glob
    val_files = sorted(glob.glob(args.val_files))
    val_tokens_list = []
    for f in val_files:
        data = np.fromfile(f, dtype=np.uint16)
        val_tokens_list.append(torch.from_numpy(data.astype(np.int32)).to(torch.int64))
    val_tokens = torch.cat(val_tokens_list)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    vocab_size = sp.get_piece_size()
    base_bytes_lut = torch.zeros(vocab_size, dtype=torch.int32)
    has_leading_space_lut = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary_token_lut = torch.zeros(vocab_size, dtype=torch.bool)
    for tid in range(vocab_size):
        piece = sp.id_to_piece(tid)
        raw = piece.encode("utf-8")
        if piece.startswith("\u2581"):
            has_leading_space_lut[tid] = True
            raw = piece[1:].encode("utf-8")
            base_bytes_lut[tid] = len(raw)
        else:
            base_bytes_lut[tid] = len(raw)
        if sp.is_unknown(tid) or sp.is_control(tid) or sp.is_unused(tid):
            is_boundary_token_lut[tid] = True
    return val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def eval_standard(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                  is_boundary_token_lut, device, seq_len=1024, batch_seqs=1):
    """Standard non-overlapping eval (same as training eval)."""
    total_seqs = (val_tokens.numel() - 1) // seq_len
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for si in range(0, total_seqs, batch_seqs):
            se = min(si + batch_seqs, total_seqs)
            bsz = se - si
            raw_start = si * seq_len
            raw_end = se * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
            x = local[:-1].reshape(bsz, seq_len)
            y = local[1:].reshape(bsz, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                  y.reshape(-1), reduction="none")
            loss_sum += nll.to(torch.float64).sum()
            token_count += float(y.numel())
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            tb = base_bytes_lut[tgt_ids].to(torch.float64)
            tb += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(torch.float64)
            byte_count += tb.sum()

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    return val_loss, bpt * tpb


def eval_sliding(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                 is_boundary_token_lut, device, seq_len=1024, stride=64, batch_seqs=8):
    """Sliding window evaluation with configurable stride."""
    total_tokens = val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= 1]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(window_starts), batch_seqs):
            batch_ws = window_starts[bi:bi + batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x_batch)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                y_batch.reshape(-1),
                reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                # Only score tokens in the "new" region (beyond overlap)
                s = 0 if ws == 0 else max(wlen - stride, 0)
                scored_nll = nll[i, s:wlen].to(torch.float64)
                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            if bi % (batch_seqs * 50) == 0 and bi > 0:
                interim_bpb = ((loss_sum / token_count).item() / math.log(2.0)) * (token_count.item() / byte_count.item())
                print(f"  slide eval progress: {bi}/{len(window_starts)} windows, interim bpb={interim_bpb:.4f}")

    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    return val_loss, bpt * tpb


def main():
    args = Hyperparameters()
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    # Load val data using the same loader as training
    print("Loading validation data...")
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    _, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = load_val_data(args)
    print(f"  val tokens: {val_tokens.numel():,}")
    vt_int = val_tokens.to(torch.int32)
    print(f"  token range: [{vt_int.min().item()}, {vt_int.max().item()}] (vocab_size={args.vocab_size})")
    base_bytes_lut = base_bytes_lut.to(device)
    has_leading_space_lut = has_leading_space_lut.to(device)
    is_boundary_token_lut = is_boundary_token_lut.to(device)

    # Build model
    print("Building model...")
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        rope_dims=args.rope_dims,
        bigram_buckets=args.bigram_buckets,
        bigram_dim=args.bigram_dim,
    ).to(device)

    results = {}

    # --- FP32 model ---
    print("\n=== Loading FP32 weights ===")
    sd = torch.load("/mnt/c/dev/parameter-golf/final_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model = model.to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("\n--- FP32: Standard eval ---")
    t0 = time.time()
    _, bpb = eval_standard(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, device, batch_seqs=1)
    results["fp32_standard"] = bpb
    print(f"  val_bpb = {bpb:.4f} [{time.time()-t0:.0f}s]")

    print("\n--- FP32: Sliding window stride=256 ---")
    t0 = time.time()
    _, bpb = eval_sliding(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                          is_boundary_token_lut, device, stride=256, batch_seqs=8)
    results["fp32_slide256"] = bpb
    print(f"  val_bpb = {bpb:.4f} [{time.time()-t0:.0f}s]", flush=True)

    # --- INT6 roundtrip model ---
    print("\n=== Loading INT6 roundtrip weights ===")
    with open("/mnt/c/dev/parameter-golf/final_model.int6.ptz", "rb") as f:
        blob = f.read()
    quant_loaded = torch.load(io.BytesIO(_decompress_auto(blob)), map_location="cpu", weights_only=False)
    sd_cpu = torch.load("/mnt/c/dev/parameter-golf/final_model.pt", map_location="cpu", weights_only=False)
    dequant_sd = dequantize_state_dict_mixed(quant_loaded["result"], quant_loaded["meta"], sd_cpu)
    model.load_state_dict(dequant_sd, strict=True)
    model = model.to(device)

    print("\n--- INT6: Standard eval ---")
    t0 = time.time()
    _, bpb = eval_standard(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                           is_boundary_token_lut, device, batch_seqs=1)
    results["int6_standard"] = bpb
    print(f"  val_bpb = {bpb:.4f} [{time.time()-t0:.0f}s]")

    print("\n--- INT6: Sliding window stride=256 ---")
    t0 = time.time()
    _, bpb = eval_sliding(model, val_tokens, base_bytes_lut, has_leading_space_lut,
                          is_boundary_token_lut, device, stride=256, batch_seqs=8)
    results["int6_slide256"] = bpb
    print(f"  val_bpb = {bpb:.4f} [{time.time()-t0:.0f}s]", flush=True)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY (exp14, WD=0.08)")
    print("=" * 60)
    for name, val in results.items():
        print(f"  {name:<25s}: {val:.4f} bpb")
    print()
    for prefix in ("fp32", "int6"):
        std_key = f"{prefix}_standard"
        slide_key = f"{prefix}_slide256"
        if std_key in results and slide_key in results:
            print(f"  Sliding window gain ({prefix}, stride=256): {results[std_key] - results[slide_key]:.4f} bpb")


if __name__ == "__main__":
    main()
