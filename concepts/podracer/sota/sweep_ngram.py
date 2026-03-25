#!/usr/bin/env python3
"""N-gram parameter sweep — loads quantized model, sweeps eval params, no retraining.

Usage:
    torchrun --standalone --nproc_per_node=8 concepts/podracer/sota/sweep_ngram.py

Env vars:
    MODEL_PATH  — path to int6 quantized model (default: final_model.int6.ptz)
    SWEEP_MAX_SECONDS — per-combo eval time budget (default: 180)
    SWEEP_RESULTS — output CSV path (default: sweep_ngram_results.csv)
"""
from __future__ import annotations
import csv
import io
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

# Import podracer components
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_gpt import (
    Hyperparameters, GPT, CastedLinear,
    build_sentencepiece_luts, load_validation_tokens,
    dequantize_mixed_int6, eval_val_sliding_hashed_ngram,
    restore_low_dim_params_to_fp32, maybe_torch_compile,
)

# ── sweep grid ────────────────────────────────────────────────────────────────

BASE = dict(
    alpha=0.30,
    alpha_min=0.05,
    alpha_max=0.60,
    entropy_center=4.0,
    entropy_scale=2.0,
    min_count=2,
    buckets=4_194_304,
    order=7,
    min_order=2,
)

def build_sweep_grid() -> list[dict]:
    combos: list[dict] = []
    tag = lambda **kw: {**BASE, **kw, "_tag": " ".join(f"{k}={v}" for k, v in kw.items())}

    # ── alpha_max sweep (most impactful single param) ────────────────────
    for am in [0.50, 0.60, 0.70, 0.80, 0.90]:
        combos.append(tag(alpha_max=am))

    # ── entropy_center sweep ─────────────────────────────────────────────
    for ec in [2.0, 2.5, 3.0, 3.5, 5.0]:
        combos.append(tag(entropy_center=ec))

    # ── buckets sweep (free lunch — eval-time memory only) ───────────────
    for b in [8_388_608, 16_777_216]:
        combos.append(tag(buckets=b))

    # ── min_count sweep ──────────────────────────────────────────────────
    for mc in [1, 3]:
        combos.append(tag(min_count=mc))

    # ── order sweep ──────────────────────────────────────────────────────
    for o in [5, 9]:
        combos.append(tag(order=o))

    # ── promising combos (alpha_max × entropy_center interaction) ────────
    combos.append(tag(alpha_max=0.70, entropy_center=3.0))
    combos.append(tag(alpha_max=0.80, entropy_center=3.0))
    combos.append(tag(alpha_max=0.80, entropy_center=2.5))
    combos.append(tag(alpha_max=0.90, entropy_center=3.0))
    combos.append(tag(alpha_max=0.90, entropy_center=2.5))

    # ── multi-param combos ───────────────────────────────────────────────
    combos.append(tag(alpha_max=0.80, entropy_center=3.0, buckets=16_777_216))
    combos.append(tag(alpha_max=0.80, entropy_center=3.0, min_count=1))
    combos.append(tag(alpha_max=0.80, entropy_center=3.0, buckets=16_777_216, min_count=1))
    combos.append(tag(alpha_max=0.80, entropy_center=3.0, buckets=16_777_216, min_count=1, order=9))

    return combos

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model_path = os.environ.get("MODEL_PATH", "final_model.int6.ptz")
    sweep_max_seconds = float(os.environ.get("SWEEP_MAX_SECONDS", "180"))
    results_path = os.environ.get("SWEEP_RESULTS", "sweep_ngram_results.csv")

    args = Hyperparameters()
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def log0(msg):
        if rank == 0:
            print(msg, flush=True)

    log0("=" * 60)
    log0("  N-GRAM PARAMETER SWEEP")
    log0(f"  model: {model_path}")
    log0(f"  per-combo budget: {sweep_max_seconds}s")
    log0(f"  world_size: {world_size}")
    log0("=" * 60)

    # ── load val data ────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_tokens: {val_tokens.numel() - 1}")

    # ── load quantized model ─────────────────────────────────────────────
    log0(f"loading {model_path}...")
    with open(model_path, "rb") as f:
        blob = f.read()
    if _COMPRESSOR == "zstd":
        raw = zstandard.ZstdDecompressor().decompress(blob)
    else:
        raw = zlib.decompress(blob)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")

    # Build template state dict from a fresh model (same architecture)
    CastedLinear._qat_enabled = False
    template_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers, mlp_act=args.mlp_act, mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank, f1_corr_scale_init=args.f1_corr_scale_init,
    )
    template_sd = {k: v.detach().cpu() for k, v in template_model.state_dict().items()
                   if "mtp_heads" not in k}
    del template_model

    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)
    del quant_state, template_sd

    eval_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
        dtg=args.dtg_enabled, ve_enabled=args.ve_enabled, ve_dim=args.ve_dim,
        ve_layers=args.ve_layers, mlp_act=args.mlp_act, mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank, f1_corr_scale_init=args.f1_corr_scale_init,
    ).to(device).bfloat16()
    for m in eval_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    del deq_state
    log0("model loaded OK")

    # ── run sweep ────────────────────────────────────────────────────────
    combos = build_sweep_grid()
    log0(f"\n{len(combos)} combos to sweep\n")

    results = []
    csv_fields = ["idx", "tag", "bpb", "val_loss", "coverage", "time_s",
                  "alpha_max", "entropy_center", "entropy_scale",
                  "min_count", "buckets", "order"]

    # Write CSV header
    if rank == 0:
        with open(results_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    for idx, combo in enumerate(combos):
        tag = combo.pop("_tag", "?")
        # Apply params to args
        args.ngram_eval_order = combo["order"]
        args.ngram_eval_min_order = combo["min_order"]
        args.ngram_eval_alpha = combo["alpha"]
        args.ngram_eval_adaptive = True
        args.ngram_eval_alpha_min = combo["alpha_min"]
        args.ngram_eval_alpha_max = combo["alpha_max"]
        args.ngram_eval_entropy_center = combo["entropy_center"]
        args.ngram_eval_entropy_scale = combo["entropy_scale"]
        args.ngram_eval_min_count = combo["min_count"]
        args.ngram_eval_buckets = combo["buckets"]
        args.ngram_eval_max_seconds = sweep_max_seconds
        args.cubric_cadence = 0  # no cubric during sweep

        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        ng_loss, ng_bpb, ng_coverage = eval_val_sliding_hashed_ngram(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=args.eval_stride,
            order=args.ngram_eval_order,
            alpha=args.ngram_eval_alpha,
            min_count=args.ngram_eval_min_count,
            buckets=args.ngram_eval_buckets,
            max_seconds=args.ngram_eval_max_seconds,
            eval_seq_len=effective_eval_seq_len,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        row = dict(
            idx=idx, tag=tag, bpb=f"{ng_bpb:.6f}", val_loss=f"{ng_loss:.6f}",
            coverage=f"{ng_coverage:.4f}", time_s=f"{elapsed:.0f}",
            alpha_max=combo["alpha_max"], entropy_center=combo["entropy_center"],
            entropy_scale=combo["entropy_scale"], min_count=combo["min_count"],
            buckets=combo["buckets"], order=combo["order"],
        )
        results.append(row)

        if rank == 0:
            cov_pct = f"{ng_coverage*100:.1f}%"
            log0(f"[{idx+1:2d}/{len(combos)}] bpb={ng_bpb:.6f} cov={cov_pct:>6s} "
                 f"t={elapsed:>4.0f}s  {tag}")
            # Append to CSV
            with open(results_path, "a", newline="") as f:
                csv.DictWriter(f, csv_fields).writerow(row)

        if distributed:
            dist.barrier()

    # ── summary ──────────────────────────────────────────────────────────
    if rank == 0:
        log0("\n" + "=" * 60)
        log0("  SWEEP COMPLETE — top 5 by BPB")
        log0("=" * 60)
        ranked = sorted(results, key=lambda r: float(r["bpb"]))
        for i, r in enumerate(ranked[:5]):
            log0(f"  #{i+1}  bpb={r['bpb']}  {r['tag']}")
        log0(f"\nBaseline (current podracer): alpha_max=0.60 center=4.0 mc=2 buckets=4M order=7")
        baseline = [r for r in results if r["tag"] == "alpha_max=0.6"]
        if baseline:
            log0(f"Baseline bpb={baseline[0]['bpb']}")
        log0(f"\nFull results: {results_path}")
        log0("=" * 60)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
