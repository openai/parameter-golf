#!/usr/bin/env python3
"""Grid sweep over n-gram eval parameters on a saved quantized model.

Loads final_model.int6.ptz once, then runs eval_val_sliding_hashed_ngram
with each parameter combination. Results written to CSV.

Usage:
    torchrun --standalone --nproc_per_node=8 experiments/B_wing/bwing_V/eval_sweep.py
"""
from __future__ import annotations
import csv
import importlib.util
import io
import itertools
import math
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

# ---------------------------------------------------------------------------
# Import train_gpt as a module (without running main)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_gpt.py"

spec = importlib.util.spec_from_file_location("train_gpt", str(TRAIN_SCRIPT))
tg = importlib.util.module_from_spec(spec)
tg.__name__ = "train_gpt"  # prevent __main__ execution
spec.loader.exec_module(tg)

# ---------------------------------------------------------------------------
# Grid definition — edit these to change the sweep
# ---------------------------------------------------------------------------
GRID = {
    "alpha_max":        [0.50, 0.60, 0.70, 0.80],
    "entropy_center":   [2.0, 2.5, 3.0],
    "high_order_mult":  [1.5, 2.0, 2.5, 3.0],
    "min_count":        [1, 2],
    "cubric":           [0, 1],
}

# Fixed params (not swept)
ALPHA_MIN = 0.03
ENTROPY_SCALE = 2.0
ENTROPY_SHIFT = True
LOW_ORDER_MULTS = (0.3, 0.3, 0.97)  # orders 2, 3, 4 — always same
BUCKETS = 8_388_608
ORDER = 9
MIN_ORDER = 2
STRIDE = 64


def build_order_mults(low: tuple, high_mult: float) -> str:
    """Build comma-separated order mults string. Orders 5-9 get high_mult."""
    return ",".join(str(x) for x in list(low) + [high_mult] * 5)


def main():
    # Distributed setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl")
    master = rank == 0

    def log0(msg):
        if master:
            print(msg, flush=True)

    # Load tokenizer + val data (once)
    args = tg.Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = tg.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_tokens:{val_tokens.numel()-1}")

    # Build fresh model for template shapes → dequantize
    tg.CastedLinear._qat_enabled = args.qat_enabled
    template_model = tg.GPT(
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
    for m in template_model.modules():
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(template_model)
    sd_cpu = {k: v.detach().cpu() for k, v in template_model.state_dict().items() if "mtp_heads" not in k}

    # Load quantized weights
    log0("loading final_model.int6.ptz...")
    with open("final_model.int6.ptz", "rb") as f:
        quant_blob = f.read()
    if _COMPRESSOR == "zstd":
        raw = zstandard.ZstdDecompressor().decompress(quant_blob)
    else:
        raw = zlib.decompress(quant_blob)
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")
    deq_state = tg.dequantize_mixed_int6(quant_state["w"], quant_state["m"], sd_cpu)

    # Build eval model
    eval_model = tg.GPT(
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
        if isinstance(m, tg.CastedLinear):
            m.float()
    tg.restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    del template_model, sd_cpu, deq_state, quant_state  # free memory
    torch.cuda.empty_cache()

    log0("model loaded. starting sweep...")

    # Build all grid combos, sorted by expected impact (high alpha_max + high mult first)
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    combos_dicts = [dict(zip(keys, vals)) for vals in combos]
    # Sort: highest alpha_max * highest high_order_mult first (most aggressive configs first)
    combos_dicts.sort(key=lambda c: -(c["alpha_max"] * c["high_order_mult"]))

    total = len(combos_dicts)
    log0(f"sweep:{total} configs")

    # CSV output
    csv_path = SCRIPT_DIR / "sweep_results.csv"
    if master:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "alpha_min", "alpha_max", "entropy_center", "entropy_scale",
                             "high_order_mult", "order_mults", "min_count", "cubric",
                             "entropy_shift", "bpb", "eval_time_s"])

    best_bpb = float("inf")
    best_config = None

    for i, cfg in enumerate(combos_dicts):
        # Build args overlay
        args.ngram_eval_alpha_min = ALPHA_MIN
        args.ngram_eval_alpha_max = cfg["alpha_max"]
        args.ngram_eval_entropy_center = cfg["entropy_center"]
        args.ngram_eval_entropy_scale = ENTROPY_SCALE
        args.ngram_eval_min_count = cfg["min_count"]
        args.ngram_eval_adaptive = True
        args.ngram_entropy_shift = ENTROPY_SHIFT
        args.cubric_cadence = cfg["cubric"]

        mults_str = build_order_mults(LOW_ORDER_MULTS, cfg["high_order_mult"])
        args.ngram_order_mults_str = mults_str

        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        ng_loss, ng_bpb, ng_coverage = tg.eval_val_sliding_hashed_ngram(
            args, eval_model, rank, world_size, device,
            val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=STRIDE, order=ORDER, alpha=0.30,
            min_count=cfg["min_count"], buckets=BUCKETS,
            max_seconds=0.0, eval_seq_len=args.train_seq_len,
        )

        elapsed = time.perf_counter() - t0

        if master:
            tag = ""
            if ng_bpb < best_bpb:
                best_bpb = ng_bpb
                best_config = cfg
                tag = " *** NEW BEST ***"

            log0(
                f"[{i+1}/{total}] bpb={ng_bpb:.6f} "
                f"amax={cfg['alpha_max']:.2f} ec={cfg['entropy_center']:.1f} "
                f"hm={cfg['high_order_mult']:.1f} mc={cfg['min_count']} "
                f"cub={cfg['cubric']} t={elapsed:.0f}s{tag}"
            )

            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    i + 1, ALPHA_MIN, cfg["alpha_max"], cfg["entropy_center"],
                    ENTROPY_SCALE, cfg["high_order_mult"], mults_str,
                    cfg["min_count"], cfg["cubric"], int(ENTROPY_SHIFT),
                    f"{ng_bpb:.8f}", f"{elapsed:.1f}",
                ])

    # Final summary
    if master:
        log0("=" * 60)
        log0(f"BEST BPB: {best_bpb:.6f}")
        log0(f"CONFIG: {best_config}")
        log0(f"results saved to {csv_path}")
        log0("=" * 60)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
