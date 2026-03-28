#!/usr/bin/env python3
"""Cubric × n-gram delta sweep (eval-only, no retraining).

Usage:
    torchrun --standalone --nproc_per_node=8 concepts/xwing/sweep_cubric_ngram_delta.py

Env vars:
    MODEL_PATH        — int6 model path (default: final_model.int6.ptz)
    SWEEP_MAX_SECONDS — per-arm n-gram eval budget (default: 180)
    DELTA_GRID        — interaction4 | delta12 (default: delta12)
    CUBRIC_CADENCE    — cadence value used when cubric-enabled arms run (default: 32)
    SWEEP_RESULTS     — CSV output path (default: sweep_cubric_ngram_delta_results.csv)
    SWEEP_SUMMARY     — JSON output path (default: sweep_cubric_ngram_delta_summary.json)
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist

try:
    import zstandard

    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from train_gpt import (  # noqa: E402
    Hyperparameters,
    GPT,
    CastedLinear,
    build_sentencepiece_luts,
    load_validation_tokens,
    dequantize_mixed_int6,
    eval_val_sliding,
    eval_val_sliding_hashed_ngram,
    restore_low_dim_params_to_fp32,
)


def _arm(
    name: str,
    *,
    ngram_enabled: bool,
    cubric_enabled: bool,
    cubric_cadence: int,
    order: int = 7,
    min_order: int = 2,
    alpha: float = 0.30,
    alpha_min: float = 0.05,
    alpha_max: float = 0.70,
    entropy_center: float = 3.0,
    entropy_scale: float = 2.0,
    min_count: int = 2,
    buckets: int = 8_388_608,
) -> dict:
    return dict(
        name=name,
        ngram_enabled=ngram_enabled,
        cubric_enabled=cubric_enabled,
        cubric_cadence=cubric_cadence if cubric_enabled else 0,
        order=order,
        min_order=min_order,
        alpha=alpha,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        entropy_center=entropy_center,
        entropy_scale=entropy_scale,
        min_count=min_count,
        buckets=buckets,
    )


def build_delta_grid(grid_name: str, cubric_cadence: int) -> list[dict]:
    if grid_name not in {"interaction4", "delta12"}:
        raise ValueError(f"Unknown DELTA_GRID={grid_name}; expected interaction4 or delta12")

    arms = [
        _arm(
            "A_ctrl_ng0_c0",
            ngram_enabled=False,
            cubric_enabled=False,
            cubric_cadence=cubric_cadence,
        ),
        _arm(
            "B_ctrl_ng0_c1",
            ngram_enabled=False,
            cubric_enabled=True,
            cubric_cadence=cubric_cadence,
        ),
        _arm(
            "C_o7_ng1_c0",
            ngram_enabled=True,
            cubric_enabled=False,
            cubric_cadence=cubric_cadence,
            order=7,
        ),
        _arm(
            "D_o7_ng1_c1",
            ngram_enabled=True,
            cubric_enabled=True,
            cubric_cadence=cubric_cadence,
            order=7,
        ),
    ]

    if grid_name == "interaction4":
        return arms

    arms.extend(
        [
            _arm(
                "E_o5_ng1_c0",
                ngram_enabled=True,
                cubric_enabled=False,
                cubric_cadence=cubric_cadence,
                order=5,
            ),
            _arm(
                "F_o5_ng1_c1",
                ngram_enabled=True,
                cubric_enabled=True,
                cubric_cadence=cubric_cadence,
                order=5,
            ),
            _arm(
                "G_o3_ng1_c0",
                ngram_enabled=True,
                cubric_enabled=False,
                cubric_cadence=cubric_cadence,
                order=3,
            ),
            _arm(
                "H_o3_ng1_c1",
                ngram_enabled=True,
                cubric_enabled=True,
                cubric_cadence=cubric_cadence,
                order=3,
            ),
            _arm(
                "I_o7_b4m_ng1_c0",
                ngram_enabled=True,
                cubric_enabled=False,
                cubric_cadence=cubric_cadence,
                order=7,
                buckets=4_194_304,
            ),
            _arm(
                "J_o7_b4m_ng1_c1",
                ngram_enabled=True,
                cubric_enabled=True,
                cubric_cadence=cubric_cadence,
                order=7,
                buckets=4_194_304,
            ),
            _arm(
                "K_o7_mc1_ng1_c0",
                ngram_enabled=True,
                cubric_enabled=False,
                cubric_cadence=cubric_cadence,
                order=7,
                min_count=1,
            ),
            _arm(
                "L_o7_mc1_ng1_c1",
                ngram_enabled=True,
                cubric_enabled=True,
                cubric_cadence=cubric_cadence,
                order=7,
                min_count=1,
            ),
        ]
    )
    return arms


def _compute_summary(results_by_name: dict[str, dict], grid_name: str) -> dict:
    def bpb(name: str) -> float | None:
        row = results_by_name.get(name)
        return float(row["bpb"]) if row is not None else None

    summary: dict = {"grid": grid_name, "deltas": {}, "order_deltas": {}}
    a = bpb("A_ctrl_ng0_c0")
    b = bpb("B_ctrl_ng0_c1")
    c = bpb("C_o7_ng1_c0")
    d = bpb("D_o7_ng1_c1")

    if all(v is not None for v in (a, b, c, d)):
        # Lower BPB is better, so "delta" is defined as improvement (positive = better).
        delta_ngram = a - c
        delta_cubric_given_ngram = c - d
        delta_cubric_without_ngram = a - b
        joint_delta = a - d
        interaction_residual = joint_delta - (delta_ngram + delta_cubric_without_ngram)
        summary["deltas"] = {
            "delta_ngram_from_control": delta_ngram,
            "delta_cubric_given_ngram": delta_cubric_given_ngram,
            "delta_cubric_without_ngram": delta_cubric_without_ngram,
            "joint_delta_ngram_plus_cubric": joint_delta,
            "interaction_residual": interaction_residual,
        }

    for off_name, on_name, label in (
        ("C_o7_ng1_c0", "D_o7_ng1_c1", "order7"),
        ("E_o5_ng1_c0", "F_o5_ng1_c1", "order5"),
        ("G_o3_ng1_c0", "H_o3_ng1_c1", "order3"),
        ("I_o7_b4m_ng1_c0", "J_o7_b4m_ng1_c1", "order7_b4m"),
        ("K_o7_mc1_ng1_c0", "L_o7_mc1_ng1_c1", "order7_mc1"),
    ):
        off_bpb = bpb(off_name)
        on_bpb = bpb(on_name)
        if off_bpb is None or on_bpb is None:
            continue
        summary["order_deltas"][label] = off_bpb - on_bpb

    return summary


def main():
    model_path = os.environ.get("MODEL_PATH", "final_model.int6.ptz")
    sweep_max_seconds = float(os.environ.get("SWEEP_MAX_SECONDS", "180"))
    grid_name = os.environ.get("DELTA_GRID", "delta12")
    cubric_cadence = int(os.environ.get("CUBRIC_CADENCE", "32"))
    results_path = os.environ.get("SWEEP_RESULTS", "sweep_cubric_ngram_delta_results.csv")
    summary_path = os.environ.get("SWEEP_SUMMARY", "sweep_cubric_ngram_delta_summary.json")
    chunk_tokens = int(os.environ.get("NGRAM_CHUNK_TOKENS", "1048576"))

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

    def log0(msg: str):
        if rank == 0:
            print(msg, flush=True)

    arms = build_delta_grid(grid_name, cubric_cadence)
    csv_fields = [
        "idx",
        "arm",
        "ngram_enabled",
        "cubric_enabled",
        "cubric_cadence",
        "order",
        "min_count",
        "buckets",
        "alpha",
        "alpha_min",
        "alpha_max",
        "entropy_center",
        "entropy_scale",
        "chunk_tokens",
        "bpb",
        "val_loss",
        "coverage",
        "time_s",
    ]

    log0("=" * 72)
    log0("  X-WING CUBRIC × NGRAM DELTA SWEEP (eval-only)")
    log0(f"  model: {model_path}")
    log0(f"  grid: {grid_name} ({len(arms)} arms)")
    log0(f"  per-ngram-arm budget: {sweep_max_seconds}s")
    log0(f"  world_size: {world_size}")
    log0("=" * 72)

    # Load val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_tokens: {val_tokens.numel() - 1}")

    # Load quantized model
    model_blob = Path(model_path).read_bytes()
    raw = None
    if _COMPRESSOR == "zstd":
        try:
            raw = zstandard.ZstdDecompressor().decompress(model_blob)
        except Exception:
            raw = None
    if raw is None:
        try:
            raw = zlib.decompress(model_blob)
        except Exception:
            if _COMPRESSOR != "zstd":
                raw = zstandard.ZstdDecompressor().decompress(model_blob)
            else:
                raise
    quant_state = torch.load(io.BytesIO(raw), map_location="cpu")

    CastedLinear._qat_enabled = False
    template_model = GPT(
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
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        mlp_act=args.mlp_act,
        mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank,
        f1_corr_scale_init=args.f1_corr_scale_init,
    )
    template_sd = {k: v.detach().cpu() for k, v in template_model.state_dict().items() if "mtp_heads" not in k}
    del template_model

    deq_state = dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)
    del quant_state, template_sd

    eval_model = GPT(
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
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
        mlp_act=args.mlp_act,
        mlp_leaky_slope=args.mlp_leaky_slope,
        f1_corr_rank=args.f1_corr_rank,
        f1_corr_scale_init=args.f1_corr_scale_init,
    ).to(device).bfloat16()
    for module in eval_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(eval_model)
    eval_model.load_state_dict(deq_state, strict=True)
    del deq_state
    log0("model loaded OK")

    # Prepare CSV
    if rank == 0:
        with open(results_path, "w", newline="") as f:
            csv.DictWriter(f, csv_fields).writeheader()

    results_by_name: dict[str, dict] = {}

    for idx, arm in enumerate(arms):
        args.cubric_cadence = int(arm["cubric_cadence"])

        if distributed:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        if arm["ngram_enabled"]:
            args.ngram_eval_order = int(arm["order"])
            args.ngram_eval_min_order = int(arm["min_order"])
            args.ngram_eval_alpha = float(arm["alpha"])
            args.ngram_eval_adaptive = True
            args.ngram_eval_alpha_min = float(arm["alpha_min"])
            args.ngram_eval_alpha_max = float(arm["alpha_max"])
            args.ngram_eval_entropy_center = float(arm["entropy_center"])
            args.ngram_eval_entropy_scale = float(arm["entropy_scale"])
            args.ngram_eval_min_count = int(arm["min_count"])
            args.ngram_eval_buckets = int(arm["buckets"])
            args.ngram_eval_max_seconds = sweep_max_seconds

            val_loss, bpb, coverage = eval_val_sliding_hashed_ngram(
                args,
                eval_model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                stride=args.eval_stride,
                order=args.ngram_eval_order,
                alpha=args.ngram_eval_alpha,
                min_count=args.ngram_eval_min_count,
                buckets=args.ngram_eval_buckets,
                max_seconds=args.ngram_eval_max_seconds,
                eval_seq_len=effective_eval_seq_len,
            )
        else:
            val_loss, bpb = eval_val_sliding(
                args,
                eval_model,
                rank,
                world_size,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                stride=args.eval_stride,
                eval_seq_len=effective_eval_seq_len,
            )
            coverage = 1.0

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        row = dict(
            idx=idx,
            arm=arm["name"],
            ngram_enabled=int(arm["ngram_enabled"]),
            cubric_enabled=int(arm["cubric_enabled"]),
            cubric_cadence=arm["cubric_cadence"],
            order=arm["order"],
            min_count=arm["min_count"],
            buckets=arm["buckets"],
            alpha=arm["alpha"],
            alpha_min=arm["alpha_min"],
            alpha_max=arm["alpha_max"],
            entropy_center=arm["entropy_center"],
            entropy_scale=arm["entropy_scale"],
            chunk_tokens=chunk_tokens,
            bpb=f"{bpb:.6f}",
            val_loss=f"{val_loss:.6f}",
            coverage=f"{coverage:.6f}",
            time_s=f"{elapsed:.0f}",
        )
        results_by_name[arm["name"]] = row

        if rank == 0:
            with open(results_path, "a", newline="") as f:
                csv.DictWriter(f, csv_fields).writerow(row)
            print(
                f"[{idx + 1:02d}/{len(arms):02d}] arm={arm['name']} "
                f"bpb={float(row['bpb']):.6f} cov={float(row['coverage']) * 100:.1f}% "
                f"t={elapsed:.0f}s",
                flush=True,
            )

        if distributed:
            dist.barrier()

    if rank == 0:
        summary = _compute_summary(results_by_name, grid_name)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        print("\n" + "=" * 72, flush=True)
        print("  DELTA SUMMARY", flush=True)
        print("=" * 72, flush=True)
        if summary.get("deltas"):
            d = summary["deltas"]
            print(f"delta_ngram_from_control: {d['delta_ngram_from_control']:.6f}", flush=True)
            print(f"delta_cubric_given_ngram: {d['delta_cubric_given_ngram']:.6f}", flush=True)
            print(f"delta_cubric_without_ngram: {d['delta_cubric_without_ngram']:.6f}", flush=True)
            print(f"joint_delta_ngram_plus_cubric: {d['joint_delta_ngram_plus_cubric']:.6f}", flush=True)
            print(f"interaction_residual: {d['interaction_residual']:.6f}", flush=True)
        else:
            print("Not enough arms present to compute interaction summary.", flush=True)

        if summary.get("order_deltas"):
            print("\norder-conditioned cubric deltas (positive = cubric improves):", flush=True)
            for key, value in sorted(summary["order_deltas"].items()):
                print(f"  {key}: {value:.6f}", flush=True)
        print(f"\nCSV: {results_path}", flush=True)
        print(f"JSON: {summary_path}", flush=True)
        print("=" * 72, flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
