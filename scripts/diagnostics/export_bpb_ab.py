#!/usr/bin/env python3
"""Export-only BPB A/B gate for mixed int5/int6 quantization.

Loads the 05c-plus float checkpoint and runs two export paths on it:
  A) uniform int6 + brotli-10
  B) conservative mixed int5/int6 + brotli-10

Reports final_int6_roundtrip_exact and sliding_window_s64 BPB for both.
This is the gate check before committing to a 3.25x training run.

Usage (on Pegasus, 1×GPU, ~10min):
  python -u scripts/diagnostics/export_bpb_ab.py \\
    --float-checkpoint /netscratch/ayach/parameter-golf/diagnostics/2026-03-31_05c_plus/final_model.pt \\
    --probe-json /netscratch/ayach/parameter-golf/diagnostics/2026-03-31_int5_probe_05c_plus/int5_tolerance_probe.json \\
    --val-dir /netscratch/ayach/parameter-golf/data/datasets/fineweb10B_sp1024 \\
    --tokenizer /netscratch/ayach/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \\
    --output-dir diagnostics/2026-04-01_export_ab
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Resolve paths and import from sibling scripts + train_gpt.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
TRAIN_SCRIPT_DIR = REPO_ROOT / "records" / "track_non_record_16mb" / "2026-03-30_training_bundle_plus"

# compress_probe provides custom_pack / custom_unpack / compress / decompress
sys.path.insert(0, str(SCRIPT_DIR))
from compress_probe import custom_pack, custom_unpack, compress, decompress  # noqa: E402

# train_gpt.py has if __name__ == "__main__": guard — safe to import
sys.path.insert(0, str(TRAIN_SCRIPT_DIR))
from train_gpt import (  # noqa: E402
    GPT,
    Hyperparameters,
    CastedLinear,
    _classify_param,
    dequantize_mixed_int6,
    mixed_quantize_int6,
    quantize_int6_per_row,
    quantize_float_tensor,
    CONTROL_TENSOR_NAME_PATTERNS,
    restore_low_dim_params_to_fp32,
    eval_val_sliding,
    build_sentencepiece_luts,
    load_validation_tokens,
    LOWP_DTYPE,
)

try:
    import sentencepiece as spm
    HAS_SPM = True
except ImportError:
    HAS_SPM = False

# ---------------------------------------------------------------------------
# Int5 quantizer — matches int6 semantics exactly, range [-16, 15]
# ---------------------------------------------------------------------------

def quantize_int5_per_row(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / 15.0).clamp_min(1.0 / 15.0).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -16, 15).to(torch.int8)
        return q, scale
    amax = t32.abs().max().item()
    scale = torch.tensor(amax / 15.0 if amax > 0 else 1.0, dtype=torch.float16)
    q = torch.clamp(torch.round(t32 / scale.float()), -16, 15).to(torch.int8)
    return q, scale


# ---------------------------------------------------------------------------
# Mixed int5/int6 quantizer — like mixed_quantize_int6 but with per-name routing
# ---------------------------------------------------------------------------

def mixed_quantize_int5_int6(
    state_dict: dict[str, torch.Tensor],
    int6_cats: set[str],
    int5_names: set[str],
) -> tuple[dict[str, torch.Tensor], dict]:
    """Quantize state_dict: int5 for names in int5_names, int6 for the rest.

    int5_names: exact tensor name strings that should use int5 (from probe JSON).
    int6_cats: categories that get int6 (e.g. {"mlp", "attn"}).
    Tensors not in int6_cats are passed through as fp16.
    """
    result: dict[str, torch.Tensor] = {}
    meta: dict[str, object] = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)

        # Mirrors mixed_quantize_int6 routing exactly — only the int5 vs int6
        # decision for mlp/attn tensors differs between Path A and Path B.
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough" if not t.is_floating_point() else "passthrough_fp16"
            continue

        if any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS):
            result[name] = t.float()
            meta[name] = "passthrough_ctrl"
            continue

        if cat in int6_cats and t.ndim >= 1:
            # mlp / attn: int5 if in schedule, else int6
            if name in int5_names:
                q, s = quantize_int5_per_row(t)
            else:
                q, s = quantize_int6_per_row(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            # embed / other big tensors: int8 (same as Path A)
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}

    return result, meta


# ---------------------------------------------------------------------------
# Model instantiation helpers
# ---------------------------------------------------------------------------

def build_eval_model(args: Hyperparameters, device: torch.device) -> torch.nn.Module:
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
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        rope_train_seq_len=args.rope_train_seq_len,
        ln_scale=args.ln_scale,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device=device, dtype=LOWP_DTYPE)
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    return model


def pack_and_compress(quant_result: dict, quant_meta: dict) -> bytes:
    """custom-shuffle + brotli-10 (the proven best path from compress_probe)."""
    packed = custom_pack(quant_result, quant_meta, shuffle=True)
    return compress(packed, "brotli", 10)


def unpack_and_decompress(blob: bytes) -> tuple[dict, dict]:
    raw = decompress(blob, "brotli")
    return custom_unpack(raw, shuffle=True)


# ---------------------------------------------------------------------------
# Single export path: quantize → pack → compress → decompress → load → eval
# ---------------------------------------------------------------------------

def run_export_path(
    label: str,
    sd_cpu: dict[str, torch.Tensor],
    quant_fn,
    args: Hyperparameters,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    code_bytes: int,
    sw_stride: int = 64,
) -> dict:
    print(f"\n{'='*60}")
    print(f"PATH {label}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    quant_result, quant_meta = quant_fn(sd_cpu)
    print(f"  Quantization: {1000*(time.perf_counter()-t0):.0f}ms")

    t0 = time.perf_counter()
    blob = pack_and_compress(quant_result, quant_meta)
    model_bytes = len(blob)
    total_bytes = code_bytes + model_bytes
    print(f"  Compression (custom-shuffle+brotli-10): {1000*(time.perf_counter()-t0):.0f}ms")
    print(f"  Model bytes: {model_bytes:,}  Total: {total_bytes:,}  "
          f"{'OK' if total_bytes < 16_000_000 else 'OVER 16MB'}")

    # Decompress and dequantize
    recovered_w, recovered_m = unpack_and_decompress(blob)
    deq_state = dequantize_mixed_int6(recovered_w, recovered_m, sd_cpu)

    # Load into eval model
    eval_model = build_eval_model(args, device)
    eval_model.load_state_dict(deq_state, strict=True)
    eval_model.eval()

    # Roundtrip eval (full context, no stride)
    dummy_args = Hyperparameters()
    dummy_args.val_batch_size = 524_288
    dummy_args.train_seq_len = 2048
    dummy_args.eval_stride = 64

    from train_gpt import eval_val
    t0 = time.perf_counter()
    rt_loss, rt_bpb = eval_val(
        dummy_args, eval_model, 0, 1, device, 1,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        eval_seq_len=2048,
    )
    rt_ms = 1000 * (time.perf_counter() - t0)
    print(f"  Roundtrip eval:  loss={rt_loss:.8f}  bpb={rt_bpb:.8f}  ({rt_ms:.0f}ms)")

    # Sliding window eval (stride configurable; s64 is the submission metric)
    t0 = time.perf_counter()
    sw_loss, sw_bpb = eval_val_sliding(
        dummy_args, eval_model, 0, 1, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=sw_stride,
        eval_seq_len=2048,
    )
    sw_ms = 1000 * (time.perf_counter() - t0)
    print(f"  Sliding s{sw_stride}:     loss={sw_loss:.8f}  bpb={sw_bpb:.8f}  ({sw_ms:.0f}ms)")

    return {
        "label": label,
        "model_bytes": model_bytes,
        "total_bytes": total_bytes,
        "fits_16mb": total_bytes < 16_000_000,
        "roundtrip_bpb": rt_bpb,
        "sliding_bpb": sw_bpb,
        "sw_stride": sw_stride,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export-only BPB A/B gate")
    parser.add_argument("--float-checkpoint", required=True,
                        help="Path to 05c-plus final_model.pt (float weights)")
    parser.add_argument("--probe-json", required=True,
                        help="Path to int5_tolerance_probe.json (conservative schedule)")
    parser.add_argument("--val-dir", required=True,
                        help="Directory containing fineweb_val_*.bin shards")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to fineweb_1024_bpe.model")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write results JSON")
    parser.add_argument("--code-bytes", type=int, default=69_000,
                        help="Estimated code bytes for total artifact size (default: 69000)")
    parser.add_argument("--sw-stride", type=int, default=64,
                        help="Sliding window stride (default: 64 = submission metric; use 512 for fast gate)")
    args_cli = parser.parse_args()

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load probe JSON to get conservative schedule tensor names ----
    print(f"\nLoading probe JSON: {args_cli.probe_json}")
    with open(args_cli.probe_json) as f:
        probe_data = json.load(f)

    # The probe JSON stores schedules_full as a list of dicts, each with a "name" key.
    # tensor_names are only in schedules_full (stripped from schedules for readability).
    schedules_full = probe_data.get("schedules_full", [])
    conservative = None
    for sc in schedules_full:
        if sc.get("name") == "conservative":
            conservative = sc
            break

    if conservative is None:
        available = [sc.get("name", "?") for sc in schedules_full]
        print(f"ERROR: 'conservative' schedule not found in probe JSON schedules_full.")
        print(f"Available schedules: {available}")
        sys.exit(1)

    int5_names: set[str] = set(conservative.get("tensor_names", []))
    print(f"Conservative schedule: {len(int5_names)} tensors → int5")
    for name in sorted(int5_names):
        print(f"  {name}")

    if not int5_names:
        print("ERROR: conservative schedule has no tensor names.")
        sys.exit(1)

    # ---- Load float checkpoint ----
    print(f"\nLoading float checkpoint: {args_cli.float_checkpoint}")
    raw_sd = torch.load(args_cli.float_checkpoint, map_location="cpu", weights_only=True)
    # Build sd_cpu in the same dtype used by the model (bfloat16 for LOWP params)
    sd_cpu = {k: v.detach().cpu().contiguous() for k, v in raw_sd.items()}
    print(f"  Loaded {len(sd_cpu)} tensors")

    # ---- Model hyperparameters (05c-plus anchor-locked values) ----
    hparams = Hyperparameters()
    # Override env-var-driven values to match 05c-plus anchor exactly
    hparams.vocab_size = 1024
    hparams.num_layers = 11
    hparams.model_dim = 512
    hparams.num_heads = 8
    hparams.num_kv_heads = 4
    hparams.mlp_mult = 3.0
    hparams.bigram_vocab_size = 2048
    hparams.bigram_dim = 128
    hparams.xsa_last_n = 11
    hparams.rope_dims = 16
    hparams.ln_scale = True
    hparams.ve_dim = 128
    hparams.ve_layers = "9,10"
    hparams.tie_embeddings = True
    hparams.tied_embed_init_std = 0.005
    hparams.logit_softcap = 30.0
    hparams.rope_base = 10000.0
    hparams.qk_gain_init = 1.5
    hparams.rope_train_seq_len = 1024
    hparams.val_batch_size = 524_288
    hparams.train_seq_len = 2048
    hparams.eval_stride = 64

    # ---- Load validation tokens + tokenizer LUTs ----
    if not HAS_SPM:
        print("ERROR: sentencepiece not available.")
        sys.exit(1)

    val_pattern = str(Path(args_cli.val_dir) / "fineweb_val_*.bin")
    print(f"\nLoading val tokens: {val_pattern}")
    val_tokens = load_validation_tokens(val_pattern, seq_len=2048)
    print(f"  Val tokens: {val_tokens.numel():,}")

    sp = spm.SentencePieceProcessor()
    sp.Load(args_cli.tokenizer)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, hparams.vocab_size, device
    )
    val_tokens = val_tokens.to(device)

    # ---- Define quantization callables ----
    def quant_a(sd):
        # Path A: uniform int6 for mlp+attn, rest passthrough
        return mixed_quantize_int6(sd, {"mlp", "attn"})

    def quant_b(sd):
        # Path B: conservative int5 for the 9 probe tensors, int6 for rest
        return mixed_quantize_int5_int6(sd, {"mlp", "attn"}, int5_names)

    code_bytes = args_cli.code_bytes

    # ---- Run A/B ----
    result_a = run_export_path(
        "A (uniform int6 + brotli-10)",
        sd_cpu, quant_a, hparams, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        code_bytes, sw_stride=args_cli.sw_stride,
    )
    result_b = run_export_path(
        "B (conservative int5/int6 + brotli-10)",
        sd_cpu, quant_b, hparams, device,
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        code_bytes, sw_stride=args_cli.sw_stride,
    )

    # ---- Decision summary ----
    delta_rt  = result_b["roundtrip_bpb"]  - result_a["roundtrip_bpb"]
    delta_sw  = result_b["sliding_bpb"] - result_a["sliding_bpb"]
    sw_stride = args_cli.sw_stride
    saved     = result_a["model_bytes"] - result_b["model_bytes"]

    print(f"\n{'='*60}")
    print("DECISION SUMMARY")
    print(f"{'='*60}")
    print(f"  Roundtrip BPB:  A={result_a['roundtrip_bpb']:.8f}  "
          f"B={result_b['roundtrip_bpb']:.8f}  delta={delta_rt:+.6f}")
    print(f"  Sliding s{sw_stride}:    A={result_a['sliding_bpb']:.8f}  "
          f"B={result_b['sliding_bpb']:.8f}  delta={delta_sw:+.6f}")
    print(f"  Model bytes:    A={result_a['model_bytes']:,}  "
          f"B={result_b['model_bytes']:,}  saved={saved:,}")
    print()

    # Gate criterion
    GATE_THRESHOLD = 0.002  # max acceptable sliding s64 BPB damage
    gate_pass = (delta_sw < GATE_THRESHOLD) and result_b["fits_16mb"]
    if gate_pass:
        print(f">>> GATE PASS  (delta_sw={delta_sw:+.6f} < {GATE_THRESHOLD})")
        print("    Proceed to 06a: mlp_mult=3.25 + conservative int5/int6 + brotli")
    else:
        if delta_sw >= GATE_THRESHOLD:
            print(f">>> GATE FAIL  (delta_sw={delta_sw:+.6f} >= {GATE_THRESHOLD})")
            print("    BPB damage is too large. Investigate which tensors are responsible.")
        if not result_b["fits_16mb"]:
            print(f">>> GATE FAIL  (total_bytes={result_b['total_bytes']:,} > 16,000,000)")
        print("    Do NOT proceed to 3.25x training. Stay at 3.0x or recheck schedule.")

    # ---- Write results JSON ----
    results = {
        "path_a": result_a,
        "path_b": result_b,
        "delta_roundtrip_bpb": delta_rt,
        f"delta_sliding_s{sw_stride}_bpb": delta_sw,
        "saved_model_bytes": saved,
        "int5_names": sorted(int5_names),
        "gate_threshold": GATE_THRESHOLD,
        "gate_pass": gate_pass,
    }
    out_path = output_dir / "export_bpb_ab.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
