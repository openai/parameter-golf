#!/usr/bin/env python3
"""Run quantization-only sweeps against an existing Parameter Golf checkpoint."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _set_env_defaults_from_summary(summary: dict[str, Any]) -> None:
    mappings = {
        "VOCAB_SIZE": "vocab_size",
        "NUM_LAYERS": "num_layers",
        "NUM_KV_HEADS": "num_kv_heads",
        "MODEL_DIM": "model_dim",
        "NUM_HEADS": "num_heads",
        "MLP_MULT": "mlp_mult",
        "MLP_HIDDEN_DIM": "mlp_hidden_dim",
        "MODEL_FAMILY": "model_family",
        "P20_RUNTIME_BACKEND": "p20_runtime_backend",
        "P20_LAYER_SCHEDULE": "p20_layer_schedule",
        "P20_STATE_BLOCKS": "p20_state_blocks_requested",
        "P20_BLOCK_PAIR_WIDTH_CAP": "p20_block_pair_width_cap",
        "P20_SCAN_CHUNK_SIZE": "p20_scan_chunk_size",
        "P20_ADAPTER_DIM": "p20_adapter_dim",
        "P20_ADAPTER_SCALE_INIT": "p20_adapter_scale_init",
        "P20_LOOP_REPEATS": "p20_loop_repeats",
        "P20_PRIMITIVE_LOOP_REPEATS": "p20_primitive_loop_repeats",
        "P20_PRIMITIVE_LOOP_DELTA_SCALE": "p20_primitive_loop_delta_scale",
        "LOOP_DELTA_SCALE": "loop_delta_scale",
        "TRAIN_SEQ_LEN": "train_seq_len",
        "VAL_BATCH_SIZE": "val_batch_size",
        "QUANT_COMPRESSION": "quant_compression",
    }
    for env_name, key in mappings.items():
        value = summary.get(key)
        if value is None or value == "":
            continue
        os.environ.setdefault(env_name, str(value))

    if "tie_embeddings" in summary:
        os.environ.setdefault("TIE_EMBEDDINGS", "1" if summary["tie_embeddings"] else "0")
    os.environ.setdefault("QUANT_FORMAT", "mixed_int6_clipsearch")
    os.environ.setdefault("MIXED_INT6_CLIP_QUANTILES", "0.999,0.9995,0.9999,0.99999,1.0")
    if "compiled_model" in summary:
        os.environ.setdefault("COMPILE_MODEL", "1" if summary["compiled_model"] else "0")
    else:
        os.environ.setdefault("COMPILE_MODEL", "0")
    os.environ.setdefault("COMPILE_MUON_BACKEND", "0")


def _parse_categories(raw: str) -> frozenset[str]:
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _variant_contracts(default_patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    p20_small_fp32_patterns = tuple(
        dict.fromkeys(
            (
                *default_patterns,
                "residual_scale",
                "input_norm",
                "output_norm",
                "primitive.in_projection.projection.bias",
                "primitive.state_transform_projection.bias",
                "readout_projection.bias",
            )
        )
    )
    return [
        {
            "name": "p20_int8_default",
            "mixed_int6_categories": "attn,mlp",
            "fp32_patterns": default_patterns,
            "note": "Reproduce the Step 3 leader: attention/MLP int6, P20 large matrices int8, small tensors passthrough.",
        },
        {
            "name": "p20_int8_p20_small_fp32",
            "mixed_int6_categories": "attn,mlp",
            "fp32_patterns": p20_small_fp32_patterns,
            "note": "Keep P20 residual/norm/bias tensors in fp32 instead of fp16 passthrough.",
        },
        {
            "name": "attn_int8_mlp_int6_p20_int8",
            "mixed_int6_categories": "mlp",
            "fp32_patterns": default_patterns,
            "note": "Spend bytes protecting attention and P20 at int8 while leaving MLP at int6.",
        },
        {
            "name": "attn_int6_mlp_int8_p20_int8",
            "mixed_int6_categories": "attn",
            "fp32_patterns": default_patterns,
            "note": "Spend bytes protecting MLP and P20 at int8 while leaving attention at int6.",
        },
        {
            "name": "all_large_int8",
            "mixed_int6_categories": "",
            "fp32_patterns": default_patterns,
            "note": "Use int8 for all large tensors; this is the upper-bound quantization-protection check.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-model-path", required=True)
    parser.add_argument("--source-summary-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    args_ns = parser.parse_args()

    source_summary_path = Path(args_ns.source_summary_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    _set_env_defaults_from_summary(source_summary)

    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root))

    import torch
    import train_gpt

    out_dir = Path(args_ns.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    args = train_gpt.Hyperparameters()
    sp = train_gpt.spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)
    val_tokens = train_gpt.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = train_gpt.build_sentencepiece_luts(
        sp,
        args.vocab_size,
        device,
    )

    base_model = train_gpt.build_model(args).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, train_gpt.CastedLinear):
            module.float()
    train_gpt.restore_low_dim_params_to_fp32(base_model)
    use_compiled_model = train_gpt.resolve_compile_model(args)
    eval_model = (
        torch.compile(base_model, dynamic=False, fullgraph=args.model_family == "baseline")
        if use_compiled_model
        else base_model
    )

    raw_model_path = Path(args_ns.raw_model_path)
    state = torch.load(raw_model_path, map_location="cpu")
    base_model.load_state_dict(state, strict=True)
    train_gpt.restore_low_dim_params_to_fp32(base_model)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in base_model.state_dict().items()}

    torch.cuda.synchronize()
    pre_t0 = time.perf_counter()
    pre_loss, pre_bpb = train_gpt.eval_val(
        args,
        eval_model,
        0,
        1,
        device,
        8,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    pre_eval_ms = 1000.0 * (time.perf_counter() - pre_t0)

    code_bytes, code_files = train_gpt.collect_submission_code_bytes(Path(train_gpt.__file__).resolve())
    default_fp32_patterns = train_gpt.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS
    variants: list[dict[str, Any]] = []

    for contract in _variant_contracts(default_fp32_patterns):
        train_gpt.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(contract["fp32_patterns"])
        categories = _parse_categories(contract["mixed_int6_categories"])
        torch.cuda.synchronize()
        quant_t0 = time.perf_counter()
        quant_obj, quant_stats = train_gpt.quantize_state_dict_mixed_int6_clipsearch(
            state,
            int6_categories=categories,
            clip_quantiles=args.mixed_int6_clip_quantiles,
        )
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_blob = train_gpt.compress_quant_payload(quant_raw, args.quant_compression)
        quant_path = out_dir / f"{args_ns.run_id}_{contract['name']}.int6clip.{args.quant_compression}.ptz"
        quant_path.write_bytes(quant_blob)
        quant_time_ms = 1000.0 * (time.perf_counter() - quant_t0)

        roundtrip_obj = torch.load(
            io.BytesIO(train_gpt.decompress_quant_payload(quant_blob, args.quant_compression)),
            map_location="cpu",
        )
        base_model.load_state_dict(train_gpt.dequantize_state_dict_export(roundtrip_obj), strict=True)
        train_gpt.restore_low_dim_params_to_fp32(base_model)

        torch.cuda.synchronize()
        eval_t0 = time.perf_counter()
        post_loss, post_bpb = train_gpt.eval_val(
            args,
            eval_model,
            0,
            1,
            device,
            8,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        post_eval_ms = 1000.0 * (time.perf_counter() - eval_t0)

        variant = {
            "name": contract["name"],
            "note": contract["note"],
            "mixed_int6_categories": sorted(categories),
            "fp32_patterns": list(contract["fp32_patterns"]),
            "post_quant_val_loss": post_loss,
            "post_quant_val_bpb": post_bpb,
            "post_minus_pre_bpb": post_bpb - pre_bpb,
            "post_quant_eval_time_ms": post_eval_ms,
            "quant_time_ms": quant_time_ms,
            "quant_file_bytes": quant_path.stat().st_size,
            "quant_raw_bytes": len(quant_raw),
            "total_submission_bytes_quantized": quant_path.stat().st_size + code_bytes,
            "quant_path": str(quant_path),
            "quant_stats": quant_stats,
        }
        variants.append(variant)
        print("variant " + json.dumps(variant, sort_keys=True), flush=True)

    best = min(variants, key=lambda item: item["post_quant_val_bpb"])
    summary = {
        "event": "checkpoint_requant_sweep_completed",
        "experiment_kind": "step4_quantization_protection_sweep",
        "run_id": args_ns.run_id,
        "source_run_id": source_summary.get("run_id"),
        "raw_model_path": str(raw_model_path),
        "source_summary_path": str(source_summary_path),
        "code_bytes": code_bytes,
        "code_files": code_files,
        "compiled_model": use_compiled_model,
        "model_family": args.model_family,
        "vocab_size": args.vocab_size,
        "num_layers": args.num_layers,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_mult": args.mlp_mult,
        "mlp_hidden_dim": args.mlp_hidden_dim if args.mlp_hidden_dim > 0 else args.mlp_mult * args.model_dim,
        "p20_layer_schedule": args.p20_layer_schedule,
        "p20_runtime_backend": args.p20_runtime_backend,
        "p20_state_blocks": args.p20_state_blocks,
        "p20_block_pair_width_cap": args.p20_block_pair_width_cap,
        "pre_quant_val_loss": pre_loss,
        "pre_quant_val_bpb": pre_bpb,
        "pre_quant_eval_time_ms": pre_eval_ms,
        "quant_compression": args.quant_compression,
        "mixed_int6_clip_quantiles": list(args.mixed_int6_clip_quantiles),
        "best_variant": best["name"],
        "best_post_quant_val_bpb": best["post_quant_val_bpb"],
        "best_post_quant_val_loss": best["post_quant_val_loss"],
        "best_total_submission_bytes_quantized": best["total_submission_bytes_quantized"],
        "variants": variants,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("summary_path " + str(summary_path), flush=True)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
