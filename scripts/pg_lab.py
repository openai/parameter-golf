#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STEP_VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iterations>\d+) "
    r"val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) "
    r"train_time:(?P<train_time_ms>\d+)ms "
    r"step_avg:(?P<step_avg_ms>[-+]?\d+(?:\.\d+)?)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip "
    r"val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?) "
    r"eval_time:(?P<eval_time_ms>\d+)ms"
)
FINAL_EXACT_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact "
    r"val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?)"
)
DELTA_RE = re.compile(
    r"final_int8_zlib_delta "
    r"val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?)"
)
DELTA_EXACT_RE = re.compile(
    r"final_int8_zlib_delta_exact "
    r"val_loss:(?P<val_loss>[-+]?\d+(?:\.\d+)?) "
    r"val_bpb:(?P<val_bpb>[-+]?\d+(?:\.\d+)?)"
)
SERIALIZED_RE = re.compile(r"Serialized model: (?P<raw_model_bytes>\d+) bytes")
CODE_SIZE_RE = re.compile(r"Code size: (?P<code_bytes>\d+) bytes")
TOTAL_SIZE_RE = re.compile(r"Total submission size: (?P<raw_submission_bytes>\d+) bytes")
SERIALIZED_INT8_RE = re.compile(
    r"Serialized model int8\+zlib: (?P<file_bytes>\d+) bytes "
    r"\(payload:(?P<payload_bytes>\d+) raw_torch:(?P<raw_torch_bytes>\d+) "
    r"payload_ratio:(?P<payload_ratio>[-+]?\d+(?:\.\d+)?)x\)"
)
MLX_SAVED_MODEL_RE = re.compile(r"saved_model:(?P<saved_model_path>\S+) bytes:(?P<raw_model_bytes>\d+)")
MLX_SERIALIZED_INT8_RE = re.compile(
    r"serialized_model_int8_zlib:(?P<file_bytes>\d+) bytes "
    r"\(payload:(?P<payload_bytes>\d+) raw_pickle:(?P<raw_pickle_bytes>\d+) "
    r"payload_ratio:(?P<payload_ratio>[-+]?\d+(?:\.\d+)?)x\)"
)
TOTAL_SIZE_INT8_RE = re.compile(r"Total submission size int8\+zlib: (?P<total_submission_bytes>\d+) bytes")
PEAK_MEM_RE = re.compile(
    r"peak memory allocated: (?P<allocated_mib>\d+) MiB reserved: (?P<reserved_mib>\d+) MiB"
)
MODEL_PARAMS_RE = re.compile(r"model_params:(?P<model_params>\d+)")
MODEL_LAYOUT_RE = re.compile(
    r"model_params:(?P<model_params>\d+) "
    r"vocab_size:(?P<vocab_size>\d+) "
    r"layers:(?P<num_layers>\d+) "
    r"dim:(?P<model_dim>\d+) "
    r"heads:(?P<num_heads>\d+) "
    r"kv_heads:(?P<num_kv_heads>\d+) "
    r"seq_len:(?P<train_seq_len>\d+) "
    r"tie_embeddings:(?P<tie_embeddings>True|False)"
)
TRAIN_META_RE = re.compile(
    r"iterations:(?P<iterations>\d+) "
    r"train_batch_tokens:(?P<train_batch_tokens>\d+) "
    r"grad_accum_steps:(?P<grad_accum_steps>\d+) "
    r"microbatch_tokens:(?P<microbatch_tokens>\d+) "
    r"microbatch_batch_size:(?P<microbatch_batch_size>\d+) "
    r"val_batch_size:(?P<val_batch_size>\d+) "
    r"warmup_steps:(?P<warmup_steps>\d+) "
    r"max_wallclock_seconds:(?P<max_wallclock_seconds>[-+]?\d+(?:\.\d+)?)"
)
TRAIN_META_LEGACY_RE = re.compile(
    r"train_batch_tokens:(?P<train_batch_tokens>\d+) "
    r"train_seq_len:(?P<train_seq_len>\d+) "
    r"iterations:(?P<iterations>\d+) "
    r"warmup_steps:(?P<warmup_steps>\d+) "
    r"max_wallclock_seconds:(?P<max_wallclock_seconds>[-+]?\d+(?:\.\d+)?)"
)
VAL_SUBSET_RE = re.compile(
    r"val_loader:subset max_seqs:(?P<val_max_seqs>\d+) "
    r"actual_seqs:(?P<actual_seqs>\d+)"
)
TIE_LR_RE = re.compile(
    r"tie_embeddings:(?P<tie_embeddings>True|False) "
    r"embed_lr:(?P<embed_lr>[-+]?\d+(?:\.\d+)?) "
    r"head_lr:(?P<head_lr>[-+]?\d+(?:\.\d+)?) "
    r"matrix_lr:(?P<matrix_lr>[-+]?\d+(?:\.\d+)?) "
    r"scalar_lr:(?P<scalar_lr>[-+]?\d+(?:\.\d+)?)"
)
ATTN_RE = re.compile(
    r"attention_mode:(?P<attention_mode>\S+) "
    r"num_heads:(?P<num_heads>\d+) "
    r"num_kv_heads:(?P<num_kv_heads>\d+)"
)
QUANT_CONFIG_RE = re.compile(
    r"quantization_config "
    r"clip_percentile:(?P<clip_percentile>[-+]?\d+(?:\.\d+)?) "
    r"keep_float_max_numel:(?P<keep_float_max_numel>\d+) "
    r"keep_float_store_dtype:(?P<keep_float_store_dtype>\S+) "
    r"per_row_scale_dtype:(?P<per_row_scale_dtype>\S+) "
    r"keep_float_fp32_patterns:(?P<keep_float_fp32_patterns>.*)"
)
QUANT_STATS_RE = re.compile(
    r"quantization_stats "
    r"num_tensors:(?P<num_tensors>\d+) "
    r"float_tensors:(?P<num_float_tensors>\d+) "
    r"nonfloat_tensors:(?P<num_nonfloat_tensors>\d+) "
    r"passthrough_float_tensors:(?P<num_passthrough_float_tensors>\d+) "
    r"per_row_tensors:(?P<num_per_row_tensors>\d+) "
    r"per_tensor_tensors:(?P<num_per_tensor_tensors>\d+) "
    r"baseline_tensor_bytes:(?P<baseline_tensor_bytes>\d+) "
    r"payload_bytes:(?P<int8_payload_bytes>\d+) "
    r"passthrough_bytes:(?P<passthrough_bytes>\d+) "
    r"quantized_value_bytes:(?P<quantized_value_bytes>\d+) "
    r"scale_bytes:(?P<scale_bytes>\d+)"
)

PROFILE_ENVS: dict[str, dict[str, str]] = {
    "mlx-smoke": {
        "VOCAB_SIZE": "1024",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_SEQ_LEN": "1024",
        "ITERATIONS": "200",
        "TRAIN_BATCH_TOKENS": "8192",
        "VAL_LOSS_EVERY": "0",
        "VAL_BATCH_SIZE": "8192",
        "VAL_MAX_SEQS": "512",
    },
    "cuda-smoke": {
        "VOCAB_SIZE": "1024",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TRAIN_SEQ_LEN": "1024",
        "ITERATIONS": "200",
        "TRAIN_BATCH_TOKENS": "8192",
        "VAL_LOSS_EVERY": "0",
        "VAL_BATCH_SIZE": "8192",
        "VAL_MAX_SEQS": "512",
        "MAX_WALLCLOCK_SECONDS": "0",
    },
    "cuda-baseline-1x": {
        "VOCAB_SIZE": "1024",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TIE_EMBEDDINGS": "1",
        "TIED_EMBED_LR": "0.05",
        "TRAIN_BATCH_TOKENS": "524288",
        "TRAIN_SEQ_LEN": "1024",
        "TRAIN_LOG_EVERY": "50",
        "VAL_LOSS_EVERY": "200",
    },
    "cuda-baseline-8x": {
        "NCCL_IB_DISABLE": "1",
        "VOCAB_SIZE": "1024",
        "NUM_LAYERS": "9",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2",
        "TIE_EMBEDDINGS": "1",
        "TIED_EMBED_LR": "0.05",
        "TRAIN_BATCH_TOKENS": "524288",
        "TRAIN_SEQ_LEN": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "TRAIN_LOG_EVERY": "50",
        "VAL_LOSS_EVERY": "200",
    },
}

PROFILE_COMMANDS: dict[str, list[str]] = {
    "mlx-smoke": ["python3", "train_gpt_mlx.py"],
    "cuda-smoke": ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
    "cuda-baseline-1x": ["torchrun", "--standalone", "--nproc_per_node=1", "train_gpt.py"],
    "cuda-baseline-8x": ["torchrun", "--standalone", "--nproc_per_node=8", "train_gpt.py"],
}


def default_paths_for_variant(variant: str) -> tuple[str, str | None]:
    dataset = f"./data/datasets/fineweb10B_{variant}"
    if variant.startswith("sp") and variant[2:].isdigit():
        vocab_size = variant[2:]
        return dataset, f"./data/tokenizers/fineweb_{vocab_size}_bpe.model"
    if variant == "byte260":
        return dataset, None
    return dataset, None


def short_tokens(value: str) -> str:
    n = int(value)
    if n >= 1_000:
        return f"{n // 1_000}k"
    return str(n)


def build_run_id(env: dict[str, str], stage: str, variant: str, focus: str, seed: str) -> str:
    return (
        f"pg_{stage}_{variant}_"
        f"l{env.get('NUM_LAYERS', 'na')}_"
        f"d{env.get('MODEL_DIM', 'na')}_"
        f"h{env.get('NUM_HEADS', 'na')}_"
        f"kv{env.get('NUM_KV_HEADS', 'na')}_"
        f"m{env.get('MLP_MULT', 'na')}_"
        f"tb{short_tokens(env.get('TRAIN_BATCH_TOKENS', '0'))}_"
        f"sl{env.get('TRAIN_SEQ_LEN', 'na')}_"
        f"{focus}_s{seed}"
    )


def expand_log_paths(items: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in items:
        if any(ch in item for ch in "*?[]"):
            for match in sorted(Path().glob(item)):
                if match.is_file():
                    paths.append(match)
            continue
        path = Path(item)
        if path.is_dir():
            candidate = path / "train.log"
            if candidate.is_file():
                paths.append(candidate)
            continue
        if path.is_file():
            paths.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def _coerce_number(value: str) -> int | float:
    if "." in value or "e" in value.lower():
        return float(value)
    return int(value)


def _apply_match(data: dict[str, Any], match: re.Match[str] | None) -> None:
    if not match:
        return
    for key, value in match.groupdict().items():
        data[key] = _coerce_number(value) if value and value[0] in "-+0123456789" else value


def parse_log(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {
        "log_path": str(path.resolve()),
        "run_name": path.parent.name if path.name == "train.log" else path.stem,
    }
    pre_quant: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    final_exact: dict[str, Any] | None = None
    delta: dict[str, Any] | None = None
    delta_exact: dict[str, Any] | None = None
    smoke_subset: dict[str, Any] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if match := STEP_VAL_RE.search(line):
                pre_quant = {k: _coerce_number(v) for k, v in match.groupdict().items()}
                continue
            if match := FINAL_RE.search(line):
                final = {k: _coerce_number(v) for k, v in match.groupdict().items()}
                continue
            if match := FINAL_EXACT_RE.search(line):
                final_exact = {k: _coerce_number(v) for k, v in match.groupdict().items()}
                continue
            if match := DELTA_RE.search(line):
                delta = {k: _coerce_number(v) for k, v in match.groupdict().items()}
                continue
            if match := DELTA_EXACT_RE.search(line):
                delta_exact = {k: _coerce_number(v) for k, v in match.groupdict().items()}
                continue
            if match := VAL_SUBSET_RE.search(line):
                smoke_subset = {k: _coerce_number(v) for k, v in match.groupdict().items()}
                continue
            _apply_match(data, SERIALIZED_RE.search(line))
            _apply_match(data, CODE_SIZE_RE.search(line))
            _apply_match(data, TOTAL_SIZE_RE.search(line))
            _apply_match(data, SERIALIZED_INT8_RE.search(line))
            _apply_match(data, MLX_SAVED_MODEL_RE.search(line))
            _apply_match(data, MLX_SERIALIZED_INT8_RE.search(line))
            _apply_match(data, TOTAL_SIZE_INT8_RE.search(line))
            _apply_match(data, PEAK_MEM_RE.search(line))
            _apply_match(data, MODEL_LAYOUT_RE.search(line))
            _apply_match(data, MODEL_PARAMS_RE.search(line))
            _apply_match(data, TRAIN_META_RE.search(line))
            _apply_match(data, TRAIN_META_LEGACY_RE.search(line))
            _apply_match(data, TIE_LR_RE.search(line))
            _apply_match(data, ATTN_RE.search(line))
            _apply_match(data, QUANT_CONFIG_RE.search(line))
            _apply_match(data, QUANT_STATS_RE.search(line))

    if pre_quant:
        data["pre_quant"] = pre_quant
    if final:
        data["post_quant"] = final
    if final_exact:
        data["post_quant_exact"] = final_exact
    if delta:
        data["post_quant_delta"] = delta
    if delta_exact:
        data["post_quant_delta_exact"] = delta_exact
    if smoke_subset:
        data["val_loader_subset"] = smoke_subset

    if delta_exact:
        data["quant_delta_bpb"] = float(delta_exact["val_bpb"])
        data["quant_delta_val_loss"] = float(delta_exact["val_loss"])
    elif final_exact and pre_quant:
        data["quant_delta_bpb"] = round(float(final_exact["val_bpb"]) - float(pre_quant["val_bpb"]), 8)
        data["quant_delta_val_loss"] = round(float(final_exact["val_loss"]) - float(pre_quant["val_loss"]), 8)
    elif pre_quant and final:
        data["quant_delta_bpb"] = round(float(final["val_bpb"]) - float(pre_quant["val_bpb"]), 8)
        data["quant_delta_val_loss"] = round(float(final["val_loss"]) - float(pre_quant["val_loss"]), 8)
    if "file_bytes" in data:
        data["int8_zlib_bytes"] = data.pop("file_bytes")
    if "payload_ratio" in data:
        data["payload_ratio"] = float(data["payload_ratio"])
    if "total_submission_bytes" in data:
        data["fits_16mb_cap"] = int(data["total_submission_bytes"]) <= 16_000_000
    return data


def render_summary(result: dict[str, Any]) -> str:
    lines = [
        f"run: {result['run_name']}",
        f"log: {result['log_path']}",
    ]
    pre_quant = result.get("pre_quant", {})
    post_quant = result.get("post_quant_exact") or result.get("post_quant", {})
    if pre_quant:
        lines.append(
            "pre_quant: "
            f"val_bpb={pre_quant.get('val_bpb')} "
            f"val_loss={pre_quant.get('val_loss')} "
            f"step={pre_quant.get('step')}/{pre_quant.get('iterations')} "
            f"step_avg_ms={pre_quant.get('step_avg_ms')}"
        )
    if post_quant:
        lines.append(
            "post_quant: "
            f"val_bpb={post_quant.get('val_bpb')} "
            f"val_loss={post_quant.get('val_loss')}"
        )
    if result.get("post_quant_delta_exact"):
        lines.append(
            "delta_exact: "
            f"val_bpb={result['post_quant_delta_exact'].get('val_bpb')} "
            f"val_loss={result['post_quant_delta_exact'].get('val_loss')}"
        )
    if "quant_delta_bpb" in result:
        lines.append(
            f"delta: val_bpb={result['quant_delta_bpb']} "
            f"val_loss={result.get('quant_delta_val_loss')}"
        )
    if "total_submission_bytes" in result:
        lines.append(
            "bytes: "
            f"int8_zlib={result.get('int8_zlib_bytes')} "
            f"code={result.get('code_bytes')} "
            f"total={result.get('total_submission_bytes')} "
            f"fits_16mb={result.get('fits_16mb_cap')}"
        )
    if "payload_ratio" in result:
        lines.append(
            "quant: "
            f"payload_ratio={result.get('payload_ratio')} "
            f"clip={result.get('clip_percentile')} "
            f"keep_float_max={result.get('keep_float_max_numel')}"
        )
    return "\n".join(lines)


def compare_logs(paths: list[Path]) -> list[dict[str, Any]]:
    rows = [parse_log(path) for path in paths]
    rows.sort(
        key=lambda row: (
            float((row.get("post_quant_exact") or row.get("post_quant") or {}).get("val_bpb", float("inf"))),
            int(row.get("total_submission_bytes", 10**18)),
        )
    )
    return rows


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "record"


def variant_vocab_size(variant: str) -> int | None:
    if variant.startswith("sp") and variant[2:].isdigit():
        return int(variant[2:])
    if variant == "byte260":
        return 260
    return None


def variant_tokenizer_path(variant: str) -> str | None:
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"./data/tokenizers/fineweb_{variant[2:]}_bpe.model"
    return None


def ensure_variant_supported_by_trainers(variant: str) -> None:
    if variant == "byte260":
        raise SystemExit(
            "byte260 is not supported by the current trainers. "
            "Both train_gpt.py and train_gpt_mlx.py currently require SentencePiece .model tokenizers, "
            "while byte260 is exported as a pure-byte JSON tokenizer. Use an sp* variant for now."
        )


def ensure_submission_ready(result: dict[str, Any]) -> None:
    if result.get("val_loader_subset"):
        subset = result["val_loader_subset"]
        raise SystemExit(
            f"Refusing to package smoke/subset validation log with VAL_MAX_SEQS={subset.get('val_max_seqs')}; "
            "real submission logs must use the full fixed validation split."
        )
    if "post_quant_exact" not in result:
        raise SystemExit("Refusing to package an incomplete log: missing final_int8_zlib_roundtrip_exact footer.")
    if "total_submission_bytes" not in result or "int8_zlib_bytes" not in result:
        raise SystemExit("Refusing to package an incomplete log: missing final submission byte totals.")


def build_submission_payload(args: argparse.Namespace, result: dict[str, Any]) -> dict[str, Any]:
    post_quant = result.get("post_quant_exact") or result.get("post_quant") or {}
    pre_quant = result.get("pre_quant") or {}
    payload: dict[str, Any] = {
        "author": args.author,
        "github_id": args.github_id,
        "name": args.record_name,
        "blurb": args.blurb,
        "date": args.date or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "val_loss": post_quant.get("val_loss"),
        "val_bpb": post_quant.get("val_bpb"),
    }
    if args.submission_track:
        payload["track"] = args.submission_track
    if pre_quant:
        payload["pre_quant_val_loss"] = pre_quant.get("val_loss")
        payload["pre_quant_val_bpb"] = pre_quant.get("val_bpb")
        payload["step_stop"] = pre_quant.get("step")
        if pre_quant.get("train_time_ms") is not None:
            payload["wallclock_seconds"] = round(float(pre_quant["train_time_ms"]) / 1000.0, 3)
    if result.get("total_submission_bytes") is not None:
        payload["bytes_total"] = result["total_submission_bytes"]
    if result.get("int8_zlib_bytes") is not None:
        payload["bytes_model_int8_zlib"] = result["int8_zlib_bytes"]
    if result.get("code_bytes") is not None:
        payload["bytes_code"] = result["code_bytes"]
    return payload


def build_record_readme(args: argparse.Namespace, result: dict[str, Any], copied_train_script_name: str) -> str:
    pre_quant = result.get("pre_quant") or {}
    post_quant = result.get("post_quant_exact") or result.get("post_quant") or {}
    lines = []
    lines.append(args.intro or "This record captures a prepared submission candidate generated from the current repo state.")
    lines.append("")
    lines.append("Configuration:")
    if args.track_label:
        lines.append(f"- Track: `{args.track_label}`")
    if result.get("model_params") is not None:
        lines.append(f"- Model params: `{result['model_params']}`")
    layout_bits = []
    vocab_size = result.get("vocab_size") or variant_vocab_size(args.variant)
    if vocab_size is not None:
        layout_bits.append(f"VOCAB_SIZE={vocab_size}")
    for key, label in (
        ("num_layers", "layers"),
        ("model_dim", "dim"),
        ("num_heads", "heads"),
        ("num_kv_heads", "kv_heads"),
        ("train_seq_len", "seq_len"),
    ):
        if result.get(key) is not None:
            layout_bits.append(f"{label}={result[key]}")
    if layout_bits:
        lines.append(f"- Layout: `{' '.join(layout_bits)}`")
    if result.get("train_batch_tokens") is not None:
        lines.append(
            f"- Batching: `TRAIN_BATCH_TOKENS={result['train_batch_tokens']}`"
            + (f" `TRAIN_SEQ_LEN={result['train_seq_len']}`" if result.get("train_seq_len") is not None else "")
        )
    if args.command:
        lines.append("")
        lines.append("Command (track-relevant params):")
        lines.append("```bash")
        lines.append(args.command)
        lines.append("```")
    lines.append("")
    lines.append("Key metrics (from `train.log`):")
    if pre_quant:
        lines.append(
            f"- Pre-quant eval at stop: `val_loss:{pre_quant.get('val_loss')}`, `val_bpb:{pre_quant.get('val_bpb')}`"
        )
    if post_quant:
        lines.append(
            f"- Post-quant roundtrip eval: `val_loss:{post_quant.get('val_loss')}`, `val_bpb:{post_quant.get('val_bpb')}`"
        )
    if result.get("quant_delta_bpb") is not None:
        lines.append(f"- Post-quant delta: `val_bpb:+{result['quant_delta_bpb']}`")
    if pre_quant.get("train_time_ms") is not None and pre_quant.get("step_avg_ms") is not None:
        lines.append(
            f"- Train time: `{pre_quant['train_time_ms']}ms` (`step_avg:{pre_quant['step_avg_ms']}ms`)"
        )
    if result.get("allocated_mib") is not None:
        lines.append(
            f"- Peak memory: `{result['allocated_mib']} MiB allocated`, `{result.get('reserved_mib')} MiB reserved`"
        )
    if result.get("int8_zlib_bytes") is not None:
        lines.append(f"- Serialized model int8+zlib: `{result['int8_zlib_bytes']} bytes`")
    if result.get("code_bytes") is not None:
        lines.append(f"- Code size: `{result['code_bytes']} bytes`")
    if result.get("total_submission_bytes") is not None:
        lines.append(f"- Total submission size int8+zlib: `{result['total_submission_bytes']} bytes`")
    lines.append("")
    lines.append("Included files:")
    lines.append(f"- `{copied_train_script_name}` (code snapshot used for the run)")
    lines.append("- `train.log` (exact run log)")
    lines.append("- `submission.json` (metadata for the PR)")
    if args.extra_files:
        for extra in args.extra_files:
            lines.append(f"- `{Path(extra).name}`")
    return "\n".join(lines) + "\n"


def print_compare_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "run",
        "post_bpb",
        "pre_bpb",
        "delta",
        "total_bytes",
        "int8_bytes",
        "step_avg_ms",
    ]
    values = [headers]
    for row in rows:
        pre = row.get("pre_quant", {})
        post = row.get("post_quant_exact") or row.get("post_quant") or {}
        values.append(
            [
                str(row["run_name"]),
                str(post.get("val_bpb", "")),
                str(pre.get("val_bpb", "")),
                str(row.get("quant_delta_bpb", "")),
                str(row.get("total_submission_bytes", "")),
                str(row.get("int8_zlib_bytes", "")),
                str(pre.get("step_avg_ms", "")),
            ]
        )
    widths = [max(len(row[i]) for row in values) for i in range(len(headers))]
    for idx, row in enumerate(values):
        line = "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(row))
        print(line)
        if idx == 0:
            print("  ".join("-" * width for width in widths))


def cmd_prepare_record(args: argparse.Namespace) -> int:
    log_path = Path(args.log).resolve()
    result = parse_log(log_path)
    ensure_submission_ready(result)
    date_prefix = args.folder_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    record_dir = Path(args.output_root) / f"{date_prefix}_{slugify(args.record_name)}"
    if record_dir.exists():
        raise SystemExit(f"Record directory already exists: {record_dir}")

    record_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(log_path, record_dir / "train.log")

    train_script_path = Path(args.train_script).resolve()
    copied_train_script_name = train_script_path.name
    shutil.copy2(train_script_path, record_dir / copied_train_script_name)

    for extra in args.extra_files:
        extra_path = Path(extra).resolve()
        shutil.copy2(extra_path, record_dir / extra_path.name)

    submission = build_submission_payload(args, result)
    with (record_dir / "submission.json").open("w", encoding="utf-8") as handle:
        json.dump(submission, handle, indent=2)
        handle.write("\n")

    with (record_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(build_record_readme(args, result, copied_train_script_name))

    print(record_dir.resolve())
    return 0


def cmd_command(args: argparse.Namespace) -> int:
    env = dict(PROFILE_ENVS[args.profile])
    if args.variant:
        ensure_variant_supported_by_trainers(args.variant)
        dataset_path, tokenizer_path = default_paths_for_variant(args.variant)
        env["DATA_PATH"] = dataset_path
        vocab_size = variant_vocab_size(args.variant)
        if vocab_size is not None:
            env["VOCAB_SIZE"] = str(vocab_size)
        if tokenizer_path:
            env["TOKENIZER_PATH"] = tokenizer_path
    if args.data_path:
        env["DATA_PATH"] = args.data_path
    if args.tokenizer_path:
        env["TOKENIZER_PATH"] = args.tokenizer_path
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    seed = env.get("SEED", args.seed)
    if args.run_id:
        env["RUN_ID"] = args.run_id
    else:
        env["RUN_ID"] = build_run_id(env, args.stage, args.variant, args.focus, seed)
        if args.stamp:
            env["RUN_ID"] = f"{env['RUN_ID']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    cmd = PROFILE_COMMANDS[args.profile]
    lines = [f"{key}={shlex.quote(value)} \\" for key, value in sorted(env.items())]
    lines.append(" ".join(shlex.quote(part) for part in cmd))
    print("\n".join(lines))
    return 0


def cmd_parse_log(args: argparse.Namespace) -> int:
    result = parse_log(Path(args.path))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_summary(result))
    return 0


def cmd_compare_logs(args: argparse.Namespace) -> int:
    paths = expand_log_paths(args.paths)
    if not paths:
        raise SystemExit("No log files found.")
    rows = compare_logs(paths)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        print_compare_table(rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parameter Golf baseline + experiment helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    command_parser = subparsers.add_parser("command", help="Print a reproducible run command.")
    command_parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(PROFILE_ENVS),
        help="Run profile to print.",
    )
    command_parser.add_argument("--variant", default="sp1024", help="Tokenizer/data variant label.")
    command_parser.add_argument("--stage", default="base", help="Run-id stage label, e.g. base/sweep/cmp/arch.")
    command_parser.add_argument("--focus", default="ref", help="Run-id focus label.")
    command_parser.add_argument("--seed", default="1337", help="Seed suffix for auto-generated run ids.")
    command_parser.add_argument("--run-id", help="Explicit run id. Overrides auto-generated naming.")
    command_parser.add_argument("--stamp", action="store_true", help="Append a UTC timestamp to the run id.")
    command_parser.add_argument("--data-path", help="Override DATA_PATH.")
    command_parser.add_argument("--tokenizer-path", help="Override TOKENIZER_PATH.")
    command_parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra env overrides. Repeatable.",
    )
    command_parser.set_defaults(func=cmd_command)

    parse_parser = subparsers.add_parser("parse-log", help="Parse one train log.")
    parse_parser.add_argument("path", help="Path to a train.log file.")
    parse_parser.add_argument("--json", action="store_true", help="Print JSON.")
    parse_parser.set_defaults(func=cmd_parse_log)

    compare_parser = subparsers.add_parser("compare-logs", help="Compare multiple logs or record folders.")
    compare_parser.add_argument("paths", nargs="+", help="Log paths, record folders, or glob patterns.")
    compare_parser.add_argument("--json", action="store_true", help="Print JSON.")
    compare_parser.set_defaults(func=cmd_compare_logs)

    record_parser = subparsers.add_parser("prepare-record", help="Create a records/ folder from a completed run log.")
    record_parser.add_argument("--log", required=True, help="Path to the completed train log.")
    record_parser.add_argument(
        "--output-root",
        default="records/track_non_record_16mb",
        help="Target records root, e.g. records/track_10min_16mb.",
    )
    record_parser.add_argument("--folder-date", help="Override the YYYY-MM-DD folder prefix.")
    record_parser.add_argument("--record-name", required=True, help="Human-readable record name.")
    record_parser.add_argument("--author", required=True, help="Submission author name.")
    record_parser.add_argument("--github-id", required=True, help="GitHub username.")
    record_parser.add_argument("--blurb", required=True, help="Short submission blurb.")
    record_parser.add_argument("--train-script", default="train_gpt.py", help="Training script to snapshot.")
    record_parser.add_argument("--command", help="Exact command to include in README.")
    record_parser.add_argument("--variant", default="sp1024", help="Variant label for README context.")
    record_parser.add_argument("--date", help="ISO8601 date for submission.json.")
    record_parser.add_argument("--track-label", help="Human-readable track label for README.")
    record_parser.add_argument("--submission-track", help="Optional submission.json track value.")
    record_parser.add_argument("--intro", help="README intro paragraph.")
    record_parser.add_argument("--extra-files", nargs="*", default=[], help="Extra dependency files to copy.")
    record_parser.set_defaults(func=cmd_prepare_record)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
