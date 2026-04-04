#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import io
import re
import statistics
import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_TRAIN_SCRIPT = SCRIPT_PATH.with_name("train_gpt.py")
DEFAULT_EXCLUDE_FILES = {
    "final_model.pt",
    "final_model.int8.ptz",
}
DEFAULT_EXCLUDE_DIRS = {
    "__pycache__",
    "logs",
}
SERIALIZED_MODEL_RE = re.compile(
    r"Serialized model .*: (?P<compressed>\d+) bytes "
    r"\(payload:(?P<payload>\d+) raw_torch:(?P<raw>\d+) payload_ratio:(?P<payload_ratio>[0-9.]+)x\)"
)


def load_train_module(train_script: Path):
    spec = importlib.util.spec_from_file_location("submission_train_gpt", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import train script from {train_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_state_dict(obj):
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()):
            return obj
        for key in ("state_dict", "model", "model_state_dict"):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(
                isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in value.items()
            ):
                return value
    raise ValueError(
        "Could not find a raw model state_dict in checkpoint. "
        "Expected a dict[str, Tensor] or a checkpoint containing one of: "
        "'state_dict', 'model', or 'model_state_dict'."
    )


def build_default_model_state_dict(module) -> dict[str, torch.Tensor]:
    args = module.Hyperparameters()
    torch.manual_seed(args.seed)
    model = module.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        rope_dims=args.rope_dims,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n,
    )
    module.restore_low_dim_params_to_fp32(model)
    return {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}


def file_bytes(path: Path) -> int:
    return path.stat().st_size


def iter_submission_files(submission_dir: Path):
    for path in sorted(submission_dir.rglob("*")):
        if not path.is_file():
            continue
        if any(part in DEFAULT_EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name in DEFAULT_EXCLUDE_FILES:
            continue
        if path.suffix == ".pyc":
            continue
        yield path


def parse_historical_compression_records(submission_dir: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    for path in sorted(submission_dir.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in SERIALIZED_MODEL_RE.finditer(text):
            compressed = int(match.group("compressed"))
            payload = int(match.group("payload"))
            raw = int(match.group("raw"))
            records.append(
                {
                    "path": str(path.relative_to(submission_dir)),
                    "compressed": compressed,
                    "payload": payload,
                    "raw": raw,
                    "compressed_over_raw": compressed / max(raw, 1),
                    "compressed_over_payload": compressed / max(payload, 1),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute submission size using the int8+zlib export path from the given train_gpt.py."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        help="PyTorch checkpoint or raw state_dict to size. Omit with --estimate-from-code.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=DEFAULT_TRAIN_SCRIPT,
        help=f"train_gpt.py to import quantization logic from (default: {DEFAULT_TRAIN_SCRIPT})",
    )
    parser.add_argument(
        "--submission-dir",
        type=Path,
        default=None,
        help="Submission directory whose code/files should count toward the total. Defaults to train script parent.",
    )
    parser.add_argument(
        "--estimate-from-code",
        action="store_true",
        help=(
            "Estimate model bytes from the model architecture in train_gpt.py without a checkpoint. "
            "This is not exact post-training zlib size."
        ),
    )
    args = parser.parse_args()

    # Convenience behavior:
    # - no positional argument defaults to estimate mode
    # - in estimate mode, a lone *.py positional is treated as --train-script
    if args.checkpoint is None:
        args.estimate_from_code = True
    elif args.estimate_from_code and args.checkpoint.suffix == ".py":
        args.train_script = args.checkpoint
        args.checkpoint = None

    train_script = args.train_script.resolve()
    if not train_script.is_file():
        raise FileNotFoundError(f"Train script not found: {train_script}")

    submission_dir = (args.submission_dir or train_script.parent).resolve()
    if not submission_dir.is_dir():
        raise NotADirectoryError(f"Submission directory not found: {submission_dir}")

    module = load_train_module(train_script)
    if not hasattr(module, "quantize_state_dict_int8"):
        raise AttributeError(f"{train_script} does not define quantize_state_dict_int8")
    module_hparams = module.Hyperparameters()

    checkpoint_path = None
    if args.estimate_from_code:
        if args.checkpoint is not None:
            raise ValueError(
                "Pass either a checkpoint or --estimate-from-code. "
                "If you want to point at a train script, use --train-script ./train_gpt.py."
            )
        state_dict = build_default_model_state_dict(module)
        mode = "estimate_from_code"
    else:
        if args.checkpoint is None:
            raise ValueError("checkpoint is required unless --estimate-from-code is set.")
        checkpoint_path = args.checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        raw_obj = torch.load(checkpoint_path, map_location="cpu")
        state_dict = extract_state_dict(raw_obj)
        mode = "checkpoint"

    quant_obj, quant_stats = module.quantize_state_dict_int8(state_dict)

    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob, compress_label = module.compress_quant_payload(quant_raw, module_hparams)

    train_script_bytes = len(train_script.read_text(encoding="utf-8").encode("utf-8"))
    submission_files = list(iter_submission_files(submission_dir))
    submission_non_model_bytes = sum(file_bytes(path) for path in submission_files)
    synthetic_compressed_bytes = len(quant_blob)
    quant_raw_bytes = len(quant_raw)
    payload_bytes = int(quant_stats["int8_payload_bytes"])
    baseline_tensor_bytes = int(quant_stats["baseline_tensor_bytes"])
    historical_records = parse_historical_compression_records(submission_dir)

    nearest_record = None
    median_raw_ratio = None
    median_payload_ratio = None
    calibrated_compressed_bytes = None
    median_compressed_bytes = None
    if historical_records:
        nearest_record = min(historical_records, key=lambda r: abs(r["payload"] - payload_bytes))
        median_raw_ratio = statistics.median(r["compressed_over_raw"] for r in historical_records)
        median_payload_ratio = statistics.median(r["compressed_over_payload"] for r in historical_records)
        calibrated_compressed_bytes = int(round(quant_raw_bytes * nearest_record["compressed_over_raw"]))
        median_compressed_bytes = int(round(quant_raw_bytes * median_raw_ratio))

    print(f"train_script={train_script}")
    print(f"submission_dir={submission_dir}")
    print(f"mode={mode}")
    if checkpoint_path is not None:
        print(f"checkpoint={checkpoint_path}")
    print(f"compressor={compress_label}")
    print(f"train_gpt_py_bytes={train_script_bytes}")
    print(f"submission_non_model_bytes={submission_non_model_bytes}")
    print(f"model_int8_payload_bytes={payload_bytes}")
    print(f"model_int8_torchsave_bytes={quant_raw_bytes}")
    print(f"model_compressed_bytes_synthetic={synthetic_compressed_bytes}")
    print(f"official_total_bytes_synthetic={train_script_bytes + synthetic_compressed_bytes}")
    print(f"whole_submission_total_bytes_synthetic={submission_non_model_bytes + synthetic_compressed_bytes}")
    if calibrated_compressed_bytes is not None:
        print(f"model_compressed_bytes_calibrated={calibrated_compressed_bytes}")
        print(f"official_total_bytes_calibrated={train_script_bytes + calibrated_compressed_bytes}")
        print(f"whole_submission_total_bytes_calibrated={submission_non_model_bytes + calibrated_compressed_bytes}")
    if median_compressed_bytes is not None:
        print(f"model_compressed_bytes_median_ratio={median_compressed_bytes}")
        print(f"official_total_bytes_median_ratio={train_script_bytes + median_compressed_bytes}")
        print(f"whole_submission_total_bytes_median_ratio={submission_non_model_bytes + median_compressed_bytes}")
    print(f"limit_bytes=16000000")
    print(f"official_within_limit_synthetic={(train_script_bytes + synthetic_compressed_bytes) <= 16_000_000}")
    print(f"whole_submission_within_limit_synthetic={(submission_non_model_bytes + synthetic_compressed_bytes) <= 16_000_000}")
    if calibrated_compressed_bytes is not None:
        print(f"official_within_limit_calibrated={(train_script_bytes + calibrated_compressed_bytes) <= 16_000_000}")
        print(f"whole_submission_within_limit_calibrated={(submission_non_model_bytes + calibrated_compressed_bytes) <= 16_000_000}")
    if median_compressed_bytes is not None:
        print(f"official_within_limit_median_ratio={(train_script_bytes + median_compressed_bytes) <= 16_000_000}")
        print(f"whole_submission_within_limit_median_ratio={(submission_non_model_bytes + median_compressed_bytes) <= 16_000_000}")
    print(f"baseline_tensor_bytes={baseline_tensor_bytes}")
    print(f"payload_compression_ratio={baseline_tensor_bytes / max(payload_bytes, 1):.6f}")
    if mode == "estimate_from_code":
        print("warning=synthetic estimate compresses initialized weights and is optimistic; use calibrated or checkpoint-based numbers when available")
    if historical_records:
        print(f"historical_records={len(historical_records)}")
        print(f"historical_raw_ratio_median={median_raw_ratio:.6f}")
        print(f"historical_payload_ratio_median={median_payload_ratio:.6f}")
        print(
            "historical_nearest="
            f"{nearest_record['path']} payload={int(nearest_record['payload'])} raw={int(nearest_record['raw'])} "
            f"compressed={int(nearest_record['compressed'])} raw_ratio={nearest_record['compressed_over_raw']:.6f}"
        )
    print("counted_submission_files:")
    for path in submission_files:
        print(f"  {path.relative_to(submission_dir)} {file_bytes(path)}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
