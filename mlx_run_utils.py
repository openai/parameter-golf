from __future__ import annotations

import argparse
from collections.abc import Callable


def build_cli_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the simple Apple Silicon MLX baseline with optional document-aware evaluation."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=defaults.max_steps,
        help="Cap local training steps. Default keeps the MLX path in prototype-run territory, not H100 territory.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=("flat", "doc", "doc_stride"),
        default=defaults.eval_mode,
        help="Validation mode for periodic `VAL_LOSS_EVERY` logging.",
    )
    parser.add_argument(
        "--eval-stride",
        type=int,
        default=defaults.eval_stride,
        help="Stride for document-aware sliding evaluation.",
    )
    parser.add_argument(
        "--eval-max-docs",
        type=int,
        default=defaults.eval_max_docs,
        help="Limit validation to the first N documents for faster local iteration. Use 0 for the full validation split.",
    )
    parser.add_argument(
        "--run-prequant-eval",
        action=argparse.BooleanOptionalAction,
        default=defaults.run_prequant_eval,
        help="Run the final pre-quant eval suite.",
    )
    parser.add_argument(
        "--run-postquant-eval",
        action=argparse.BooleanOptionalAction,
        default=defaults.run_postquant_eval,
        help="Run the final post-quant int8+zlib roundtrip eval suite.",
    )
    return parser


def apply_cli_overrides(args, cli_args: argparse.Namespace) -> None:
    if cli_args.max_steps <= 0:
        raise ValueError(f"--max-steps must be positive, got {cli_args.max_steps}")
    if cli_args.eval_stride <= 0:
        raise ValueError(f"--eval-stride must be positive, got {cli_args.eval_stride}")
    if cli_args.eval_max_docs < 0:
        raise ValueError(f"--eval-max-docs must be non-negative, got {cli_args.eval_max_docs}")
    args.max_steps = int(cli_args.max_steps)
    args.eval_mode = str(cli_args.eval_mode)
    args.eval_stride = int(cli_args.eval_stride)
    args.eval_max_docs = int(cli_args.eval_max_docs)
    args.run_prequant_eval = bool(cli_args.run_prequant_eval)
    args.run_postquant_eval = bool(cli_args.run_postquant_eval)
    args.iterations = min(args.iterations, args.max_steps)


def print_run_summary(
    log_fn: Callable[[str], None],
    *,
    train_steps_completed: int,
    train_tokens_processed: int,
    prequant_results: dict[str, dict[str, float]] | None,
    postquant_results: dict[str, dict[str, float]] | None,
    artifact_metrics: dict[str, object],
) -> None:
    def maybe_metric(result_map: dict[str, dict[str, float]] | None, mode: str, key: str) -> str:
        if not result_map or mode not in result_map:
            return "na"
        value = result_map[mode][key]
        return f"{value:.8f}" if key in {"loss", "bpb"} else f"{value:.0f}"

    def maybe_gap(mode: str, key: str) -> str:
        if not prequant_results or not postquant_results:
            return "na"
        if mode not in prequant_results or mode not in postquant_results:
            return "na"
        return f"{postquant_results[mode][key] - prequant_results[mode][key]:.8f}"

    lines = [
        "run_summary:",
        f"  train_steps_completed:{train_steps_completed}",
        f"  train_tokens_processed:{train_tokens_processed}",
    ]
    for phase_name, result_map in (("prequant", prequant_results), ("postquant", postquant_results)):
        for mode in ("flat", "doc", "doc_stride"):
            lines.append(f"  {phase_name}_{mode}_val_loss:{maybe_metric(result_map, mode, 'loss')}")
            lines.append(f"  {phase_name}_{mode}_val_bpb:{maybe_metric(result_map, mode, 'bpb')}")
            lines.append(f"  {phase_name}_{mode}_eval_time_ms:{maybe_metric(result_map, mode, 'eval_time_ms')}")
    for mode in ("flat", "doc", "doc_stride"):
        lines.append(f"  quantization_gap_{mode}_loss:{maybe_gap(mode, 'loss')}")
        lines.append(f"  quantization_gap_{mode}_bpb:{maybe_gap(mode, 'bpb')}")
    lines.extend(
        [
            f"  model_bytes:{artifact_metrics['model_bytes']}",
            f"  quant_serialized_bytes:{artifact_metrics['quant_serialized_bytes']}",
            f"  compressed_bytes:{artifact_metrics['compressed_bytes']}",
        ]
    )
    for line in lines:
        log_fn(line)
