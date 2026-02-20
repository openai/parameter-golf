#!/usr/bin/env python3
"""Search train_gpt_simple config grids against target parameter/byte budgets.

This wraps the shape math in `calc_train_gpt_simple_params.py` and ranks by
distance to targets. Defaults are tuned for under-30MB searches and match
current train_gpt_simple.py save dtypes.
"""

from __future__ import annotations

import argparse
import csv
import itertools

from calc_train_gpt_simple_params import (
    BYTE_POLICY_CHOICES,
    COMPONENT_ORDER,
    DEFAULT_INT8_ON_DISK_RATIO,
    DEFAULT_INT8_ZLIB_ON_DISK_RATIO,
    DEFAULT_NORMAL_ASPECT_RATIO_MAX,
    DEFAULT_NORMAL_ASPECT_RATIO_MIN,
    ModelConfig,
    component_bytes,
    component_params,
    derive_serialized_size_bytes,
    default_submission_overhead_bytes,
    model_dim_per_layer_aspect_ratio,
    parse_component_list,
    parse_int_spec,
    SIZE_BUDGET_CHOICES,
)


def format_int(n: int) -> str:
    return f"{n:,}"


def format_mb(nbytes: int) -> str:
    return f"{(nbytes / 1_000_000):.2f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dim", required=True, help="One of: 768 | 256,512,768 | 192:1024:64")
    parser.add_argument("--num-layers", required=True, help="One of: 12 | 8,10,12 | 6:24:2")
    parser.add_argument("--num-heads", required=True, help="One of: 6 | 4,6,8 | 2:16:2")
    parser.add_argument("--mlp-mult", default="4", help="Default 4. Supports lists/ranges.")
    parser.add_argument("--vocab-size", default="50304", help="Default matches train_gpt_simple.py.")
    parser.add_argument(
        "--normal-aspect-ratio",
        action="store_true",
        help=(
            "Enable a built-in width/depth filter: "
            f"{DEFAULT_NORMAL_ASPECT_RATIO_MIN:.2f} <= model_dim/num_layers <= "
            f"{DEFAULT_NORMAL_ASPECT_RATIO_MAX:.2f}."
        ),
    )
    parser.add_argument(
        "--aspect-ratio-min",
        type=float,
        default=None,
        help="Minimum width/depth ratio (model_dim / num_layers).",
    )
    parser.add_argument(
        "--aspect-ratio-max",
        type=float,
        default=None,
        help="Maximum width/depth ratio (model_dim / num_layers).",
    )
    parser.add_argument(
        "--size-budget",
        choices=SIZE_BUDGET_CHOICES,
        default="raw_model",
        help=(
            "Which byte metric to use for --target-* / --max-* byte filters and ranking: "
            "'raw_model' (default) or 'submission' (post-scale model bytes + submission overhead)."
        ),
    )
    parser.add_argument(
        "--byte-policy",
        choices=BYTE_POLICY_CHOICES,
        default="train_gpt_simple_save",
        help=(
            "How to estimate serialized tensor bytes. Default matches current "
            "train_gpt_simple.py save dtypes (CastedLinear weights fp32, others bf16)."
        ),
    )
    parser.add_argument(
        "--assume-int8-on-disk",
        action="store_true",
        help=(
            "Apply the built-in int8 on-disk ratio from the provided baseline "
            f"({DEFAULT_INT8_ON_DISK_RATIO:.6f}x of raw model bytes)."
        ),
    )
    parser.add_argument(
        "--assume-int8-zlib-on-disk",
        action="store_true",
        help=(
            "Apply the built-in int8+zlib on-disk ratio from the provided baseline "
            f"({DEFAULT_INT8_ZLIB_ON_DISK_RATIO:.6f}x of raw model bytes)."
        ),
    )
    parser.add_argument(
        "--on-disk-ratio",
        type=float,
        default=1.0,
        help=(
            "Scale raw estimated model bytes to approximate exported on-disk model file bytes. "
            "Use with --size-budget submission. Default 1.0."
        ),
    )
    parser.add_argument(
        "--submission-overhead-bytes",
        type=int,
        default=default_submission_overhead_bytes(),
        help=(
            "Extra bytes added on top of model file bytes when --size-budget submission "
            "(defaults to local train_gpt_simple.py file size)."
        ),
    )
    parser.add_argument(
        "--bytes-per-param",
        type=int,
        default=2,
        help="Bytes/param only for --byte-policy uniform (default 2 = bf16).",
    )
    parser.add_argument(
        "--target-params",
        type=int,
        default=None,
        help="Optional target total parameter count used for ranking.",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=None,
        help="Target checkpoint bytes used for ranking.",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=30.0,
        help="Target checkpoint MB (decimal). Default 30. Overrides --target-bytes if set.",
    )
    parser.add_argument(
        "--weight-params",
        type=float,
        default=1.0,
        help="Weight on normalized parameter error in score.",
    )
    parser.add_argument(
        "--weight-bytes",
        type=float,
        default=1.0,
        help="Weight on normalized byte error in score.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=30_000_000,
        help="Optional hard filter on checkpoint bytes (default 30,000,000).",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=30.0,
        help="Optional hard filter on checkpoint MB (decimal). Default 30. Overrides --max-bytes if set.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=None,
        help="Optional hard filter on parameter count.",
    )
    parser.add_argument(
        "--under-target-only",
        action="store_true",
        help=(
            "Only keep rows under active target(s): total_bytes<=target and, "
            "if --target-params is set, total_params<=target."
        ),
    )
    parser.add_argument(
        "--strict-runtime-constraints",
        action="store_true",
        help=(
            "Only keep configs valid for current train_gpt_simple.py assumptions: "
            "num_layers>=1, positive dims, model_dim divisible by num_heads, and even head_dim."
        ),
    )
    parser.add_argument("--exclude", default="", help="Comma-separated components to exclude from totals.")
    parser.add_argument("--top-k", type=int, default=25, help="Rows to print after sorting.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path for ranked rows.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_dims = parse_int_spec(args.model_dim, name="--model-dim")
    num_layers = parse_int_spec(args.num_layers, name="--num-layers")
    num_heads = parse_int_spec(args.num_heads, name="--num-heads")
    mlp_mults = parse_int_spec(args.mlp_mult, name="--mlp-mult")
    vocab_sizes = parse_int_spec(args.vocab_size, name="--vocab-size")
    exclude = parse_component_list(args.exclude)

    if args.weight_params < 0 or args.weight_bytes < 0:
        parser.error("--weight-params and --weight-bytes must be non-negative.")
    if args.target_params is not None and args.target_params <= 0:
        parser.error("--target-params must be positive if provided.")
    if args.byte_policy != "uniform" and args.bytes_per_param != 2:
        parser.error(
            "--bytes-per-param is only used with --byte-policy uniform. "
            "Use --byte-policy uniform to override bytes/param."
        )
    if args.normal_aspect_ratio:
        if args.aspect_ratio_min is None:
            args.aspect_ratio_min = DEFAULT_NORMAL_ASPECT_RATIO_MIN
        if args.aspect_ratio_max is None:
            args.aspect_ratio_max = DEFAULT_NORMAL_ASPECT_RATIO_MAX
    if args.aspect_ratio_min is not None and args.aspect_ratio_max is not None:
        if args.aspect_ratio_min > args.aspect_ratio_max:
            parser.error("--aspect-ratio-min cannot be greater than --aspect-ratio-max.")
    num_on_disk_presets = int(bool(args.assume_int8_on_disk)) + int(
        bool(args.assume_int8_zlib_on_disk)
    )
    if num_on_disk_presets > 1:
        parser.error(
            "Use at most one built-in on-disk preset: --assume-int8-on-disk or "
            "--assume-int8-zlib-on-disk."
        )
    if num_on_disk_presets and args.on_disk_ratio != 1.0:
        parser.error(
            "Use either a built-in on-disk preset (--assume-int8-on-disk or "
            "--assume-int8-zlib-on-disk) or --on-disk-ratio, not both."
        )
    if args.assume_int8_zlib_on_disk:
        on_disk_ratio = DEFAULT_INT8_ZLIB_ON_DISK_RATIO
        on_disk_ratio_source = "int8_zlib_preset"
    elif args.assume_int8_on_disk:
        on_disk_ratio = DEFAULT_INT8_ON_DISK_RATIO
        on_disk_ratio_source = "int8_preset"
    else:
        on_disk_ratio = args.on_disk_ratio
        on_disk_ratio_source = "custom" if args.on_disk_ratio != 1.0 else "identity"
    if on_disk_ratio <= 0:
        parser.error("--on-disk-ratio must be positive.")
    if args.submission_overhead_bytes < 0:
        parser.error("--submission-overhead-bytes must be non-negative.")

    unknown_excludes = exclude - set(COMPONENT_ORDER)
    if unknown_excludes:
        parser.error(
            "Unknown components in --exclude: "
            + ", ".join(sorted(unknown_excludes))
            + f". Valid: {', '.join(COMPONENT_ORDER)}"
        )

    target_bytes = int(round(args.target_mb * 1_000_000)) if args.target_mb is not None else args.target_bytes
    max_bytes = int(round(args.max_mb * 1_000_000)) if args.max_mb is not None else args.max_bytes
    if target_bytes is None and args.target_params is None:
        parser.error("Provide at least one target (--target-mb/--target-bytes or --target-params).")

    rows: list[dict[str, int | float | bool]] = []
    for model_dim, layers, heads, mlp_mult, vocab_size in itertools.product(
        model_dims, num_layers, num_heads, mlp_mults, vocab_sizes
    ):
        cfg = ModelConfig(
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            mlp_mult=mlp_mult,
            vocab_size=vocab_size,
        )
        aspect_ratio = model_dim_per_layer_aspect_ratio(cfg)
        if args.aspect_ratio_min is not None and aspect_ratio < args.aspect_ratio_min:
            continue
        if args.aspect_ratio_max is not None and aspect_ratio > args.aspect_ratio_max:
            continue
        runtime_valid = cfg.runtime_valid_for_current_train_gpt_simple
        if args.strict_runtime_constraints and not runtime_valid:
            continue

        comps = component_params(cfg)
        bytes_by_comp = component_bytes(
            comps,
            bytes_per_param=args.bytes_per_param,
            byte_policy=args.byte_policy,
        )
        total_params = sum(v for k, v in comps.items() if k not in exclude)
        raw_model_bytes = sum(v for k, v in bytes_by_comp.items() if k not in exclude)
        model_file_bytes, submission_bytes = derive_serialized_size_bytes(
            raw_model_bytes,
            on_disk_ratio=on_disk_ratio,
            submission_overhead_bytes=args.submission_overhead_bytes,
        )
        total_bytes = raw_model_bytes if args.size_budget == "raw_model" else submission_bytes

        if args.max_params is not None and total_params > args.max_params:
            continue
        if max_bytes is not None and total_bytes > max_bytes:
            continue

        if args.under_target_only:
            if target_bytes is not None and total_bytes > target_bytes:
                continue
            if args.target_params is not None and total_params > args.target_params:
                continue

        delta_params = 0
        score = 0.0
        if args.target_params is not None:
            delta_params = total_params - args.target_params
            score += args.weight_params * (abs(delta_params) / args.target_params)

        delta_bytes = 0
        if target_bytes is not None:
            delta_bytes = total_bytes - target_bytes
            score += args.weight_bytes * (abs(delta_bytes) / max(1, target_bytes))

        rows.append(
            {
                "score": score,
                "total_params": total_params,
                "total_bytes": total_bytes,
                "raw_model_bytes": raw_model_bytes,
                "model_file_bytes": model_file_bytes,
                "submission_bytes": submission_bytes,
                "delta_params": delta_params,
                "delta_bytes": delta_bytes,
                "model_dim": model_dim,
                "num_layers": layers,
                "num_heads": heads,
                "head_dim": cfg.head_dim,
                "aspect_ratio": aspect_ratio,
                "mlp_mult": mlp_mult,
                "vocab_size": vocab_size,
                "runtime_valid": runtime_valid,
            }
        )

    if not rows:
        print("No configs matched the current filters.")
        return 0

    rows.sort(
        key=lambda r: (
            float(r["score"]),
            abs(int(r["delta_bytes"])),
            abs(int(r["delta_params"])),
            -int(r["total_bytes"]),
            -int(r["total_params"]),
        )
    )

    if args.csv_out:
        fieldnames = [
            "score",
            "total_params",
            "total_bytes",
            "raw_model_bytes",
            "model_file_bytes",
            "submission_bytes",
            "delta_params",
            "delta_bytes",
            "model_dim",
            "num_layers",
            "num_heads",
            "head_dim",
            "aspect_ratio",
            "mlp_mult",
            "vocab_size",
            "runtime_valid",
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.csv_out} ({len(rows)} rows)")

    shown = rows[: max(1, args.top_k)]
    print(f"Showing top {len(shown)} config(s) by target score.")
    if args.aspect_ratio_min is not None or args.aspect_ratio_max is not None:
        lo = "-inf" if args.aspect_ratio_min is None else f"{args.aspect_ratio_min:.2f}"
        hi = "+inf" if args.aspect_ratio_max is None else f"{args.aspect_ratio_max:.2f}"
        print(f"Aspect-ratio filter: {lo} <= model_dim/num_layers <= {hi}")
    print()
    print(
        f"{'rank':>4} "
        f"{'score':>8} "
        f"{'budget_bytes':>12} "
        f"{'budget_mb':>8} "
        f"{'raw_model_mb':>12} "
        f"{'submission_mb':>13} "
        f"{'delta_bytes':>12} "
        f"{'total_params':>12} "
        f"{'delta_params':>12} "
        f"{'model_dim':>9} "
        f"{'num_layers':>10} "
        f"{'num_heads':>9} "
        f"{'head_dim':>8} "
        f"{'aspect':>7} "
        f"{'mlp_mult':>8} "
        f"{'vocab_size':>10} "
        f"{'runtime_valid':>13}"
    )
    for idx, row in enumerate(shown, start=1):
        print(
            f"{idx:>4} "
            f"{float(row['score']):>8.6f} "
            f"{format_int(int(row['total_bytes'])):>12} "
            f"{format_mb(int(row['total_bytes'])):>8} "
            f"{format_mb(int(row['raw_model_bytes'])):>12} "
            f"{format_mb(int(row['submission_bytes'])):>13} "
            f"{format_int(int(row['delta_bytes'])):>12} "
            f"{format_int(int(row['total_params'])):>12} "
            f"{format_int(int(row['delta_params'])):>12} "
            f"{int(row['model_dim']):>9} "
            f"{int(row['num_layers']):>10} "
            f"{int(row['num_heads']):>9} "
            f"{int(row['head_dim']):>8} "
            f"{float(row['aspect_ratio']):>7.2f} "
            f"{int(row['mlp_mult']):>8} "
            f"{int(row['vocab_size']):>10} "
            f"{str(bool(row['runtime_valid'])):>13}"
        )

    print()
    print(
        "Tip: defaults are set for under-30MB search; adjust --target-mb/--max-mb for other budgets."
    )
    if args.byte_policy == "train_gpt_simple_save":
        print(
            "Byte estimate matches current train_gpt_simple.py save dtypes "
            "(CastedLinear weights fp32, other params bf16) and excludes "
            "torch.save metadata overhead before optional on-disk scaling."
        )
    else:
        print(
            f"Byte estimate assumes uniform {args.bytes_per_param} bytes/param and excludes "
            "torch.save metadata overhead before optional on-disk scaling."
        )
    print(
        f"Budget byte metric: {args.size_budget} (raw_model -> on_disk_ratio={on_disk_ratio:.6f}"
        f" [{on_disk_ratio_source}], "
        f"submission_overhead_bytes={args.submission_overhead_bytes})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
