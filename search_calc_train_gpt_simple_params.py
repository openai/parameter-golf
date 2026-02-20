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
    ModelConfig,
    component_bytes,
    component_params,
    parse_component_list,
    parse_int_spec,
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
        "--byte-policy",
        choices=BYTE_POLICY_CHOICES,
        default="train_gpt_simple_save",
        help=(
            "How to estimate serialized tensor bytes. Default matches current "
            "train_gpt_simple.py save dtypes (CastedLinear weights fp32, others bf16)."
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
        total_bytes = sum(v for k, v in bytes_by_comp.items() if k not in exclude)

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
                "delta_params": delta_params,
                "delta_bytes": delta_bytes,
                "model_dim": model_dim,
                "num_layers": layers,
                "num_heads": heads,
                "head_dim": cfg.head_dim,
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
            "delta_params",
            "delta_bytes",
            "model_dim",
            "num_layers",
            "num_heads",
            "head_dim",
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
    print()
    print(
        "rank score total_bytes total_mb delta_bytes total_params delta_params "
        "model_dim num_layers num_heads head_dim mlp_mult vocab_size runtime_valid"
    )
    for idx, row in enumerate(shown, start=1):
        print(
            f"{idx:>4} "
            f"{float(row['score']):>8.6f} "
            f"{format_int(int(row['total_bytes'])):>12} "
            f"{format_mb(int(row['total_bytes'])):>8} "
            f"{format_int(int(row['delta_bytes'])):>12} "
            f"{format_int(int(row['total_params'])):>12} "
            f"{format_int(int(row['delta_params'])):>12} "
            f"{int(row['model_dim']):>9} "
            f"{int(row['num_layers']):>10} "
            f"{int(row['num_heads']):>9} "
            f"{int(row['head_dim']):>8} "
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
            "torch.save metadata/code-size overhead."
        )
    else:
        print(
            f"Byte estimate assumes uniform {args.bytes_per_param} bytes/param and excludes "
            "torch.save metadata/code-size overhead."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
