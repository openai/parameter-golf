#!/usr/bin/env python3
"""Search model grids against target parameter/byte budgets.

This wraps the shape math in `calc_train_gpt_params.py` and adds
target-oriented ranking (closest-to-target search).
"""

from __future__ import annotations

import argparse
import csv
import itertools

from calc_train_gpt_params import (
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
    parser.add_argument("--model-dim", required=True, help="One of: 768 | 512,768 | 256:1024:128")
    parser.add_argument("--num-layers", required=True, help="One of: 11 | 4,6,8 | 4:12")
    parser.add_argument("--num-heads", required=True, help="One of: 6 | 4,6,8 | 2:12:2")
    parser.add_argument("--head-dim", default="128", help="Default 128. Supports lists/ranges.")
    parser.add_argument("--mlp-mult", default="4", help="Default 4. Supports lists/ranges.")
    parser.add_argument("--vocab-size", required=True, help="Model vocab size (exact; no implicit padding).")
    parser.add_argument("--world-size", default="8", help="Supports lists/ranges like 1,2,4,8.")
    parser.add_argument(
        "--target-params",
        type=int,
        required=True,
        help="Target total parameter count used for ranking.",
    )
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=None,
        help="Optional target checkpoint bytes used in the joint score.",
    )
    parser.add_argument(
        "--target-mb",
        type=float,
        default=None,
        help="Optional target checkpoint MB (decimal). Overrides --target-bytes if set.",
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
        help="Weight on normalized byte error in score (only used with --target-bytes/--target-mb).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Optional hard filter on checkpoint bytes.",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=None,
        help="Optional hard filter on checkpoint MB (decimal). Overrides --max-bytes if set.",
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
        help="Only keep rows with total_params <= target_params.",
    )
    parser.add_argument(
        "--strict-runtime-constraints",
        action="store_true",
        help="Only keep configs valid for current train_gpt.py runtime assumptions.",
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
    head_dims = parse_int_spec(args.head_dim, name="--head-dim")
    mlp_mults = parse_int_spec(args.mlp_mult, name="--mlp-mult")
    vocab_sizes = parse_int_spec(args.vocab_size, name="--vocab-size")
    world_sizes = parse_int_spec(args.world_size, name="--world-size")
    exclude = parse_component_list(args.exclude)

    if any(w <= 0 for w in world_sizes):
        parser.error("--world-size values must be positive integers.")
    if args.target_params <= 0:
        parser.error("--target-params must be positive.")
    if args.weight_params < 0 or args.weight_bytes < 0:
        parser.error("--weight-params and --weight-bytes must be non-negative.")

    unknown_excludes = exclude - set(COMPONENT_ORDER)
    if unknown_excludes:
        parser.error(
            "Unknown components in --exclude: "
            + ", ".join(sorted(unknown_excludes))
            + f". Valid: {', '.join(COMPONENT_ORDER)}"
        )

    target_bytes = int(round(args.target_mb * 1_000_000)) if args.target_mb is not None else args.target_bytes
    max_bytes = int(round(args.max_mb * 1_000_000)) if args.max_mb is not None else args.max_bytes

    rows: list[dict[str, int | float | bool]] = []
    for model_dim, layers, heads, head_dim, mlp_mult, vocab_size, world_size in itertools.product(
        model_dims, num_layers, num_heads, head_dims, mlp_mults, vocab_sizes, world_sizes
    ):
        cfg = ModelConfig(
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            mlp_mult=mlp_mult,
            world_size=world_size,
        )
        runtime_valid = cfg.runtime_valid_for_current_train_gpt
        if args.strict_runtime_constraints and not runtime_valid:
            continue

        comps = component_params(cfg)
        bytes_by_comp = component_bytes(comps)
        total_params = sum(v for k, v in comps.items() if k not in exclude)
        total_bytes = sum(v for k, v in bytes_by_comp.items() if k not in exclude)

        if args.under_target_only and total_params > args.target_params:
            continue
        if args.max_params is not None and total_params > args.max_params:
            continue
        if max_bytes is not None and total_bytes > max_bytes:
            continue

        delta_params = total_params - args.target_params
        score = args.weight_params * (abs(delta_params) / args.target_params)

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
                "head_dim": head_dim,
                "mlp_mult": mlp_mult,
                "vocab_size": vocab_size,
                "world_size": world_size,
                "runtime_valid": runtime_valid,
            }
        )

    if not rows:
        print("No configs matched the current filters.")
        return 0

    rows.sort(
        key=lambda r: (
            float(r["score"]),
            abs(int(r["delta_params"])),
            abs(int(r["delta_bytes"])),
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
            "world_size",
            "runtime_valid",
        ]
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.csv_out} ({len(rows)} rows)")

    shown = rows[: max(1, args.top_k)]
    print(f"Showing top {len(shown)} config(s) by target score.")
    print()
    print(
        "rank score total_params delta_params total_bytes total_mb delta_bytes "
        "model_dim num_layers num_heads head_dim mlp_mult vocab_size world_size runtime_valid"
    )
    for idx, row in enumerate(shown, start=1):
        print(
            f"{idx:>4} "
            f"{float(row['score']):>8.6f} "
            f"{format_int(int(row['total_params'])):>12} "
            f"{format_int(int(row['delta_params'])):>12} "
            f"{format_int(int(row['total_bytes'])):>12} "
            f"{format_mb(int(row['total_bytes'])):>8} "
            f"{format_int(int(row['delta_bytes'])):>12} "
            f"{int(row['model_dim']):>9} "
            f"{int(row['num_layers']):>10} "
            f"{int(row['num_heads']):>9} "
            f"{int(row['head_dim']):>8} "
            f"{int(row['mlp_mult']):>8} "
            f"{int(row['vocab_size']):>10} "
            f"{int(row['world_size']):>10} "
            f"{str(bool(row['runtime_valid'])):>13}"
        )

    print()
    print("Tip: use --strict-runtime-constraints to keep only train_gpt-runnable configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
