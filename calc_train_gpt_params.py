#!/usr/bin/env python3
"""Estimate train_gpt checkpoint size from config knobs.

This mirrors tensor shapes in the current `GPT` class in `train_gpt.py` so you
can quickly scan config grids and find the largest model under a checkpoint
budget.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass

def parse_int_spec(spec: str, *, name: str) -> list[int]:
    """Parse specs like '768', '512,768,1024', or '256:1024:128'."""
    spec = spec.strip()
    if not spec:
        raise ValueError(f"{name} is empty")

    values: list[int] = []
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if ":" not in token:
            values.append(int(token))
            continue

        parts = [p.strip() for p in token.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid range token '{token}' for {name}. Expected start:end[:step]."
            )

        start = int(parts[0])
        end = int(parts[1])
        if len(parts) == 2:
            step = 1 if end >= start else -1
        else:
            step = int(parts[2])
            if step == 0:
                raise ValueError(f"Step cannot be 0 in token '{token}' for {name}.")

        if (end > start and step < 0) or (end < start and step > 0):
            raise ValueError(
                f"Step sign does not move from {start} to {end} in token '{token}' for {name}."
            )

        stop = end + (1 if step > 0 else -1)
        values.extend(range(start, stop, step))

    if not values:
        raise ValueError(f"No values parsed for {name}")

    deduped: list[int] = []
    seen = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def parse_component_list(raw: str) -> set[str]:
    if not raw.strip():
        return set()
    return {piece.strip() for piece in raw.split(",") if piece.strip()}


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int
    mlp_mult: int
    world_size: int

    @property
    def hdim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def num_decoder_layers(self) -> int:
        return self.num_layers - (self.num_layers // 2)

    @property
    def runtime_valid_for_current_train_gpt(self) -> bool:
        # Current train_gpt.py assumptions:
        # - model_dim must equal num_heads * head_dim
        # - head_dim must be even for RoPE
        # - world_size must divide 8 (grad_accum parity assertion)
        # - num_layers and mlp_mult must be positive
        if self.model_dim <= 0:
            return False
        if self.num_heads <= 0 or self.head_dim <= 0:
            return False
        if self.model_dim != self.hdim:
            return False
        if self.head_dim % 2 != 0:
            return False
        if self.world_size <= 0 or 8 % self.world_size != 0:
            return False
        if self.num_layers < 1:
            return False
        if self.mlp_mult < 1:
            return False
        return True


COMPONENT_ORDER = [
    "tok_emb",
    "skip_weights",
    "block_attn_linears",
    "block_attn_v_mix",
    "block_mlp_linears",
    "block_resid_mix",
    "lm_head",
]

# Current train_gpt.py casts the model to bf16 before training/saving.
COMPONENT_BYTES_PER_PARAM = {name: 2 for name in COMPONENT_ORDER}


def component_params(cfg: ModelConfig) -> dict[str, int]:
    d = cfg.model_dim
    layers = cfg.num_layers
    comps: dict[str, int] = {}
    comps["tok_emb"] = cfg.vocab_size * d
    comps["skip_weights"] = cfg.num_decoder_layers
    # Per block: q/k/v/proj, each d x d, bias=False.
    comps["block_attn_linears"] = layers * (4 * d * d)
    # Per block: CausalSelfAttention.v_mix (scalar) and Block.resid_mix (2 scalars).
    comps["block_attn_v_mix"] = layers
    # Per block MLP has fc(d -> mlp_mult*d) and proj(mlp_mult*d -> d), bias=False.
    comps["block_mlp_linears"] = layers * (2 * cfg.mlp_mult * d * d)
    comps["block_resid_mix"] = 2 * layers
    comps["lm_head"] = d * cfg.vocab_size
    return comps


def component_bytes(comps: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, nparams in comps.items():
        out[name] = nparams * COMPONENT_BYTES_PER_PARAM[name]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dim", required=True, help="One of: 768 | 512,768 | 256:1024:128")
    parser.add_argument("--num-layers", required=True, help="One of: 11 | 4,6,8 | 4:12")
    parser.add_argument("--num-heads", required=True, help="One of: 6 | 4,6,8 | 2:12:2")
    parser.add_argument(
        "--head-dim",
        default="128",
        help="Default 128. Supports lists/ranges; used as a runtime-valid consistency check.",
    )
    parser.add_argument("--mlp-mult", default="4", help="Default 4. Supports lists/ranges.")
    parser.add_argument("--vocab-size", required=True, help="Model vocab size (exact; no implicit padding).")
    parser.add_argument(
        "--world-size",
        default="8",
        help="Affects runtime-valid filtering only (current train_gpt.py requires world_size dividing 8).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=30_000_000,
        help="Checkpoint-size threshold in bytes (default: 30,000,000).",
    )
    parser.add_argument(
        "--max-mb",
        type=float,
        default=None,
        help="Checkpoint-size threshold in MB (decimal). Overrides --max-bytes if set.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=None,
        help="Optional extra filter on parameter count.",
    )
    parser.add_argument(
        "--sort-by",
        choices=("bytes", "params"),
        default="bytes",
        help="Primary ranking key (default: bytes).",
    )
    parser.add_argument("--show-all", action="store_true", help="Show configs above budget filters too.")
    parser.add_argument("--top-k", type=int, default=25, help="Rows to print after sorting.")
    parser.add_argument("--exclude", default="", help="Comma-separated components to exclude from total.")
    parser.add_argument("--csv-out", default="", help="Optional CSV output path for all filtered rows.")
    parser.add_argument(
        "--strict-runtime-constraints",
        action="store_true",
        help=(
            "Only keep configs valid for current train_gpt.py runtime assumptions: "
            "num_layers>=1, mlp_mult>=1, model_dim==num_heads*head_dim, "
            "head_dim even for RoPE, and world_size dividing 8."
        ),
    )
    parser.add_argument(
        "--head-dim-multiple-of-8",
        action="store_true",
        help="Only keep configs with head_dim divisible by 8.",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="Print per-component parameter counts for shown rows.",
    )
    return parser


def format_int(n: int) -> str:
    return f"{n:,}"


def format_mb(nbytes: int) -> str:
    return f"{(nbytes / 1_000_000):.2f}"


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
    if any(w <= 0 for w in world_sizes):
        parser.error("--world-size values must be positive integers.")
    exclude = parse_component_list(args.exclude)

    unknown_excludes = exclude - set(COMPONENT_ORDER)
    if unknown_excludes:
        parser.error(
            "Unknown components in --exclude: "
            + ", ".join(sorted(unknown_excludes))
            + f". Valid: {', '.join(COMPONENT_ORDER)}"
        )

    max_bytes = int(round(args.max_mb * 1_000_000)) if args.max_mb is not None else args.max_bytes

    rows: list[dict[str, int | bool]] = []
    for model_dim, layers, heads, head_dim, mlp_mult, vocab_size, world_size in itertools.product(
        model_dims, num_layers, num_heads, head_dims, mlp_mults, vocab_sizes, world_sizes
    ):
        if args.head_dim_multiple_of_8 and head_dim % 8 != 0:
            continue
        cfg = ModelConfig(
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            mlp_mult=mlp_mult,
            world_size=world_size,
        )

        comps = component_params(cfg)
        comp_bytes = component_bytes(comps)
        total_params = sum(v for k, v in comps.items() if k not in exclude)
        total_bytes = sum(v for k, v in comp_bytes.items() if k not in exclude)
        row: dict[str, int | bool] = {
            "total_params": total_params,
            "total_bytes": total_bytes,
            "slack_to_max_bytes": max_bytes - total_bytes,
            "slack_to_max_params": (
                (args.max_params - total_params) if args.max_params is not None else 0
            ),
            "model_dim": model_dim,
            "num_layers": layers,
            "num_heads": heads,
            "head_dim": head_dim,
            "hdim": cfg.hdim,
            "mlp_mult": mlp_mult,
            "vocab_size": vocab_size,
            "world_size": world_size,
            "runtime_valid": cfg.runtime_valid_for_current_train_gpt,
        }
        row.update(comps)
        row.update({f"bytes_{k}": v for k, v in comp_bytes.items()})
        rows.append(row)

    if args.strict_runtime_constraints:
        rows = [r for r in rows if bool(r["runtime_valid"])]

    if not args.show_all:
        rows = [r for r in rows if int(r["total_bytes"]) <= max_bytes]
        if args.max_params is not None:
            rows = [r for r in rows if int(r["total_params"]) <= args.max_params]

    rows.sort(
        key=lambda r: int(r["total_bytes"] if args.sort_by == "bytes" else r["total_params"]),
        reverse=True,
    )

    if not rows:
        print("No configs matched the current filters.")
        return 0

    if args.csv_out:
        fieldnames = [
            "total_params",
            "total_bytes",
            "slack_to_max_bytes",
            "slack_to_max_params",
            "model_dim",
            "num_layers",
            "num_heads",
            "head_dim",
            "hdim",
            "mlp_mult",
            "vocab_size",
            "world_size",
            "runtime_valid",
            *COMPONENT_ORDER,
            *[f"bytes_{name}" for name in COMPONENT_ORDER],
        ]
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote CSV: {args.csv_out} ({len(rows)} rows)")

    shown = rows[: max(1, args.top_k)]
    print(
        f"Showing top {len(shown)} config(s) by total {args.sort_by} "
        f"(byte filter: {'off' if args.show_all else format_int(max_bytes)})."
    )
    if args.max_params is not None:
        print(
            f"Optional param filter: {'off' if args.show_all else format_int(args.max_params)}"
        )
    print()
    print(
        "rank total_bytes total_mb slack_to_max_bytes total_params model_dim "
        "num_layers num_heads head_dim mlp_mult vocab_size runtime_valid"
    )
    for idx, row in enumerate(shown, start=1):
        print(
            f"{idx:>4} "
            f"{format_int(int(row['total_bytes'])):>12} "
            f"{format_mb(int(row['total_bytes'])):>8} "
            f"{format_int(int(row['slack_to_max_bytes'])):>18} "
            f"{format_int(int(row['total_params'])):>12} "
            f"{int(row['model_dim']):>9} "
            f"{int(row['num_layers']):>10} "
            f"{int(row['num_heads']):>9} "
            f"{int(row['head_dim']):>8} "
            f"{int(row['mlp_mult']):>8} "
            f"{int(row['vocab_size']):>10} "
            f"{str(bool(row['runtime_valid'])):>13}"
        )

    if args.breakdown:
        for idx, row in enumerate(shown, start=1):
            print()
            print(
                f"[rank {idx}] d={row['model_dim']} L={row['num_layers']} "
                f"h={row['num_heads']} hd={row['head_dim']} "
                f"mlp={row['mlp_mult']} vocab={row['vocab_size']}"
            )
            for name in COMPONENT_ORDER:
                marker = " (excluded)" if name in exclude else ""
                print(
                    f"  - {name:>14}: "
                    f"params={format_int(int(row[name])):<12} "
                    f"bytes={format_int(int(row[f'bytes_{name}']))}{marker}"
                )
            print(
                f"  - {'TOTAL':>14}: "
                f"params={format_int(int(row['total_params'])):<12} "
                f"bytes={format_int(int(row['total_bytes']))} "
                f"({format_mb(int(row['total_bytes']))} MB)"
            )

    if not all(bool(r["runtime_valid"]) for r in shown):
        print()
        print(
            "Note: runtime_valid=False means current train_gpt.py would reject that config "
            "(requires num_layers>=1, mlp_mult>=1, model_dim==num_heads*head_dim, "
            "head_dim even for RoPE, and world_size dividing 8)."
        )
    print()
    print(
        "Byte estimate assumes bf16 (2B) for all model parameters in current "
        "train_gpt.py; excludes torch.save metadata overhead."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
