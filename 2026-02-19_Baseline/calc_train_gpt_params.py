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
from pathlib import Path

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
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    mlp_mult: int
    world_size: int

    @property
    def hdim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads <= 0 else self.num_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.effective_num_kv_heads * self.head_dim

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
        if self.effective_num_kv_heads <= 0:
            return False
        if self.effective_num_kv_heads > self.num_heads:
            return False
        if self.num_heads % self.effective_num_kv_heads != 0:
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

SIZE_BUDGET_CHOICES = ("raw_model", "submission")
DEFAULT_NORMAL_ASPECT_RATIO_MIN = 40.0
DEFAULT_NORMAL_ASPECT_RATIO_MAX = 120.0

# Baseline provided by user for train_gpt.py with 2048 tokenizer:
#   raw model file bytes:     144,740,630
#   int8+zlib model bytes:     29,307,854
DEFAULT_INT8_ZLIB_ON_DISK_RATIO = 29_307_854 / 144_740_630

# Current train_gpt.py save path:
#   raw_model = GPT(...).bfloat16()
#   for module in raw_model.modules():
#       if isinstance(module, CastedLinear):
#           module.float()
# So CastedLinear weights are saved in fp32, while embedding/scalars are bf16.
COMPONENT_BYTES_PER_PARAM = {
    "tok_emb": 2,
    "skip_weights": 2,
    "block_attn_linears": 4,
    "block_attn_v_mix": 2,
    "block_mlp_linears": 4,
    "block_resid_mix": 2,
    "lm_head": 4,
}


def component_params(cfg: ModelConfig) -> dict[str, int]:
    d = cfg.model_dim
    layers = cfg.num_layers
    comps: dict[str, int] = {}
    comps["tok_emb"] = cfg.vocab_size * d
    comps["skip_weights"] = cfg.num_decoder_layers
    # Per block: q(d->d), k(d->kv_dim), v(d->kv_dim), proj(d->d), bias=False.
    kv_dim = cfg.kv_dim
    comps["block_attn_linears"] = layers * ((2 * d * d) + (2 * d * kv_dim))
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


def default_submission_overhead_bytes() -> int:
    """Best-effort code-size overhead for final submission bytes."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "train_gpt.py",  # repo root current script
        here.parent / "train_gpt.py",      # same dir fallback
    ]
    for path in candidates:
        try:
            return path.stat().st_size
        except OSError:
            continue
    return 0


def derive_serialized_size_bytes(
    raw_model_bytes: int,
    *,
    on_disk_ratio: float,
    submission_overhead_bytes: int,
) -> tuple[int, int]:
    """Return (estimated_model_file_bytes, estimated_submission_bytes)."""
    if raw_model_bytes < 0:
        raise ValueError(f"raw_model_bytes must be non-negative, got {raw_model_bytes}")
    if on_disk_ratio <= 0:
        raise ValueError(f"on_disk_ratio must be positive, got {on_disk_ratio}")
    if submission_overhead_bytes < 0:
        raise ValueError(
            f"submission_overhead_bytes must be non-negative, got {submission_overhead_bytes}"
        )
    model_file_bytes = int(round(raw_model_bytes * on_disk_ratio))
    submission_bytes = model_file_bytes + submission_overhead_bytes
    return model_file_bytes, submission_bytes


def model_dim_per_layer_aspect_ratio(cfg: ModelConfig) -> float:
    """Width/depth ratio used for reporting: model_dim / num_layers."""
    if cfg.num_layers <= 0:
        return float("inf")
    return float(cfg.model_dim) / float(cfg.num_layers)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dim", required=True, help="One of: 768 | 512,768 | 256:1024:128")
    parser.add_argument("--num-layers", required=True, help="One of: 11 | 4,6,8 | 4:12")
    parser.add_argument("--num-heads", required=True, help="One of: 6 | 4,6,8 | 2:12:2")
    parser.add_argument(
        "--num-kv-heads",
        default="0",
        help="0 means full MHA (NUM_KV_HEADS=NUM_HEADS). Supports lists/ranges for GQA/MQA.",
    )
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
            "Which byte metric to use for --max-* filtering and byte sorting: "
            "'raw_model' (default) or 'submission' (post-scale model bytes + submission overhead)."
        ),
    )
    parser.add_argument(
        "--assume-int8-zlib-on-disk",
        action="store_true",
        help=(
            "Apply the built-in int8+zlib on-disk ratio from the provided 2048-tokenizer baseline "
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
            "(defaults to local train_gpt.py file size)."
        ),
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
    num_kv_heads = parse_int_spec(args.num_kv_heads, name="--num-kv-heads")
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
    if args.normal_aspect_ratio:
        if args.aspect_ratio_min is None:
            args.aspect_ratio_min = DEFAULT_NORMAL_ASPECT_RATIO_MIN
        if args.aspect_ratio_max is None:
            args.aspect_ratio_max = DEFAULT_NORMAL_ASPECT_RATIO_MAX
    if args.aspect_ratio_min is not None and args.aspect_ratio_max is not None:
        if args.aspect_ratio_min > args.aspect_ratio_max:
            parser.error("--aspect-ratio-min cannot be greater than --aspect-ratio-max.")

    if args.assume_int8_zlib_on_disk and args.on_disk_ratio != 1.0:
        parser.error(
            "Use either --assume-int8-zlib-on-disk or --on-disk-ratio, not both."
        )
    if args.assume_int8_zlib_on_disk:
        on_disk_ratio = DEFAULT_INT8_ZLIB_ON_DISK_RATIO
        on_disk_ratio_source = "int8_zlib_preset"
    else:
        on_disk_ratio = args.on_disk_ratio
        on_disk_ratio_source = "custom" if args.on_disk_ratio != 1.0 else "identity"
    if on_disk_ratio <= 0:
        parser.error("--on-disk-ratio must be positive.")
    if args.submission_overhead_bytes < 0:
        parser.error("--submission-overhead-bytes must be non-negative.")

    max_bytes = int(round(args.max_mb * 1_000_000)) if args.max_mb is not None else args.max_bytes

    rows: list[dict[str, int | float | bool]] = []
    for model_dim, layers, heads, kv_heads, head_dim, mlp_mult, vocab_size, world_size in itertools.product(
        model_dims, num_layers, num_heads, num_kv_heads, head_dims, mlp_mults, vocab_sizes, world_sizes
    ):
        if args.head_dim_multiple_of_8 and head_dim % 8 != 0:
            continue
        cfg = ModelConfig(
            model_dim=model_dim,
            num_layers=layers,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            mlp_mult=mlp_mult,
            world_size=world_size,
        )
        aspect_ratio = model_dim_per_layer_aspect_ratio(cfg)
        if args.aspect_ratio_min is not None and aspect_ratio < args.aspect_ratio_min:
            continue
        if args.aspect_ratio_max is not None and aspect_ratio > args.aspect_ratio_max:
            continue

        comps = component_params(cfg)
        comp_bytes = component_bytes(comps)
        total_params = sum(v for k, v in comps.items() if k not in exclude)
        raw_model_bytes = sum(v for k, v in comp_bytes.items() if k not in exclude)
        model_file_bytes, submission_bytes = derive_serialized_size_bytes(
            raw_model_bytes,
            on_disk_ratio=on_disk_ratio,
            submission_overhead_bytes=args.submission_overhead_bytes,
        )
        total_bytes = raw_model_bytes if args.size_budget == "raw_model" else submission_bytes
        row: dict[str, int | float | bool] = {
            "total_params": total_params,
            "total_bytes": total_bytes,
            "slack_to_max_bytes": max_bytes - total_bytes,
            "slack_to_max_params": (
                (args.max_params - total_params) if args.max_params is not None else 0
            ),
            "raw_model_bytes": raw_model_bytes,
            "model_file_bytes": model_file_bytes,
            "submission_bytes": submission_bytes,
            "model_dim": model_dim,
            "num_layers": layers,
            "num_heads": heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
            "hdim": cfg.hdim,
            "aspect_ratio": aspect_ratio,
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
            "raw_model_bytes",
            "model_file_bytes",
            "submission_bytes",
            "model_dim",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "hdim",
            "aspect_ratio",
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
        f"Showing top {len(shown)} config(s) by {args.size_budget} {args.sort_by} "
        f"(byte filter: {'off' if args.show_all else format_int(max_bytes)})."
    )
    if args.max_params is not None:
        print(
            f"Optional param filter: {'off' if args.show_all else format_int(args.max_params)}"
        )
    if args.aspect_ratio_min is not None or args.aspect_ratio_max is not None:
        lo = "-inf" if args.aspect_ratio_min is None else f"{args.aspect_ratio_min:.2f}"
        hi = "+inf" if args.aspect_ratio_max is None else f"{args.aspect_ratio_max:.2f}"
        print(f"Aspect-ratio filter: {lo} <= model_dim/num_layers <= {hi}")
    print()
    print(
        f"{'rank':>4} "
        f"{'budget_bytes':>12} "
        f"{'budget_mb':>8} "
        f"{'raw_model_mb':>12} "
        f"{'submission_mb':>13} "
        f"{'slack_to_max_bytes':>18} "
        f"{'total_params':>12} "
        f"{'model_dim':>9} "
        f"{'num_layers':>10} "
        f"{'num_heads':>9} "
        f"{'num_kv':>8} "
        f"{'head_dim':>8} "
        f"{'aspect':>7} "
        f"{'mlp_mult':>8} "
        f"{'vocab_size':>10} "
        f"{'runtime_valid':>13}"
    )
    for idx, row in enumerate(shown, start=1):
        print(
            f"{idx:>4} "
            f"{format_int(int(row['total_bytes'])):>12} "
            f"{format_mb(int(row['total_bytes'])):>8} "
            f"{format_mb(int(row['raw_model_bytes'])):>12} "
            f"{format_mb(int(row['submission_bytes'])):>13} "
            f"{format_int(int(row['slack_to_max_bytes'])):>18} "
            f"{format_int(int(row['total_params'])):>12} "
            f"{int(row['model_dim']):>9} "
            f"{int(row['num_layers']):>10} "
            f"{int(row['num_heads']):>9} "
            f"{int(row['num_kv_heads']):>8} "
            f"{int(row['head_dim']):>8} "
            f"{float(row['aspect_ratio']):>7.2f} "
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
                f"aspect={float(row['aspect_ratio']):.2f} "
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
                f"budget_bytes={format_int(int(row['total_bytes']))} "
                f"({format_mb(int(row['total_bytes']))} MB)"
            )
            print(
                f"  - {'RAW_MODEL':>14}: "
                f"bytes={format_int(int(row['raw_model_bytes']))} "
                f"({format_mb(int(row['raw_model_bytes']))} MB)"
            )
            print(
                f"  - {'SUBMISSION':>14}: "
                f"bytes={format_int(int(row['submission_bytes']))} "
                f"({format_mb(int(row['submission_bytes']))} MB)"
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
        "Byte estimate matches current train_gpt.py save dtypes "
        "(CastedLinear weights fp32, embedding/scalars bf16) and excludes "
        "torch.save metadata overhead before optional on-disk scaling."
    )
    print(
        f"Budget byte metric: {args.size_budget} (raw_model -> on_disk_ratio={on_disk_ratio:.6f}"
        f" [{on_disk_ratio_source}], submission_overhead_bytes={args.submission_overhead_bytes})."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
