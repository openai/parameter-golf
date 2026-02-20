#!/usr/bin/env python3
"""Estimate train_gpt_simple checkpoint size from config knobs.

This mirrors parameter shapes in GPT.__init__ in `train_gpt_simple.py` so you
can quickly scan grids and find the largest model under a checkpoint budget.
Defaults match current `train_gpt_simple.py` save dtypes (most linear weights
saved in fp32, remaining params in bf16).
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path


def parse_int_spec(spec: str, *, name: str) -> list[int]:
    """Parse specs like '768', '512,768,1024', or '8:16:2'."""
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
    mlp_mult: int
    vocab_size: int

    @property
    def head_dim(self) -> int:
        if self.num_heads <= 0:
            return 0
        return self.model_dim // self.num_heads

    @property
    def hidden_dim(self) -> int:
        return self.mlp_mult * self.model_dim

    @property
    def num_decoder_layers(self) -> int:
        # train_gpt_simple uses:
        # num_encoder_layers = num_layers // 2
        # num_decoder_layers = num_layers - num_encoder_layers
        return self.num_layers - (self.num_layers // 2)

    @property
    def runtime_valid_for_current_train_gpt_simple(self) -> bool:
        # Current train_gpt_simple.py assumptions:
        # - num_layers >= 1
        # - model_dim, num_heads, mlp_mult, vocab_size > 0
        # - model_dim divisible by num_heads
        # - head_dim even (RoPE requirement)
        if self.num_layers < 1:
            return False
        if self.model_dim <= 0 or self.num_heads <= 0 or self.mlp_mult <= 0 or self.vocab_size <= 0:
            return False
        if self.model_dim % self.num_heads != 0:
            return False
        if self.head_dim % 2 != 0:
            return False
        return True


COMPONENT_ORDER = [
    "tok_emb",
    "lm_head",
    "block_attn_qkv",
    "block_attn_proj",
    "block_mlp_fc",
    "block_mlp_proj",
    "skip_weights",
    "block_v_mix",
    "block_resid_mix",
]

BYTE_POLICY_CHOICES = ("train_gpt_simple_save", "uniform")
SIZE_BUDGET_CHOICES = ("raw_model", "submission")
DEFAULT_NORMAL_ASPECT_RATIO_MIN = 40.0
DEFAULT_NORMAL_ASPECT_RATIO_MAX = 120.0

# Baseline provided by user for parity-safe custom int8 export:
#   original file bytes: 571,579,806
#   int8 file bytes:     162,269,315
DEFAULT_INT8_ON_DISK_RATIO = 162_269_315 / 571_579_806

# Baseline provided by user for int8+zlib export:
#   raw model file bytes:     103,836,542
#   int8+zlib model bytes:     21,391,571
DEFAULT_INT8_ZLIB_ON_DISK_RATIO = 21_391_571 / 103_836_542

# train_gpt_simple.py does:
#   raw_model = GPT(...).bfloat16()
#   for module in raw_model.modules():
#       if isinstance(module, CastedLinear):
#           module.float()
# so all CastedLinear weights are serialized as fp32, while embeddings/scalars
# stay bf16.
TRAIN_GPT_SIMPLE_FP32_SAVE_COMPONENTS = frozenset(
    {
        "lm_head",
        "block_attn_qkv",
        "block_attn_proj",
        "block_mlp_fc",
        "block_mlp_proj",
    }
)


def component_params(cfg: ModelConfig) -> dict[str, int]:
    d = cfg.model_dim
    layers = cfg.num_layers
    hidden = cfg.hidden_dim
    vocab = cfg.vocab_size

    comps: dict[str, int] = {}
    comps["tok_emb"] = vocab * d
    comps["lm_head"] = d * vocab
    comps["block_attn_qkv"] = layers * (3 * d * d)
    comps["block_attn_proj"] = layers * (d * d)
    comps["block_mlp_fc"] = layers * (d * hidden)
    comps["block_mlp_proj"] = layers * (hidden * d)
    # RMSNorm in train_gpt_simple.py is parameter-free (uses F.rms_norm directly).
    comps["skip_weights"] = cfg.num_decoder_layers
    comps["block_v_mix"] = layers  # one scalar per block attention
    comps["block_resid_mix"] = layers * 2  # two scalars per block
    return comps


def component_bytes(
    comps: dict[str, int],
    *,
    bytes_per_param: int,
    byte_policy: str = "train_gpt_simple_save",
) -> dict[str, int]:
    if byte_policy == "uniform":
        if bytes_per_param <= 0:
            raise ValueError(f"bytes_per_param must be positive, got {bytes_per_param}")
        return {name: nparams * bytes_per_param for name, nparams in comps.items()}

    if byte_policy == "train_gpt_simple_save":
        return {
            name: nparams * (4 if name in TRAIN_GPT_SIMPLE_FP32_SAVE_COMPONENTS else 2)
            for name, nparams in comps.items()
        }

    raise ValueError(
        f"Unknown byte_policy={byte_policy!r}. Expected one of: {', '.join(BYTE_POLICY_CHOICES)}"
    )


def default_submission_overhead_bytes() -> int:
    """Best-effort code-size overhead for final submission bytes."""
    base = Path(__file__).resolve().parent
    for candidate in ("train_gpt_simple.py", "train_gpt_simple_muon.py"):
        path = base / candidate
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
    """Width/depth ratio used for shape filtering: model_dim / num_layers."""
    if cfg.num_layers <= 0:
        return float("inf")
    return float(cfg.model_dim) / float(cfg.num_layers)


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
            "Which byte metric to use for --max-* filtering and byte sorting: "
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
            "Only keep configs valid for current train_gpt_simple.py assumptions: "
            "num_layers>=1, positive dims, model_dim divisible by num_heads, and even head_dim."
        ),
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
    mlp_mults = parse_int_spec(args.mlp_mult, name="--mlp-mult")
    vocab_sizes = parse_int_spec(args.vocab_size, name="--vocab-size")
    exclude = parse_component_list(args.exclude)

    unknown_excludes = exclude - set(COMPONENT_ORDER)
    if unknown_excludes:
        parser.error(
            "Unknown components in --exclude: "
            + ", ".join(sorted(unknown_excludes))
            + f". Valid: {', '.join(COMPONENT_ORDER)}"
        )
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

    max_bytes = int(round(args.max_mb * 1_000_000)) if args.max_mb is not None else args.max_bytes

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

        comps = component_params(cfg)
        comp_bytes = component_bytes(
            comps,
            bytes_per_param=args.bytes_per_param,
            byte_policy=args.byte_policy,
        )
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
            "head_dim": cfg.head_dim,
            "aspect_ratio": aspect_ratio,
            "mlp_mult": mlp_mult,
            "vocab_size": vocab_size,
            "runtime_valid": cfg.runtime_valid_for_current_train_gpt_simple,
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
            "head_dim",
            "aspect_ratio",
            "mlp_mult",
            "vocab_size",
            "runtime_valid",
            *COMPONENT_ORDER,
            *[f"bytes_{name}" for name in COMPONENT_ORDER],
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
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
                f"mlp_mult={row['mlp_mult']} vocab={row['vocab_size']}"
            )
            for name in COMPONENT_ORDER:
                marker = " (excluded)" if name in exclude else ""
                print(
                    f"  - {name:>16}: "
                    f"params={format_int(int(row[name])):<12} "
                    f"bytes={format_int(int(row[f'bytes_{name}']))}{marker}"
                )
            print(
                f"  - {'TOTAL':>16}: "
                f"params={format_int(int(row['total_params'])):<12} "
                f"budget_bytes={format_int(int(row['total_bytes']))} ({format_mb(int(row['total_bytes']))} MB)"
            )
            print(
                f"  - {'RAW_MODEL':>16}: "
                f"bytes={format_int(int(row['raw_model_bytes']))} ({format_mb(int(row['raw_model_bytes']))} MB)"
            )
            print(
                f"  - {'SUBMISSION':>16}: "
                f"bytes={format_int(int(row['submission_bytes']))} ({format_mb(int(row['submission_bytes']))} MB)"
            )

    if not all(bool(r["runtime_valid"]) for r in shown):
        print()
        print(
            "Note: runtime_valid=False means current train_gpt_simple.py would reject that config "
            "(requires num_layers>=1, positive dims, model_dim divisible by num_heads, even head_dim)."
        )
    print()
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
