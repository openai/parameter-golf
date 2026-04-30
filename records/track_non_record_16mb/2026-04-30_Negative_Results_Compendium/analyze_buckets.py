#!/usr/bin/env python3
"""Analyze GPTQ integer bucket distributions for PR-1787 quant configs.

This is local-only and does not run eval. It loads the saved PR-1787 bundle,
re-runs GPTQ for selected configs, and writes symbol histograms/entropy summaries.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "axes/quantization/pr1787_repro"))
import quantize_bundle as qb  # noqa: E402


DEFAULT_BUNDLE = Path.home() / "parameter-golf-bundles/pr1787_bundle_seed42"
OUT_DIR = ROOT / "axes/quantization/quant_bucket_analysis/results"


@dataclass(frozen=True)
class QuantSpec:
    name: str
    mlp_k: float = 12.0
    attn_k: float = 13.0
    matrix_k: float = 12.85
    embed_k: float = 15.0
    matrix_bits: int = 6
    embed_bits: int = 7
    mlp_scale: str = "row"
    mlp_scale_block_size: int = 128
    loop_mlp_k: float | None = None
    nonloop_mlp_k: float | None = None
    loop_attn_k: float | None = None
    nonloop_attn_k: float | None = None


SPECS = {
    "stock": QuantSpec("stock"),
    "loop_best_row": QuantSpec(
        "loop_best_row",
        loop_mlp_k=11.0,
        nonloop_mlp_k=12.0,
        loop_attn_k=12.85,
        nonloop_attn_k=13.0,
    ),
    "tensor_mlp11p6": QuantSpec(
        "tensor_mlp11p6",
        mlp_k=11.6,
        mlp_scale="tensor",
    ),
    "block256_int6": QuantSpec(
        "block256_int6",
        mlp_scale="block",
        mlp_scale_block_size=256,
    ),
}

LOOP_LAYERS = {3, 4, 5}


def layer_index(name: str) -> int | None:
    if not name.startswith("blocks."):
        return None
    try:
        return int(name.split(".", 2)[1])
    except (IndexError, ValueError):
        return None


def family(name: str) -> str:
    if name == "tok_emb.weight":
        return "embed"
    if ".mlp.fc.weight" in name:
        return "mlp_fc"
    if ".mlp.proj.weight" in name:
        return "mlp_proj"
    if ".attn.c_q.weight" in name:
        return "attn_q"
    if ".attn.c_k.weight" in name:
        return "attn_k"
    if ".attn.c_v.weight" in name:
        return "attn_v"
    if ".attn.proj.weight" in name:
        return "attn_proj"
    return "other"


def quant_params(name: str, spec: QuantSpec) -> tuple[int, float, str, int]:
    fam = family(name)
    layer = layer_index(name)
    if fam == "embed":
        return spec.embed_bits, spec.embed_k, "row", 128
    if fam.startswith("mlp_"):
        if layer in LOOP_LAYERS and spec.loop_mlp_k is not None:
            k = spec.loop_mlp_k
        elif layer not in LOOP_LAYERS and spec.nonloop_mlp_k is not None:
            k = spec.nonloop_mlp_k
        else:
            k = spec.mlp_k
        return spec.matrix_bits, k, spec.mlp_scale, spec.mlp_scale_block_size
    if fam.startswith("attn_"):
        if layer in LOOP_LAYERS and spec.loop_attn_k is not None:
            k = spec.loop_attn_k
        elif layer not in LOOP_LAYERS and spec.nonloop_attn_k is not None:
            k = spec.nonloop_attn_k
        else:
            k = spec.attn_k
        return spec.matrix_bits, k, "row", 128
    return spec.matrix_bits, spec.matrix_k, "row", 128


def is_quantized_tensor(name: str, tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() and tensor.ndim == 2 and tensor.numel() > 65536


def entropy_from_counts(counts: collections.Counter[int]) -> float:
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def summarize_counts(
    config: str,
    group: str,
    bits: int | str,
    counts: collections.Counter[int],
) -> dict[str, object]:
    total = sum(counts.values())
    abs_counts = collections.Counter()
    for q, n in counts.items():
        abs_counts[abs(q)] += n
    entropy = entropy_from_counts(counts)
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:16]
    rng = 2 ** (int(bits) - 1) - 1 if isinstance(bits, int) else None
    return {
        "config": config,
        "group": group,
        "bits": bits,
        "n_values": total,
        "unique_symbols": len(counts),
        "entropy_bits_per_symbol": entropy,
        "zero_frac": counts.get(0, 0) / total,
        "abs_le_1_frac": sum(n for q, n in abs_counts.items() if q <= 1) / total,
        "abs_le_2_frac": sum(n for q, n in abs_counts.items() if q <= 2) / total,
        "abs_le_3_frac": sum(n for q, n in abs_counts.items() if q <= 3) / total,
        "abs_le_4_frac": sum(n for q, n in abs_counts.items() if q <= 4) / total,
        "abs_ge_8_frac": sum(n for q, n in abs_counts.items() if q >= 8) / total,
        "saturation_frac": (
            (counts.get(-rng, 0) + counts.get(rng, 0)) / total if rng else ""
        ),
        "top_symbols": " ".join(f"{q}:{n / total:.4f}" for q, n in top),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["stock", "loop_best_row", "tensor_mlp11p6", "block256_int6"],
        choices=sorted(SPECS),
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = torch.load(args.bundle_dir / "ema_weights.pt", map_location="cpu")
    hessians = torch.load(args.bundle_dir / "hessians.pt", map_location="cpu")
    readme = OUT_DIR / "README.md"
    readme.write_text("# PR-1787 Quantized Bucket Analysis\n\n")

    for config_name in args.configs:
        spec = SPECS[config_name]
        t0 = time.perf_counter()
        rows = []
        tensor_rows = []
        counters: dict[tuple[str, int | str], collections.Counter[int]] = collections.defaultdict(collections.Counter)
        for idx, (name, tensor) in enumerate(weights.items(), 1):
            if not torch.is_tensor(tensor) or not is_quantized_tensor(name, tensor):
                continue
            bits, k, scale_mode, scale_block_size = quant_params(name, spec)
            clip_range = 2 ** (bits - 1) - 1
            q, scale = qb._gptq_quantize_weight_scale_granularity(
                tensor.detach().cpu().contiguous(),
                hessians[name],
                clip_sigmas=k,
                clip_range=clip_range,
                scale_granularity=scale_mode,
                scale_block_size=scale_block_size,
            )
            values = q.reshape(-1).to(torch.int16)
            bincount = torch.bincount((values + 128).to(torch.int64), minlength=256)
            counts = collections.Counter(
                {i - 128: int(n) for i, n in enumerate(bincount.tolist()) if n}
            )
            fam = family(name)
            loop_tag = "loop" if layer_index(name) in LOOP_LAYERS else "nonloop"
            keys = {
                ("all", "mixed"),
                (fam, bits),
                (fam.split("_", 1)[0], bits),
                (f"{loop_tag}_{fam.split('_', 1)[0]}", bits),
            }
            for key in keys:
                counters[key].update(counts)
            tensor_rows.append(
                summarize_counts(config_name, name, bits, counts)
                | {
                    "family": fam,
                    "loop_tag": loop_tag,
                    "k": k,
                    "scale_mode": scale_mode,
                    "scale_shape": "x".join(map(str, scale.shape)),
                }
            )
            print(
                f"{config_name}: {idx:03d} {name} bits={bits} k={k:g} "
                f"scale={scale_mode} H={entropy_from_counts(counts):.3f}",
                flush=True,
            )

        for (group, bits), counts in sorted(counters.items(), key=lambda item: str(item[0])):
            rows.append(summarize_counts(config_name, group, bits, counts))

        hist_rows = []
        for (group, bits), counts in sorted(counters.items(), key=lambda item: str(item[0])):
            total = sum(counts.values())
            for q, n in sorted(counts.items()):
                hist_rows.append(
                    {
                        "config": config_name,
                        "group": group,
                        "bits": bits,
                        "bucket": q,
                        "count": n,
                        "frac": n / total,
                    }
                )

        fields = list(rows[0].keys())
        with (OUT_DIR / f"{config_name}_bucket_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        hist_fields = list(hist_rows[0].keys())
        with (OUT_DIR / f"{config_name}_bucket_histograms.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=hist_fields)
            writer.writeheader()
            writer.writerows(hist_rows)

        tensor_fields = list(tensor_rows[0].keys())
        with (OUT_DIR / f"{config_name}_tensor_buckets.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tensor_fields)
            writer.writeheader()
            writer.writerows(tensor_rows)

        with readme.open("a") as f:
            f.write(f"\n## {config_name}\n\n")
            f.write(f"Generated in {time.perf_counter() - t0:.1f}s.\n\n")
            f.write("| Group | Bits | Entropy | Zero | |q|<=1 | |q|<=2 | |q|>=8 | Top symbols |\n")
            f.write("|-------|-----:|--------:|-----:|-------:|-------:|-------:|-------------|\n")
            important = {"all", "mlp", "attn", "embed", "loop_mlp", "nonloop_mlp"}
            for row in rows:
                if row["group"] not in important:
                    continue
                f.write(
                    f"| {row['group']} | {row['bits']} | "
                    f"{row['entropy_bits_per_symbol']:.3f} | "
                    f"{row['zero_frac']:.3f} | "
                    f"{row['abs_le_1_frac']:.3f} | "
                    f"{row['abs_le_2_frac']:.3f} | "
                    f"{row['abs_ge_8_frac']:.3f} | "
                    f"`{row['top_symbols']}` |\n"
                )

    print(OUT_DIR)


if __name__ == "__main__":
    main()
