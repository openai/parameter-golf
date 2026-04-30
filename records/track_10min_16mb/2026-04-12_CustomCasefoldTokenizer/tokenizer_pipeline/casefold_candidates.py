"""Merge case variants in the candidate pool for case-folded tokenizer training.

Reads candidates_25gb.json, lowercases all candidate byte sequences,
merges frequencies of case variants, and writes a new file.
Preserves the original file untouched.

Usage:
    uv run data/casefold_candidates.py
    uv run data/casefold_candidates.py --input data/tokenizers/candidates_25gb.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent
DEFAULT_INPUT = DATA_DIR / "tokenizers" / "candidates_25gb.json"
DEFAULT_OUTPUT = DATA_DIR / "tokenizers" / "candidates_25gb_casefold.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Loading {args.input} ...", flush=True)
    t0 = time.perf_counter()
    with open(args.input, "r") as f:
        raw = json.load(f)

    # Handle both formats: {"candidates": {...}} or flat dict
    if "candidates" in raw:
        hex_cands = raw["candidates"]
        metadata = {k: v for k, v in raw.items() if k != "candidates"}
    else:
        hex_cands = raw
        metadata = {}

    elapsed = time.perf_counter() - t0
    print(f"  Loaded {len(hex_cands):,} candidates in {elapsed:.1f}s", flush=True)

    # Merge: lowercase each candidate's bytes, sum frequencies
    merged: dict[str, int] = {}
    upper_merged = 0
    already_lower = 0

    for hex_key, freq in hex_cands.items():
        try:
            raw_bytes = bytes.fromhex(hex_key)
            text = raw_bytes.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            # Keep non-UTF8 candidates as-is (rare byte sequences)
            merged[hex_key] = merged.get(hex_key, 0) + freq
            continue

        lowered = text.lower()
        if lowered != text:
            upper_merged += 1
        else:
            already_lower += 1

        low_hex = lowered.encode("utf-8").hex()
        merged[low_hex] = merged.get(low_hex, 0) + freq

    elapsed2 = time.perf_counter() - t0
    print(f"\nMerge results ({elapsed2 - elapsed:.1f}s):", flush=True)
    print(f"  Original candidates: {len(hex_cands):,}", flush=True)
    print(f"  Already lowercase:   {already_lower:,}", flush=True)
    print(f"  Had uppercase (merged in): {upper_merged:,}", flush=True)
    print(f"  Final candidates:    {len(merged):,}", flush=True)
    print(f"  Reduction:           {len(hex_cands) - len(merged):,} "
          f"({(len(hex_cands) - len(merged)) / len(hex_cands) * 100:.1f}%)", flush=True)

    # Write output
    output_data = {**metadata, "candidates": merged} if metadata else merged
    print(f"\nWriting {args.output} ...", flush=True)
    # Write to temp file then rename for atomicity
    tmp = args.output.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(output_data, f)
    tmp.rename(args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(f"  Done: {size_mb:.0f} MB, {len(merged):,} candidates", flush=True)
    print(f"  Total time: {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
