#!/usr/bin/env python3
"""
Parameter Golf Ablation Framework
Tracks every experiment, records results, computes deltas.
"""

import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List

ABLATION_DIR = Path("./ablation_results")
ABLATION_DIR.mkdir(exist_ok=True)


@dataclass
class AblationResult:
    run_id: str
    techniques: List[str]
    base_checkpoint: str
    seed: int
    val_loss: float
    val_bpb: float
    artifact_size_bytes: int
    training_steps: int
    wall_clock_seconds: float
    gpu_config: str
    notes: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def delta_vs(self, other: "AblationResult") -> dict:
        return {
            "bpb_delta": self.val_bpb - other.val_bpb,
            "loss_delta": self.val_loss - other.val_loss,
            "size_delta": self.artifact_size_bytes - other.artifact_size_bytes,
            "added_techniques": sorted(set(self.techniques) - set(other.techniques)),
            "removed_techniques": sorted(set(other.techniques) - set(self.techniques)),
        }


def save_result(result: AblationResult):
    path = ABLATION_DIR / f"{result.run_id}.json"
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[ABLATION] Saved: {path}")
    return path


def load_result(run_id: str) -> AblationResult:
    path = ABLATION_DIR / f"{run_id}.json"
    with open(path) as f:
        return AblationResult(**json.load(f))


def load_all() -> List[AblationResult]:
    results = []
    for p in sorted(ABLATION_DIR.glob("*.json")):
        with open(p) as f:
            results.append(AblationResult(**json.load(f)))
    return results


def print_leaderboard():
    results = load_all()
    results.sort(key=lambda r: r.val_bpb)
    print(f"\n{'=' * 80}")
    print(f"{'RUN_ID':<35} {'BPB':>8} {'LOSS':>8} {'SIZE':>10} {'TECHNIQUES'}")
    print(f"{'=' * 80}")
    for r in results:
        techs = ",".join(r.techniques[:8])
        if len(r.techniques) > 8:
            techs += f"...+{len(r.techniques) - 8}"
        print(
            f"{r.run_id:<35} {r.val_bpb:>8.4f} {r.val_loss:>8.4f} {r.artifact_size_bytes:>10,} {techs}"
        )
    print()


def print_ablation_table():
    """Show impact of each technique by comparing runs that differ by exactly one technique."""
    results = load_all()
    print(f"\n{'=' * 80}")
    print("SINGLE-TECHNIQUE ABLATIONS (pairs differing by exactly 1 technique)")
    print(f"{'=' * 80}")
    pairs = []
    for i, a in enumerate(results):
        for b in results[i + 1 :]:
            delta = a.delta_vs(b)
            if len(delta["added_techniques"]) + len(delta["removed_techniques"]) == 1:
                pairs.append((a, b, delta))
    pairs.sort(key=lambda x: abs(x[2]["bpb_delta"]), reverse=True)
    for a, b, d in pairs:
        direction = "+" if d["added_techniques"] else "-"
        tech = (d["added_techniques"] or d["removed_techniques"])[0]
        sign = "-" if d["bpb_delta"] < 0 else "+"
        print(
            f"  {direction}{tech:<12} {sign}{abs(d['bpb_delta']):.4f} BPB  "
            f"({a.run_id} vs {b.run_id})"
        )
    if not pairs:
        print("  No single-technique pairs found yet. Run more ablations!")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "leaderboard":
        print_leaderboard()
    elif len(sys.argv) > 1 and sys.argv[1] == "ablations":
        print_ablation_table()
    else:
        print("Usage: python ablation.py [leaderboard|ablations]")
