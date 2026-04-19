from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run2/timing_probe_rows.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run2/trigger_overlap_audit.md")
ACTIVE_TRIGGERS = ("first_drift", "persist_2", "high_uncertainty", "low_top_gap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze v1 timing probe rows.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.input)
    by_arm_sample: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_arm_sample[(str(row["arm"]), str(row["sample_id"]))].append(row)

    lines = [
        "# Timing Probe v1 Audit",
        "",
        f"- input: `{args.input}`",
        "",
    ]

    arms = sorted({str(row["arm"]) for row in rows})
    for arm in arms:
        overlap_counter: Counter[tuple[str, ...]] = Counter()
        lines.append(f"## {arm}")
        lines.append("")
        arm_samples = [sample for (candidate_arm, sample) in by_arm_sample if candidate_arm == arm]
        for sample_id in sorted(arm_samples):
            sample_rows = by_arm_sample[(arm, sample_id)]
            active_rows = {str(row["trigger_name"]): row for row in sample_rows if row["trigger_name"] in ACTIVE_TRIGGERS}
            fired = tuple(name for name in ACTIVE_TRIGGERS if bool(active_rows[name]["trigger_fired"]))
            overlap_counter[fired] += 1
        lines.append("### Fired overlap patterns")
        lines.append("")
        for pattern, count in overlap_counter.most_common():
            label = ", ".join(pattern) if pattern else "none"
            lines.append(f"- `{label}`: {count}")
        lines.append("")
        lines.append("### Sample detail")
        lines.append("")
        for sample_id in sorted(arm_samples):
            sample_rows = by_arm_sample[(arm, sample_id)]
            active_rows = {str(row["trigger_name"]): row for row in sample_rows if row["trigger_name"] in ACTIVE_TRIGGERS}
            fired = tuple(name for name in ACTIVE_TRIGGERS if bool(active_rows[name]["trigger_fired"]))
            moves = {name: active_rows[name]["target_move"] for name in ACTIVE_TRIGGERS}
            audits = {name: active_rows[name]["target_audit"] for name in ACTIVE_TRIGGERS}
            lines.append(f"- `{sample_id}` fired=`{', '.join(fired) if fired else 'none'}` moves=`{json.dumps(moves, ensure_ascii=False)}` audits=`{json.dumps(audits, ensure_ascii=False)}`")
        lines.append("")

    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
