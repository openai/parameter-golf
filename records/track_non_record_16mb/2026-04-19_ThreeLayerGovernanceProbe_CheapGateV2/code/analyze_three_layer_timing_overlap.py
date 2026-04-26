from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run1/timing_probe_rows.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run1/trigger_overlap_audit.md")

ACTIVE_TRIGGERS = ("first_drift", "persist_2", "high_uncertainty", "low_top_gap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze overlap/collinearity in three-layer timing probe results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.input)
    by_sample: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row["sample_id"])].append(row)

    overlap_counter: Counter[tuple[str, ...]] = Counter()
    scheduled_counter: Counter[tuple[str, ...]] = Counter()
    sample_notes: list[dict[str, object]] = []
    trigger_pair_equal_rows: Counter[tuple[str, str]] = Counter()
    trigger_pair_total_rows: Counter[tuple[str, str]] = Counter()

    for sample_id, sample_rows in sorted(by_sample.items()):
        active_rows = {str(row["trigger_name"]): row for row in sample_rows if row["trigger_name"] in ACTIVE_TRIGGERS}
        fired = tuple(name for name in ACTIVE_TRIGGERS if bool(active_rows[name]["trigger_fired"]))
        scheduled = tuple(name for name in ACTIVE_TRIGGERS if bool(active_rows[name]["trigger_scheduled"]))
        overlap_counter[fired] += 1
        scheduled_counter[scheduled] += 1

        for i, left in enumerate(ACTIVE_TRIGGERS):
            for right in ACTIVE_TRIGGERS[i + 1 :]:
                left_row = active_rows[left]
                right_row = active_rows[right]
                trigger_pair_total_rows[(left, right)] += 1
                if (
                    left_row["trigger_fired"] == right_row["trigger_fired"]
                    and left_row["target_move"] == right_row["target_move"]
                    and left_row["target_audit"] == right_row["target_audit"]
                    and left_row["cael_trace"] == right_row["cael_trace"]
                ):
                    trigger_pair_equal_rows[(left, right)] += 1

        sample_notes.append(
            {
                "sample_id": sample_id,
                "error_mode": active_rows["first_drift"]["cael_trace"]["error_mode"],
                "uncertainty_band": active_rows["first_drift"]["cael_trace"]["uncertainty_band"],
                "top_gap_band": active_rows["first_drift"]["cael_trace"]["top_gap_band"],
                "fired": fired,
                "scheduled": scheduled,
                "moves": {name: active_rows[name]["target_move"] for name in ACTIVE_TRIGGERS},
                "audits": {name: active_rows[name]["target_audit"] for name in ACTIVE_TRIGGERS},
            }
        )

    boundary_cases = [
        note
        for note in sample_notes
        if len(set(note["moves"].values())) > 1
        or len(set(note["audits"].values())) > 1
        or len(note["fired"]) not in (0, len(ACTIVE_TRIGGERS))
    ]

    lines = [
        "# Trigger Overlap Audit",
        "",
        f"- input: `{args.input}`",
        f"- sample_count: {len(by_sample)}",
        f"- active_triggers: {', '.join(ACTIVE_TRIGGERS)}",
        "",
        "## Fired overlap patterns",
        "",
    ]
    for pattern, count in overlap_counter.most_common():
        label = ", ".join(pattern) if pattern else "none"
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Scheduled overlap patterns", ""])
    for pattern, count in scheduled_counter.most_common():
        label = ", ".join(pattern) if pattern else "none"
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Pairwise equivalence", ""])
    for pair in sorted(trigger_pair_total_rows):
        equal = trigger_pair_equal_rows[pair]
        total = trigger_pair_total_rows[pair]
        lines.append(f"- `{pair[0]} == {pair[1]}` on full trace/move/audit outcome: {equal}/{total}")

    lines.extend(["", "## Sample-level detail", ""])
    for note in sample_notes:
        lines.append(f"### {note['sample_id']}")
        lines.append(f"- error_mode: `{note['error_mode']}`")
        lines.append(f"- uncertainty_band: `{note['uncertainty_band']}`")
        lines.append(f"- top_gap_band: `{note['top_gap_band']}`")
        lines.append(f"- scheduled: `{', '.join(note['scheduled']) if note['scheduled'] else 'none'}`")
        lines.append(f"- fired: `{', '.join(note['fired']) if note['fired'] else 'none'}`")
        lines.append(f"- moves: `{json.dumps(note['moves'], ensure_ascii=False)}`")
        lines.append(f"- audits: `{json.dumps(note['audits'], ensure_ascii=False)}`")
        lines.append("")

    lines.extend(["## Boundary cases", ""])
    if not boundary_cases:
        lines.append("- none")
    else:
        for note in boundary_cases:
            lines.append(f"- `{note['sample_id']}`: fired=`{', '.join(note['fired']) if note['fired'] else 'none'}` moves=`{json.dumps(note['moves'], ensure_ascii=False)}` audits=`{json.dumps(note['audits'], ensure_ascii=False)}`")

    lines.extend(
        [
            "",
            "## Minimal reading",
            "",
            "- If all active triggers fire together and produce the same move/audit, the issue is trigger collinearity in the current trace space.",
            "- If a trigger is scheduled but does not fire because `target_move=no`, that is a useful boundary: the rule sees something, but policy still has no actionable branch.",
            "- Boundary cases are the only cheap places where trigger v1 should learn to separate early drift, persistent drift, uncertainty, and candidate competition.",
            "",
        ]
    )

    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
