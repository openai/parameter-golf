from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_three_layer_prototype_data import mismatch_score


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run2/timing_probe_rows.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_timing_probe_run2/boundary_microscope.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect boundary samples across signal/policy/audit layers.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", action="append", dest="sample_ids", default=["world_006"])
    return parser


def load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_report(path: Path, rows: list[dict[str, object]], sample_ids: list[str]) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["sample_id"] in sample_ids:
            grouped[str(row["sample_id"])].append(row)

    lines = [
        "# Boundary Sample Microscope",
        "",
        "- purpose: determine whether `high_uncertainty` / `low_top_gap` separation is visible at signal layer or only appears after policy translation.",
        "",
    ]

    for sample_id in sample_ids:
        sample_rows = grouped.get(sample_id, [])
        if not sample_rows:
            continue
        first = sample_rows[0]
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"- prefix: `{first['prefix']}`",
                f"- gold: `{first['gold_continuation']}`",
                "",
                "| arm | trigger | scheduled | fired | error_mode | uncertainty | gap | move | audit | post_continuation |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in sample_rows:
            trace = row["cael_trace"]
            lines.append(
                "| "
                f"{row['arm']} | {row['trigger_name']} | {row['trigger_scheduled']} | {row['trigger_fired']} | "
                f"{trace['error_mode']} | {trace['uncertainty_band']} | {trace['top_gap_band']} | "
                f"{row['target_move']} | {row['target_audit']} | `{str(row['post_monday_continuation']).replace('|', '/')}` |"
            )
        lines.append("")

        lines.extend(
            [
                "### Layer Read",
                "",
            ]
        )

        v1_old = [r for r in sample_rows if r["arm"] == "v1_old_policy" and r["trigger_scheduled"]]
        v1_new = [r for r in sample_rows if r["arm"] == "v1_translation_v0" and r["trigger_scheduled"]]

        lines.append(f"- signal_layer_v1_old_policy: `{[(r['trigger_name'], r['trigger_fired']) for r in v1_old]}`")
        lines.append(f"- signal_plus_translation_v1: `{[(r['trigger_name'], r['target_move'], r['target_audit']) for r in v1_new]}`")
        gold = str(first["gold_continuation"])
        downstream = [
            (
                r["trigger_name"],
                r["target_move"],
                mismatch_score(str(r["post_monday_continuation"]), gold),
                str(r["post_monday_continuation"]),
            )
            for r in v1_new
        ]
        lines.append(f"- downstream_effects_v1_translation: `{downstream}`")

        if v1_old and all(r["target_move"] == "no" for r in v1_old) and any(r["target_move"] != "no" for r in v1_new):
            lines.append("- read: signal is already present under v1 trigger logic, but the separation only becomes actionable once translation v0 allows light moves.")
        elif any(r["trigger_fired"] for r in v1_old):
            lines.append("- read: part of the split is already present before translation; translation then refines it further.")
        else:
            lines.append("- read: current split is mostly a translation-layer effect; signal-layer separation is still weak or invisible in this sample.")
        unique_downstream = {item[2:] for item in downstream}
        if len(unique_downstream) == 1 and len(downstream) > 1:
            lines.append("- downstream read: the two light moves are named differently, but their post-move continuation and mismatch score are still identical here; downstream separation has not yet appeared.")
        elif len(unique_downstream) > 1:
            lines.append("- downstream read: the two light moves already produce different post-move effects, so policy translation is beginning to correspond to distinct downstream outcomes.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.input)
    # Preserve order and uniqueness for repeated --sample-id use.
    sample_ids = list(dict.fromkeys(args.sample_ids))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, rows, sample_ids)


if __name__ == "__main__":
    main()
