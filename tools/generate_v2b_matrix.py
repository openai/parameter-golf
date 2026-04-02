#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MatrixRun:
    name: str
    stage: str
    intent: str
    flags: tuple[str, ...]

    def shell_command(self, python_bin: str, script_path: str, output_dir: str) -> str:
        result_path = f"{output_dir}/{self.name}/result.json"
        model_artifact_path = f"{output_dir}/{self.name}/model_int8.npz"
        parts = [
            python_bin,
            script_path,
            "--output-json",
            result_path,
            "--model-artifact-path",
            model_artifact_path,
            *self.flags,
        ]
        return " ".join(shlex.quote(part) for part in parts)


def build_matrix() -> list[MatrixRun]:
    return [
        MatrixRun(
            name="baseline_memread1_nomaint",
            stage="0-sanity",
            intent="Persistent hidden memory with immediate reads and no maintenance; establishes the raw online signal.",
            flags=(
                "--memory-min-read-count",
                "1",
                "--maintenance-passes",
                "0",
                "--maintenance-max-slots",
                "0",
            ),
        ),
        MatrixRun(
            name="gate2_nomaint",
            stage="1-read-gate",
            intent="Require one prior scored write before reading from memory; tests whether early-chunk sacrifice helps.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "0",
                "--maintenance-max-slots",
                "0",
            ),
        ),
        MatrixRun(
            name="gate4_nomaint",
            stage="1-read-gate",
            intent="Stronger read gate to protect the early stream more aggressively.",
            flags=(
                "--memory-min-read-count",
                "4",
                "--maintenance-passes",
                "0",
                "--maintenance-max-slots",
                "0",
            ),
        ),
        MatrixRun(
            name="gate2_maint1_slots64",
            stage="2-maintenance",
            intent="First flop-push setting: one maintenance pass over a moderate touched-slot budget.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "1",
                "--maintenance-max-slots",
                "64",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots64",
            stage="2-maintenance",
            intent="Increase maintenance depth while holding the touched-slot budget fixed.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-max-slots",
                "64",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128",
            stage="2-maintenance",
            intent="Increase both maintenance depth and touched-slot budget to spend more eval compute in memory refinement.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-max-slots",
                "128",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_hits",
            stage="2-maintenance",
            intent="Switch maintenance prioritization from write counts to read hits to bias compute toward frequently consulted slots.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "hits",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_nograd",
            stage="2-maintenance",
            intent="Ablate gradient-assisted maintenance to test whether extra compute needs the EMA signal or just slot mixing.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-max-slots",
                "128",
                "--no-maintenance-use-grad",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_table96k",
            stage="3-capacity",
            intent="Increase table capacity once gating and maintenance are in a plausible range.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-max-slots",
                "128",
                "--memory-table-size",
                "98304",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_readscale125",
            stage="3-capacity",
            intent="Probe whether a modestly stronger read path helps once the memory is warm and refined.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-max-slots",
                "128",
                "--memory-read-scale",
                "1.25",
            ),
        ),
    ]


def render_table(runs: list[MatrixRun]) -> str:
    headers = ("name", "stage", "intent", "flags")
    rows: list[dict[str, str]] = []
    widths = {header: len(header) for header in headers}
    for run in runs:
        row = {
            "name": run.name,
            "stage": run.stage,
            "intent": run.intent,
            "flags": " ".join(run.flags),
        }
        rows.append(row)
        for header, value in row.items():
            widths[header] = max(widths[header], len(value))
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate a curated Spectral Flood Walk v2b experiment matrix")
    parser.add_argument("--format", choices=("table", "json", "shell"), default="table")
    parser.add_argument("--python-bin", default="python3")
    parser.add_argument("--script-path", default="spectral_flood_walk_v2b.py")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args(argv[1:])

    runs = build_matrix()
    if args.format == "table":
        print(render_table(runs))
        return 0
    if args.format == "json":
        print(json.dumps([asdict(run) for run in runs], indent=2))
        return 0
    for run in runs:
        print(f"# {run.name}: {run.intent}")
        print(run.shell_command(args.python_bin, args.script_path, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
