#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import itertools
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class AxisValue:
    token: str
    flags: tuple[str, ...]
    intent: str = ""


@dataclass(frozen=True)
class MatrixAxis:
    name: str
    values: tuple[AxisValue, ...]


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
            intent="First replay-sharpen setting: one maintenance pass over a moderate touched-slot budget with pure replay updates.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "1",
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "64",
                "--maintenance-metric",
                "loss",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "2",
                "--maintenance-replay-candidates",
                "32",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots64",
            stage="2-maintenance",
            intent="Increase replay-sharpen depth while holding the touched-slot budget fixed, still using pure replay updates.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "64",
                "--maintenance-metric",
                "loss",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "2",
                "--maintenance-replay-candidates",
                "32",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128",
            stage="2-maintenance",
            intent="Increase both replay-sharpen depth and touched-slot budget to spend more eval compute on hard cases without EMA smoothing.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "loss",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "2",
                "--maintenance-replay-candidates",
                "64",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_hits",
            stage="2-maintenance",
            intent="Switch replay-sharpen prioritization from hard cases to read hits to bias compute toward frequently consulted slots.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "hits",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "2",
                "--maintenance-replay-candidates",
                "64",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_withgrad25",
            stage="2-maintenance",
            intent="Re-introduce a small EMA contribution during replay sharpening to test whether gentle bias helps without washing out spikes.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "loss",
                "--maintenance-replay-candidates",
                "64",
                "--maintenance-use-grad",
                "--maintenance-grad-mix",
                "0.25",
                "--maintenance-replay-depth",
                "2",
            ),
        ),
        MatrixRun(
            name="gate2_maint2_slots128_depth3",
            stage="2-maintenance",
            intent="Keep pure replay sharpening but let each slot hold one more hard trace so extra maintenance passes see more than one example.",
            flags=(
                "--memory-min-read-count",
                "2",
                "--maintenance-passes",
                "2",
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "loss",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "3",
                "--maintenance-replay-candidates",
                "64",
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
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "loss",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "2",
                "--maintenance-replay-candidates",
                "64",
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
                "--maintenance-mode",
                "replay",
                "--maintenance-step-size",
                "0.05",
                "--maintenance-max-slots",
                "128",
                "--maintenance-metric",
                "loss",
                "--no-maintenance-use-grad",
                "--maintenance-replay-depth",
                "2",
                "--maintenance-replay-candidates",
                "64",
                "--memory-read-scale",
                "1.25",
            ),
        ),
    ]


def _require_string_list(values: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(values)


def _load_axis_value(raw: object, *, axis_name: str) -> AxisValue:
    if not isinstance(raw, dict):
        raise ValueError(f"axis value for {axis_name} must be an object")
    token = raw.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError(f"axis value for {axis_name} requires a non-empty token")
    flags = _require_string_list(raw.get("flags", []), field=f"{axis_name}.{token}.flags")
    intent = raw.get("intent", "")
    if not isinstance(intent, str):
        raise ValueError(f"{axis_name}.{token}.intent must be a string")
    return AxisValue(token=token, flags=flags, intent=intent)


def _load_axis(raw: object) -> MatrixAxis:
    if not isinstance(raw, dict):
        raise ValueError("axis entry must be an object")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("axis entry requires a non-empty name")
    raw_values = raw.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"axis {name} requires a non-empty values list")
    values = tuple(_load_axis_value(item, axis_name=name) for item in raw_values)
    return MatrixAxis(name=name, values=values)


def _selection_matches_rule(selection: dict[str, str], rule: object) -> bool:
    if not isinstance(rule, dict):
        raise ValueError("exclude entries must be objects")
    for key, expected in rule.items():
        if not isinstance(expected, str):
            expected = str(expected)
        if selection.get(str(key)) != expected:
            return False
    return True


def expand_matrix_grid(
    *,
    stage: str,
    intent: str,
    base_flags: tuple[str, ...],
    name_prefix: str,
    axes: tuple[MatrixAxis, ...],
    excludes: tuple[dict[str, str], ...],
) -> list[MatrixRun]:
    runs: list[MatrixRun] = []
    for combo in itertools.product(*(axis.values for axis in axes)):
        selection = {axis.name: value.token for axis, value in zip(axes, combo)}
        if any(_selection_matches_rule(selection, rule) for rule in excludes):
            continue
        name_parts: list[str] = []
        if name_prefix:
            name_parts.append(name_prefix)
        name_parts.extend(value.token for value in combo)
        run_intent = intent
        detail_parts = [f"{axis.name}={value.token}" for axis, value in zip(axes, combo)]
        value_intents = [value.intent for value in combo if value.intent]
        if detail_parts:
            run_intent = f"{run_intent} [{' '.join(detail_parts)}]"
        if value_intents:
            run_intent = f"{run_intent} {' '.join(value_intents)}"
        flags = list(base_flags)
        for value in combo:
            flags.extend(value.flags)
        runs.append(
            MatrixRun(
                name="_".join(name_parts),
                stage=stage,
                intent=run_intent,
                flags=tuple(flags),
            )
        )
    return runs


def load_matrix_file(path: Path) -> list[MatrixRun]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("matrix file must contain a JSON object")

    runs: list[MatrixRun] = []
    raw_runs = payload.get("runs", [])
    if raw_runs:
        if not isinstance(raw_runs, list):
            raise ValueError("runs must be a list")
        for raw_run in raw_runs:
            if not isinstance(raw_run, dict):
                raise ValueError("run entries must be objects")
            name = raw_run.get("name")
            stage = raw_run.get("stage")
            intent = raw_run.get("intent")
            if not isinstance(name, str) or not name:
                raise ValueError("run.name must be a non-empty string")
            if not isinstance(stage, str) or not stage:
                raise ValueError(f"run {name} stage must be a non-empty string")
            if not isinstance(intent, str) or not intent:
                raise ValueError(f"run {name} intent must be a non-empty string")
            flags = _require_string_list(raw_run.get("flags", []), field=f"run {name}.flags")
            runs.append(MatrixRun(name=name, stage=stage, intent=intent, flags=flags))

    raw_axes = payload.get("axes", [])
    if raw_axes:
        if not isinstance(raw_axes, list):
            raise ValueError("axes must be a list")
        axes = tuple(_load_axis(item) for item in raw_axes)
        stage = payload.get("stage", "search")
        intent = payload.get("intent", "Generated v2b search space")
        name_prefix = payload.get("name_prefix", "")
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        if not isinstance(intent, str) or not intent:
            raise ValueError("intent must be a non-empty string")
        if not isinstance(name_prefix, str):
            raise ValueError("name_prefix must be a string")
        base_flags = _require_string_list(payload.get("base_flags", []), field="base_flags")
        raw_excludes = payload.get("exclude", [])
        if not isinstance(raw_excludes, list):
            raise ValueError("exclude must be a list")
        excludes: tuple[dict[str, str], ...] = tuple(
            {str(key): str(value) for key, value in rule.items()} for rule in raw_excludes if isinstance(rule, dict)
        )
        if len(excludes) != len(raw_excludes):
            raise ValueError("exclude entries must be objects")
        runs.extend(
            expand_matrix_grid(
                stage=stage,
                intent=intent,
                base_flags=base_flags,
                name_prefix=name_prefix,
                axes=axes,
                excludes=excludes,
            )
        )

    if not runs:
        raise ValueError("matrix file must define runs and/or axes")
    unique_runs: list[MatrixRun] = []
    seen_flags: set[tuple[str, ...]] = set()
    for run in runs:
        if run.flags in seen_flags:
            continue
        seen_flags.add(run.flags)
        unique_runs.append(run)
    return unique_runs


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
    parser.add_argument(
        "--matrix-json",
        help="Optional JSON file defining explicit runs and/or a cartesian search space over tweakable candidates.",
    )
    args = parser.parse_args(argv[1:])

    runs = load_matrix_file(Path(args.matrix_json)) if args.matrix_json else build_matrix()
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
