#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_RECORD_ROOT = REPO_ROOT / "records" / "track_non_record_16mb" / "2026-04-03_DeepFloor"
DEFAULT_RUN_ROOT = DEFAULT_RECORD_ROOT / "runs"


def run_checked(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write_fixture_bytes(path: Path, size: int) -> None:
    payload = bytes((idx % 256 for idx in range(size)))
    path.write_bytes(payload)


def ensure_fixture(enwik8_path: Path | None) -> Path:
    if enwik8_path is not None:
        return enwik8_path
    fixture_dir = Path(tempfile.mkdtemp(prefix="deepfloor-fixture-"))
    path = fixture_dir / "enwik8"
    write_fixture_bytes(path, size=32768)
    return path


def deepfloor_cmd(
    python_bin: Path,
    *,
    enwik8_path: Path,
    output_json: Path,
    device: str,
    cross_token_mode: str,
    kernel_feature_map: str | None,
    recurrent_dim: int,
    view_count: int,
    num_distinct_blocks: int,
    train_steps: int,
    eval_batches: int,
    train_recurrence_steps: int,
    eval_recurrence_steps: int,
    train_floor_interval: int,
) -> list[str]:
    cmd = [
        str(python_bin),
        str(REPO_ROOT / "spectral_flood_walk_v3.py"),
        "--device",
        device,
        "--seq-len",
        "16",
        "--stride",
        "8",
        "--batch-size",
        "2",
        "--train-steps",
        str(train_steps),
        "--eval-batches",
        str(eval_batches),
        "--report-every",
        "1",
        "--recurrent-dim",
        str(recurrent_dim),
        "--recurrent-heads",
        "4",
        "--num-distinct-blocks",
        str(num_distinct_blocks),
        "--view-count",
        str(view_count),
        "--train-recurrence-steps",
        str(train_recurrence_steps),
        "--eval-recurrence-steps",
        str(eval_recurrence_steps),
        "--floor-min-interval",
        "1",
        "--floor-max-interval",
        "3",
        "--floor-threshold",
        "0.01",
        "--no-cache-dataset-on-device",
        "--enwik8-path",
        str(enwik8_path),
        "--cross-token-mode",
        cross_token_mode,
        "--output-json",
        str(output_json),
    ]
    if cross_token_mode == "floor":
        cmd.extend(["--train-floor-interval", str(train_floor_interval)])
    if kernel_feature_map is not None:
        cmd.extend(["--kernel-feature-map", kernel_feature_map])
    return cmd


def run_local_unit(python_bin: Path) -> None:
    run_checked(
        [
            str(python_bin),
            "-m",
            "unittest",
            "tests.test_evolutionary_benchmark",
            "tests.test_spectral_flood_walk_v3",
            "-v",
        ]
    )


def run_local_smoke(
    python_bin: Path,
    output_dir: Path,
    enwik8_path: Path | None,
    *,
    device: str,
    train_steps: int,
    eval_batches: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = ensure_fixture(enwik8_path)
    run_checked(
        deepfloor_cmd(
            python_bin,
            enwik8_path=fixture,
            output_json=output_dir / "floor_smoke.json",
            device=device,
            cross_token_mode="floor",
            kernel_feature_map=None,
            recurrent_dim=32,
            view_count=2,
            num_distinct_blocks=2,
            train_steps=train_steps,
            eval_batches=eval_batches,
            train_recurrence_steps=4,
            eval_recurrence_steps=6,
            train_floor_interval=2,
        )
    )
    run_checked(
        deepfloor_cmd(
            python_bin,
            enwik8_path=fixture,
            output_json=output_dir / "fused_smoke.json",
            device=device,
            cross_token_mode="fused",
            kernel_feature_map="identity",
            recurrent_dim=32,
            view_count=2,
            num_distinct_blocks=2,
            train_steps=train_steps,
            eval_batches=eval_batches,
            train_recurrence_steps=4,
            eval_recurrence_steps=6,
            train_floor_interval=2,
        )
    )


def run_matrix(
    python_bin: Path,
    output_dir: Path,
    enwik8_path: Path | None,
    *,
    device: str,
    train_steps: int,
    eval_batches: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = ensure_fixture(enwik8_path)
    experiments = [
        {
            "name": "floor_d32_v2",
            "cross_token_mode": "floor",
            "kernel_feature_map": None,
            "recurrent_dim": 32,
            "view_count": 2,
            "num_distinct_blocks": 1,
            "train_recurrence_steps": 6,
            "eval_recurrence_steps": 8,
            "train_floor_interval": 2,
        },
        {
            "name": "fused_d32_v2",
            "cross_token_mode": "fused",
            "kernel_feature_map": "identity",
            "recurrent_dim": 32,
            "view_count": 2,
            "num_distinct_blocks": 1,
            "train_recurrence_steps": 6,
            "eval_recurrence_steps": 8,
            "train_floor_interval": 2,
        },
        {
            "name": "floor_d64_v2",
            "cross_token_mode": "floor",
            "kernel_feature_map": None,
            "recurrent_dim": 64,
            "view_count": 2,
            "num_distinct_blocks": 2,
            "train_recurrence_steps": 8,
            "eval_recurrence_steps": 12,
            "train_floor_interval": 4,
        },
        {
            "name": "fused_d64_v2",
            "cross_token_mode": "fused",
            "kernel_feature_map": "elu_plus_1",
            "recurrent_dim": 64,
            "view_count": 2,
            "num_distinct_blocks": 2,
            "train_recurrence_steps": 8,
            "eval_recurrence_steps": 12,
            "train_floor_interval": 4,
        },
    ]
    for experiment in experiments:
        run_checked(
            deepfloor_cmd(
                python_bin,
                enwik8_path=fixture,
                output_json=output_dir / f"{experiment['name']}.json",
                device=device,
                cross_token_mode=str(experiment["cross_token_mode"]),
                kernel_feature_map=(
                    None if experiment["kernel_feature_map"] is None else str(experiment["kernel_feature_map"])
                ),
                recurrent_dim=int(experiment["recurrent_dim"]),
                view_count=int(experiment["view_count"]),
                num_distinct_blocks=int(experiment["num_distinct_blocks"]),
                train_steps=train_steps,
                eval_batches=eval_batches,
                train_recurrence_steps=int(experiment["train_recurrence_steps"]),
                eval_recurrence_steps=int(experiment["eval_recurrence_steps"]),
                train_floor_interval=int(experiment["train_floor_interval"]),
            )
        )


def print_report(run_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        rows.append(
            {
                "name": path.name,
                "mode": config["cross_token_mode"],
                "views": config["view_count"],
                "dim": config["recurrent_dim"],
                "val_bpb": round(float(payload["val"]["bpb"]), 4),
                "test_bpb": round(float(payload["test"]["bpb"]), 4),
                "artifact_mb": round(float(payload["artifact"]["estimated_mb"]), 4),
            }
        )
    if not rows:
        print(f"no json runs found in {run_dir}")
        return
    headers = ("name", "mode", "views", "dim", "val_bpb", "test_bpb", "artifact_mb")
    print("\t".join(headers))
    for row in rows:
        print("\t".join(str(row[header]) for header in headers))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local DeepFloor test and smoke suite")
    parser.add_argument("--python-bin", default=str(DEFAULT_PYTHON))
    subparsers = parser.add_subparsers(dest="command", required=True)

    local_unit = subparsers.add_parser("local-unit", help="Run unit tests")
    local_unit.set_defaults(handler="local-unit")

    local_smoke = subparsers.add_parser("local-smoke", help="Run fixed floor/fused smoke experiments")
    local_smoke.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT / "local_smoke"))
    local_smoke.add_argument("--enwik8-path")
    local_smoke.add_argument("--device", default="cpu")
    local_smoke.add_argument("--train-steps", type=int, default=2)
    local_smoke.add_argument("--eval-batches", type=int, default=2)
    local_smoke.set_defaults(handler="local-smoke")

    local_all = subparsers.add_parser("local-all", help="Run unit tests and smoke experiments")
    local_all.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT / "local_smoke"))
    local_all.add_argument("--enwik8-path")
    local_all.add_argument("--device", default="cpu")
    local_all.add_argument("--train-steps", type=int, default=2)
    local_all.add_argument("--eval-batches", type=int, default=2)
    local_all.set_defaults(handler="local-all")

    matrix = subparsers.add_parser("matrix", help="Run a fixed DeepFloor comparison matrix")
    matrix.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT / "matrix"))
    matrix.add_argument("--enwik8-path")
    matrix.add_argument("--device", default="cpu")
    matrix.add_argument("--train-steps", type=int, default=2)
    matrix.add_argument("--eval-batches", type=int, default=2)
    matrix.set_defaults(handler="matrix")

    report = subparsers.add_parser("report", help="Summarize DeepFloor json outputs")
    report.add_argument("--run-dir", default=str(DEFAULT_RUN_ROOT / "matrix"))
    report.set_defaults(handler="report")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    python_bin = Path(args.python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"python bin not found: {python_bin}")
    if args.handler == "local-unit":
        run_local_unit(python_bin)
        return
    if args.handler == "local-smoke":
        run_local_smoke(
            python_bin,
            Path(args.output_dir),
            Path(args.enwik8_path) if args.enwik8_path else None,
            device=args.device,
            train_steps=args.train_steps,
            eval_batches=args.eval_batches,
        )
        return
    if args.handler == "local-all":
        run_local_unit(python_bin)
        run_local_smoke(
            python_bin,
            Path(args.output_dir),
            Path(args.enwik8_path) if args.enwik8_path else None,
            device=args.device,
            train_steps=args.train_steps,
            eval_batches=args.eval_batches,
        )
        return
    if args.handler == "matrix":
        run_matrix(
            python_bin,
            Path(args.output_dir),
            Path(args.enwik8_path) if args.enwik8_path else None,
            device=args.device,
            train_steps=args.train_steps,
            eval_batches=args.eval_batches,
        )
        return
    if args.handler == "report":
        print_report(Path(args.run_dir))
        return
    raise ValueError(f"unsupported handler: {args.handler}")


if __name__ == "__main__":
    main()
