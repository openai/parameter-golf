from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_REGISTRY = ROOT / "record_registry.json"
DEFAULT_OUTPUT_DIR = ROOT / "skydiscover_bootstrap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SkyDiscover-compatible checkpoint from the Parameter Golf record registry."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_0",
        help="Name of the checkpoint folder to create inside output-dir.",
    )
    parser.add_argument(
        "--seed-id",
        action="append",
        default=[],
        help="Specific record_id to include. May be passed multiple times. Defaults to registry initial_population_default.",
    )
    parser.add_argument(
        "--include-nonrecord",
        action="store_true",
        help="Include non-record seeds when selecting from the default initial population.",
    )
    parser.add_argument(
        "--source-override",
        action="append",
        default=[],
        metavar="RECORD_ID=PATH",
        help="Override the source train script for a record seed. Path is relative to parameter-golf/ unless absolute.",
    )
    parser.add_argument(
        "--json-indent",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--normalize-evaluator",
        type=Path,
        default=None,
        help=(
            "Optional evaluator Python file with evaluate(program_path) used to rescore seed programs "
            "under the same local conditions before writing the checkpoint."
        ),
    )
    return parser.parse_args()


def load_registry(path: Path) -> dict:
    return json.loads(path.read_text())


def choose_seed_ids(registry: dict, explicit_seed_ids: list[str], include_nonrecord: bool) -> list[str]:
    if explicit_seed_ids:
        return explicit_seed_ids

    chosen = []
    records_by_id = {record["record_id"]: record for record in registry["records"]}
    for seed_id in registry.get("initial_population_default", []):
        record = records_by_id.get(seed_id)
        if record is None:
            continue
        if not include_nonrecord and record["track"] != "track_10min_16mb":
            continue
        chosen.append(seed_id)
    return chosen


def parse_source_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected RECORD_ID=PATH, got: {item}")
        record_id, path = item.split("=", 1)
        overrides[record_id] = path
    return overrides


def metric_payload(record: dict) -> dict:
    val_bpb = float(record["val_bpb"])
    return {
        "combined_score": -val_bpb,
        "val_bpb": val_bpb,
        "candidate_family": record["candidate_family"],
        "track": record["track"],
        "risk": record["risk"],
    }


def artifact_payload(record: dict) -> dict:
    return {
        "record_id": record["record_id"],
        "runnable_path": record["runnable_path"],
        "train_script": record["train_script"],
        "readme": record["readme"],
    }


def metadata_payload(record: dict) -> dict:
    return {
        "source": "parameter_golf_record_registry",
        "warm_start_role": record["warm_start_role"],
        "rank_hint": record["rank_hint"],
        "author": record["author"],
        "date": record["date"],
        "key_mechanisms": record["key_mechanisms"],
        "likely_layers": record["likely_layers"],
        "legal_eval_notes": record["legal_eval_notes"],
    }


def load_evaluator(path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("parameter_golf_seed_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "evaluate"):
        raise RuntimeError(f"Evaluator {path} must define evaluate(program_path)")
    return module.evaluate


def normalized_eval_payload(
    record: dict,
    source_overrides: dict[str, str],
    evaluate_fn,
) -> tuple[dict, dict]:
    train_script_path = resolve_source_path(record, source_overrides)
    raw_solution = train_script_path.read_text()
    is_proxy_source = record["record_id"] in source_overrides
    header_lines = [
        f"# Warm-start seed: {record['record_id']}",
        f"# Candidate family: {record['candidate_family']}",
        f"# Record val_bpb: {record['val_bpb']}",
    ]
    if is_proxy_source:
        header_lines.append(f"# Proxy source: {relative_to_workspace(train_script_path)}")
    solution = "\n".join(header_lines) + "\n\n" + raw_solution

    with tempfile.TemporaryDirectory(prefix="pg_seed_eval_") as tmp_dir:
        temp_program = Path(tmp_dir) / "train_gpt.py"
        temp_program.write_text(solution)
        result = evaluate_fn(str(temp_program))

    structured = result.get("structured_result", {})
    metrics = dict(result)
    metrics.pop("structured_result", None)
    metrics["seed_source_record_id"] = record["record_id"]
    metrics["candidate_family"] = record["candidate_family"]
    metrics["track"] = record["track"]
    metrics["risk"] = record["risk"]

    artifacts = {
        "normalization_status": structured.get("status", result.get("status")),
        "normalization_log_path": result.get("log_path"),
        "normalization_run_id": result.get("run_id"),
        "normalization_eval_mode": result.get("evaluation_mode"),
    }
    return metrics, artifacts


def resolve_source_path(record: dict, source_overrides: dict[str, str]) -> Path:
    source = source_overrides.get(record["record_id"], record["train_script"])
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    return source_path


def make_program(
    record: dict,
    iteration_found: int,
    source_overrides: dict[str, str],
    evaluate_fn=None,
) -> dict:
    train_script_path = resolve_source_path(record, source_overrides)
    raw_solution = train_script_path.read_text()
    is_proxy_source = record["record_id"] in source_overrides
    header_lines = [
        f"# Warm-start seed: {record['record_id']}",
        f"# Candidate family: {record['candidate_family']}",
        f"# Record val_bpb: {record['val_bpb']}",
    ]
    if is_proxy_source:
        header_lines.append(f"# Proxy source: {relative_to_workspace(train_script_path)}")
    solution = "\n".join(header_lines) + "\n\n" + raw_solution
    metrics = metric_payload(record)
    artifacts = artifact_payload(record)
    if evaluate_fn is not None:
        metrics, normalization_artifacts = normalized_eval_payload(
            record,
            source_overrides=source_overrides,
            evaluate_fn=evaluate_fn,
        )
        artifacts = {**artifacts, **normalization_artifacts}
    return {
        "id": record["record_id"],
        "solution": solution,
        "language": "python",
        "metrics": metrics,
        "iteration_found": iteration_found,
        "parent_id": None,
        "other_context_ids": None,
        "parent_info": None,
        "context_info": None,
        "metadata": {
            **metadata_payload(record),
            "source_train_script": relative_to_workspace(train_script_path),
            "is_proxy_source": is_proxy_source,
            "normalized_by_evaluator": evaluate_fn is not None,
        },
        "artifacts": artifacts,
        "prompts": None,
        "generation": 0,
    }


def select_records(registry: dict, seed_ids: list[str]) -> list[dict]:
    records_by_id = {record["record_id"]: record for record in registry["records"]}
    missing = [seed_id for seed_id in seed_ids if seed_id not in records_by_id]
    if missing:
        raise ValueError(f"Unknown seed ids: {', '.join(missing)}")
    return [records_by_id[seed_id] for seed_id in seed_ids]


def best_program(programs: list[dict]) -> dict:
    return max(programs, key=lambda program: float(program["metrics"]["combined_score"]))


def relative_to_workspace(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def write_checkpoint(
    output_dir: Path,
    checkpoint_name: str,
    registry_path: Path,
    records: list[dict],
    source_overrides: dict[str, str],
    json_indent: int,
    normalize_evaluator: Path | None,
) -> Path:
    checkpoint_dir = output_dir / checkpoint_name
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    programs_dir = checkpoint_dir / "programs"
    programs_dir.mkdir(parents=True, exist_ok=True)
    evaluate_fn = load_evaluator(normalize_evaluator) if normalize_evaluator is not None else None

    programs = []
    for iteration_found, record in enumerate(records):
        program = make_program(
            record,
            iteration_found=iteration_found,
            source_overrides=source_overrides,
            evaluate_fn=evaluate_fn,
        )
        programs.append(program)
        program_path = programs_dir / f"{record['record_id']}.json"
        program_path.write_text(json.dumps(program, indent=json_indent))

    best = best_program(programs)
    metadata = {
        "best_program_id": best["id"],
        "last_iteration": 0,
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=json_indent))

    best_record_id = best["id"]
    best_record = next(record for record in records if record["record_id"] == best_record_id)
    best_source = resolve_source_path(best_record, source_overrides)
    best_target = checkpoint_dir / best_source.name
    best_target.write_text(best_source.read_text())

    best_info = {
        "id": best["id"],
        "iteration": programs.index(best),
        "generation": 0,
        "metrics": best["metrics"],
        "language": "python",
        "source_registry": relative_to_workspace(registry_path),
    }
    (checkpoint_dir / "best_program_info.json").write_text(json.dumps(best_info, indent=json_indent))

    manifest = {
        "registry_path": relative_to_workspace(registry_path),
        "checkpoint_dir": relative_to_workspace(checkpoint_dir),
        "seed_ids": [record["record_id"] for record in records],
        "tracks": sorted({record["track"] for record in records}),
        "best_record_id": best["id"],
        "record_count": len(records),
        "normalized_by_evaluator": (
            relative_to_workspace(normalize_evaluator) if normalize_evaluator is not None else None
        ),
        "source_overrides": {
            record_id: relative_to_workspace(resolve_source_path({"record_id": record_id, "train_script": path}, source_overrides={}))
            for record_id, path in source_overrides.items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=json_indent))
    return checkpoint_dir


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    seed_ids = choose_seed_ids(registry, args.seed_id, include_nonrecord=args.include_nonrecord)
    source_overrides = parse_source_overrides(args.source_override)
    if not seed_ids:
        raise SystemExit("No seeds selected. Pass --seed-id or allow default record seeds.")

    records = select_records(registry, seed_ids)
    checkpoint_dir = write_checkpoint(
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        registry_path=args.registry,
        records=records,
        source_overrides=source_overrides,
        json_indent=args.json_indent,
        normalize_evaluator=args.normalize_evaluator,
    )

    result = {
        "status": "ok",
        "checkpoint_dir": str(checkpoint_dir),
        "seed_ids": seed_ids,
        "normalized_by_evaluator": (
            str(args.normalize_evaluator) if args.normalize_evaluator is not None else None
        ),
    }
    json.dump(result, sys.stdout, indent=args.json_indent)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
