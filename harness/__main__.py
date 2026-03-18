from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bootstrap import bootstrap_records
from .common import HISTORY_PATH, JOURNAL_PATH, ensure_lab_layout, write_json
from .history import append_record, load_history, record_evidence_tier, record_is_planner_eligible
from .journal import append_run_entry, ensure_harness_section
from .planner import plan_next_experiment
from .preflight import run_preflight
from .profiles import available_profiles, default_profile_name
from .runner import run_experiment


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Override must look like KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        overrides[key.strip()] = raw.strip()
    return overrides


def _cmd_bootstrap(_: argparse.Namespace) -> int:
    ensure_lab_layout()
    imported = bootstrap_records()
    print(f"Imported {len(imported)} historical record(s) into {HISTORY_PATH}")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    ensure_lab_layout()
    history = load_history()
    spec = plan_next_experiment(history, args.profile, _parse_overrides(args.override), args.code_mutation)
    if args.write:
        write_json(args.write, spec)
        print(f"Wrote plan to {args.write}")
    else:
        print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    ensure_lab_layout()
    ensure_harness_section()
    history = load_history()
    spec = plan_next_experiment(history, args.profile, _parse_overrides(args.override), args.code_mutation)
    record = run_experiment(spec, dry_run=args.dry_run)
    append_record(record)
    if record["result"]["status"] in {"completed", "failed", "blocked"}:
        append_run_entry(record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["result"]["returncode"] == 0 else record["result"]["returncode"]


def _cmd_loop(args: argparse.Namespace) -> int:
    ensure_lab_layout()
    ensure_harness_section()
    exit_code = 0
    for _ in range(args.max_runs):
        history = load_history()
        spec = plan_next_experiment(history, args.profile, _parse_overrides(args.override), args.code_mutation)
        record = run_experiment(spec, dry_run=args.dry_run)
        append_record(record)
        if record["result"]["status"] in {"completed", "failed", "blocked"}:
            append_run_entry(record)
        print(f"{record['spec']['experiment_id']}: {record['result']['status']}")
        if record["result"]["returncode"] != 0:
            exit_code = record["result"]["returncode"]
            if args.stop_on_failure:
                break
    return exit_code


def _cmd_preflight(args: argparse.Namespace) -> int:
    history = load_history()
    spec = plan_next_experiment(history, args.profile, _parse_overrides(args.override), args.code_mutation)
    payload = {"spec": spec, "preflight": run_preflight(spec)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    history = load_history()
    profile_names = [args.profile] if args.profile else sorted(available_profiles())
    payload: dict[str, object] = {"status": "ok", "profiles": {}}
    for profile_name in profile_names:
        spec = plan_next_experiment(history, profile_name, _parse_overrides(args.override), args.code_mutation)
        preflight = run_preflight(spec)
        payload["profiles"][profile_name] = {
            "track": spec["track"],
            "supports_autoloop": spec["supports_autoloop"],
            "require_challenge_ready": spec["require_challenge_ready"],
            "comparability": preflight["comparability"],
            "challenge_ready": preflight["challenge_ready"],
            "can_launch": preflight["can_launch"],
            "ready_for_execution": preflight["ready_for_execution"],
            "fatal_issues": preflight["fatal_issues"],
            "warnings": preflight["warnings"],
        }
        if not preflight["ready_for_execution"]:
            payload["status"] = "blocked"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


def _cmd_selfcheck(_: argparse.Namespace) -> int:
    from .log_parser import parse_log

    checks: dict[str, object] = {"profiles": {}, "parser": {}, "status": "ok"}
    for profile in ("mlx_smoke", "torch_single_gpu_smoke", "torch_record_8gpu", "torch_nonrecord_8gpu"):
        spec = plan_next_experiment(load_history(), profile)
        checks["profiles"][profile] = run_preflight(spec)
    record_metrics = parse_log(Path("records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log"))
    nonrecord_metrics = parse_log(
        Path("records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/train.log")
    )
    checks["parser"]["record_roundtrip_bpb"] = record_metrics.get("final_roundtrip_val_bpb")
    checks["parser"]["nonrecord_roundtrip_bpb"] = nonrecord_metrics.get("final_roundtrip_val_bpb")
    if record_metrics.get("final_roundtrip_val_bpb") != 1.2243657:
        checks["status"] = "failed"
    if nonrecord_metrics.get("final_roundtrip_val_bpb") != 1.20737944:
        checks["status"] = "failed"
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["status"] == "ok" else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    history = load_history()
    profile_name = args.profile
    if profile_name:
        history = [row for row in history if row.get("spec", {}).get("profile") == profile_name]
    print(f"History path: {HISTORY_PATH}")
    print(f"Journal path: {JOURNAL_PATH}")
    print(f"Total records: {len(history)}")
    if not history:
        return 0
    rows = sorted(
        history,
        key=lambda row: row.get("result", {}).get("completed_at") or "",
    )[-args.limit :]
    for row in rows:
        spec = row.get("spec", {})
        result = row.get("result", {})
        metrics = row.get("metrics", {})
        planner_eligible = result.get("planner_eligible")
        if planner_eligible is None:
            planner_eligible = record_is_planner_eligible(row)
        evidence_tier = result.get("evidence_tier")
        if evidence_tier is None:
            evidence_tier = record_evidence_tier(row)
        warnings = ",".join((result.get("parse_warnings") or metrics.get("parse_warnings") or [])[:2]) or "-"
        print(
            f"{spec.get('experiment_id')} | {spec.get('profile')} | {result.get('run_state') or result.get('status')} | "
            f"eligible={planner_eligible} | "
            f"tier={evidence_tier} | "
            f"family={spec.get('hypothesis_family')} | "
            f"mutation_kind={spec.get('mutation_kind')} | "
            f"code_mutation={((spec.get('code_mutation') or {}).get('name'))} | "
            f"comparability={result.get('comparability')} | "
            f"roundtrip_bpb={metrics.get('final_roundtrip_val_bpb')} | "
            f"pre_quant_bpb={metrics.get('last_pre_quant_val_bpb')} | "
            f"bytes={metrics.get('serialized_model_int8_zlib_bytes')} | "
            f"failure_stage={result.get('failure_stage')} | "
            f"failure={result.get('failure_reason')} | "
            f"warnings={warnings}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Autonomous experiment harness for Parameter Golf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Import existing record logs into structured harness history")
    bootstrap.set_defaults(func=_cmd_bootstrap)

    plan = subparsers.add_parser("plan", help="Plan the next experiment without executing it")
    plan.add_argument("--profile", default=default_profile_name())
    plan.add_argument("--override", action="append", default=[], help="Override planner env values as KEY=VALUE")
    plan.add_argument("--code-mutation", default=None, help="Force a named code mutation onto the planned run")
    plan.add_argument("--write", type=Path, help="Write the planned spec to a file")
    plan.set_defaults(func=_cmd_plan)

    run = subparsers.add_parser("run", help="Plan and execute a single experiment")
    run.add_argument("--profile", default=default_profile_name())
    run.add_argument("--override", action="append", default=[], help="Override planner env values as KEY=VALUE")
    run.add_argument("--code-mutation", default=None, help="Force a named code mutation onto the planned run")
    run.add_argument("--dry-run", action="store_true", help="Create the run record without launching the trainer")
    run.set_defaults(func=_cmd_run)

    loop = subparsers.add_parser("loop", help="Run the autonomous feedback loop for N experiments")
    loop.add_argument("--profile", default=default_profile_name())
    loop.add_argument("--override", action="append", default=[], help="Override planner env values as KEY=VALUE")
    loop.add_argument("--code-mutation", default=None, help="Force a named code mutation onto each planned run")
    loop.add_argument("--max-runs", type=int, default=3)
    loop.add_argument("--dry-run", action="store_true")
    loop.add_argument("--stop-on-failure", action="store_true")
    loop.set_defaults(func=_cmd_loop)

    inspect_cmd = subparsers.add_parser("inspect", help="Show recent structured harness history")
    inspect_cmd.add_argument("--profile", default=None)
    inspect_cmd.add_argument("--limit", type=int, default=10)
    inspect_cmd.set_defaults(func=_cmd_inspect)

    preflight = subparsers.add_parser("preflight", help="Run preflight for the next planned spec")
    preflight.add_argument("--profile", default=default_profile_name())
    preflight.add_argument("--override", action="append", default=[], help="Override planner env values as KEY=VALUE")
    preflight.add_argument("--code-mutation", default=None, help="Force a named code mutation onto the planned run")
    preflight.set_defaults(func=_cmd_preflight)

    doctor = subparsers.add_parser("doctor", help="Show readiness across one or all harness profiles")
    doctor.add_argument("--profile", default=None)
    doctor.add_argument("--override", action="append", default=[], help="Override planner env values as KEY=VALUE")
    doctor.add_argument("--code-mutation", default=None, help="Force a named code mutation onto the planned run")
    doctor.set_defaults(func=_cmd_doctor)

    selfcheck = subparsers.add_parser("selfcheck", help="Run built-in harness self-checks")
    selfcheck.set_defaults(func=_cmd_selfcheck)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
