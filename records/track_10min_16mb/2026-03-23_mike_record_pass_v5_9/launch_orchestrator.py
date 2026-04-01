#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def expand_tokens(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for token, replacement in mapping.items():
            out = out.replace(token, replacement)
        return out
    if isinstance(value, list):
        return [expand_tokens(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: expand_tokens(v, mapping) for k, v in value.items()}
    return value


def detect_repo_root(config_path: Path) -> Path:
    env_repo_root = os.environ.get("REPO_ROOT")
    if env_repo_root:
        return Path(env_repo_root).expanduser().resolve()

    config_dir = config_path.resolve().parent
    try:
        proc = subprocess.run(
            ["git", "-C", str(config_dir), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = Path(proc.stdout.strip()).resolve()
        if candidate.exists():
            return candidate
    except Exception:
        pass

    for candidate in (config_dir, *config_dir.parents):
        if (candidate / "analysis").exists() and (candidate / "records").exists():
            return candidate

    parents = config_path.resolve().parents
    return parents[3] if len(parents) > 3 else config_dir


def load_config(config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    record_dir = Path(os.environ.get("RECORD_DIR", str(config_path.resolve().parent))).expanduser().resolve()
    repo_root = detect_repo_root(config_path)
    mapping = {
        "${REPO_ROOT}": str(repo_root),
        "${RECORD_DIR}": str(record_dir),
    }
    return expand_tokens(raw, mapping)


def command_str(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], *, cwd: Path | None = None, execute: bool = True) -> None:
    print(command_str(cmd))
    if not execute:
        return
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {command_str(cmd)}")


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def evaluate_donor_gate(payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    gate_cfg = cfg.get("donor_gate", {})
    gate_payload = payload.get("donor_gate", {})

    quality_margin = to_float(gate_payload.get("quality_margin_bpb"), float("inf"))
    required_quality_margin = to_float(gate_cfg.get("quality_margin_bpb"), -0.000012)

    reruns_with_same_sign = to_int(gate_payload.get("reruns_with_same_sign"), 0)
    required_reruns = to_int(gate_cfg.get("required_reruns"), 2)

    donor_cap_safe = to_int(gate_payload.get("donor_cap_safe"), 0)
    bytes_delta = to_int(gate_payload.get("bytes_delta"), 10**9)
    max_bytes_delta = to_int(gate_cfg.get("max_bytes_delta"), 8192)

    broad_qat_required = to_int(gate_payload.get("broad_qat_required"), 1)
    require_no_broad_qat = bool(gate_cfg.get("require_no_broad_qat", True))

    quality_ok = quality_margin <= required_quality_margin
    reruns_ok = reruns_with_same_sign >= required_reruns
    cap_ok = donor_cap_safe == 1
    bytes_ok = bytes_delta <= max_bytes_delta
    qat_ok = (not require_no_broad_qat) or (broad_qat_required == 0)

    allowed = bool(quality_ok and reruns_ok and cap_ok and bytes_ok and qat_ok)

    return {
        "allowed": allowed,
        "checks": {
            "quality_ok": quality_ok,
            "reruns_ok": reruns_ok,
            "cap_ok": cap_ok,
            "bytes_ok": bytes_ok,
            "qat_ok": qat_ok,
        },
        "inputs": {
            "quality_margin_bpb": quality_margin,
            "required_quality_margin_bpb": required_quality_margin,
            "reruns_with_same_sign": reruns_with_same_sign,
            "required_reruns": required_reruns,
            "donor_cap_safe": donor_cap_safe,
            "bytes_delta": bytes_delta,
            "max_bytes_delta": max_bytes_delta,
            "broad_qat_required": broad_qat_required,
            "require_no_broad_qat": require_no_broad_qat,
        },
    }


def evaluate_decision(payload: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    decision_block = payload.get("decision_block", {})
    stop_rule_outcome = str(decision_block.get("stop_rule_outcome", "")).strip().upper()
    if stop_rule_outcome not in {"STOP", "HOLD_RECHECK", "RUN_T1"}:
        raise RuntimeError("decision JSON missing valid decision_block.stop_rule_outcome")

    donor_eval = evaluate_donor_gate(payload, cfg)
    donor_probe_requested = bool(
        to_int(payload.get("donor_probe_requested"), 0)
        or to_int(decision_block.get("donor_probe_requested"), 0)
    )
    if stop_rule_outcome == "RUN_T1":
        if donor_probe_requested and (not donor_eval["allowed"]):
            operator_token = "DONOR_NOT_ALLOWED"
        else:
            operator_token = "RUN_T1"
    else:
        operator_token = stop_rule_outcome

    return {
        "operator_token": operator_token,
        "stop_rule_outcome": stop_rule_outcome,
        "donor_probe_requested": donor_probe_requested,
        "donor_gate": donor_eval,
        "decision_block": decision_block,
    }


def find_latest_decision_json(cfg: dict[str, Any]) -> Path:
    output_root = Path(cfg["paths"]["output_root"]).resolve()
    candidates = sorted(output_root.glob("**/export_decision.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No export_decision.json files under {output_root}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V5.9 launch orchestrator")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["preflight", "t0", "export", "decision", "full_t0_cycle", "print_only"],
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent / "launch_config.json"))
    parser.add_argument("--checkpoint", default="", help="Checkpoint to export; used for mode=export")
    parser.add_argument("--output-dir", default="", help="Export output dir; used for mode=export")
    parser.add_argument("--decision-json", default="", help="Decision JSON path; used for mode=decision")
    parser.add_argument("--policy-id", default="v5_9_t0_export", help="Export policy id label")
    parser.add_argument("--training-instability", action="store_true", help="Forward instability signal into export decision")
    parser.add_argument("--dry-run", action="store_true", help="Use preflight dry-run behavior")
    parser.add_argument("--json", action="store_true", help="Emit full JSON for mode=decision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    record_dir = Path(cfg["paths"]["record_dir"]).resolve()
    output_root = Path(cfg["paths"]["output_root"]).resolve()
    preflight_script = record_dir / "runpod_preflight.py"
    t0_script = record_dir / "launch_t0.sh"
    export_script = record_dir / "launch_export_sweep.sh"

    t0_checkpoint = output_root / "T0" / "final_model.pt"
    t0_output_dir = output_root / "T0"
    t0_decision_json = t0_output_dir / "export_decision.json"

    preflight_cmd = ["python3", str(preflight_script), "--config", str(config_path)]
    if args.dry_run:
        preflight_cmd.append("--dry-run")

    t0_cmd = ["bash", str(t0_script), "--config", str(config_path)]

    export_checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else t0_checkpoint
    export_output_dir = Path(args.output_dir).resolve() if args.output_dir else t0_output_dir
    export_cmd = [
        "bash",
        str(export_script),
        "--config",
        str(config_path),
        "--checkpoint",
        str(export_checkpoint),
        "--output-dir",
        str(export_output_dir),
        "--policy-id",
        str(args.policy_id),
    ]
    if args.training_instability:
        export_cmd.append("--training-instability")

    if args.mode == "preflight":
        run_command(preflight_cmd, cwd=record_dir, execute=True)
        return

    if args.mode == "t0":
        run_command(t0_cmd, cwd=record_dir, execute=True)
        return

    if args.mode == "export":
        if not export_checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {export_checkpoint}")
        run_command(export_cmd, cwd=record_dir, execute=True)
        return

    if args.mode == "decision":
        decision_json = Path(args.decision_json).resolve() if args.decision_json else find_latest_decision_json(cfg)
        if not decision_json.exists():
            raise FileNotFoundError(f"Decision JSON does not exist: {decision_json}")
        payload = json.loads(decision_json.read_text(encoding="utf-8"))
        evaluated = evaluate_decision(payload, cfg)
        if args.json:
            out = {
                "decision_json": str(decision_json),
                **evaluated,
            }
            print(json.dumps(out, indent=2, sort_keys=True))
        else:
            print(evaluated["operator_token"])
        return

    if args.mode == "full_t0_cycle":
        run_command(preflight_cmd, cwd=record_dir, execute=True)
        run_command(t0_cmd, cwd=record_dir, execute=True)

        if not t0_checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint after T0: {t0_checkpoint}")

        run_command(export_cmd, cwd=record_dir, execute=True)
        if not t0_decision_json.exists():
            raise FileNotFoundError(f"Missing decision JSON after export: {t0_decision_json}")

        payload = json.loads(t0_decision_json.read_text(encoding="utf-8"))
        evaluated = evaluate_decision(payload, cfg)
        print(evaluated["operator_token"])
        return

    if args.mode == "print_only":
        run_command(preflight_cmd, cwd=record_dir, execute=False)
        run_command(t0_cmd, cwd=record_dir, execute=False)
        run_command(export_cmd, cwd=record_dir, execute=False)
        decision_cmd = [
            "python3",
            str(record_dir / "launch_orchestrator.py"),
            "--mode",
            "decision",
            "--config",
            str(config_path),
            "--decision-json",
            str(t0_decision_json),
        ]
        run_command(decision_cmd, cwd=record_dir, execute=False)
        return

    raise RuntimeError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
