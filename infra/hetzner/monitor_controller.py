#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_ID_PATTERN = re.compile(r"^(?P<tag>.+)_(?P<num>\d+)$")


class MonitorError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    repo_dir: Path
    env_file: Path
    service_name: str
    trace_root: Path
    results_file: Path
    reviews_file: Path
    harness_log: Path
    remote_log_dir: Path
    remote_host: str
    remote_port: int
    remote_identity: str
    remote_force_tty: bool
    max_wallclock_seconds: int
    codex_binary: str
    codex_model: str
    max_idle_minutes: int
    remote_silence_minutes: int
    run_grace_minutes: int
    codex_cooldown_minutes: int
    auto_restart_service: bool
    journal_lines: int
    monitor_dir: Path
    incidents_dir: Path
    health_file: Path
    checks_file: Path
    codex_state_file: Path
    lock_file: Path


def iso_now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_shell_assignments(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise MonitorError(f"invalid assignment in {path}: {raw_line}")
        key, rhs = line.split("=", 1)
        key = key.strip()
        rhs = rhs.strip()
        if not key:
            raise MonitorError(f"invalid assignment in {path}: {raw_line}")
        if rhs[:1] in {"'", '"'}:
            tokens = shlex.split(rhs, posix=True)
            if len(tokens) != 1:
                raise MonitorError(f"invalid shell assignment in {path}: {raw_line}")
            values[key] = tokens[0]
        else:
            values[key] = rhs
    return values


def env_flag(values: dict[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise MonitorError(f"invalid boolean value for {name}: {raw}")


def env_path(repo_dir: Path, values: dict[str, str], name: str, default: str) -> Path:
    raw = values.get(name, default)
    path = Path(raw)
    return path if path.is_absolute() else repo_dir / path


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def load_config(argv: list[str]) -> Config:
    parser = argparse.ArgumentParser(description="Monitor the Hetzner autoresearch controller.")
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--service-name", default="parameter-golf-autoresearch")
    parser.add_argument("--codex-model", default=None)
    args = parser.parse_args(argv)

    repo_dir = args.repo_dir.resolve()
    env_file = args.env_file.resolve()
    env_values = parse_shell_assignments(env_file)
    trace_root = env_path(repo_dir, env_values, "TRACE_ROOT", "controller_state/autoresearch")
    monitor_dir = trace_root / "monitor"
    return Config(
        repo_dir=repo_dir,
        env_file=env_file,
        service_name=args.service_name,
        trace_root=trace_root,
        results_file=env_path(repo_dir, env_values, "RESULTS_FILE", "results.tsv"),
        reviews_file=env_path(repo_dir, env_values, "REVIEWS_FILE", "reviews.tsv"),
        harness_log=env_path(repo_dir, env_values, "HARNESS_LOG", "logs/autoresearch_hetzner.log"),
        remote_log_dir=env_path(repo_dir, env_values, "REMOTE_LOG_DIR", "remote_logs"),
        remote_host=env_values.get("REMOTE_HOST", ""),
        remote_port=int(env_values.get("REMOTE_PORT", "22")),
        remote_identity=env_values.get("REMOTE_IDENTITY", ""),
        remote_force_tty=env_flag(
            env_values,
            "REMOTE_SSH_FORCE_TTY",
            "runpod.io" in env_values.get("REMOTE_HOST", ""),
        ),
        max_wallclock_seconds=int(env_values.get("MAX_WALLCLOCK_SECONDS", "600")),
        codex_binary=env_values.get("CODEX_BIN", "codex"),
        codex_model=args.codex_model or env_values.get("MONITOR_CODEX_MODEL", "gpt-5.4"),
        max_idle_minutes=int(env_values.get("MONITOR_MAX_IDLE_MINUTES", "25")),
        remote_silence_minutes=int(env_values.get("MONITOR_REMOTE_SILENCE_MINUTES", "20")),
        run_grace_minutes=int(env_values.get("MONITOR_RUN_GRACE_MINUTES", "20")),
        codex_cooldown_minutes=int(env_values.get("MONITOR_CODEX_COOLDOWN_MINUTES", "60")),
        auto_restart_service=env_flag(env_values, "MONITOR_AUTO_RESTART_SERVICE", True),
        journal_lines=int(env_values.get("MONITOR_JOURNAL_LINES", "80")),
        monitor_dir=monitor_dir,
        incidents_dir=monitor_dir / "incidents",
        health_file=monitor_dir / "health.json",
        checks_file=monitor_dir / "checks.jsonl",
        codex_state_file=monitor_dir / "codex_state.json",
        lock_file=monitor_dir / "monitor.lock",
    )


def file_age_seconds(path: Path, *, now: float) -> float | None:
    if not path.exists():
        return None
    return max(0.0, now - path.stat().st_mtime)


def latest_run_dir(runs_dir: Path) -> Path | None:
    best: tuple[int, Path] | None = None
    if not runs_dir.exists():
        return None
    for path in runs_dir.iterdir():
        if not path.is_dir():
            continue
        match = RUN_ID_PATTERN.match(path.name)
        if not match:
            continue
        number = int(match.group("num"))
        if best is None or number > best[0]:
            best = (number, path)
    return best[1] if best else None


def latest_recorded_run_id(results_file: Path) -> str:
    last = ""
    if not results_file.exists():
        return last
    for line in results_file.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 5:
            last = parts[4]
    return last


def service_state(service_name: str) -> dict[str, str]:
    result = run_cmd(
        [
            "systemctl",
            "--user",
            "show",
            service_name,
            "--property=ActiveState,SubState,ExecMainPID,Result,ActiveEnterTimestamp",
        ],
        check=False,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    values["show_exit_code"] = str(result.returncode)
    return values


def journal_tail(service_name: str, lines: int) -> str:
    result = run_cmd(
        [
            "journalctl",
            "--user",
            "-u",
            service_name,
            "-n",
            str(lines),
            "--no-pager",
            "--output=cat",
        ],
        check=False,
    )
    return result.stdout


def restart_service(service_name: str) -> bool:
    run_cmd(["systemctl", "--user", "restart", service_name], check=False)
    time.sleep(3)
    state = service_state(service_name)
    return state.get("ActiveState") == "active"


def build_ssh_cmd(config: Config, remote_cmd: str | None = None) -> list[str]:
    cmd = ["ssh", "-p", str(config.remote_port)]
    if config.remote_identity:
        cmd.extend(["-i", config.remote_identity])
    cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
    if config.remote_force_tty:
        cmd.append("-tt")
    cmd.append(config.remote_host)
    if remote_cmd:
        cmd.append(remote_cmd)
    return cmd


def remote_reachable(config: Config) -> tuple[bool, str]:
    if not config.remote_host:
        return False, "REMOTE_HOST not configured"
    result = run_cmd(build_ssh_cmd(config, "true"), check=False)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def diagnose_remote_processes(config: Config) -> dict[str, Any]:
    if not config.remote_host:
        return {"reachable": False, "reason": "REMOTE_HOST not configured"}
    probe = (
        "bash -lc "
        + shlex.quote(
            "ps -eo pid,cmd | grep -E 'torchrun|train_gpt.py' | grep -v grep || true\n"
            "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total "
            "--format=csv,noheader,nounits || true"
        )
    )
    result = run_cmd(build_ssh_cmd(config, probe), check=False)
    output = (result.stdout + result.stderr).strip()
    return {
        "reachable": result.returncode == 0,
        "exit_code": result.returncode,
        "output": output,
    }


def active_run_snapshot(config: Config, *, now: float) -> dict[str, Any]:
    runs_dir = config.trace_root / "runs"
    run_dir = latest_run_dir(runs_dir)
    if run_dir is None:
        return {"run_dir": "", "run_id": "", "state": "none"}
    run_id = run_dir.name
    failure = run_dir / "failure.json"
    manifest = run_dir / "manifest.json"
    remote_log = config.remote_log_dir / f"{run_id}.log"
    state = "active"
    if manifest.exists():
        state = "finalized"
    elif failure.exists():
        state = "failed"
    remote_log_age = file_age_seconds(remote_log, now=now)
    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "state": state,
        "run_dir_age_seconds": file_age_seconds(run_dir, now=now),
        "remote_log": str(remote_log),
        "remote_log_exists": remote_log.exists(),
        "remote_log_age_seconds": remote_log_age,
        "manifest_exists": manifest.exists(),
        "failure_exists": failure.exists(),
    }


def maybe_invoke_codex(config: Config, payload: dict[str, Any]) -> dict[str, Any]:
    fingerprint = "|".join(sorted(payload["anomalies"]))
    now = iso_now()
    state = {}
    if config.codex_state_file.exists():
        state = json.loads(config.codex_state_file.read_text(encoding="utf-8"))
    last_fingerprint = state.get("fingerprint", "")
    last_timestamp = state.get("timestamp", "")
    if last_fingerprint == fingerprint and last_timestamp:
        previous = datetime.fromisoformat(last_timestamp).timestamp()
        if time.time() - previous < config.codex_cooldown_minutes * 60:
            return {"invoked": False, "reason": "cooldown_active", "fingerprint": fingerprint}

    incident_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    incident_dir = config.incidents_dir / incident_id
    incident_dir.mkdir(parents=True, exist_ok=True)
    context_file = incident_dir / "context.json"
    prompt_file = incident_dir / "codex_prompt.txt"
    diagnosis_file = incident_dir / "diagnosis.md"
    log_file = incident_dir / "codex.log"
    write_json(context_file, payload)
    prompt = "\n".join(
        [
            "This is a monitor escalation for the Parameter Golf controller host.",
            "",
            "Read the incident context JSON and inspect the local repository state if needed.",
            "Do not edit code, do not restart services, and do not modify the repo.",
            "",
            f"Incident context: {context_file}",
            f"Write a short diagnosis markdown file to: {diagnosis_file}",
            "",
            "The diagnosis should include:",
            "- Assessment",
            "- Evidence",
            "- Most likely cause",
            "- Suggested next action",
        ]
    )
    prompt_file.write_text(prompt, encoding="utf-8")
    result = run_cmd(
        [
            config.codex_binary,
            "exec",
            "-m",
            config.codex_model,
            "--dangerously-bypass-approvals-and-sandbox",
            prompt,
        ],
        cwd=config.repo_dir,
        check=False,
    )
    log_file.write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    write_json(
        config.codex_state_file,
        {
            "fingerprint": fingerprint,
            "timestamp": now,
            "incident_dir": str(incident_dir),
            "exit_code": result.returncode,
        },
    )
    return {
        "invoked": True,
        "fingerprint": fingerprint,
        "incident_dir": str(incident_dir),
        "exit_code": result.returncode,
        "diagnosis_file": str(diagnosis_file),
        "log_file": str(log_file),
    }


def collect_health(config: Config) -> dict[str, Any]:
    now = time.time()
    service = service_state(config.service_name)
    harness_age = file_age_seconds(config.harness_log, now=now)
    run = active_run_snapshot(config, now=now)
    latest_results_run = latest_recorded_run_id(config.results_file)
    journal = journal_tail(config.service_name, config.journal_lines)

    anomalies: list[str] = []
    actions: list[str] = []
    facts: dict[str, Any] = {
        "timestamp": iso_now(),
        "service": service,
        "harness_log": str(config.harness_log),
        "harness_log_exists": config.harness_log.exists(),
        "harness_log_age_seconds": harness_age,
        "latest_results_run_id": latest_results_run,
        "run": run,
        "results_file": str(config.results_file),
        "reviews_file": str(config.reviews_file),
    }

    if service.get("ActiveState") != "active":
        anomalies.append("service_not_active")
        if config.auto_restart_service:
            restarted = restart_service(config.service_name)
            actions.append(f"service_restart_attempted={restarted}")
            service = service_state(config.service_name)
            facts["service_after_restart"] = service
            if service.get("ActiveState") == "active":
                anomalies = [item for item in anomalies if item != "service_not_active"]
                actions.append("service_restart_succeeded")
            else:
                anomalies.append("service_restart_failed")

    if harness_age is None:
        anomalies.append("missing_harness_log")
    elif harness_age > config.max_idle_minutes * 60:
        anomalies.append("controller_idle_too_long")

    active_run = run.get("state") == "active"
    if active_run:
        remote_log_age = run.get("remote_log_age_seconds")
        if remote_log_age is None:
            anomalies.append("missing_remote_log_for_active_run")
        elif remote_log_age > config.remote_silence_minutes * 60:
            anomalies.append("remote_log_silent_too_long")
        run_age = run.get("run_dir_age_seconds")
        max_age = None
        if config.max_wallclock_seconds > 0:
            max_age = config.max_wallclock_seconds + config.run_grace_minutes * 60
        if max_age is not None and isinstance(run_age, (int, float)) and run_age > max_age:
            anomalies.append("active_run_exceeded_expected_window")

    remote_probe: dict[str, Any] | None = None
    if active_run or anomalies:
        remote_probe = diagnose_remote_processes(config)
        facts["remote_probe"] = remote_probe
        if active_run and not remote_probe.get("reachable", False):
            anomalies.append("remote_host_unreachable")

    payload = {
        "timestamp": facts["timestamp"],
        "ok": not anomalies,
        "anomalies": sorted(set(anomalies)),
        "actions": actions,
        "facts": facts,
        "journal_tail": journal,
    }
    codex_result = maybe_invoke_codex(config, payload) if anomalies else {"invoked": False}
    payload["codex"] = codex_result
    return payload


def main(argv: list[str]) -> int:
    try:
        config = load_config(argv)
        config.monitor_dir.mkdir(parents=True, exist_ok=True)
        with config.lock_file.open("w", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            payload = collect_health(config)
            write_json(config.health_file, payload)
            append_jsonl(config.checks_file, payload)
        return 0 if payload["ok"] else 1
    except BlockingIOError:
        print("monitor already running", file=sys.stderr)
        return 0
    except MonitorError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
