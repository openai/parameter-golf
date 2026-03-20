from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable

from search.config import RunnerConfig


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    run_env: dict[str, str]
    argv: list[str]
    display_command: str
    log_path: Path
    stdout_path: Path
    status_path: Path


@dataclass(frozen=True)
class CompletedRun:
    prepared: PreparedRun
    return_code: int
    started_at: str
    finished_at: str


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.12g}"
    return str(value)


def build_run_env(fixed_env: dict[str, Any], overrides: dict[str, Any], *, run_id: str) -> dict[str, str]:
    env = {key: _stringify(value) for key, value in fixed_env.items()}
    env["RUN_ID"] = run_id
    for key, value in overrides.items():
        if key == "WARMDOWN_FRACTION":
            continue
        env[key] = _stringify(value)
    if "WARMDOWN_FRACTION" in overrides:
        iterations = int(env["ITERATIONS"])
        warmdown_iters = round(float(overrides["WARMDOWN_FRACTION"]) * iterations)
        env["WARMDOWN_ITERS"] = str(warmdown_iters)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_prepared_run(
    runner: RunnerConfig,
    fixed_env: dict[str, Any],
    overrides: dict[str, Any],
    *,
    run_id: str,
    output_root: Path,
) -> PreparedRun:
    run_env = build_run_env(fixed_env, overrides, run_id=run_id)
    python_bin = shlex.quote(str(runner.python_bin))
    script_path = shlex.quote(str(runner.script_path))
    if runner.gpus > 1:
        command = f"{python_bin} -m torch.distributed.run --standalone --nproc_per_node={runner.gpus} {script_path}"
    else:
        command = f"{python_bin} {script_path}"
    env_exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(run_env.items()))
    if runner.activate_script is not None:
        shell_command = f"source {shlex.quote(str(runner.activate_script))} && exec {command}"
        argv = ["/bin/bash", "-lc", shell_command]
        display_command = (
            f"cd {shlex.quote(str(runner.workdir))} && "
            f"env {env_exports} /bin/bash -lc {shlex.quote(shell_command)}"
        )
    else:
        shell_command = f"exec {command}"
        argv = [str(runner.python_bin), str(runner.script_path)] if runner.gpus == 1 else [
            str(runner.python_bin), "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={runner.gpus}", str(runner.script_path)
        ]
        display_command = f"cd {shlex.quote(str(runner.workdir))} && env {env_exports} {shell_command}"
    stdout_dir = output_root / "stdout"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    return PreparedRun(
        run_id=run_id,
        run_env=run_env,
        argv=argv,
        display_command=display_command,
        log_path=(runner.logs_dir / f"{run_id}.txt").resolve(),
        stdout_path=(stdout_dir / f"{run_id}.stdout").resolve(),
        status_path=(output_root / "run_status" / f"{run_id}.json").resolve(),
    )


def launch_run(
    prepared: PreparedRun,
    *,
    workdir: Path,
    tee_to_stdout: bool = True,
    on_output_line: Callable[[str], None] | None = None,
) -> CompletedRun:
    launched_env = os.environ.copy()
    launched_env.update(prepared.run_env)
    started_at = datetime.now(timezone.utc).isoformat()
    prepared.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.status_path.parent.mkdir(parents=True, exist_ok=True)
    with prepared.stdout_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            prepared.argv,
            cwd=workdir,
            env=launched_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if tee_to_stdout:
                print(line, end="")
            if on_output_line is not None:
                on_output_line(line.rstrip("\n"))
        process.stdout.close()
        return_code = process.wait()
    finished_at = datetime.now(timezone.utc).isoformat()
    return CompletedRun(
        prepared=prepared,
        return_code=return_code,
        started_at=started_at,
        finished_at=finished_at,
    )
