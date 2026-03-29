from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_BASE = "/workspace/hyperactive-octupus"
DEFAULT_REMOTE_EVAL = f"{DEFAULT_REMOTE_BASE}/parameter-golf/evaluator/runpod_remote_eval.py"
DEFAULT_SSH_HOST = os.environ.get("RUNPOD_SSH_HOST", "216.243.220.220")
DEFAULT_SSH_PORT = os.environ.get("RUNPOD_SSH_PORT", "15917")
DEFAULT_SSH_USER = os.environ.get("RUNPOD_SSH_USER", "root")
DEFAULT_SSH_KEY = os.environ.get(
    "RUNPOD_SSH_KEY", str(Path.home() / ".ssh" / "google_compute_engine")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ship a candidate to a RunPod evaluator and return JSON metrics.")
    parser.add_argument("candidate_program")
    parser.add_argument("--ssh-host", default=DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-port", default=DEFAULT_SSH_PORT)
    parser.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--remote-base", default=os.environ.get("RUNPOD_REMOTE_BASE", DEFAULT_REMOTE_BASE))
    parser.add_argument("--remote-eval", default=os.environ.get("RUNPOD_REMOTE_EVAL", DEFAULT_REMOTE_EVAL))
    parser.add_argument("--env-json", default="{}")
    parser.add_argument("--family", default="runpod_remote_candidate")
    parser.add_argument("--json-indent", type=int, default=2)
    return parser.parse_args()


def _ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        args.ssh_key,
        "-p",
        str(args.ssh_port),
        f"{args.ssh_user}@{args.ssh_host}",
    ]


def _scp_base(args: argparse.Namespace) -> list[str]:
    return [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        args.ssh_key,
        "-P",
        str(args.ssh_port),
    ]


def evaluate_remote(
    candidate_program: str | Path,
    *,
    ssh_host: str = DEFAULT_SSH_HOST,
    ssh_port: str = DEFAULT_SSH_PORT,
    ssh_user: str = DEFAULT_SSH_USER,
    ssh_key: str | None = DEFAULT_SSH_KEY,
    remote_base: str = DEFAULT_REMOTE_BASE,
    remote_eval: str = DEFAULT_REMOTE_EVAL,
    env_json: str = "{}",
    family: str = "runpod_remote_candidate",
) -> dict:
    ssh_key = ssh_key or DEFAULT_SSH_KEY
    candidate = Path(candidate_program).resolve()
    if not candidate.exists():
        raise SystemExit(f"Candidate program not found: {candidate}")

    class Args:
        pass

    args = Args()
    args.ssh_host = ssh_host
    args.ssh_port = ssh_port
    args.ssh_user = ssh_user
    args.ssh_key = ssh_key
    args.remote_base = remote_base
    args.remote_eval = remote_eval
    args.env_json = env_json
    args.family = family

    remote_tmp = f"{args.remote_base}/tmp/{uuid.uuid4().hex}.py"
    subprocess.run(
        _ssh_base(args) + [f"mkdir -p {shlex.quote(args.remote_base)}/tmp"],
        check=True,
    )
    subprocess.run(
        _scp_base(args) + [str(candidate), f"{args.ssh_user}@{args.ssh_host}:{remote_tmp}"],
        check=True,
    )

    remote_cmd = (
        f"cd {shlex.quote(args.remote_base)} && "
        f"python {shlex.quote(args.remote_eval)} "
        f"{shlex.quote(remote_tmp)} "
        f"--family {shlex.quote(args.family)} "
        f"--env-json {shlex.quote(args.env_json)}"
    )
    proc = subprocess.run(
        _ssh_base(args) + [remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout)

    return json.loads(proc.stdout)


def main() -> int:
    args = parse_args()
    payload = evaluate_remote(
        args.candidate_program,
        ssh_host=args.ssh_host,
        ssh_port=str(args.ssh_port),
        ssh_user=args.ssh_user,
        ssh_key=args.ssh_key,
        remote_base=args.remote_base,
        remote_eval=args.remote_eval,
        env_json=args.env_json,
        family=args.family,
    )
    json.dump(payload, sys.stdout, indent=args.json_indent)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
