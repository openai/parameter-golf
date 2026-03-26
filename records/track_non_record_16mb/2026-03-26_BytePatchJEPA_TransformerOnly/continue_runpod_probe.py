from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REMOTE_REPO = "/workspace/parameter-golf"
REMOTE_PARENT = f"{REMOTE_REPO}/records/track_non_record_16mb"
REMOTE_DIR = f"{REMOTE_PARENT}/{ROOT.name}"


def ssh_base(ip: str, port: int, key_path: Path) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-i",
        str(key_path),
        "-p",
        str(port),
        f"root@{ip}",
    ]


def scp_base(ip: str, port: int, key_path: Path) -> list[str]:
    return [
        "scp",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-i",
        str(key_path),
        "-P",
        str(port),
    ]


def run_local(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True)


def run_remote(ip: str, port: int, key_path: Path, command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(ssh_base(ip, port, key_path) + [command], check=check, text=True)


def load_session(session_path: Path) -> dict:
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    ready = payload.get("ready") or {}
    if not ready.get("public_ip") or not ready.get("ssh_port"):
        raise RuntimeError(f"session missing SSH details: {session_path}")
    return payload


def sync_code(ip: str, port: int, key_path: Path) -> None:
    run_remote(ip, port, key_path, f"bash -lc {shlex.quote(f'mkdir -p {REMOTE_DIR}')}")
    upload_files = [
        ROOT / "README.md",
        ROOT / "JEPA_SUMMARY.md",
        ROOT / "bootstrap_byte260_subset.py",
        ROOT / "continue_runpod_probe.py",
        ROOT / "launch_runpod_probe.py",
        ROOT / "run_probe_pair.sh",
        ROOT / "summarize_sweep.py",
        ROOT / "train_gpt.py",
    ]
    run_local(scp_base(ip, port, key_path) + [*(str(path) for path in upload_files), f"root@{ip}:{REMOTE_DIR}/"])


def sync_phase_results(ip: str, port: int, key_path: Path, phase: str) -> bool:
    remote_archive = f"{REMOTE_DIR}/continue_{phase}.tgz"
    archive_cmd = (
        f"if [ -d {shlex.quote(f'{REMOTE_DIR}/results/{phase}')} ]; then "
        f"cd {shlex.quote(REMOTE_DIR)} && "
        f"tar czf {shlex.quote(Path(remote_archive).name)} results/{shlex.quote(phase)} results/{shlex.quote(phase)}/summary.json 2>/dev/null || "
        f"tar czf {shlex.quote(Path(remote_archive).name)} results/{shlex.quote(phase)}; "
        f"fi"
    )
    run_remote(ip, port, key_path, f"bash -lc {shlex.quote(archive_cmd)}", check=False)
    local_archive = ROOT / f"continue_{phase}.tgz"
    copy = subprocess.run(
        scp_base(ip, port, key_path) + [f"root@{ip}:{remote_archive}", str(local_archive)],
        check=False,
        text=True,
        capture_output=True,
    )
    if copy.returncode != 0:
        return False
    run_local(["tar", "xzf", str(local_archive), "-C", str(ROOT)])
    local_archive.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("backbone_screen", "objective_screen", "ablate", "scale", "data_scale"),
        required=True,
    )
    parser.add_argument("--session-file", default=str(ROOT / "runpod_session.json"))
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--skip-sync-code", action="store_true")
    parser.add_argument("--run-env", action="append", default=[], help="Extra KEY=VALUE env passed to run_probe_pair.sh")
    args = parser.parse_args()

    key_path = Path.home() / ".ssh/id_ed25519"
    if not key_path.exists():
        raise RuntimeError(f"Missing private key: {key_path}")

    session_path = Path(args.session_file).resolve()
    base_session = load_session(session_path)
    ready = base_session["ready"]
    ip = ready["public_ip"]
    port = int(ready["ssh_port"])
    gpu_count = args.gpu_count if args.gpu_count is not None else int(base_session.get("requested_gpu_count", 1))

    if not args.skip_sync_code:
        sync_code(ip, port, key_path)

    phase_session_path = ROOT / f"continue_{args.phase}_session.json"
    phase_session = {
        "phase": args.phase,
        "pod_id": base_session.get("pod_id"),
        "public_ip": ip,
        "ssh_port": port,
        "gpu_count": gpu_count,
        "run_env": args.run_env,
        "started_at_unix": time.time(),
    }
    phase_session_path.write_text(json.dumps(phase_session, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env_items = [f"RUN_PHASE={args.phase}", f"BACKBONE_GPU_COUNT={gpu_count}"]
    env_items.extend(args.run_env)
    env_prefix = "env " + " ".join(shlex.quote(item) for item in env_items) + " "
    remote_cmd = f"cd {REMOTE_DIR} && chmod +x run_probe_pair.sh && {env_prefix}bash run_probe_pair.sh"
    remote_run = run_remote(ip, port, key_path, f"bash -lc {shlex.quote(remote_cmd)}", check=False)
    phase_session["remote_returncode"] = remote_run.returncode

    synced_results = sync_phase_results(ip, port, key_path, args.phase)
    phase_session["synced_results"] = synced_results
    phase_session["completed_at_unix"] = time.time()
    phase_session_path.write_text(json.dumps(phase_session, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if remote_run.returncode != 0 and not synced_results:
        raise RuntimeError(f"remote phase {args.phase} failed with code {remote_run.returncode} and no results synced")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
