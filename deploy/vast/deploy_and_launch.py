from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SshTarget:
    user: str
    host: str
    port: int

    @property
    def destination(self) -> str:
        return f"{self.user}@{self.host}"


class TeeLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def stream_process(self, argv: list[str], *, cwd: Path | None = None) -> int:
        self.log(f"RUN: {shlex.join(argv)}")
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            self.handle.write(line)
        rc = process.wait()
        self.handle.flush()
        self.log(f"RC={rc}: {shlex.join(argv)}")
        return rc

    def capture(self, argv: list[str], *, cwd: Path | None = None) -> str:
        self.log(f"CAPTURE: {shlex.join(argv)}")
        result = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip():
            self.handle.write(result.stdout)
            self.handle.flush()
        return result.stdout

    def close(self) -> None:
        self.handle.close()


def parse_ssh_target(raw: str) -> SshTarget:
    value = raw.strip()
    if value.startswith("ssh://"):
        parsed = urlparse(value)
        if not parsed.hostname or not parsed.username or not parsed.port:
            raise ValueError(f"Could not parse ssh url: {value!r}")
        return SshTarget(parsed.username, parsed.hostname, parsed.port)

    if value.startswith("ssh "):
        tokens = shlex.split(value)
        user_host = None
        port = 22
        idx = 0
        while idx < len(tokens):
            token = tokens[idx]
            if token == "-p":
                port = int(tokens[idx + 1])
                idx += 2
                continue
            if "@" in token and not token.startswith("-"):
                user_host = token
            idx += 1
        if not user_host:
            raise ValueError(f"Could not parse ssh command: {value!r}")
        user, host = user_host.split("@", 1)
        return SshTarget(user, host, port)

    if "@" in value:
        user, host_port = value.split("@", 1)
    else:
        user, host_port = "root", value
    if ":" in host_port:
        host, port_raw = host_port.rsplit(":", 1)
        return SshTarget(user, host, int(port_raw))
    return SshTarget(user, host_port, 22)


def git_metadata(repo_root: Path) -> dict[str, object]:
    def safe_capture(*argv: str) -> str:
        try:
            return subprocess.run(list(argv), cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()
        except Exception:
            return ""

    return {
        "head": safe_capture("git", "rev-parse", "HEAD"),
        "branch": safe_capture("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "status": safe_capture("git", "status", "--short"),
    }


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Sync current Parameter Golf worktree to a Vast.ai instance and launch the remote workflow")
    parser.add_argument("--instance-id", type=int, help="Vast instance id")
    parser.add_argument("--ssh-target", help="Override SSH target, for example ssh://root@host:port")
    parser.add_argument("--vast-bin", default=str(repo_root / ".venv-vastai/bin/vastai"))
    parser.add_argument("--remote-root", default="/workspace/parameter-golf")
    parser.add_argument("--config", default="search_configs/metastack_v2_wd_sliding_remote.yaml")
    parser.add_argument("--mode", choices=["search", "bootstrap-only"], default="search")
    parser.add_argument("--variant", default="sp1024")
    parser.add_argument("--train-shards", type=int, default=80)
    parser.add_argument("--max-runs", type=int, default=2)
    parser.add_argument("--launch-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--prebuilt-python-bin", default="")
    parser.add_argument("--matched-fineweb-repo-id", default=os.environ.get("MATCHED_FINEWEB_REPO_ID", "willdepueoai/parameter-golf"))
    parser.add_argument("--no-sync-back", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    local_run_root = repo_root / "deploy_runs" / args.launch_id / "local"
    local_run_root.mkdir(parents=True, exist_ok=True)
    logger = TeeLogger(local_run_root / "deploy.log")

    manifest = {
        "launch_id": args.launch_id,
        "started_at": utc_now(),
        "repo_root": str(repo_root),
        "config": args.config,
        "mode": args.mode,
        "variant": args.variant,
        "train_shards": args.train_shards,
        "max_runs": args.max_runs,
        "remote_root": args.remote_root,
        "git": git_metadata(repo_root),
    }

    try:
        if not args.ssh_target and not args.instance_id:
            raise ValueError("Either --instance-id or --ssh-target is required")

        if args.ssh_target:
            raw_target = args.ssh_target
        else:
            raw_target = logger.capture([args.vast_bin, "ssh-url", str(args.instance_id)], cwd=repo_root).strip()
        ssh_target = parse_ssh_target(raw_target)
        manifest["ssh_target"] = asdict(ssh_target)
        manifest["raw_ssh_target"] = raw_target
        (local_run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        ssh_base = [
            "ssh",
            "-p",
            str(ssh_target.port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=6",
            ssh_target.destination,
        ]
        rsync_ssh = " ".join(shlex.quote(token) for token in ssh_base[:-1])

        mkdir_cmd = ssh_base + [f"mkdir -p {shlex.quote(args.remote_root)}"]
        remote_cmd = [
            "bash",
            str((repo_root / "deploy/vast/remote_bootstrap_and_run.sh").relative_to(repo_root)),
            "--launch-id",
            args.launch_id,
            "--mode",
            args.mode,
            "--config",
            args.config,
            "--workdir",
            args.remote_root,
            "--variant",
            args.variant,
            "--train-shards",
            str(args.train_shards),
            "--world-size",
            "8",
            "--matched-fineweb-repo-id",
            args.matched_fineweb_repo_id,
        ]
        if args.max_runs is not None:
            remote_cmd += ["--max-runs", str(args.max_runs)]
        if args.prebuilt_python_bin:
            remote_cmd += ["--prebuilt-python-bin", args.prebuilt_python_bin]

        rsync_cmd = [
            "rsync",
            "-az",
            "--info=progress2",
            "--exclude-from",
            str(repo_root / "deploy/vast/rsync_excludes.txt"),
            "-e",
            rsync_ssh,
            f"{repo_root}/",
            f"{ssh_target.destination}:{args.remote_root}/",
        ]
        launch_ssh_cmd = ssh_base + [f"cd {shlex.quote(args.remote_root)} && {shlex.join(remote_cmd)}"]

        logger.log(f"Resolved SSH target: {ssh_target.destination}:{ssh_target.port}")
        logger.log(f"Remote bootstrap command: {shlex.join(remote_cmd)}")
        if args.dry_run:
            logger.log("Dry run complete")
            return

        rc = logger.stream_process(mkdir_cmd, cwd=repo_root)
        if rc != 0:
            raise RuntimeError(f"remote mkdir failed with rc={rc}")

        rc = logger.stream_process(rsync_cmd, cwd=repo_root)
        if rc != 0:
            raise RuntimeError(f"rsync failed with rc={rc}")

        remote_rc = logger.stream_process(launch_ssh_cmd, cwd=repo_root)

        if not args.no_sync_back:
            sync_back_cmd = [
                "rsync",
                "-az",
                "-e",
                rsync_ssh,
                f"{ssh_target.destination}:{args.remote_root}/deploy_runs/{args.launch_id}/",
                f"{repo_root}/deploy_runs/{args.launch_id}/remote/",
            ]
            sync_rc = logger.stream_process(sync_back_cmd, cwd=repo_root)
            manifest["sync_back_rc"] = sync_rc
        else:
            manifest["sync_back_rc"] = None

        manifest["finished_at"] = utc_now()
        manifest["remote_rc"] = remote_rc
        (local_run_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if remote_rc != 0:
            raise SystemExit(remote_rc)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
