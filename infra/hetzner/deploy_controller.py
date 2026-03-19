#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

SERVICE_PATH = "%h/.local/bin:%h/.npm/bin:%h/bin:/usr/local/bin:/usr/bin:/bin"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def run(cmd: list[str], *, dry_run: bool = False, input_text: str | None = None) -> None:
    print("+", shell_join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, input=input_text, text=input_text is not None, check=True)


def capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[2]


def require_clean_git(root: Path) -> None:
    status = capture(["git", "status", "--short"], cwd=root)
    if status:
        raise SystemExit("local git worktree must be clean before deployment")


def build_ssh_prefix(args: argparse.Namespace) -> list[str]:
    prefix = ["ssh", "-p", str(args.port)]
    if args.identity:
        prefix.extend(["-i", str(args.identity)])
    prefix.append(args.host)
    return prefix


def remote_shell_path(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return f"$HOME/{path[2:]}"
    return path


def remote_shell_expr(path: str) -> str:
    shell_path = remote_shell_path(path)
    if shell_path == "$HOME" or shell_path.startswith("$HOME/"):
        return f'"{shell_path}"'
    return shlex.quote(shell_path)


def remote_service_path(path: str) -> str:
    if path == "~":
        return "%h"
    if path.startswith("~/"):
        return f"%h/{path[2:]}"
    return path


def build_rsync_ssh(args: argparse.Namespace) -> str:
    parts = ["ssh", "-p", str(args.port)]
    if args.identity:
        parts.extend(["-i", str(args.identity)])
    return shell_join(parts)


def ssh_run(args: argparse.Namespace, script: str, *, dry_run: bool) -> None:
    run([*build_ssh_prefix(args), "bash", "-lc", script], dry_run=dry_run)


def create_bundle(root: Path, *, dry_run: bool) -> tuple[Path, str]:
    deploy_ref = capture(["git", "branch", "--show-current"], cwd=root)
    if not deploy_ref:
        raise SystemExit("deploy script requires a checked out branch, not detached HEAD")
    with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as tmp:
        bundle_path = Path(tmp.name)
    run(["git", "bundle", "create", str(bundle_path), deploy_ref], dry_run=dry_run)
    return bundle_path, deploy_ref


def render_service(args: argparse.Namespace, remote_repo_dir: str, remote_env_file: str) -> str:
    escaped_args = shell_join(shlex.split(args.controller_args.strip() or "--forever"))
    service_repo_dir = remote_service_path(remote_repo_dir)
    service_env_file = remote_service_path(remote_env_file)
    exec_start = (
        '/bin/bash -lc "cd '
        f"{service_repo_dir}/autoresearch"
        f' && uv run python run_pgolf_experiment.py {escaped_args}"'
    )
    return f"""[Unit]
Description=Parameter Golf Autoresearch Controller
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={service_repo_dir}
Environment=PATH={SERVICE_PATH}
EnvironmentFile={service_env_file}
ExecStart={exec_start}
Restart=always
RestartSec=15

[Install]
WantedBy=default.target
"""


def upload_file(args: argparse.Namespace, local_path: Path, remote_path: str) -> None:
    remote_target = f"{args.host}:{remote_path}"
    cmd = [
        "rsync",
        "-az",
        "--mkpath",
        "-e",
        build_rsync_ssh(args),
        str(local_path),
        remote_target,
    ]
    run(cmd, dry_run=args.dry_run)


def remote_prepare_script(args: argparse.Namespace) -> str:
    repo_dir = remote_shell_expr(args.remote_repo_dir)
    env_dir = remote_shell_expr(str(Path(args.remote_env_file).parent))
    systemd_dir = remote_shell_expr(args.remote_systemd_dir)
    state_dir = remote_shell_expr(args.remote_state_dir)
    service_name = shlex.quote(args.service_name)
    require_python = (
        "command -v python3 >/dev/null 2>&1 || "
        "{ echo 'python3 is required on the controller host'; exit 1; }"
    )
    require_git = (
        "command -v git >/dev/null 2>&1 || "
        "{ echo 'git is required on the controller host'; exit 1; }"
    )
    export_path = f'export PATH="{SERVICE_PATH.replace("%h", "$HOME")}"'
    require_uv = "command -v uv >/dev/null 2>&1 || { echo 'uv install failed'; exit 1; }"
    require_codex = (
        "command -v codex >/dev/null 2>&1 || "
        "{ echo 'codex CLI missing on controller host'; exit 1; }"
    )
    return "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {repo_dir} {env_dir} {systemd_dir} {state_dir}",
            f"if [ ! -d {repo_dir}/.git ]; then git init {repo_dir}; fi",
            require_python,
            require_git,
            "if ! command -v uv >/dev/null 2>&1; then",
            "  curl -LsSf https://astral.sh/uv/install.sh | sh",
            "fi",
            export_path,
            require_uv,
            require_codex,
            f"git config --global user.name {shlex.quote(args.git_user_name)}",
            f"git config --global user.email {shlex.quote(args.git_user_email)}",
            f"git config --global --add safe.directory {repo_dir}",
            f"git config --global --add safe.directory {repo_dir}/.git",
            f"cd {repo_dir}",
            "git diff --quiet",
            "git diff --cached --quiet",
            f"systemctl --user stop {service_name} >/dev/null 2>&1 || true",
        ]
    )


def remote_finalize_script(
    args: argparse.Namespace,
    *,
    remote_bundle_path: str,
    deploy_ref: str,
) -> str:
    service_name = shlex.quote(args.service_name)
    remote_systemd_dir = remote_shell_path(args.remote_systemd_dir).rstrip("/")
    service_file = shlex.quote(f"{remote_systemd_dir}/{args.service_name}.service")
    remote_repo_dir = remote_shell_expr(args.remote_repo_dir)
    remote_bundle_expr = remote_shell_expr(remote_bundle_path)
    export_path = f'export PATH="{SERVICE_PATH.replace("%h", "$HOME")}"'
    require_systemctl = (
        "command -v systemctl >/dev/null 2>&1 || "
        "{ echo 'systemctl is required on the controller host'; exit 1; }"
    )
    commands = [
        "set -euo pipefail",
        export_path,
        require_systemctl,
        f"test -f {service_file}",
        f"chown -R \"$USER\":\"$USER\" {remote_repo_dir}",
        f"git config --global user.name {shlex.quote(args.git_user_name)}",
        f"git config --global user.email {shlex.quote(args.git_user_email)}",
        f"git config --global --add safe.directory {remote_repo_dir}",
        f"git config --global --add safe.directory {remote_repo_dir}/.git",
        f"cd {remote_repo_dir}",
        "git diff --quiet",
        "git diff --cached --quiet",
        f"git fetch {remote_bundle_expr} {shlex.quote(deploy_ref)}:refs/heads/codex-deploy",
        (
            "if git show-ref --verify --quiet refs/heads/main; then "
            "git checkout --quiet main; "
            "else git checkout --quiet -B main codex-deploy; fi"
        ),
        (
            "if git merge-base --is-ancestor HEAD codex-deploy; then "
            "git merge --ff-only codex-deploy; "
            "else git merge --no-edit codex-deploy; fi"
        ),
        "git branch -D codex-deploy >/dev/null 2>&1 || true",
        f"rm -f {remote_bundle_expr}",
        "systemctl --user daemon-reload",
        f"systemctl --user enable {service_name}",
    ]
    if args.enable_linger:
        commands.append("sudo loginctl enable-linger \"$USER\"")
    if args.start:
        commands.append(f"systemctl --user restart {service_name}")
        commands.append(f"systemctl --user --no-pager --full status {service_name} || true")
    return "\n".join(commands)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy the Parameter Golf autoresearch controller to a Hetzner host."
    )
    parser.add_argument("--host", required=True, help="SSH destination for the controller host")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--identity", type=Path, default=None, help="Optional SSH identity file")
    parser.add_argument(
        "--git-user-name",
        default=os.environ.get("PGOLF_GIT_USER_NAME", "carlulsoe"),
        help="Git committer name configured on the controller host",
    )
    parser.add_argument(
        "--git-user-email",
        default=os.environ.get("PGOLF_GIT_USER_EMAIL", "carlulsoe@gmail.com"),
        help="Git committer email configured on the controller host",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="Local env file to upload as the controller runtime contract",
    )
    parser.add_argument(
        "--remote-repo-dir",
        default="~/parameter-golf",
        help="Where the repository should live on the controller host",
    )
    parser.add_argument(
        "--remote-env-file",
        default="~/.config/parameter-golf/autoresearch.env",
        help="Remote path for the controller env file",
    )
    parser.add_argument(
        "--remote-systemd-dir",
        default="~/.config/systemd/user",
        help="Remote systemd user unit directory",
    )
    parser.add_argument(
        "--remote-state-dir",
        default="~/.local/state/parameter-golf",
        help="Remote controller log directory",
    )
    parser.add_argument(
        "--service-name",
        default="parameter-golf-autoresearch",
        help="systemd user service name",
    )
    parser.add_argument(
        "--controller-args",
        default="--forever",
        help="Arguments passed to run_pgolf_experiment.py inside the service",
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start or restart the service after deployment",
    )
    parser.add_argument(
        "--enable-linger",
        action="store_true",
        help="Run `sudo loginctl enable-linger $USER` on the controller host",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    env_file = args.env_file.resolve()
    if not env_file.exists():
        raise SystemExit(f"env file does not exist: {env_file}")
    require_clean_git(root)
    bundle_path, deploy_ref = create_bundle(root, dry_run=args.dry_run)
    remote_bundle_path = f"{args.remote_state_dir.rstrip('/')}/deploy.bundle"
    service_path: Path | None = None

    try:
        ssh_run(args, remote_prepare_script(args), dry_run=args.dry_run)
        upload_file(args, bundle_path, remote_bundle_path)
        upload_file(args, env_file, args.remote_env_file)

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
            tmp.write(
                render_service(
                    args,
                    remote_repo_dir=args.remote_repo_dir,
                    remote_env_file=args.remote_env_file,
                )
            )
            service_path = Path(tmp.name)
        upload_file(
            args,
            service_path,
            f"{args.remote_systemd_dir.rstrip('/')}/{args.service_name}.service",
        )
    finally:
        bundle_path.unlink(missing_ok=True)
        if service_path is not None:
            service_path.unlink(missing_ok=True)

    ssh_run(
        args,
        remote_finalize_script(
            args,
            remote_bundle_path=remote_bundle_path,
            deploy_ref=deploy_ref,
        ),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
