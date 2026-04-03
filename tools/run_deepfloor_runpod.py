#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_SCRIPT = (
    REPO_ROOT
    / "records"
    / "track_non_record_16mb"
    / "2026-04-03_DeepFloor"
    / "run_smallbox_suite.sh"
)


def run_json(cmd: list[str]) -> dict | list:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def run_passthrough(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_create_command(args: argparse.Namespace) -> list[str]:
    name = args.name or f"deepfloor-smallbox-{int(time.time())}"
    cmd = [
        "runpodctl",
        "pod",
        "create",
        "--template-id",
        args.template_id,
        "--gpu-id",
        args.gpu_id,
        "--gpu-count",
        str(args.gpu_count),
        "--name",
        name,
        "--ports",
        args.ports,
        "--container-disk-in-gb",
        str(args.container_disk_gb),
        "--volume-in-gb",
        str(args.volume_gb),
        "-o",
        "json",
    ]
    if args.cloud_type:
        cmd.extend(["--cloud-type", args.cloud_type])
    if args.public_ip:
        cmd.append("--public-ip")
    if args.global_networking:
        cmd.append("--global-networking")
    return cmd


def cmd_create(args: argparse.Namespace) -> None:
    cmd = build_create_command(args)
    if args.print_only:
        print(" ".join(cmd))
        return
    payload = run_json(cmd)
    print(json.dumps(payload, indent=2))


def cmd_lifecycle(args: argparse.Namespace, verb: str) -> None:
    run_passthrough(["runpodctl", "pod", verb, args.pod_id])


def cmd_ssh_info(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "ssh", "info", args.pod_id])


def cmd_bootstrap(args: argparse.Namespace) -> None:
    script_path = str(args.script_path)
    print("Remote bootstrap commands:")
    print(f"cd /workspace/parameter-golf")
    print(f"bash {script_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod helpers for the DeepFloor small-box workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a small-box pod with DeepFloor defaults")
    create.add_argument("--template-id", default="y5cejece4j")
    create.add_argument("--gpu-id", default="NVIDIA H100 80GB HBM3")
    create.add_argument("--gpu-count", type=int, default=2)
    create.add_argument("--name")
    create.add_argument("--ports", default="22/tcp")
    create.add_argument("--cloud-type", default="COMMUNITY")
    create.add_argument("--public-ip", action=argparse.BooleanOptionalAction, default=True)
    create.add_argument("--global-networking", action=argparse.BooleanOptionalAction, default=False)
    create.add_argument("--container-disk-gb", type=int, default=50)
    create.add_argument("--volume-gb", type=int, default=50)
    create.add_argument("--print-only", action="store_true")
    create.set_defaults(handler=cmd_create)

    stop = subparsers.add_parser("stop", help="Stop a pod by id")
    stop.add_argument("pod_id")
    stop.set_defaults(handler=lambda args: cmd_lifecycle(args, "stop"))

    start = subparsers.add_parser("start", help="Start a pod by id")
    start.add_argument("pod_id")
    start.set_defaults(handler=lambda args: cmd_lifecycle(args, "start"))

    restart = subparsers.add_parser("restart", help="Restart a pod by id")
    restart.add_argument("pod_id")
    restart.set_defaults(handler=lambda args: cmd_lifecycle(args, "restart"))

    delete = subparsers.add_parser("delete", help="Delete a pod by id")
    delete.add_argument("pod_id")
    delete.set_defaults(handler=lambda args: cmd_lifecycle(args, "delete"))

    ssh_info = subparsers.add_parser("ssh-info", help="Show ssh info for a pod")
    ssh_info.add_argument("pod_id")
    ssh_info.set_defaults(handler=cmd_ssh_info)

    bootstrap = subparsers.add_parser("bootstrap", help="Print the remote DeepFloor bootstrap command")
    bootstrap.add_argument("--script-path", default=str(DEFAULT_RECORD_SCRIPT))
    bootstrap.set_defaults(handler=cmd_bootstrap)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
