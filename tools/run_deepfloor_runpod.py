#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_SCRIPT = (
    REPO_ROOT
    / "records"
    / "track_non_record_16mb"
    / "2026-04-03_DeepFloor"
    / "run_smallbox_suite.sh"
)
LEASE_STATE_ROOT = Path.home() / ".cache" / "parameter-golf" / "runpod_leases"
DEFAULT_LEASE_MINUTES = 120
WATCH_POLL_SECONDS = 30
LEASE_SCHEMA_VERSION = 1


def run_json(cmd: list[str]) -> dict[str, Any] | list[Any]:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def run_passthrough(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_local_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def epoch_now() -> int:
    return int(time.time())


def iso_utc(epoch_seconds: int) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def owner_label(explicit_owner: str | None = None) -> str:
    if explicit_owner:
        return explicit_owner
    return f"{os.getenv('USER', 'unknown')}@{socket.gethostname()}:{os.getpid()}"


def ensure_lease_state_root() -> Path:
    LEASE_STATE_ROOT.mkdir(parents=True, exist_ok=True)
    return LEASE_STATE_ROOT


def lease_state_path(pod_id: str) -> Path:
    return LEASE_STATE_ROOT / f"{pod_id}.json"


def load_lease_state(pod_id: str) -> dict[str, Any] | None:
    path = lease_state_path(pod_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_lease_state(state: dict[str, Any]) -> Path:
    ensure_lease_state_root()
    path = lease_state_path(str(state["pod_id"]))
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    return path


def clear_lease_state(pod_id: str) -> None:
    path = lease_state_path(pod_id)
    if path.exists():
        path.unlink()


def find_nested_value(payload: Any, candidate_keys: tuple[str, ...]) -> Any | None:
    if isinstance(payload, dict):
        for key in candidate_keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for value in payload.values():
            found = find_nested_value(value, candidate_keys)
            if found not in (None, ""):
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_nested_value(item, candidate_keys)
            if found not in (None, ""):
                return found
    return None


def extract_pod_id(payload: dict[str, Any] | list[Any]) -> str:
    pod_id = find_nested_value(payload, ("id", "podId", "pod_id"))
    if not isinstance(pod_id, str) or not pod_id:
        raise RuntimeError(f"Unable to determine pod id from RunPod response: {payload}")
    return pod_id


def extract_pod_name(payload: dict[str, Any] | list[Any]) -> str | None:
    pod_name = find_nested_value(payload, ("name", "podName", "pod_name"))
    return pod_name if isinstance(pod_name, str) and pod_name else None


def get_pod_details(pod_id: str) -> dict[str, Any]:
    payload = run_json(["runpodctl", "pod", "get", pod_id, "-o", "json"])
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected pod detail payload for {pod_id}: {payload}")
    return payload


def get_ssh_info(pod_id: str) -> dict[str, Any]:
    payload = run_json(["runpodctl", "ssh", "info", pod_id, "-v", "-o", "json"])
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected ssh info payload for {pod_id}: {payload}")
    if payload.get("error"):
        raise RuntimeError(f"Unable to fetch ssh info for {pod_id}: {payload['error']}")
    return payload


def get_pod_name_with_fallback(pod_id: str, fallback_payload: dict[str, Any] | list[Any] | None = None) -> str:
    try:
        details = get_pod_details(pod_id)
        return str(details.get("name") or extract_pod_name(fallback_payload or {}) or pod_id)
    except Exception:
        return str(extract_pod_name(fallback_payload or {}) or pod_id)


def new_lease_record(
    lease_minutes: int,
    owner: str,
    reason: str,
    now_epoch: int,
    lease_id: str | None = None,
) -> dict[str, Any]:
    if lease_minutes <= 0:
        raise ValueError("lease_minutes must be positive")
    return {
        "lease_id": lease_id or uuid.uuid4().hex,
        "owner": owner,
        "reason": reason,
        "created_at": now_epoch,
        "expires_at": now_epoch + (lease_minutes * 60),
    }


def append_lease_to_state(
    existing_state: dict[str, Any] | None,
    *,
    pod_id: str,
    pod_name: str | None,
    lease_record: dict[str, Any],
    watch_token: str,
    now_epoch: int,
) -> dict[str, Any]:
    leases = list(existing_state.get("leases", [])) if existing_state else []
    leases.append(lease_record)
    state = {
        "schema_version": LEASE_SCHEMA_VERSION,
        "pod_id": pod_id,
        "pod_name": pod_name or (existing_state or {}).get("pod_name") or pod_id,
        "watch_token": watch_token,
        "updated_at": now_epoch,
        "leases": leases,
    }
    if existing_state and "created_at" in existing_state:
        state["created_at"] = existing_state["created_at"]
    else:
        state["created_at"] = now_epoch
    return state


def active_leases(state: dict[str, Any], now_epoch: int) -> list[dict[str, Any]]:
    return [
        lease
        for lease in state.get("leases", [])
        if isinstance(lease, dict) and int(lease.get("expires_at", 0)) > now_epoch
    ]


def summarize_lease_state(state: dict[str, Any], now_epoch: int) -> dict[str, Any]:
    leases = [lease for lease in state.get("leases", []) if isinstance(lease, dict)]
    active = active_leases(state, now_epoch)
    next_expiry = max((int(lease["expires_at"]) for lease in active), default=None)
    return {
        "pod_id": state.get("pod_id"),
        "pod_name": state.get("pod_name"),
        "active_lease_count": len(active),
        "total_lease_count": len(leases),
        "next_expiry_epoch": next_expiry,
        "next_expiry_iso": iso_utc(next_expiry) if next_expiry is not None else None,
        "owners": sorted({str(lease.get("owner", "unknown")) for lease in active}),
    }


def format_lease_summary(state: dict[str, Any], now_epoch: int) -> str:
    summary = summarize_lease_state(state, now_epoch)
    if summary["next_expiry_iso"] is None:
        return (
            f"pod={summary['pod_id']} name={summary['pod_name']} active_leases=0 total_leases="
            f"{summary['total_lease_count']}"
        )
    return (
        f"pod={summary['pod_id']} name={summary['pod_name']} active_leases="
        f"{summary['active_lease_count']} total_leases={summary['total_lease_count']} "
        f"next_expiry={summary['next_expiry_iso']} owners={','.join(summary['owners'])}"
    )


def build_watch_command(pod_id: str, watch_token: str) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "watch", pod_id, "--watch-token", watch_token]


def spawn_watchdog(pod_id: str, watch_token: str) -> None:
    cmd = build_watch_command(pod_id, watch_token)
    subprocess.Popen(  # noqa: S603
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )


def maybe_stop_expired_pod(pod_id: str) -> None:
    try:
        run_passthrough(["runpodctl", "pod", "stop", pod_id])
    except subprocess.CalledProcessError:
        details = get_pod_details(pod_id)
        desired_status = str(details.get("desiredStatus", "")).upper()
        if desired_status != "EXITED":
            raise


def arm_lease(
    pod_id: str,
    *,
    pod_name: str | None,
    lease_minutes: int,
    owner: str,
    reason: str,
) -> dict[str, Any]:
    now_epoch = epoch_now()
    lease_record = new_lease_record(
        lease_minutes=lease_minutes,
        owner=owner,
        reason=reason,
        now_epoch=now_epoch,
    )
    existing_state = load_lease_state(pod_id)
    watch_token = uuid.uuid4().hex
    state = append_lease_to_state(
        existing_state,
        pod_id=pod_id,
        pod_name=pod_name,
        lease_record=lease_record,
        watch_token=watch_token,
        now_epoch=now_epoch,
    )
    save_lease_state(state)
    spawn_watchdog(pod_id, watch_token)
    print(
        f"armed lease {lease_record['lease_id']} for pod {pod_id} until "
        f"{iso_utc(int(lease_record['expires_at']))} ({lease_minutes}m)"
    )
    print(format_lease_summary(state, now_epoch))
    return state


def collect_lease_states() -> list[dict[str, Any]]:
    if not LEASE_STATE_ROOT.exists():
        return []
    states: list[dict[str, Any]] = []
    for path in sorted(LEASE_STATE_ROOT.glob("*.json")):
        states.append(json.loads(path.read_text()))
    return states


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
        print(f"# lease-minutes={args.lease_minutes} owner={owner_label(args.owner)}")
        return
    payload = run_json(cmd)
    pod_id = extract_pod_id(payload)
    arm_lease(
        pod_id,
        pod_name=get_pod_name_with_fallback(pod_id, fallback_payload=payload),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason=args.reason or "create",
    )
    print(json.dumps(payload, indent=2))


def cmd_start(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "pod", "start", args.pod_id])
    arm_lease(
        args.pod_id,
        pod_name=get_pod_name_with_fallback(args.pod_id),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason=args.reason or "start",
    )


def cmd_restart(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "pod", "restart", args.pod_id])
    arm_lease(
        args.pod_id,
        pod_name=get_pod_name_with_fallback(args.pod_id),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason=args.reason or "restart",
    )


def cmd_stop(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "pod", "stop", args.pod_id])
    clear_lease_state(args.pod_id)


def cmd_delete(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "pod", "delete", args.pod_id])
    clear_lease_state(args.pod_id)


def cmd_extend(args: argparse.Namespace) -> None:
    arm_lease(
        args.pod_id,
        pod_name=get_pod_name_with_fallback(args.pod_id),
        lease_minutes=args.lease_minutes,
        owner=owner_label(args.owner),
        reason=args.reason or "extend",
    )


def cmd_lease_status(args: argparse.Namespace) -> None:
    now_epoch = epoch_now()
    if args.pod_id:
        state = load_lease_state(args.pod_id)
        if state is None:
            print(f"no lease state found for pod {args.pod_id}")
            return
        print(format_lease_summary(state, now_epoch))
        return
    states = collect_lease_states()
    if not states:
        print("no tracked pod leases")
        return
    for state in states:
        print(format_lease_summary(state, now_epoch))


def cmd_ssh_info(args: argparse.Namespace) -> None:
    run_passthrough(["runpodctl", "ssh", "info", args.pod_id])


def build_rsync_command(
    *,
    pod_id: str,
    remote_path: str,
    local_path: Path,
) -> list[str]:
    ssh_info = get_ssh_info(pod_id)
    host = str(ssh_info["ip"])
    port = str(ssh_info["port"])
    ssh_key = str(ssh_info["ssh_key"]["path"])
    ensure_local_parent(local_path)
    return [
        "rsync",
        "-az",
        "--progress",
        "-e",
        (
            "ssh -o StrictHostKeyChecking=no "
            "-o UserKnownHostsFile=/dev/null "
            f"-i {ssh_key} -p {port}"
        ),
        f"root@{host}:{remote_path}",
        str(local_path),
    ]


def cmd_sync_path(args: argparse.Namespace) -> None:
    local_path = Path(args.local_path).expanduser().resolve()
    if args.remote_path.endswith("/"):
        local_path.mkdir(parents=True, exist_ok=True)
    else:
        ensure_local_parent(local_path)
    cmd = build_rsync_command(
        pod_id=args.pod_id,
        remote_path=args.remote_path,
        local_path=local_path,
    )
    run_passthrough(cmd)


def cmd_harvest_stop(args: argparse.Namespace) -> None:
    sync_args = argparse.Namespace(
        pod_id=args.pod_id,
        remote_path=args.remote_path,
        local_path=args.local_path,
    )
    cmd_sync_path(sync_args)
    run_passthrough(["runpodctl", "pod", "stop", args.pod_id])
    clear_lease_state(args.pod_id)


def cmd_bootstrap(args: argparse.Namespace) -> None:
    script_path = str(args.script_path)
    print("Remote bootstrap commands:")
    print("cd /workspace/parameter-golf")
    print(f"bash {script_path}")


def cmd_watch(args: argparse.Namespace) -> None:
    while True:
        state = load_lease_state(args.pod_id)
        if state is None:
            return
        if state.get("watch_token") != args.watch_token:
            return
        now_epoch = epoch_now()
        active = active_leases(state, now_epoch)
        if not active:
            maybe_stop_expired_pod(args.pod_id)
            clear_lease_state(args.pod_id)
            return
        next_expiry = max(int(lease["expires_at"]) for lease in active)
        sleep_seconds = max(1, min(WATCH_POLL_SECONDS, next_expiry - now_epoch))
        time.sleep(sleep_seconds)


def add_common_lease_arguments(parser: argparse.ArgumentParser, *, default_minutes: int) -> None:
    parser.add_argument(
        "--lease-minutes",
        type=int,
        default=default_minutes,
        help=f"Lease window in minutes before auto-stop (default: {default_minutes})",
    )
    parser.add_argument("--owner", help="Optional owner label for the lease record")
    parser.add_argument("--reason", help="Optional reason for the lease record")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lease-aware RunPod helpers for the DeepFloor small-box workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a small-box pod and arm an auto-stop lease")
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
    add_common_lease_arguments(create, default_minutes=DEFAULT_LEASE_MINUTES)
    create.set_defaults(handler=cmd_create)

    start = subparsers.add_parser("start", help="Start a stopped pod and arm an auto-stop lease")
    start.add_argument("pod_id")
    add_common_lease_arguments(start, default_minutes=DEFAULT_LEASE_MINUTES)
    start.set_defaults(handler=cmd_start)

    restart = subparsers.add_parser("restart", help="Restart a pod and arm an auto-stop lease")
    restart.add_argument("pod_id")
    add_common_lease_arguments(restart, default_minutes=DEFAULT_LEASE_MINUTES)
    restart.set_defaults(handler=cmd_restart)

    stop = subparsers.add_parser("stop", help="Stop a pod by id and clear any tracked lease")
    stop.add_argument("pod_id")
    stop.set_defaults(handler=cmd_stop)

    delete = subparsers.add_parser("delete", help="Delete a pod by id and clear any tracked lease")
    delete.add_argument("pod_id")
    delete.set_defaults(handler=cmd_delete)

    extend = subparsers.add_parser("extend", help="Add more lease time to a running pod without restarting it")
    extend.add_argument("pod_id")
    add_common_lease_arguments(extend, default_minutes=60)
    extend.set_defaults(handler=cmd_extend)

    lease_status = subparsers.add_parser("lease-status", help="Show lease status for one pod or all tracked pods")
    lease_status.add_argument("pod_id", nargs="?")
    lease_status.set_defaults(handler=cmd_lease_status)

    ssh_info = subparsers.add_parser("ssh-info", help="Show ssh info for a pod")
    ssh_info.add_argument("pod_id")
    ssh_info.set_defaults(handler=cmd_ssh_info)

    sync_path = subparsers.add_parser("sync-path", help="Sync a remote pod path to a local path over rsync")
    sync_path.add_argument("pod_id")
    sync_path.add_argument("--remote-path", required=True)
    sync_path.add_argument("--local-path", required=True)
    sync_path.set_defaults(handler=cmd_sync_path)

    harvest_stop = subparsers.add_parser(
        "harvest-stop",
        help="Sync a remote pod path locally and only then stop the pod",
    )
    harvest_stop.add_argument("pod_id")
    harvest_stop.add_argument("--remote-path", required=True)
    harvest_stop.add_argument("--local-path", required=True)
    harvest_stop.set_defaults(handler=cmd_harvest_stop)

    bootstrap = subparsers.add_parser("bootstrap", help="Print the remote DeepFloor bootstrap command")
    bootstrap.add_argument("--script-path", default=str(DEFAULT_RECORD_SCRIPT))
    bootstrap.set_defaults(handler=cmd_bootstrap)

    watch = subparsers.add_parser("watch", help=argparse.SUPPRESS)
    watch.add_argument("pod_id")
    watch.add_argument("--watch-token", required=True)
    watch.set_defaults(handler=cmd_watch)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
