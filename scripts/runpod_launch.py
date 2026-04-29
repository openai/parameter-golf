#!/usr/bin/env python3
"""RunPod pod launcher for Parameter Golf experiments.

Usage:
    python3 scripts/runpod_launch.py launch [--gpu-count 8] [--gpu-type "NVIDIA H100 80GB HBM3"] [--name "pgolf-eval"]
    python3 scripts/runpod_launch.py smoke-1gpu [--name "pgolf-smoke-1gpu"] [--dwell-seconds 15]

Launches a pod with the proteus-pytorch Docker image and SSH access.
Prints the pod ID and SSH command when ready.
"""
import argparse
import os
import sys
import time

# Use the correct Python that has runpod installed
try:
    import runpod
except ImportError:
    print("ERROR: runpod SDK not installed. Run: pip install runpod")
    sys.exit(1)

# Pod-side self-termination helpers (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pod_selfterm import (  # noqa: E402
    POD_HARD_DEADLINE_SECONDS,
    selfterm_env_dict,
)

API_KEY_ENV = "RUNPOD_API_KEY"
DOCKER_IMAGE = "matotezitanka/proteus-pytorch:community"
DEFAULT_GPU_TYPE = "NVIDIA H100 80GB HBM3"  # H100 SXM
DEFAULT_SMOKE_NAME = "pgolf-smoke-1gpu"
DEFAULT_SMOKE_VOLUME_SIZE = 20
DEFAULT_SMOKE_DWELL_SECONDS = 15
TERMINATE_WAIT_SECONDS = 45
TERMINATE_POLL_SECONDS = 5


def _require_api_key():
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if api_key:
        return api_key
    raise RuntimeError(f"{API_KEY_ENV} is required in the environment")


def _configure_runpod_auth():
    runpod.api_key = _require_api_key()


def _pod_present(pod_id):
    return any(p.get("id") == pod_id for p in (runpod.get_pods() or []))


def _terminate_and_wait(pod_id, timeout=TERMINATE_WAIT_SECONDS,
                        poll_interval=TERMINATE_POLL_SECONDS):
    try:
        runpod.terminate_pod(pod_id)
    except Exception as exc:
        if "not found" in str(exc).lower():
            print(f"Pod {pod_id} already absent.")
            return True
        raise
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            if not _pod_present(pod_id):
                print(f"Pod {pod_id} terminated.")
                return True
        except Exception as exc:
            last_error = exc
        time.sleep(poll_interval)
    if last_error:
        print(f"Terminate requested for {pod_id}, but verification failed before timeout: {last_error}")
    else:
        print(f"Terminate requested for {pod_id}, but it still exists after {timeout}s")
    return False


def _runtime_ready(status):
    if not status:
        return False
    runtime = status.get("runtime", {}) or {}
    return runtime.get("uptimeInSeconds", 0) > 0


def launch_pod(name="pgolf-run", gpu_count=8, gpu_type=DEFAULT_GPU_TYPE,
               volume_size=50, container_disk=20, return_pod_on_failure=False):
    """Launch a RunPod GPU pod."""
    _configure_runpod_auth()

    # Build env with self-termination vars so pod can kill itself
    api_key = _require_api_key()
    st_env = selfterm_env_dict(api_key, POD_HARD_DEADLINE_SECONDS)
    pod_env = {
        "PGOLF_DATA": "sp8192",
        "PGOLF_HARD_DEADLINE_SEC": st_env["PGOLF_HARD_DEADLINE_SEC"],
        "RUNPOD_API_KEY": st_env["RUNPOD_API_KEY"],
        "PUBLIC_KEY": open(os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go.pub")).read().strip(),
    }

    pod = runpod.create_pod(
        name=name,
        image_name=DOCKER_IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=gpu_count,
        volume_in_gb=volume_size,
        container_disk_in_gb=container_disk,
        env=pod_env,
        ports="22/tcp,8888/http",
    )

    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    print(f"Name: {name}")
    print(f"GPU: {gpu_count}x {gpu_type}")
    print(f"Image: {DOCKER_IMAGE}")
    print(f"Waiting for pod to be ready...")

    def _post_create_failure(message, cause=None):
        print(message)
        if return_pod_on_failure:
            return pod_id
        try:
            terminated = terminate_pod(pod_id)
        except Exception as cleanup_exc:
            raise RuntimeError(
                f"{message.strip()} Cleanup failed for pod {pod_id}: {cleanup_exc}"
            ) from cleanup_exc
        if not terminated:
            raise RuntimeError(
                f"{message.strip()} Termination could not be verified for pod {pod_id}"
            ) from cause
        raise RuntimeError(f"{message.strip()} Pod {pod_id} was terminated after launch failure")

    try:
        # Poll until running
        for _ in range(120):  # 10 min max wait
            status = runpod.get_pod(pod_id) or {}
            if not status:
                if not _pod_present(pod_id):
                    return _post_create_failure(
                        f"\nPod {pod_id} disappeared before reaching runtime readiness."
                    )
                time.sleep(5)
                continue
            runtime = status.get("runtime", {}) or {}
            desired = status.get("desiredStatus", "")
            if runtime and _runtime_ready(status):
                ports = runtime.get("ports", [])
                ssh_port = None
                for p in ports:
                    if p.get("privatePort") == 22:
                        ssh_port = p.get("publicPort")
                        break
                ip = runtime.get("gpus", [{}])[0].get("ip", "?")
                print(f"\nPod RUNNING!")
                print(f"  Pod ID:   {pod_id}")
                print(f"  SSH:      ssh -i ~/.runpod/ssh/RunPod-Key-Go root@{ip} -p {ssh_port}")
                if ssh_port:
                    print(f"  Quick:    runpodctl ssh --podId {pod_id}")
                return pod_id
            if desired == "EXITED":
                return _post_create_failure(f"\nPod failed to start! Status: {status}")
            time.sleep(5)
    except Exception as exc:
        return _post_create_failure(
            f"\nPod {pod_id} hit an error during readiness polling: {exc}",
            cause=exc,
        )

    return _post_create_failure("\nTimeout waiting for pod. Check RunPod dashboard.")


def smoke_1gpu(name=DEFAULT_SMOKE_NAME, gpu_type=DEFAULT_GPU_TYPE,
               volume_size=DEFAULT_SMOKE_VOLUME_SIZE,
               dwell_seconds=DEFAULT_SMOKE_DWELL_SECONDS):
    """Launch a cheap 1-GPU smoke pod, dwell briefly, and always tear it down."""
    pod_id = None
    body_error = None
    try:
        pod_id = launch_pod(
            name=name,
            gpu_count=1,
            gpu_type=gpu_type,
            volume_size=volume_size,
            return_pod_on_failure=True,
        )
        if not pod_id:
            raise RuntimeError("RunPod did not return a pod ID for the smoke test")

        status = runpod.get_pod(pod_id) or {}
        if not _runtime_ready(status):
            raise RuntimeError(f"Smoke pod {pod_id} did not reach runtime readiness")

        print(f"Smoke pod {pod_id} is ready. Dwelling for {dwell_seconds}s before teardown...")
        time.sleep(dwell_seconds)
        print(f"Smoke dwell complete for pod {pod_id}.")
        return pod_id
    except Exception as exc:
        body_error = exc
        raise
    finally:
        if pod_id:
            print(f"Tearing down smoke pod {pod_id}...")
            try:
                terminated = terminate_pod(pod_id)
            except Exception as exc:
                print(f"ERROR: failed to terminate smoke pod {pod_id}: {exc}", file=sys.stderr)
                if body_error is None:
                    raise
            else:
                if not terminated:
                    message = f"Smoke pod {pod_id} teardown could not be verified"
                    print(f"ERROR: {message}", file=sys.stderr)
                    if body_error is None:
                        raise RuntimeError(message)


def stop_pod(pod_id):
    """Stop a running pod."""
    _configure_runpod_auth()
    runpod.stop_pod(pod_id)
    print(f"Pod {pod_id} stopped.")


def terminate_pod(pod_id):
    """Terminate (delete) a pod."""
    _configure_runpod_auth()
    return _terminate_and_wait(pod_id)


def list_pods():
    """List all active pods."""
    _configure_runpod_auth()
    pods = runpod.get_pods()
    if not pods:
        print("No active pods.")
        return
    for p in pods:
        rt = p.get("runtime", {}) or {}
        uptime = rt.get("uptimeInSeconds", 0)
        gpu_count = p.get("gpuCount", "?")
        print(f"  {p['id']:20s}  {p.get('name','?'):20s}  GPUs={gpu_count}  uptime={uptime}s  status={p.get('desiredStatus','?')}")


def check_balance():
    """Check account spending."""
    _configure_runpod_auth()
    me = runpod.get_user()
    print(f"User ID: {me.get('id', '?')}")
    # The API may not expose balance directly; we check pods for cost
    pods = runpod.get_pods()
    active = [p for p in pods if p.get("desiredStatus") == "RUNNING"]
    print(f"Active pods: {len(active)}")
    total_cost_hr = sum(p.get("costPerHr", 0) for p in active)
    print(f"Current burn rate: ${total_cost_hr:.2f}/hr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RunPod pod management for Parameter Golf")
    sub = parser.add_subparsers(dest="command")

    launch = sub.add_parser("launch", help="Launch a new pod")
    launch.add_argument("--name", default="pgolf-run")
    launch.add_argument("--gpu-count", type=int, default=8)
    launch.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    launch.add_argument("--volume-size", type=int, default=50)

    smoke = sub.add_parser("smoke-1gpu", help="Launch a 1xH100 SXM smoke pod, dwell briefly, then tear it down")
    smoke.add_argument("--name", default=DEFAULT_SMOKE_NAME)
    smoke.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    smoke.add_argument("--volume-size", type=int, default=DEFAULT_SMOKE_VOLUME_SIZE)
    smoke.add_argument("--dwell-seconds", type=int, default=DEFAULT_SMOKE_DWELL_SECONDS)

    stop = sub.add_parser("stop", help="Stop a pod")
    stop.add_argument("pod_id")

    term = sub.add_parser("terminate", help="Terminate a pod")
    term.add_argument("pod_id")

    sub.add_parser("list", help="List active pods")
    sub.add_parser("balance", help="Check account balance")

    args = parser.parse_args()
    try:
        if args.command == "launch":
            launch_pod(args.name, args.gpu_count, args.gpu_type, args.volume_size)
        elif args.command == "smoke-1gpu":
            smoke_1gpu(args.name, args.gpu_type, args.volume_size, args.dwell_seconds)
        elif args.command == "stop":
            stop_pod(args.pod_id)
        elif args.command == "terminate":
            terminate_pod(args.pod_id)
        elif args.command == "list":
            list_pods()
        elif args.command == "balance":
            check_balance()
        else:
            parser.print_help()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
