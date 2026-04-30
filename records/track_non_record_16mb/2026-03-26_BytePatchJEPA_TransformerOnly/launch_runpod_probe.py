from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
WORKSPACE_ROOT = ROOT.parents[3]
REMOTE_REPO = "/workspace/parameter-golf"
REMOTE_PARENT = f"{REMOTE_REPO}/records/track_non_record_16mb"
REMOTE_DIR = f"{REMOTE_PARENT}/{ROOT.name}"
RUNPOD_API = "https://rest.runpod.io/v1/pods"
DEFAULT_GPU_TYPES = ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe", "NVIDIA H100 NVL"]
DEFAULT_IMAGE = "runpod/parameter-golf:latest"
PHASE_TO_TRAIN_SHARDS = {
    "smoke": 1,
    "backbone_screen": 10,
    "objective_screen": 10,
    "encoder_screen": 10,
    "scale": 10,
    "data_scale": 10,
    "ablate": 10,
}
PHASE_TO_VAL_DOCS = {
    "smoke": 1024,
    "backbone_screen": None,
    "objective_screen": None,
    "encoder_screen": None,
    "scale": None,
    "data_scale": None,
    "ablate": None,
}
PHASE_TO_GPU_COUNT = {
    "smoke": 1,
    "backbone_screen": 1,
    "objective_screen": 1,
    "encoder_screen": 1,
    "scale": 1,
    "data_scale": 1,
    "ablate": 1,
}


def load_runpod_api_key() -> str:
    for key in ("RUNPOD_API_KEY", "RUNPOD_TOKEN", "RUNPOD_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    env_path = WORKSPACE_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "RUNPOD_API_KEY":
                return value.strip().strip('"').strip("'")
    raise RuntimeError("RUNPOD_API_KEY not found in environment or top-level .env")


def load_public_key() -> str:
    for path in (Path.home() / ".ssh/id_ed25519.pub", Path.home() / ".ssh/id_rsa.pub"):
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    raise RuntimeError("No SSH public key found in ~/.ssh")


def api_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Runpod API {method} {url} failed: {exc.code} {detail}") from exc


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


def run_local(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def run_remote(ip: str, port: int, key_path: Path, command: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(ssh_base(ip, port, key_path) + [command], check=check, text=True)


def create_pod(
    token: str,
    public_key: str,
    gpu_types: list[str],
    image_name: str,
    cloud_type: str,
    phase: str,
    gpu_count: int,
) -> dict:
    payload = {
        "name": f"pure-jepa-{phase}-{int(time.time())}",
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "custom",
        "gpuCount": gpu_count,
        "imageName": image_name,
        "containerDiskInGb": 60,
        "volumeInGb": 40,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"SSH_PUBLIC_KEY": public_key},
    }
    return api_request("POST", RUNPOD_API, token, payload)


def poll_pod_ready(token: str, pod_id: str, timeout_seconds: int = 1800) -> dict:
    deadline = time.time() + timeout_seconds
    last = None
    while time.time() < deadline:
        pod = api_request("GET", f"{RUNPOD_API}/{pod_id}", token)
        last = pod
        if pod.get("desiredStatus") == "RUNNING" and pod.get("publicIp") and str(22) in (pod.get("portMappings") or {}):
            return pod
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not become SSH-ready: {last}")


def terminate_pod(token: str, pod_id: str) -> None:
    api_request("DELETE", f"{RUNPOD_API}/{pod_id}", token)


def run_cuda_sanity(ip: str, port: int, key_path: Path) -> None:
    sanity = """python3 - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('count', torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print('device', idx, torch.cuda.get_device_name(idx))
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
    print('norm', float(x.float().norm()))
PY"""
    run_remote(ip, port, key_path, f"bash -lc {shlex.quote(sanity)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("smoke", "backbone_screen", "objective_screen", "encoder_screen", "ablate", "scale", "data_scale"),
        default="backbone_screen",
    )
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument("--gpu-type", action="append", dest="gpu_types", help="Preferred GPU type; repeat to set fallback order")
    parser.add_argument("--image-name", default=DEFAULT_IMAGE)
    parser.add_argument("--cloud-type", default="SECURE")
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--train-shards", type=int, default=None)
    parser.add_argument("--run-env", action="append", default=[], help="Extra KEY=VALUE env passed to run_probe_pair.sh")
    args = parser.parse_args()

    token = load_runpod_api_key()
    public_key = load_public_key()
    key_path = Path.home() / ".ssh/id_ed25519"
    if not key_path.exists():
        raise RuntimeError(f"Missing private key: {key_path}")

    phase = args.phase
    gpu_types = args.gpu_types or DEFAULT_GPU_TYPES
    gpu_count = args.gpu_count if args.gpu_count is not None else PHASE_TO_GPU_COUNT[phase]
    train_shards = args.train_shards if args.train_shards is not None else PHASE_TO_TRAIN_SHARDS[phase]
    num_val_docs = PHASE_TO_VAL_DOCS[phase]

    session_path = ROOT / "runpod_session.json"
    pod = create_pod(token, public_key, gpu_types, args.image_name, args.cloud_type, phase, gpu_count)
    pod_id = pod["id"]
    start_ts = time.time()
    session = {
        "pod_id": pod_id,
        "phase": phase,
        "requested_gpu_types": gpu_types,
        "requested_image_name": args.image_name,
        "requested_cloud_type": args.cloud_type,
        "requested_gpu_count": gpu_count,
        "requested_train_shards": train_shards,
        "requested_num_val_docs": num_val_docs,
        "requested_run_env": args.run_env,
        "create_response": {
            "id": pod.get("id"),
            "name": pod.get("name"),
            "cost_per_hr": pod.get("costPerHr"),
            "image": pod.get("image"),
            "gpu_display_name": (pod.get("gpu") or {}).get("displayName"),
        },
        "started_at_unix": start_ts,
    }
    session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    try:
        ready = poll_pod_ready(token, pod_id)
        ip = ready["publicIp"]
        port = int(ready["portMappings"]["22"])
        session["ready"] = {
            "public_ip": ip,
            "ssh_port": port,
            "cost_per_hr": ready.get("costPerHr"),
            "gpu_display_name": ((ready.get("machine") or {}).get("gpuType") or {}).get("displayName"),
        }
        session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        run_cuda_sanity(ip, port, key_path)
        bootstrap = """
set -euo pipefail
cd /workspace
rm -rf parameter-golf
git clone --depth 1 https://github.com/openai/parameter-golf.git
cd /workspace/parameter-golf
if ! python3 - <<'PY'
missing = []
for name in ('huggingface_hub',):
    try:
        __import__(name)
    except Exception:
        missing.append(name)
if missing:
    raise SystemExit('MISSING:' + ' '.join(missing))
print('deps_ok')
PY
then
  python3 -m pip install --break-system-packages huggingface-hub
fi
mkdir -p records/track_non_record_16mb
"""
        run_remote(ip, port, key_path, f"bash -lc {shlex.quote(bootstrap)}")
        run_remote(ip, port, key_path, f"bash -lc {shlex.quote(f'rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR}')}")

        upload_files = [
            ROOT / "README.md",
            ROOT / "bootstrap_byte260_subset.py",
            ROOT / "launch_runpod_probe.py",
            ROOT / "run_probe_pair.sh",
            ROOT / "summarize_sweep.py",
            ROOT / "train_gpt.py",
        ]
        run_local(scp_base(ip, port, key_path) + [*(str(path) for path in upload_files), f"root@{ip}:{REMOTE_DIR}/"])

        data_bootstrap = (
            f"cd {REMOTE_REPO} && "
            f"python3 records/track_non_record_16mb/{ROOT.name}/bootstrap_byte260_subset.py "
            f"--output-dir {REMOTE_REPO}/data/datasets/fineweb10B_byte260 --train-shards {train_shards}"
        )
        if num_val_docs is not None:
            data_bootstrap += f" --num-val-docs {num_val_docs}"
        run_remote(ip, port, key_path, f"bash -lc {shlex.quote(data_bootstrap)}")

        env_items = [f"RUN_PHASE={phase}", f"BACKBONE_GPU_COUNT={gpu_count}"]
        env_items.extend(args.run_env)
        env_prefix = "env " + " ".join(shlex.quote(item) for item in env_items) + " "
        remote_run = run_remote(
            ip,
            port,
            key_path,
            f"bash -lc {shlex.quote(f'cd {REMOTE_DIR} && chmod +x run_probe_pair.sh && {env_prefix}bash run_probe_pair.sh')}",
            check=False,
        )
        session["remote_returncode"] = remote_run.returncode
        session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        remote_summary = f"{REMOTE_DIR}/results/{phase}/summary.json"
        remote_curves = f"{REMOTE_DIR}/results/{phase}/curves.tsv"
        remote_archive = f"{REMOTE_DIR}/probe_results.tgz"
        archive_cmd = (
            f"if [ -f {shlex.quote(remote_summary)} ]; then "
            f"cd {shlex.quote(REMOTE_DIR)} && "
            f"tar czf probe_results.tgz results/{shlex.quote(phase)} runpod_session.json 2>/dev/null || "
            f"tar czf probe_results.tgz results/{shlex.quote(phase)}; "
            f"fi"
        )
        run_remote(ip, port, key_path, f"bash -lc {shlex.quote(archive_cmd)}", check=False)

        archive_copy = subprocess.run(
            scp_base(ip, port, key_path) + [f"root@{ip}:{remote_archive}", str(ROOT / "probe_results.tgz")],
            check=False,
            text=True,
            capture_output=True,
        )
        synced_results = archive_copy.returncode == 0
        if synced_results:
            run_local(["tar", "xzf", str(ROOT / "probe_results.tgz"), "-C", str(ROOT)])
            (ROOT / "probe_results.tgz").unlink(missing_ok=True)
        session["synced_results"] = synced_results
        session["remote_summary_expected"] = remote_summary
        session["remote_curves_expected"] = remote_curves
        session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        if remote_run.returncode != 0 and not synced_results:
            raise RuntimeError(f"remote run failed with code {remote_run.returncode} and no results were copied back")

        end_ts = time.time()
        cost_per_hr = float(session["ready"]["cost_per_hr"])
        session["completed_at_unix"] = end_ts
        session["elapsed_hours"] = (end_ts - start_ts) / 3600.0
        session["estimated_cost_usd"] = session["elapsed_hours"] * cost_per_hr
        session_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        if not args.keep_pod:
            try:
                terminate_pod(token, pod_id)
            except Exception as exc:  # noqa: BLE001
                print(f"warning: failed to terminate pod {pod_id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
