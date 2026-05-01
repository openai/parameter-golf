#!/usr/bin/env python3
"""Run bounded RunPod HTTP-bootstrap jobs (1–8 GPU) without Jupyter writes or SSH.

This script packages a small local bundle into a base64 env var, launches a pod
(1–8 GPUs) with dockerArgs that reconstruct the bundle and run a user command on
boot, then retrieves artifacts back to this HPC through a simple HTTP server
running inside the pod on port 30000.

Examples:
    # 1-GPU retrieval rehearsal (original default behavior)
    python scripts/runpod_http_rehearsal.py --max-minutes 8

    # 8-GPU production train run
    python scripts/runpod_http_rehearsal.py --gpus 8 --max-minutes 25 \\
        --cmd 'cd /root/rehearsal_src && pip install -r requirements.txt && ...'
"""

import argparse
import base64
import datetime
import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request

from pathlib import Path

import pod_selfterm
from runpod_safe import (
    UA, RUNTIME_WAIT_SECONDS, _make_ssl_ctx, _ssh_upload, balance, create_pod,
    get_pods, terminate_and_wait, wait_runtime, GPU_SKU_TABLE,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FILES_TO_BUNDLE = [
    Path("train_gpt.py"),
    Path("data/cached_challenge_fineweb.py"),
    Path("data/tokenizer_specs.json"),
    Path("requirements.txt"),
]
LAUNCHER_STATE_FILENAME = "launcher_state.json"
HTTP_TERMINAL_STATUSES = ("DONE", "FAIL", "TIMEOUT")
HTTP_STARTUP_READINESS_STATUSES = HTTP_TERMINAL_STATUSES + ("RUNNING", "AWAITING_SSH")
EARLY_HTTP_READINESS_TIMEOUT_SECONDS = 60


def _utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _message_metadata(message):
    text = "" if message is None else str(message)
    return {
        "message_length": len(text),
        "message_sha256": _sha256_text(text),
    }


def build_launcher_state(
    launcher,
    pod_id,
    pod_name,
    gpus,
    max_minutes,
    results_dir,
    hard_deadline_sec=None,
    bundle_b64=None,
    command=None,
    docker_args=None,
    docker_image=None,
    runtime_timeout_sec=None,
):
    """Build non-secret local launch metadata for durable cleanup/recovery."""
    now = _utc_now()
    state = {
        "schema_version": 1,
        "launcher": launcher,
        "created_at_utc": now,
        "updated_at_utc": now,
        "phase": "pod_created",
        "pod_id": pod_id,
        "pod_name": pod_name,
        "gpus": gpus,
        "max_minutes": max_minutes,
        "results_dir": str(results_dir),
        "launcher_pid": os.getpid(),
        "hard_deadline_sec": hard_deadline_sec,
        "runtime_timeout_sec": runtime_timeout_sec,
        "docker_image": docker_image,
        "cleanup_attempted": False,
        "cleanup_status": "not_started",
    }
    if bundle_b64 is not None:
        state["bundle_b64_length"] = len(bundle_b64)
        state["bundle_b64_sha256"] = _sha256_text(bundle_b64)
    if command is not None:
        state["command_length"] = len(command)
        state["command_sha256"] = _sha256_text(command)
    if docker_args is not None:
        state["docker_args_length"] = len(docker_args)
        state["docker_args_sha256"] = _sha256_text(docker_args)
    return state


def write_launcher_state(results_dir, state):
    """Atomically write launcher_state.json with stable keys.

    The state intentionally stores only metadata and hashes/lengths for large
    or sensitive payloads.  Raw environment, GraphQL headers, API keys, and the
    base64 bundle must never be placed in this dictionary.
    """
    path = Path(results_dir) / LAUNCHER_STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    state_to_write = dict(state)
    state_to_write["updated_at_utc"] = _utc_now()
    fd, tmp_name = tempfile.mkstemp(
        prefix=".launcher_state.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state_to_write, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, str(path))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    state.clear()
    state.update(state_to_write)
    return path


def record_launcher_exception(results_dir, state, exc):
    state["phase"] = "exception"
    state["last_exception_type"] = exc.__class__.__name__
    metadata = _message_metadata(exc)
    state["last_exception_message_length"] = metadata["message_length"]
    state["last_exception_message_sha256"] = metadata["message_sha256"]
    write_launcher_state(results_dir, state)


def _write_launcher_state_best_effort(results_dir, state, pod_id, phase):
    try:
        write_launcher_state(results_dir, state)
    except BaseException as exc:
        print(
            "WARNING: failed to write launcher state for pod {} during {}: {}".format(
                pod_id, phase, exc.__class__.__name__
            ),
            file=sys.stderr,
        )
        return exc
    return None


def _reraise_control_flow_bookkeeping_exception(exc, original_exc):
    if original_exc is None and isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc


def terminate_pod_with_launcher_state(results_dir, state, pod_id, terminate_func, original_exc=None):
    """Terminate a pod while recording cleanup status without masking original exceptions."""
    cleanup_reason = original_exc.__class__.__name__ if original_exc is not None else "normal_exit"
    state["phase"] = "cleanup_started"
    state["cleanup_attempted"] = True
    state["cleanup_status"] = "started"
    state["cleanup_reason"] = cleanup_reason
    state["cleanup_started_at_utc"] = _utc_now()
    cleanup_started_exc = _write_launcher_state_best_effort(
        results_dir, state, pod_id, "cleanup_started"
    )
    try:
        terminate_func(pod_id)
    except BaseException as cleanup_exc:
        state["phase"] = "cleanup_failed"
        state["cleanup_status"] = "failed"
        state["cleanup_finished_at_utc"] = _utc_now()
        state["cleanup_exception_type"] = cleanup_exc.__class__.__name__
        metadata = _message_metadata(cleanup_exc)
        state["cleanup_exception_message_length"] = metadata["message_length"]
        state["cleanup_exception_message_sha256"] = metadata["message_sha256"]
        _write_launcher_state_best_effort(results_dir, state, pod_id, "cleanup_failed")
        if original_exc is None:
            raise
        print(
            "WARNING: cleanup failed for pod {} after {}: {}".format(
                pod_id, cleanup_reason, cleanup_exc.__class__.__name__
            ),
            file=sys.stderr,
        )
    else:
        state["phase"] = "cleanup_completed"
        state["cleanup_status"] = "succeeded"
        state["cleanup_finished_at_utc"] = _utc_now()
        cleanup_completed_exc = _write_launcher_state_best_effort(
            results_dir, state, pod_id, "cleanup_completed"
        )
        _reraise_control_flow_bookkeeping_exception(cleanup_started_exc, original_exc)
        _reraise_control_flow_bookkeeping_exception(cleanup_completed_exc, original_exc)


def build_bundle_b64(train_script=None, extra_files=None):
    """Build base64-encoded tar.gz bundle of files to upload to the pod.

    Args:
        train_script: Override path for train_gpt.py (e.g. a record's version).
        extra_files: List of (local_path, arcname) tuples for additional files.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel_path in FILES_TO_BUNDLE:
            if rel_path.name == "train_gpt.py" and train_script:
                tf.add(str(train_script), arcname="train_gpt.py")
            else:
                tf.add(str(REPO_ROOT / rel_path), arcname=rel_path.name)
        if extra_files:
            for local_path, arcname in extra_files:
                tf.add(str(local_path), arcname=arcname)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_boot_command(user_cmd):
    shell = r"""set -uo pipefail
__PGOLF_SELFTERM_PREAMBLE__
mkdir -p /root/rehearsal_src /root/rehearsal_out
printf 'RUNNING\n' > /root/rehearsal_out/status.txt
python3 - <<'HTTPSERVER' > /root/rehearsal_out/http_server.log 2>&1 &
import http.server, os, threading
class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory='/root/rehearsal_out', **kw)
    def do_POST(self):
        path = self.path.lstrip('/')
        if path.startswith('upload/'):
            dest = '/root/rehearsal_src/' + path[7:]
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            length = int(self.headers.get('Content-Length', 0))
            with open(dest, 'wb') as f:
                remaining = length
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 1048576))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK\n')
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, fmt, *args):
        if 'upload' in (args[0] if args else ''):
            super().log_message(fmt, *args)
http.server.HTTPServer(('', 30000), H).serve_forever()
HTTPSERVER
PGOLF_HTTP_PID=$!
extract_ec=0
python3 - <<'PY' > /root/rehearsal_out/pgolf_extract_stdout.txt 2>&1 || extract_ec=$?
import base64
import io
import os
import tarfile
n_parts = int(os.environ.get('PGOLF_BUNDLE_PARTS', '0'))
if n_parts > 0:
    chunks = []
    for i in range(n_parts):
        key = 'PGOLF_BUNDLE_PART_{:03d}'.format(i)
        if key not in os.environ:
            raise RuntimeError('missing bundle env var: ' + key)
        chunks.append(os.environ[key])
    b64 = ''.join(chunks)
else:
    b64 = os.environ['PGOLF_BUNDLE_B64']
payload = base64.b64decode(b64)
print('bundle bytes:', len(payload), 'parts:', n_parts)
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as tf:
    tf.extractall('/root/rehearsal_src')
PY
if [ "${extract_ec}" -ne 0 ]; then
    {
        printf 'ERROR: bundle extraction failed with exit_code=%s\n' "${extract_ec}"
        cat /root/rehearsal_out/pgolf_extract_stdout.txt 2>/dev/null || true
    } > /root/rehearsal_out/pgolf_stdout.txt
    printf '%s\n' "${extract_ec}" > /root/rehearsal_out/pgolf_exit_code.txt
    printf '%s\n' "${extract_ec}" > /root/rehearsal_out/overall_exit_code.txt
    printf 'FAIL\n' > /root/rehearsal_out/status.txt
    wait "${PGOLF_HTTP_PID}"
    exit "${extract_ec}"
fi
# Wait for upload sentinel if launcher signaled it will upload large files.
if [ "${PGOLF_AWAIT_SSH_UPLOAD:-0}" = "1" ] || [ "${PGOLF_AWAIT_HTTP_UPLOAD:-0}" = "1" ]; then
    printf 'AWAITING_UPLOAD\n' > /root/rehearsal_out/status.txt
    upload_wait_seconds="${PGOLF_UPLOAD_WAIT_SECONDS:-600}"
    ssh_wait_deadline=$(( $(date +%s) + upload_wait_seconds ))
    while [ ! -f /root/rehearsal_src/.ssh_upload_complete ] && [ ! -f /root/rehearsal_src/.http_upload_complete ]; do
        if [ "$(date +%s)" -ge "${ssh_wait_deadline}" ]; then
            printf 'TIMEOUT waiting for upload sentinel\n' >> /root/rehearsal_out/pgolf_stdout.txt
            printf '124\n' > /root/rehearsal_out/pgolf_exit_code.txt
            printf 'FAIL\n' > /root/rehearsal_out/status.txt
            wait "${PGOLF_HTTP_PID}"
            exit 124
        fi
        sleep 2
    done
    printf 'RUNNING\n' > /root/rehearsal_out/status.txt
fi
set +e
setsid bash -lc __PGOLF_USER_CMD__ > /root/rehearsal_out/pgolf_stdout.txt 2>&1 &
pgolf_cmd_pid=$!
pgolf_timed_out=0
timeout_sec=0
if [ -n "${PGOLF_MAX_MINUTES:-}" ]; then
    timeout_sec=$((PGOLF_MAX_MINUTES * 60))
fi
if [ "${timeout_sec}" -gt 0 ]; then
    pgolf_deadline=$(( $(date +%s) + timeout_sec ))
    while kill -0 "${pgolf_cmd_pid}" 2>/dev/null; do
        if [ "$(date +%s)" -ge "${pgolf_deadline}" ]; then
            pgolf_timed_out=1
            printf '\nTIMEOUT: user command exceeded PGOLF_MAX_MINUTES=%s\n' "${PGOLF_MAX_MINUTES}" >> /root/rehearsal_out/pgolf_stdout.txt
            kill -TERM "-${pgolf_cmd_pid}" 2>/dev/null || kill -TERM "${pgolf_cmd_pid}" 2>/dev/null || true
            for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
                kill -0 "${pgolf_cmd_pid}" 2>/dev/null || break
                sleep 5
            done
            kill -0 "${pgolf_cmd_pid}" 2>/dev/null && { kill -KILL "-${pgolf_cmd_pid}" 2>/dev/null || kill -KILL "${pgolf_cmd_pid}" 2>/dev/null || true; }
            break
        fi
        sleep 5
    done
fi
wait "${pgolf_cmd_pid}"
ec=$?
if [ "${pgolf_timed_out}" -eq 1 ]; then
    ec=124
fi
printf '%s\n' "$ec" > /root/rehearsal_out/pgolf_exit_code.txt
printf '%s\n' "$ec" > /root/rehearsal_out/overall_exit_code.txt
if [ "${pgolf_timed_out}" -eq 1 ]; then
    printf 'TIMEOUT\n' > /root/rehearsal_out/status.txt
elif [ "$ec" -eq 0 ]; then
  printf 'DONE\n' > /root/rehearsal_out/status.txt
else
  printf 'FAIL\n' > /root/rehearsal_out/status.txt
fi
wait "${PGOLF_HTTP_PID}"
exit ${ec}"""
    shell = shell.replace("__PGOLF_SELFTERM_PREAMBLE__", pod_selfterm.selfterm_bash_preamble().strip())
    shell = shell.replace("__PGOLF_USER_CMD__", shlex.quote(user_cmd))
    return "bash -lc {}".format(shlex.quote(shell))


def wait_http_proxy(pod_id, port, timeout=180, startup_readiness=False):
    url = "https://{pod}-{port}.proxy.runpod.net/status.txt".format(pod=pod_id, port=port)
    deadline = time.time() + timeout
    accepted_statuses = HTTP_STARTUP_READINESS_STATUSES if startup_readiness else HTTP_TERMINAL_STATUSES
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", UA)
            with urllib.request.urlopen(req, timeout=15, context=_make_ssl_ctx()) as r:
                body = r.read().decode("utf-8", errors="replace").strip()
            if body in accepted_statuses:
                return body
        except Exception:
            pass
        time.sleep(5)
    mode = "startup/readiness" if startup_readiness else "terminal"
    raise RuntimeError(
        "HTTP rehearsal endpoint did not become ready within {}s ({})".format(timeout, mode)
    )


def download_file(pod_id, port, name, out_dir, optional=False, local_name=None):
    url = "https://{pod}-{port}.proxy.runpod.net/{name}".format(pod=pod_id, port=port, name=name)
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    # Retry briefly on proxy/filesystem races after the on-pod payload writes a
    # file but before the proxy consistently serves it.
    attempts = 6
    backoff = 3
    last_err = None
    transient_http_codes = {404, 408, 425, 429, 500, 502, 503, 504}
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120, context=_make_ssl_ctx()) as r:
                data = r.read()
            out_path = out_dir / (local_name or name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
            return out_path
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in transient_http_codes and i < attempts - 1:
                time.sleep(backoff)
                continue
            if optional:
                return None
            raise
        except urllib.error.URLError as e:
            last_err = e
            if i < attempts - 1:
                time.sleep(backoff)
                continue
            if optional:
                return None
            raise
    if last_err is not None and optional:
        return None
    raise last_err if last_err is not None else RuntimeError("download_file unknown error")


def wait_startup_readiness_and_maybe_download_status(
    pod_id,
    port,
    out_dir,
    timeout=EARLY_HTTP_READINESS_TIMEOUT_SECONDS,
    wait_func=None,
    download_func=None,
):
    """Best-effort early proof that the artifact HTTP server is serving status.txt.

    This intentionally never raises: launchers still rely on the later terminal
    wait for DONE/FAIL/TIMEOUT, while this helper captures lightweight startup
    evidence when the server exposes the initial RUNNING state early.
    """
    wait_func = wait_func or wait_http_proxy
    download_func = download_func or download_file
    try:
        status = wait_func(pod_id, port, timeout=timeout, startup_readiness=True)
    except BaseException as exc:
        print(
            "Early HTTP readiness not observed within {}s: {}".format(
                timeout, exc.__class__.__name__
            ),
            file=sys.stderr,
        )
        return None
    print("HTTP endpoint early status: {}".format(status))
    if status == "RUNNING":
        try:
            path = download_func(
                pod_id,
                port,
                "status.txt",
                out_dir,
                optional=True,
                local_name="early_status.txt",
            )
            if path:
                print("  {} ({})".format(path.name, path.stat().st_size))
        except BaseException as exc:
            print(
                "WARNING: early status.txt download failed for pod {}: {}".format(
                    pod_id, exc.__class__.__name__
                ),
                file=sys.stderr,
            )
    return status


H100_COST_PER_GPU_HR = 2.99

SKU_NOMINAL_COST_PER_HR = {
    "a100-1x": 1.89,
    "a100-2x": 3.78,
    "h100-1x": 2.99,
}


def main():
    parser = argparse.ArgumentParser(description="Run bounded RunPod HTTP-bootstrap jobs (1–8 GPU).")
    parser.add_argument("--gpus", type=int, default=None, choices=[1, 2, 4, 8],
                        help="Number of GPUs to request (default: 1, or inferred from --gpu-sku)")
    parser.add_argument("--gpu-sku", default=None, choices=list(GPU_SKU_TABLE.keys()),
                        help="GPU SKU selector (e.g. a100-1x, h100-1x). Sets gpu_type_id and validates --gpus.")
    parser.add_argument("--pod-name", default=None,
                        help="RunPod pod display name (default: pgolf-http-<gpus>gpu)")
    parser.add_argument("--max-minutes", type=int, default=8)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--cmd", default="nvidia-smi > /root/rehearsal_out/nvidia_smi.txt; python3 --version > /root/rehearsal_out/python_version.txt; sha256sum /root/rehearsal_src/train_gpt.py /root/rehearsal_src/cached_challenge_fineweb.py /root/rehearsal_src/tokenizer_specs.json /root/rehearsal_src/requirements.txt > /root/rehearsal_out/upload_manifest.txt; wc -c /root/rehearsal_src/train_gpt.py /root/rehearsal_src/cached_challenge_fineweb.py /root/rehearsal_src/tokenizer_specs.json /root/rehearsal_src/requirements.txt > /root/rehearsal_out/upload_sizes.txt")
    parser.add_argument("--download", nargs="*", default=["status.txt", "pgolf_exit_code.txt", "overall_exit_code.txt", "pgolf_stdout.txt", "nvidia_smi.txt", "python_version.txt", "upload_manifest.txt", "upload_sizes.txt"])
    parser.add_argument("--train-script", default=None,
                        help="Override train_gpt.py with a different script path (e.g. a record's version)")
    parser.add_argument("--extra-file", action="append", default=[],
                        help="Extra file to bundle (env var). Format: 'local_path' (uses basename) or 'local_path:arcname'. Repeatable.")
    parser.add_argument("--ssh-upload", action="append", default=[],
                        help="Large file to upload via SSH after pod boot (avoids GraphQL env-var size limit). Format: 'local_path' (uses basename) or 'local_path:arcname'. Lands at /root/rehearsal_src/<arcname>. Repeatable.")
    parser.add_argument("--docker-image", default=None,
                        help="Docker image override (default: PGOLF_DOCKER_IMAGE env or base community image)")
    parser.add_argument("--runtime-timeout-sec", type=int, default=RUNTIME_WAIT_SECONDS,
                        help="Seconds to wait for RunPod runtime startup (default: {})".format(RUNTIME_WAIT_SECONDS))
    args = parser.parse_args()

    # Resolve GPU count and type from --gpu-sku / --gpus.
    effective_gpu_type_id = None
    if args.gpu_sku is not None:
        sku_info = GPU_SKU_TABLE[args.gpu_sku]
        effective_gpu_type_id = sku_info["gpu_type_id"]
        sku_gpu_count = sku_info["gpu_count"]
        if args.gpus is not None and args.gpus != sku_gpu_count:
            raise SystemExit(
                "ERROR: --gpus {} conflicts with --gpu-sku {} (expected {} GPUs)".format(
                    args.gpus, args.gpu_sku, sku_gpu_count
                )
            )
        effective_gpus = sku_gpu_count
    else:
        effective_gpus = args.gpus if args.gpus is not None else 1

    # SKU-aware nominal cost estimate (authoritative costPerHr comes from API post-creation).
    if args.gpu_sku is not None:
        nominal_cost_per_hr = SKU_NOMINAL_COST_PER_HR.get(args.gpu_sku, H100_COST_PER_GPU_HR)
        cost_est = nominal_cost_per_hr * args.max_minutes / 60.0
    else:
        cost_est = effective_gpus * H100_COST_PER_GPU_HR * args.max_minutes / 60.0

    pod_name = args.pod_name or "pgolf-http-{n}gpu".format(n=effective_gpus)

    pod_id = None
    out_dir = None
    launcher_state = None
    original_exc = None
    try:
        bal, _ = balance()
        print("Balance: ${:.2f}  Est cost: ${:.2f}  ({} GPU(s), {} min)".format(
            bal, cost_est, effective_gpus, args.max_minutes))
        balance_mult = float(os.environ.get("PGOLF_BALANCE_MULT", "2"))
        if bal < cost_est * balance_mult:
            raise SystemExit("ERROR: Insufficient balance (need >= {:.1f}× est cost = ${:.2f})".format(
                balance_mult, cost_est * balance_mult))

        train_script = Path(args.train_script) if args.train_script else None
        extra_files = []
        for spec in args.extra_file:
            if ":" in spec:
                lp, arc = spec.split(":", 1)
            else:
                lp = spec
                arc = Path(spec).name
            lp_path = Path(lp)
            if not lp_path.exists():
                raise SystemExit("ERROR: --extra-file path does not exist: {}".format(lp))
            extra_files.append((lp_path, arc))
        # Parse --ssh-upload specs (large files delivered post-boot via SSH).
        ssh_uploads = []
        for spec in args.ssh_upload:
            if ":" in spec:
                lp, arc = spec.split(":", 1)
            else:
                lp = spec
                arc = Path(spec).name
            lp_path = Path(lp)
            if not lp_path.exists():
                raise SystemExit("ERROR: --ssh-upload path does not exist: {}".format(lp))
            ssh_uploads.append((lp_path, arc))
        bundle_b64 = build_bundle_b64(train_script=train_script, extra_files=extra_files)
        # Chunk bundle into 256KB pieces to keep individual env vars and total
        # GraphQL request size under RunPod limits. Single env var > ~1MB tends
        # to cause HTTP 413; total request > a few MB also rejected.
        CHUNK_SIZE = 32 * 1024
        chunk_env = {"PGOLF_MAX_MINUTES": str(args.max_minutes)}
        if ssh_uploads:
            chunk_env["PGOLF_AWAIT_HTTP_UPLOAD"] = "1"
            chunk_env["PGOLF_AWAIT_SSH_UPLOAD"] = "1"
            chunk_env["PGOLF_UPLOAD_WAIT_SECONDS"] = os.environ.get(
                "PGOLF_UPLOAD_WAIT_SECONDS", "1800"
            )
            chunk_env["PGOLF_SSH_WAIT_ATTEMPTS"] = os.environ.get(
                "PGOLF_SSH_WAIT_ATTEMPTS", "120"
            )
        if len(bundle_b64) <= CHUNK_SIZE:
            chunk_env["PGOLF_BUNDLE_B64"] = bundle_b64
            chunk_env["PGOLF_BUNDLE_PARTS"] = "0"
        else:
            n_parts = (len(bundle_b64) + CHUNK_SIZE - 1) // CHUNK_SIZE
            chunk_env["PGOLF_BUNDLE_PARTS"] = str(n_parts)
            for i in range(n_parts):
                chunk_env["PGOLF_BUNDLE_PART_{:03d}".format(i)] = bundle_b64[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
            print("Bundle chunked: {} bytes -> {} parts of {} bytes".format(
                len(bundle_b64), n_parts, CHUNK_SIZE))
        docker_args = build_boot_command(args.cmd)
        hard_deadline_sec = args.max_minutes * 60 + 120
        pod = create_pod(
            name=pod_name,
            gpus=effective_gpus,
            max_minutes=args.max_minutes,
            docker_args=docker_args,
            extra_env=chunk_env,
            ports="30000/http,22/tcp",
            start_ssh=bool(ssh_uploads),
            deadline_sec=hard_deadline_sec,
            image=args.docker_image,
            gpu_type_id=effective_gpu_type_id,
        )
        pod_id = pod["id"]
        out_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results" / ("pod_{pod}_http".format(pod=pod_id))
        launcher_state = build_launcher_state(
            launcher="runpod_http_rehearsal",
            pod_id=pod_id,
            pod_name=pod_name,
            gpus=effective_gpus,
            max_minutes=args.max_minutes,
            results_dir=out_dir,
            hard_deadline_sec=hard_deadline_sec,
            bundle_b64=bundle_b64,
            command=args.cmd,
            docker_args=docker_args,
            docker_image=args.docker_image,
            runtime_timeout_sec=args.runtime_timeout_sec,
        )
        launcher_state["cost_per_hr"] = pod.get("costPerHr")
        launcher_state["gpu_sku"] = args.gpu_sku
        launcher_state["gpu_type_id"] = effective_gpu_type_id
        write_launcher_state(out_dir, launcher_state)
        print("Pod: {}  ${}/hr  name={}".format(pod_id, pod.get("costPerHr", "?"), pod_name))
        rt = wait_runtime(pod_id, timeout=args.runtime_timeout_sec)
        print("Pod RUNNING (uptime={}s)".format(rt["uptimeInSeconds"]))
        if ssh_uploads:
            # Try HTTP upload first (works through RunPod proxy without SSH).
            # Wait for the HTTP server to be reachable on the pod.
            proxy_base = "https://{}-30000.proxy.runpod.net".format(pod_id)
            http_upload_ok = False
            print("Waiting for HTTP upload endpoint...")
            for attempt in range(60):
                try:
                    req = urllib.request.Request(proxy_base + "/status.txt")
                    req.add_header("User-Agent", UA)
                    with urllib.request.urlopen(req, timeout=10, context=_make_ssl_ctx()) as r:
                        body = r.read().decode().strip()
                    if body:
                        print("HTTP endpoint ready (status={}, attempt={})".format(body, attempt))
                        http_upload_ok = True
                        break
                except Exception:
                    pass
                time.sleep(5)
            if http_upload_ok:
                try:
                    print("Uploading {} large file(s) via HTTP proxy...".format(len(ssh_uploads)))
                    for lp_path, arc in ssh_uploads:
                        size = lp_path.stat().st_size
                        print("  {} -> /root/rehearsal_src/{} ({:.1f} MB)".format(
                            lp_path.name, arc, size / 1048576))
                        upload_url = proxy_base + "/upload/" + arc
                        with open(str(lp_path), 'rb') as f:
                            file_data = f.read()
                        req = urllib.request.Request(upload_url, data=file_data, method='POST')
                        req.add_header("User-Agent", UA)
                        req.add_header("Content-Type", "application/octet-stream")
                        req.add_header("Content-Length", str(len(file_data)))
                        with urllib.request.urlopen(req, timeout=600, context=_make_ssl_ctx()) as r:
                            resp = r.read().decode().strip()
                        print("    uploaded ({})".format(resp))
                    # Drop HTTP sentinel
                    sentinel_url = proxy_base + "/upload/.http_upload_complete"
                    req = urllib.request.Request(sentinel_url, data=b'done', method='POST')
                    req.add_header("User-Agent", UA)
                    req.add_header("Content-Length", "4")
                    with urllib.request.urlopen(req, timeout=30, context=_make_ssl_ctx()) as r:
                        r.read()
                    print("HTTP upload complete; sentinel dropped.")
                except Exception as exc:
                    http_upload_ok = False
                    print("HTTP upload failed ({}: {}); falling back to SSH...".format(
                        type(exc).__name__, exc))
            if not http_upload_ok:
                # Fall back to SSH if HTTP upload endpoint not reachable.
                # Re-fetch runtime to ensure SSH port info is populated.
                for _ in range(30):
                    pods_now = [p for p in get_pods() if p["id"] == pod_id]
                    if pods_now and pods_now[0].get("runtime"):
                        rt = pods_now[0]["runtime"]
                        has_ssh = any(p.get("privatePort") == 22 and p.get("publicPort") for p in rt.get("ports", []))
                        if has_ssh:
                            break
                    time.sleep(5)
                else:
                    raise RuntimeError("Neither HTTP nor SSH upload available")
                from runpod_safe import _ssh_run
                ssh_ready = False
                ssh_max_attempts = int(os.environ.get("PGOLF_SSH_WAIT_ATTEMPTS", "60"))
                for attempt in range(ssh_max_attempts):
                    try:
                        _ssh_run(rt, "true", timeout=15)
                        ssh_ready = True
                        print("SSH ready after {}s".format(attempt * 5))
                        break
                    except Exception as exc:
                        if attempt == 0:
                            print("Waiting for sshd ({})...".format(type(exc).__name__))
                        time.sleep(5)
                if not ssh_ready:
                    raise RuntimeError("SSH not reachable after {}s; cannot upload large files".format(ssh_max_attempts * 5))
                print("Uploading {} large file(s) via SSH...".format(len(ssh_uploads)))
                for lp_path, arc in ssh_uploads:
                    size = lp_path.stat().st_size
                    print("  {} -> /root/rehearsal_src/{} ({} bytes)".format(lp_path, arc, size))
                    _ssh_upload(rt, str(lp_path), "rehearsal_src/{}".format(arc))
                _ssh_run(rt, "touch /root/rehearsal_src/.ssh_upload_complete", timeout=30)
                print("SSH upload complete; sentinel dropped.")
        wait_startup_readiness_and_maybe_download_status(pod_id, 30000, out_dir)
        wait_http_proxy(pod_id, 30000, timeout=max(180, args.max_minutes * 60 + 60))
        print("HTTP rehearsal endpoint ready")
        for name in args.download:
            optional = name == "early_status.txt" or name.endswith((".ptz", ".pt", "_log.txt", "_exit.txt", ".npz", ".json"))
            path = download_file(pod_id, 30000, name, out_dir, optional=optional)
            if path:
                print("  {} ({})".format(path.name, path.stat().st_size))
            else:
                print("  {} (not found, skipped)".format(name))
    except BaseException as exc:
        original_exc = exc
        if pod_id is not None and out_dir is not None and launcher_state is not None:
            try:
                record_launcher_exception(out_dir, launcher_state, exc)
            except BaseException as state_exc:
                print(
                    "WARNING: failed to record launcher exception for pod {}: {}".format(
                        pod_id, state_exc.__class__.__name__
                    ),
                    file=sys.stderr,
                )
        raise
    finally:
        if pod_id is not None and out_dir is not None and launcher_state is not None:
            print("Terminating pod {}...".format(pod_id))
            try:
                terminate_pod_with_launcher_state(
                    out_dir,
                    launcher_state,
                    pod_id,
                    terminate_and_wait,
                    original_exc=original_exc,
                )
            except BaseException as cleanup_exc:
                if original_exc is None:
                    raise
                print(
                    "WARNING: failed during cleanup bookkeeping for pod {} after {}: {}".format(
                        pod_id, original_exc.__class__.__name__, cleanup_exc.__class__.__name__
                    ),
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
