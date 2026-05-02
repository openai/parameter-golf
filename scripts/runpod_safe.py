#!/usr/bin/env python3
"""Safe RunPod launcher — HTTPS proxy + Jupyter API, auto-shutdown.

SSH is blocked from this HPC so we use RunPod's HTTPS proxy to
access Jupyter on the pod. Pods auto-terminate after --max-minutes.

Usage:
    python3 scripts/runpod_safe.py list
    python3 scripts/runpod_safe.py test-1gpu
    python3 scripts/runpod_safe.py run --gpus 1 --max-minutes 10 --cmd "nvidia-smi"
    python3 scripts/runpod_safe.py terminate-all
"""

import argparse
import base64
import http.cookiejar
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

# Pod-side self-termination helpers (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pod_selfterm import (  # noqa: E402
    POD_HARD_DEADLINE_SECONDS,
    RETRIEVAL_BUFFER_SECONDS,
    selfterm_bash_preamble,
    selfterm_env_dict,
)

API_KEY_ENV = "RUNPOD_API_KEY"
GQL_URL = "https://api.runpod.io/graphql"
DEFAULT_IMAGE = "matotezitanka/proteus-pytorch:community"
IMAGE = os.environ.get("PGOLF_DOCKER_IMAGE", DEFAULT_IMAGE)
GPU_SKU_TABLE = {
    "a100-1x": {"gpu_type_id": "NVIDIA A100-SXM4-80GB", "gpu_count": 1},
    "a100-2x": {"gpu_type_id": "NVIDIA A100-SXM4-80GB", "gpu_count": 2},
    "h100-1x": {"gpu_type_id": "NVIDIA H100 80GB HBM3", "gpu_count": 1},
}
GPU_TYPE = GPU_SKU_TABLE["h100-1x"]["gpu_type_id"]  # backward-compat alias
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"  # Cloudflare blocks non-browser UAs
JUPYTER_TOKEN = ""  # Empty string = disable Jupyter auth
TERMINATE_WAIT_SECONDS = 45
TERMINATE_POLL_SECONDS = 5
RUNTIME_WAIT_SECONDS = 600
JUPYTER_WAIT_SECONDS = 180
JOB_POLL_GRACE_SECONDS = 120
WATCHDOG_DOWNLOAD_GRACE_SECONDS = 300
WATCHDOG_POLL_SECONDS = 5
JUPYTER_COOKIES = http.cookiejar.CookieJar()
JUPYTER_OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(JUPYTER_COOKIES))
MAX_SAFE_JOB_MINUTES = (POD_HARD_DEADLINE_SECONDS - RETRIEVAL_BUFFER_SECONDS) // 60


def _require_api_key():
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if api_key:
        return api_key
    raise RuntimeError(f"{API_KEY_ENV} is required in the environment")


def _validate_max_minutes(max_minutes):
    max_safe_seconds = POD_HARD_DEADLINE_SECONDS - RETRIEVAL_BUFFER_SECONDS
    if max_minutes * 60 > max_safe_seconds:
        raise ValueError(
            f"--max-minutes must be <= {MAX_SAFE_JOB_MINUTES} "
            f"to preserve the {RETRIEVAL_BUFFER_SECONDS}s retrieval buffer "
            f"within the {POD_HARD_DEADLINE_SECONDS}s hard deadline"
        )


# ── SSL context for HPC environments with certificate interception ──
import ssl as _ssl

def _make_ssl_ctx():
    ctx = _ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = _ssl.CERT_NONE
    return ctx


# ── HTTP helpers ─────────────────────────────────────────────────
def _gql(query, variables=None):
    """RunPod GraphQL call."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    body = json.dumps(payload).encode()
    req = urllib.request.Request(GQL_URL, data=body, method="POST")
    req.add_header("User-Agent", UA)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {_require_api_key()}")
    try:
        with urllib.request.urlopen(req, timeout=30, context=_make_ssl_ctx()) as r:
            result = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # GraphQL validation errors return 400 with a JSON body
        if e.code == 400:
            try:
                result = json.loads(e.read().decode())
            except Exception:
                raise
        else:
            raise
    if "errors" in result:
        raise RuntimeError(f"GraphQL: {result['errors']}")
    return result["data"]


def _jupyter_req(pod_id, path, data=None, method="GET", timeout=30):
    """HTTP request to Jupyter via RunPod HTTPS proxy."""
    url = f"https://{pod_id}-8888.proxy.runpod.net/{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("User-Agent", UA)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"token {JUPYTER_TOKEN}")
    for k, v in _jupyter_xsrf_headers(pod_id).items():
        req.add_header(k, v)
    with JUPYTER_OPENER.open(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw.decode()) if raw else {}


def _jupyter_upload(pod_id, name, text_content):
    """Upload text file to pod via Jupyter contents API."""
    url = f"https://{pod_id}-8888.proxy.runpod.net/api/contents/{name}"
    payload = json.dumps({"type": "file", "format": "text", "content": text_content}).encode()
    req = urllib.request.Request(url, data=payload, method="PUT")
    req.add_header("User-Agent", UA)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"token {JUPYTER_TOKEN}")
    for k, v in _jupyter_xsrf_headers(pod_id).items():
        req.add_header(k, v)
    with JUPYTER_OPENER.open(req, timeout=60) as r:
        r.read()


def _jupyter_upload_binary(pod_id, name, raw_bytes):
    """Upload binary file to pod via Jupyter contents API."""
    url = f"https://{pod_id}-8888.proxy.runpod.net/api/contents/{name}"
    b64 = base64.b64encode(raw_bytes).decode()
    payload = json.dumps({"type": "file", "format": "base64", "content": b64}).encode()
    req = urllib.request.Request(url, data=payload, method="PUT")
    req.add_header("User-Agent", UA)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"token {JUPYTER_TOKEN}")
    for k, v in _jupyter_xsrf_headers(pod_id).items():
        req.add_header(k, v)
    with JUPYTER_OPENER.open(req, timeout=120) as r:
        r.read()


def _jupyter_download(pod_id, name):
    """Download file from pod."""
    url = f"https://{pod_id}-8888.proxy.runpod.net/api/contents/{name}?content=1"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", UA)
    req.add_header("Authorization", f"token {JUPYTER_TOKEN}")
    for k, v in _jupyter_xsrf_headers(pod_id).items():
        req.add_header(k, v)
    with JUPYTER_OPENER.open(req, timeout=60) as r:
        result = json.loads(r.read().decode())
    if result.get("format") == "base64":
        return base64.b64decode(result["content"])
    return result.get("content", "").encode()


def _jupyter_xsrf_headers(pod_id):
    domain = f"{pod_id}-8888.proxy.runpod.net"
    xsrf_token = None
    cookies = []
    for cookie in JUPYTER_COOKIES:
        if domain.endswith(cookie.domain.lstrip(".")) or cookie.domain.lstrip(".").endswith(domain):
            cookies.append(f"{cookie.name}={cookie.value}")
            if cookie.name == "_xsrf":
                xsrf_token = cookie.value
    if xsrf_token is None:
        req = urllib.request.Request(f"https://{domain}/")
        req.add_header("User-Agent", UA)
        req.add_header("Authorization", f"token {JUPYTER_TOKEN}")
        with JUPYTER_OPENER.open(req, timeout=30) as r:
            r.read()
        cookies = []
        for cookie in JUPYTER_COOKIES:
            if domain.endswith(cookie.domain.lstrip(".")) or cookie.domain.lstrip(".").endswith(domain):
                cookies.append(f"{cookie.name}={cookie.value}")
                if cookie.name == "_xsrf":
                    xsrf_token = cookie.value
    headers = {}
    if cookies:
        headers["Cookie"] = "; ".join(cookies)
    if xsrf_token:
        headers["X-XSRFToken"] = xsrf_token
    return headers


def _ssh_target(runtime):
    key_path = os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go")
    if not os.path.isfile(key_path):
        raise RuntimeError(f"SSH fallback requires private key at {key_path}")
    for port in runtime.get("ports", []):
        if port.get("privatePort") == 22:
            ip = port.get("ip")
            public_port = port.get("publicPort")
            if ip and public_port:
                return key_path, ip, int(public_port)
    raise RuntimeError("SSH fallback requires a public SSH port on the pod runtime")


def _ssh_args(runtime):
    key_path, ip, public_port = _ssh_target(runtime)
    return [
        "ssh",
        "-i",
        key_path,
        "-p",
        str(public_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        f"root@{ip}",
    ]


def _ssh_run(runtime, command, *, input_bytes=None, timeout=120, check=True):
    proc = subprocess.run(
        _ssh_args(runtime) + [command],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            "SSH command failed ({code}): {command}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}".format(
                code=proc.returncode,
                command=command,
                stdout=proc.stdout.decode("utf-8", errors="replace"),
                stderr=proc.stderr.decode("utf-8", errors="replace"),
            )
        )
    return proc


def _ssh_upload(runtime, local_path, remote_name):
    with open(local_path, "rb") as f:
        raw = f.read()
    remote_path = "/root/{name}".format(name=remote_name)
    _ssh_run(runtime, "cat > {path}".format(path=shlex.quote(remote_path)), input_bytes=raw, timeout=180)
    return len(raw)


def _ssh_download(runtime, remote_name):
    remote_path = "/root/{name}".format(name=remote_name)
    proc = _ssh_run(runtime, "cat {path}".format(path=shlex.quote(remote_path)), timeout=120)
    return proc.stdout


def _start_job_via_ssh(runtime):
    _ssh_run(
        runtime,
        "bash -lc 'chmod +x /root/pgolf_job.sh && nohup bash /root/pgolf_job.sh > /dev/null 2>&1 &'",
        timeout=60,
    )


def _poll_status_via_ssh(runtime):
    proc = _ssh_run(
        runtime,
        "bash -lc 'if [ -f /root/pgolf_status.txt ]; then cat /root/pgolf_status.txt; fi'",
        timeout=30,
        check=False,
    )
    return proc.stdout.decode("utf-8", errors="replace").strip()


# ── RunPod API ───────────────────────────────────────────────────
def balance():
    d = _gql("{ myself { clientBalance currentSpendPerHr } }")["myself"]
    return d["clientBalance"], d["currentSpendPerHr"]


def get_pods():
    return _gql("""{ myself { pods { id name desiredStatus costPerHr gpuCount
        runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } }
        machine { gpuDisplayName }
    } } }""")["myself"]["pods"]


def create_pod(name, gpus, max_minutes, docker_args=None, extra_env=None, ports=None, start_ssh=True, deadline_sec=None, image=None, gpu_type_id=None, cloud_type=None):
    ssh_pub = ""
    p = os.path.expanduser("~/.runpod/ssh/RunPod-Key-Go.pub")
    if os.path.exists(p):
        ssh_pub = open(p).read().strip()

    # Self-termination env vars — pod will kill itself after the hard deadline
    api_key = _require_api_key()
    effective_deadline = deadline_sec or max(POD_HARD_DEADLINE_SECONDS, max_minutes * 60 + 120)
    st_env = selfterm_env_dict(api_key, effective_deadline)

    env = [
        {"key": "PGOLF_MAX_MINUTES", "value": str(max_minutes)},
        {"key": "PGOLF_HARD_DEADLINE_SEC", "value": st_env["PGOLF_HARD_DEADLINE_SEC"]},
        {"key": "RUNPOD_API_KEY", "value": st_env["RUNPOD_API_KEY"]},
        {"key": "JUPYTER_TOKEN", "value": ""},
        {"key": "JUPYTER_SERVER_TOKEN", "value": ""},
    ]
    if extra_env:
        for key, value in extra_env.items():
            env.append({"key": str(key), "value": str(value)})
    if ssh_pub:
        env.append({"key": "PUBLIC_KEY", "value": ssh_pub})

    mut = """mutation($i: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $i) { id costPerHr machineId }
    }"""
    inp = {
        "name": name, "imageName": image or IMAGE, "gpuTypeId": gpu_type_id or GPU_TYPE,
        "gpuCount": gpus, "containerDiskInGb": 50, "volumeInGb": 0,
        "env": env, "ports": ports or "8888/http,22/tcp",
        "cloudType": cloud_type or "SECURE", "startSsh": bool(start_ssh),
    }
    if docker_args:
        inp["dockerArgs"] = docker_args
    return _gql(mut, {"i": inp})["podFindAndDeployOnDemand"]


def wait_runtime(pod_id, timeout=RUNTIME_WAIT_SECONDS):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = _gql(f'{{ pod(input: {{ podId: "{pod_id}" }}) {{ desiredStatus runtime {{ uptimeInSeconds ports {{ ip privatePort publicPort }} }} }} }}')
        pod = d.get("pod")
        if pod is None:
            elapsed = int(time.time() - t0)
            print(f"  [{elapsed}s] not yet visible...", end="\r", flush=True)
            time.sleep(10)
            continue
        if pod["desiredStatus"] == "EXITED":
            raise RuntimeError("Pod exited")
        rt = pod.get("runtime")
        if rt and rt.get("uptimeInSeconds", 0) > 0:
            print()
            return rt
        elapsed = int(time.time() - t0)
        print(f"  [{elapsed}s] starting...", end="\r", flush=True)
        time.sleep(10)
    raise TimeoutError(f"Pod not ready in {timeout}s")


def wait_jupyter(pod_id, timeout=JUPYTER_WAIT_SECONDS):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            # Check root URL (doesn't require token)
            url = f"https://{pod_id}-8888.proxy.runpod.net/"
            req = urllib.request.Request(url)
            req.add_header("User-Agent", UA)
            with urllib.request.urlopen(req, timeout=10, context=_make_ssl_ctx()) as r:
                r.read()
            print()
            return True
        except Exception:
            elapsed = int(time.time() - t0)
            print(f"  [{elapsed}s] jupyter starting...", end="\r", flush=True)
            time.sleep(10)
    return False


def _pod_present(pod_id):
    return any(p.get("id") == pod_id for p in get_pods() or [])


def terminate_and_wait(pod_id, timeout=TERMINATE_WAIT_SECONDS,
                       poll_interval=TERMINATE_POLL_SECONDS):
    _gql(f'mutation {{ podTerminate(input: {{ podId: "{pod_id}" }}) }}')
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            if not _pod_present(pod_id):
                print(f"  Terminated {pod_id}")
                return True
        except Exception as exc:
            last_error = exc
        time.sleep(poll_interval)
    if last_error:
        print(f"  Terminate requested for {pod_id}, but verification failed before timeout: {last_error}")
    else:
        print(f"  Terminate requested for {pod_id}, but it still exists after {timeout}s")
    return False


def terminate(pod_id):
    return terminate_and_wait(pod_id)


def _watchdog_window_seconds(max_minutes):
    return (
        RUNTIME_WAIT_SECONDS
        + JUPYTER_WAIT_SECONDS
        + max_minutes * 60
        + JOB_POLL_GRACE_SECONDS
        + WATCHDOG_DOWNLOAD_GRACE_SECONDS
        + TERMINATE_WAIT_SECONDS
    )


def _watchdog_arm_file(pod_id):
    safe_pod_id = "".join(ch for ch in pod_id if ch.isalnum() or ch in ("-", "_"))
    return os.path.join(tempfile.gettempdir(), f"runpod_safe_{safe_pod_id}.arm")


def arm_watchdog(pod_id, deadline_epoch):
    arm_file = _watchdog_arm_file(pod_id)
    with open(arm_file, "w") as f:
        json.dump({"pod_id": pod_id, "deadline": int(deadline_epoch)}, f)
    subprocess.Popen(
        [
            sys.executable,
            os.path.abspath(__file__),
            "_watchdog",
            "--pod-id",
            pod_id,
            "--deadline",
            str(int(deadline_epoch)),
            "--arm-file",
            arm_file,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    return arm_file


def disarm_watchdog(arm_file):
    if not arm_file:
        return
    try:
        os.remove(arm_file)
    except FileNotFoundError:
        pass


def _watchdog_main(pod_id, deadline_epoch, arm_file):
    while time.time() < deadline_epoch:
        if not os.path.exists(arm_file):
            return
        time.sleep(min(WATCHDOG_POLL_SECONDS, max(0, deadline_epoch - time.time())))
    if not os.path.exists(arm_file):
        return
    try:
        terminate_and_wait(pod_id)
    except Exception:
        pass
    finally:
        disarm_watchdog(arm_file)


def _cleanup_pod(pod_id, watchdog_arm_file=None):
    print(f"\nTerminating pod {pod_id}...")
    terminated = terminate_and_wait(pod_id)
    if terminated:
        disarm_watchdog(watchdog_arm_file)
    return terminated


# ── Job execution via websocket terminal ─────────────────────────
def start_job_via_ws(pod_id, command):
    """Start a job on pod using Jupyter terminal websocket."""
    try:
        import websocket
    except ImportError:
        # Install it
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "websocket-client", "-q"],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        import websocket

    # Create a terminal
    term = _jupyter_req(pod_id, "api/terminals", data={}, method="POST")
    term_name = term.get("name", "1")
    print(f"  Terminal: {term_name}")

    # Connect websocket
    ws_url = f"wss://{pod_id}-8888.proxy.runpod.net/terminals/websocket/{term_name}?token={JUPYTER_TOKEN}"
    ws = websocket.create_connection(ws_url, timeout=15,
        header=[f"User-Agent: {UA}"])

    # Wait for initial prompt
    time.sleep(2)
    try:
        while ws.timeout == 0 or True:
            ws.settimeout(1)
            msg = ws.recv()
            # just drain initial output
    except Exception:
        pass

    # Send the command
    ws.settimeout(5)
    ws.send(json.dumps(["stdin", command + "\r"]))
    time.sleep(1)

    # Read a bit of output
    output = []
    try:
        for _ in range(10):
            ws.settimeout(2)
            msg = ws.recv()
            data = json.loads(msg)
            if data[0] == "stdout":
                output.append(data[1])
    except Exception:
        pass

    ws.close()
    return "".join(output)


def _run_ws_command(pod_id, command, timeout=60):
    """Run a shell command over the Jupyter terminal websocket and capture stdout."""
    try:
        import websocket
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "websocket-client", "-q"],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        import websocket

    term = _jupyter_req(pod_id, "api/terminals", data={}, method="POST")
    term_name = term.get("name", "1")
    ws_url = f"wss://{pod_id}-8888.proxy.runpod.net/terminals/websocket/{term_name}?token={JUPYTER_TOKEN}"
    ws = websocket.create_connection(ws_url, timeout=15, header=[f"User-Agent: {UA}"])
    sentinel = "__PGOLF_DONE_{stamp}__".format(stamp=int(time.time() * 1000))
    try:
        time.sleep(1)
        try:
            while True:
                ws.settimeout(1)
                ws.recv()
        except Exception:
            pass
        payload = "{command}\nprintf '{sentinel}\\n'\n".format(command=command, sentinel=sentinel)
        ws.settimeout(5)
        ws.send(json.dumps(["stdin", payload]))
        output = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            ws.settimeout(min(5, max(1, deadline - time.time())))
            msg = ws.recv()
            data = json.loads(msg)
            if data[0] != "stdout":
                continue
            output.append(data[1])
            combined = "".join(output)
            if sentinel in combined:
                return combined.split(sentinel, 1)[0]
        raise TimeoutError(f"Timed out waiting for terminal output after {timeout}s")
    finally:
        ws.close()


def _ws_upload_text(pod_id, remote_name, text_content):
    marker = "__PGOLF_EOF__"
    while marker in text_content:
        marker += "_X"
    command = "cat > /root/{name} <<'{marker}'\n{text}\n{marker}".format(
        name=remote_name,
        marker=marker,
        text=text_content,
    )
    _run_ws_command(pod_id, command, timeout=max(60, min(300, len(text_content) // 200 + 30)))


def _ws_download(pod_id, remote_name):
    command = """python3 - <<'PY'
import base64
from pathlib import Path
path = Path('/root/{name}')
print(base64.b64encode(path.read_bytes()).decode('ascii'))
PY""".format(name=remote_name)
    output = _run_ws_command(pod_id, command, timeout=120).strip()
    if not output:
        return b""
    return base64.b64decode(output)


def _poll_status_via_ws(pod_id):
    command = "bash -lc 'if [ -f /root/pgolf_status.txt ]; then cat /root/pgolf_status.txt; fi'"
    return _run_ws_command(pod_id, command, timeout=30).strip()


def launch_job(name, gpus, max_minutes, shell_script, upload_files=None, download_after=None, image=None):
    """Full lifecycle: create pod → upload → execute → poll → download → terminate."""
    _validate_max_minutes(max_minutes)
    bal, _ = balance()
    cost_est = gpus * 2.69 * max_minutes / 60
    print(f"Balance: ${bal:.2f}  Est cost: ${cost_est:.2f}")
    if bal < cost_est * 2:
        print("ERROR: Insufficient balance"); return None

    # Wrap script with pod-side self-termination + watchdog
    selfterm_snippet = selfterm_bash_preamble()
    wrapped = f"""#!/bin/bash
set -o pipefail
exec > >(tee /root/pgolf_stdout.txt) 2>&1
echo "=== PGOLF JOB START $(date -u) ==="
echo "GPUs: {gpus}, Max: {max_minutes} min"

{selfterm_snippet}

# Watchdog: write TIMEOUT status after {max_minutes} min
( sleep {max_minutes * 60}; echo TIMEOUT > /root/pgolf_status.txt ) &

set +e
{shell_script}
EC=$?
set -e

echo $EC > /root/pgolf_exit_code.txt
echo DONE > /root/pgolf_status.txt
echo "=== PGOLF JOB END $(date -u) exit=$EC ==="
"""

    print(f"\nLaunching {gpus}×H100 SXM (max {max_minutes} min)...")
    pod = create_pod(name, gpus, max_minutes, image=image)
    pod_id = pod["id"]
    cost_hr = pod.get("costPerHr", "?")
    print(f"Pod: {pod_id}  ${cost_hr}/hr")

    watchdog_arm_file = None
    result = None
    results_saved_msg = None
    caught_exc = None
    cleanup_exc = None

    try:
        watchdog_arm_file = arm_watchdog(
            pod_id,
            time.time() + _watchdog_window_seconds(max_minutes),
        )

        rt = wait_runtime(pod_id)
        print(f"Pod RUNNING (uptime={rt['uptimeInSeconds']}s)")

        print("Waiting for Jupyter...")
        if not wait_jupyter(pod_id):
            raise RuntimeError("Jupyter not available after 3 min")
        print("Jupyter ready!")

        use_ssh_fallback = False
        use_ws_fallback = False
        try:
            # Upload job script
            _jupyter_upload(pod_id, "pgolf_job.sh", wrapped)
            print("  Uploaded pgolf_job.sh via Jupyter")

            # Upload additional files
            if upload_files:
                for lf in upload_files:
                    fname = os.path.basename(lf)
                    with open(lf, "rb") as f:
                        raw = f.read()
                    try:
                        _jupyter_upload(pod_id, fname, raw.decode())
                    except UnicodeDecodeError:
                        _jupyter_upload_binary(pod_id, fname, raw)
                    print(f"  Uploaded {fname} via Jupyter ({len(raw)} bytes)")

            # Start the job
            print("Starting job via websocket...")
            out = start_job_via_ws(pod_id,
                "chmod +x /root/pgolf_job.sh && nohup bash /root/pgolf_job.sh > /dev/null 2>&1 &")
            print(f"  WS output: {out[:200]}")
        except urllib.error.HTTPError as e:
            if e.code != 403:
                raise
            use_ws_fallback = True
            print("  Jupyter contents API returned 403; falling back to terminal websocket upload/exec.")

        if use_ws_fallback:
            _ws_upload_text(pod_id, "pgolf_job.sh", wrapped)
            print(f"  Uploaded pgolf_job.sh via terminal websocket ({len(wrapped.encode('utf-8'))} bytes)")
            if upload_files:
                for lf in upload_files:
                    fname = os.path.basename(lf)
                    with open(lf, "r", encoding="utf-8") as f:
                        text = f.read()
                    _ws_upload_text(pod_id, fname, text)
                    print(f"  Uploaded {fname} via terminal websocket ({len(text.encode('utf-8'))} bytes)")
            print("Starting job via terminal websocket...")
            out = _run_ws_command(
                pod_id,
                "chmod +x /root/pgolf_job.sh && nohup bash /root/pgolf_job.sh > /dev/null 2>&1 &",
                timeout=30,
            )
            print(f"  WS output: {out[:200]}")

        if use_ssh_fallback:
            with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
                tmp.write(wrapped)
                tmp_path = tmp.name
            try:
                size = _ssh_upload(rt, tmp_path, "pgolf_job.sh")
                print(f"  Uploaded pgolf_job.sh via SSH ({size} bytes)")
            finally:
                os.remove(tmp_path)
            if upload_files:
                for lf in upload_files:
                    fname = os.path.basename(lf)
                    size = _ssh_upload(rt, lf, fname)
                    print(f"  Uploaded {fname} via SSH ({size} bytes)")
            print("Starting job via SSH...")
            _start_job_via_ssh(rt)

        # Poll for completion
        print(f"\nPolling for completion (max {max_minutes} min)...")
        poll_start = time.time()
        while time.time() - poll_start < max_minutes * 60 + 120:
            time.sleep(30)
            try:
                if use_ws_fallback:
                    status = _poll_status_via_ws(pod_id)
                elif use_ssh_fallback:
                    status = _poll_status_via_ssh(rt)
                else:
                    content = _jupyter_download(pod_id, "pgolf_status.txt")
                    status = content.decode().strip()
                if status in ("DONE", "TIMEOUT"):
                    elapsed = int(time.time() - poll_start)
                    print(f"\n  Job {status} after {elapsed}s")
                    break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    elapsed = int(time.time() - poll_start)
                    print(f"  [{elapsed}s] running...", end="\r", flush=True)
            except Exception as e:
                print(f"  Error polling: {e}")
        else:
            print("\n  WARNING: Timed out")

        # Download results
        results_dir = f"results/pod_{pod_id}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"\nDownloading results to {results_dir}/")

        for fname in ["pgolf_stdout.txt", "pgolf_exit_code.txt", "pgolf_status.txt"]:
            try:
                if use_ws_fallback:
                    data = _ws_download(pod_id, fname)
                elif use_ssh_fallback:
                    data = _ssh_download(rt, fname)
                else:
                    data = _jupyter_download(pod_id, fname)
                with open(f"{results_dir}/{fname}", "wb") as f:
                    f.write(data)
                print(f"  {fname} ({len(data)} bytes)")
            except Exception as e:
                print(f"  {fname}: {e}")

        if download_after:
            for rf in download_after:
                try:
                    if use_ws_fallback:
                        data = _ws_download(pod_id, rf)
                    elif use_ssh_fallback:
                        data = _ssh_download(rt, rf)
                    else:
                        data = _jupyter_download(pod_id, rf)
                    local = f"{results_dir}/{os.path.basename(rf)}"
                    with open(local, "wb") as f:
                        f.write(data)
                    print(f"  {rf} ({len(data)} bytes)")
                except Exception as e:
                    print(f"  {rf}: {e}")

        # Show output
        stdout_path = f"{results_dir}/pgolf_stdout.txt"
        if os.path.exists(stdout_path):
            print("\n=== OUTPUT (last 50 lines) ===")
            with open(stdout_path) as f:
                for line in f.readlines()[-50:]:
                    print(line, end="")

        results_saved_msg = f"Results saved to {results_dir}/"
        result = pod_id

    except Exception as e:
        caught_exc = e
        print(f"\nERROR: {e}")
    finally:
        if pod_id:
            try:
                _cleanup_pod(pod_id, watchdog_arm_file)
            except Exception as e:
                cleanup_exc = e
                print(f"  Cleanup failed for {pod_id}: {e}")

    if cleanup_exc is not None and caught_exc is None:
        raise cleanup_exc
    if caught_exc is not None:
        raise caught_exc
    if results_saved_msg:
        print(results_saved_msg)
    return result


# ── CLI commands ─────────────────────────────────────────────────
def cmd_list(_args):
    bal, spend = balance()
    print(f"Balance: ${bal:.2f}  Burn: ${spend:.2f}/hr\n")
    ps = get_pods()
    for p in ps:
        rt = p.get("runtime") or {}
        up = rt.get("uptimeInSeconds", 0)
        c = p.get("costPerHr", 0)
        g = p.get("gpuCount", "?")
        gn = (p.get("machine") or {}).get("gpuDisplayName", "?")
        tot = c * up / 3600
        print(f"  {p['id']}  {p.get('name','?'):25s}  {g}×{gn}  up={up}s  "
              f"${c:.2f}/hr  ~${tot:.2f}  {p.get('desiredStatus','?')}")
    if not ps:
        print("  No pods.")


def cmd_test(args):
    script = """
nvidia-smi
python3 -c "
import torch
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU{i}: {torch.cuda.get_device_name(i)}')
print(f'PyTorch: {torch.__version__}')
"
echo TEST_COMPLETE
"""
    launch_job("pgolf-test-1gpu", gpus=1, max_minutes=args.max_minutes,
               shell_script=script, image=getattr(args, 'docker_image', None))


def cmd_run(args):
    if args.script:
        with open(args.script) as f:
            script = f.read()
    elif args.cmd:
        script = args.cmd
    else:
        print("Need --script or --cmd"); return
    launch_job(f"pgolf-{args.gpus}gpu", gpus=args.gpus,
               max_minutes=args.max_minutes, shell_script=script,
               upload_files=args.upload, download_after=args.download,
               image=getattr(args, 'docker_image', None))


def cmd_terminate(_args):
    ps = get_pods()
    if not ps:
        print("No pods to terminate.")
    for p in ps:
        print(f"Terminating {p['id']} ({p.get('name','?')})...")
        try:
            terminate(p["id"])
        except Exception as e:
            print(f"  {e}")
    bal, _ = balance()
    print(f"Balance: ${bal:.2f}")


def cmd_watchdog(args):
    _watchdog_main(args.pod_id, args.deadline, args.arm_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe RunPod launcher")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List pods + balance")
    t = sub.add_parser("test-1gpu", help="Test 1 GPU connectivity")
    t.add_argument("--max-minutes", type=int, default=10)
    t.add_argument("--docker-image", default=None,
                   help="Docker image override (default: PGOLF_DOCKER_IMAGE env or base community image)")

    r = sub.add_parser("run", help="Run job on pod")
    r.add_argument("--gpus", type=int, default=1)
    r.add_argument("--max-minutes", type=int, default=30)
    r.add_argument("--script", help="Shell script to run")
    r.add_argument("--cmd", help="Inline command")
    r.add_argument("--upload", nargs="*", help="Files to upload")
    r.add_argument("--download", nargs="*", help="Files to download after")
    r.add_argument("--docker-image", default=None,
                   help="Docker image override (default: PGOLF_DOCKER_IMAGE env or base community image)")

    sub.add_parser("terminate-all", help="Kill all pods")
    w = sub.add_parser("_watchdog", help=argparse.SUPPRESS)
    w.add_argument("--pod-id", required=True)
    w.add_argument("--deadline", type=int, required=True)
    w.add_argument("--arm-file", required=True)

    args = parser.parse_args()
    cmd_map = {
        "list": cmd_list, "test-1gpu": cmd_test,
        "run": cmd_run, "terminate-all": cmd_terminate,
        "_watchdog": cmd_watchdog,
    }
    try:
        fn = cmd_map.get(args.command)
        if fn:
            fn(args)
        else:
            parser.print_help()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
