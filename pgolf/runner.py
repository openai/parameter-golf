"""Run a single training trial and parse results."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

from .config import PENALTY, PROJECT_DIR, RESULTS_FILE, TRAIN_SCRIPT


def run_trial(
    config: dict,
    max_wallclock: int,
    iterations: int,
    label: str = "",
    skip_compile: bool = False,
) -> dict:
    """Run one training+eval via subprocess. Returns a results dict."""
    base = {
        "ITERATIONS": iterations,
        "MAX_WALLCLOCK_SECONDS": max_wallclock,
        "TRAIN_BATCH_TOKENS": 65536,
        "VAL_BATCH_SIZE": 65536,
        "VAL_LOSS_EVERY": 0,
        "TRAIN_LOG_EVERY": 50,
        "WARMUP_STEPS": 0,
        "ENABLE_TURBOQUANT": 1,
    }
    if skip_compile:
        base["SKIP_COMPILE"] = 1
    base.update(config)

    env = os.environ.copy()
    env.update({k: str(v) for k, v in base.items()})

    timeout = max_wallclock + 300
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["torchrun", "--standalone", "--nproc_per_node=1", TRAIN_SCRIPT],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=PROJECT_DIR,
        )
        elapsed = time.time() - t0
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return _fail(label, "TIMEOUT", time.time() - t0, config)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return _fail(label, f"EXCEPTION: {e}", time.time() - t0, config)

    if proc.returncode != 0:
        err_lines = (proc.stderr or "").strip().split("\n")[-10:]
        return _fail(label, "CRASHED", elapsed, config, error="\n".join(err_lines))

    return _parse_output(label, output, elapsed, config)


def _fail(label, status, elapsed, config, error=""):
    r = {
        "label": label,
        "status": status,
        "val_bpb": PENALTY,
        "elapsed": elapsed,
        "config": config,
    }
    if error:
        r["error"] = error
    return r


def _parse_output(label, output, elapsed, config):
    val_bpb = None
    for pattern in [
        r"final_ngram\+knn_exact val_loss:\S+ val_bpb:(\S+)",
        r"final_ngram_exact val_loss:\S+ val_bpb:(\S+)",
        r"final_knn_exact val_loss:\S+ val_bpb:(\S+)",
        r"final_ttt_lora_exact val_loss:\S+ val_bpb:(\S+)",
        r"final_int8_zlib_roundtrip_exact val_loss:\S+ val_bpb:(\S+)",
        r"val_bpb:(\S+)",
    ]:
        m = re.search(pattern, output)
        if m:
            val_bpb = float(m.group(1))
            break

    size_match = re.search(r"Total submission size int8\+zlib: (\d+) bytes", output)
    actual_size = int(size_match.group(1)) if size_match else None

    return {
        "label": label,
        "status": "OK",
        "val_bpb": val_bpb if val_bpb else PENALTY,
        "artifact_size": actual_size,
        "elapsed": elapsed,
        "config": config,
    }


def save_result(result: dict) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result, default=str) + "\n")


def load_results() -> list[dict]:
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            for line in f:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def load_completed_labels() -> set[str]:
    return {r["label"] for r in load_results() if r.get("status") == "OK"}
