"""
Modal orchestration for Parameter Golf autoresearch lanes.

This launches finite autoresearch batches on Modal GPUs, seeded from local lane
state, and persists outputs into a Modal volume so runs can be monitored and
pulled back later.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
AUTORESEARCH_DIR = ROOT / "autoresearch"
ENV_LOCAL = ROOT / ".env.local"

WORKDIR = "/root/parameter-golf"
REMOTE_DATA = f"{WORKDIR}/data"
REMOTE_OUT = "/root/out"

VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "parameter-golf-autoresearch")
APP_NAME = os.environ.get("MODAL_APP_NAME", "parameter-golf-autoresearch")


def load_env_local() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_LOCAL.exists():
        return values
    for line in ENV_LOCAL.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip()
    return values


ENV_LOCAL_VALUES = load_env_local()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or ENV_LOCAL_VALUES.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("Missing ANTHROPIC_API_KEY in environment or .env.local")

anthropic_secret = modal.Secret.from_dict({"ANTHROPIC_API_KEY": ANTHROPIC_API_KEY})
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def ignore_repo_files(path: Path) -> bool:
    parts = path.parts
    rel = path.as_posix()
    blocked = (
        ".git",
        ".venv-modal",
        "__pycache__",
        "records",
        "autoresearch/experiments",
        "autoresearch/core_discovery",
        "autoresearch/core_promotion",
        "autoresearch/eval_time_discovery",
        "autoresearch/storage_discovery",
        "autoresearch/logs",
    )
    return any(part in parts for part in (".git", "__pycache__", ".venv-modal")) or any(rel.startswith(b) for b in blocked)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git", "nodejs", "npm")
    .pip_install("torch>=2.5", "numpy", "sentencepiece", "huggingface_hub")
    .run_commands("npm install -g @anthropic-ai/claude-code")
    .add_local_file(ROOT / "train_gpt.py", remote_path=f"{WORKDIR}/train_gpt.py", copy=True)
    .add_local_file(ROOT / "autoresearch.py", remote_path=f"{WORKDIR}/autoresearch.py", copy=True)
    .add_local_file(ROOT / "run_lane.sh", remote_path=f"{WORKDIR}/run_lane.sh", copy=True)
    .add_local_file(ROOT / "program.md", remote_path=f"{WORKDIR}/program.md", copy=True)
    .add_local_dir(DATA_DIR, remote_path=REMOTE_DATA, copy=True)
    .run_commands(
        f"mkdir -p {WORKDIR}/autoresearch {REMOTE_OUT}",
        f"chmod +x {WORKDIR}/run_lane.sh",
        f"cd {WORKDIR} && python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10",
    )
)

app = modal.App(APP_NAME, image=image)


LANE_CONFIGS: dict[str, dict[str, str | int]] = {
    "core_record_discovery": {
        "lane": "core",
        "stage": "discovery",
        "namespace": "core_record_discovery",
        "program_file": "program_record.md",
        "gpus": 1,
        "seconds": 180,
        "max_experiments": 20,
    },
    "eval_time_discovery": {
        "lane": "eval_time",
        "stage": "discovery",
        "namespace": "eval_time_discovery",
        "program_file": "program_eval.md",
        "gpus": 1,
        "seconds": 180,
        "max_experiments": 12,
    },
    "eval_window_discovery": {
        "lane": "eval_time",
        "stage": "discovery",
        "namespace": "eval_window_discovery",
        "program_file": "program_eval_window.md",
        "gpus": 1,
        "seconds": 180,
        "max_experiments": 8,
    },
    "storage_discovery": {
        "lane": "storage",
        "stage": "discovery",
        "namespace": "storage_discovery",
        "program_file": "program_storage.md",
        "gpus": 1,
        "seconds": 180,
        "max_experiments": 12,
    },
    "storage_export_discovery": {
        "lane": "storage",
        "stage": "discovery",
        "namespace": "storage_export_discovery",
        "program_file": "program_storage_export.md",
        "gpus": 1,
        "seconds": 180,
        "max_experiments": 8,
    },
    "core_nonrecord_promotion": {
        "lane": "core",
        "stage": "promotion",
        "namespace": "core_nonrecord_promotion",
        "program_file": "program_nonrecord.md",
        "gpus": 1,
        "seconds": 600,
        "max_experiments": 6,
    },
}


def read_seed(namespace: str) -> tuple[str | None, str | None]:
    best = AUTORESEARCH_DIR / namespace / "train_gpt.best.py"
    hist = AUTORESEARCH_DIR / namespace / "history.jsonl"
    best_text = best.read_text() if best.exists() else None
    hist_text = hist.read_text() if hist.exists() else None
    return best_text, hist_text


def best_bpb_from_history(history_text: str | None) -> str | None:
    if not history_text:
        return None
    best = None
    for line in history_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kept") and row.get("val_bpb") is not None:
            val = float(row["val_bpb"])
            if best is None or val < best:
                best = val
    return None if best is None else str(best)


@app.function(
    gpu="H100",
    timeout=6 * 60 * 60,
    memory=65536,
    volumes={REMOTE_OUT: volume},
    secrets=[anthropic_secret],
)
def run_lane_job(
    lane_key: str,
    seed_best: str | None,
    seed_history: str | None,
    program_text: str | None,
    extra_env: dict[str, str],
):
    import subprocess
    import time

    cfg = LANE_CONFIGS[lane_key]
    workdir = WORKDIR
    namespace = str(cfg["namespace"])
    ns_dir = Path(workdir) / "autoresearch" / namespace
    ns_dir.mkdir(parents=True, exist_ok=True)

    if seed_best:
        (ns_dir / "train_gpt.best.py").write_text(seed_best)
    if seed_history:
        (ns_dir / "history.jsonl").write_text(seed_history)

    if program_text:
        (Path(workdir) / "program.md").write_text(program_text)

    env = os.environ.copy()
    env.update(
        {
            "AUTORESEARCH_LANE": str(cfg["lane"]),
            "AUTORESEARCH_STAGE": str(cfg["stage"]),
            "AUTORESEARCH_NAMESPACE": namespace,
            "EXPERIMENT_SECONDS": str(cfg["seconds"]),
            "MAX_EXPERIMENTS": str(cfg["max_experiments"]),
            "GPUS": str(cfg["gpus"]),
            "VAL_LOSS_EVERY": "0",
            "PYTHONUNBUFFERED": "1",
            "AUTORESEARCH_MODEL": env.get("AUTORESEARCH_MODEL", "opus"),
            "CLAUDE_EFFORT": env.get("CLAUDE_EFFORT", "high"),
            "ANTHROPIC_API_KEY": env["ANTHROPIC_API_KEY"],
        }
    )
    env.update(extra_env)

    start = time.time()
    cmd = ["bash", "./run_lane.sh", lane_key]
    print(f"Launching lane {lane_key}: {' '.join(shlex.quote(c) for c in cmd)}")
    process = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip())
        lines.append(line)
    process.wait()
    elapsed = time.time() - start

    out_root = Path(REMOTE_OUT) / namespace
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "autoresearch.out").write_text("".join(lines))

    local_ns = Path(workdir) / "autoresearch" / namespace
    for rel in ("history.jsonl", "train_gpt.best.py"):
        src = local_ns / rel
        if src.exists():
            (out_root / rel).write_text(src.read_text())

    experiments_dir = local_ns / "experiments"
    if experiments_dir.exists():
        target = out_root / "experiments"
        target.mkdir(exist_ok=True)
        for item in experiments_dir.glob("*"):
            if item.is_dir():
                dst_dir = target / item.name
                dst_dir.mkdir(exist_ok=True)
                for child in item.iterdir():
                    if child.is_file():
                        (dst_dir / child.name).write_text(child.read_text())

    volume.commit()
    return {
        "lane_key": lane_key,
        "namespace": namespace,
        "elapsed_seconds": elapsed,
        "exit_code": process.returncode,
        "volume_path": f"{VOLUME_NAME}/{namespace}",
    }


def load_program_text(name: str) -> str | None:
    path = ROOT / name
    return path.read_text() if path.exists() else None


def resolve_seed(lane_key: str) -> tuple[str | None, str | None]:
    if lane_key == "core_record_discovery":
        best_text, _ = read_seed("core_promotion")
        return best_text, None
    if lane_key == "core_nonrecord_promotion":
        return read_seed("core_promotion")
    if lane_key == "eval_time_discovery":
        return read_seed("eval_time_discovery")
    if lane_key == "eval_window_discovery":
        return read_seed("eval_time_discovery")
    if lane_key == "storage_discovery":
        return read_seed("storage_discovery")
    if lane_key == "storage_export_discovery":
        return read_seed("storage_discovery")
    return read_seed("autoresearch")


@app.local_entrypoint()
def main(
    action: str = "run",
    lanes: str = "core_record_discovery",
):
    lane_list = [lane.strip() for lane in lanes.split(",") if lane.strip()]
    for lane_key in lane_list:
        if lane_key not in LANE_CONFIGS:
            raise SystemExit(f"Unknown lane {lane_key}. Known: {', '.join(sorted(LANE_CONFIGS))}")
    if action not in {"run", "launch"}:
        raise SystemExit(f"Unknown action {action}")

    print(f"{action.title()} lanes on Modal: {', '.join(lane_list)}")
    for lane_key in lane_list:
        cfg = LANE_CONFIGS[lane_key]
        seed_best, seed_history = resolve_seed(lane_key)
        extra_env: dict[str, str] = {}
        best_bpb = best_bpb_from_history(seed_history)
        if best_bpb:
            extra_env["BASELINE_BPB"] = best_bpb
        program_text = load_program_text(str(cfg["program_file"]))

        if action == "launch":
            handle = run_lane_job.spawn(lane_key, seed_best, seed_history, program_text, extra_env)
            print(f"{lane_key}: spawned function call {handle.object_id}")
        else:
            result = run_lane_job.remote(lane_key, seed_best, seed_history, program_text, extra_env)
            print(json.dumps(result, indent=2))
