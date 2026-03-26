#!/usr/bin/env python3
"""
Auto-evolving research agent for the OpenAI Parameter Golf challenge.

Inspired by Andrej Karpathy's autoresearch project. Uses an LLM to autonomously
propose, test, and iterate on modifications to the training script.

Usage:
    # Proxy mode on 1x GPU:
    python3 autoevolve/evolve.py --nproc 1

    # Final-fidelity mode on multi-GPU:
    python3 autoevolve/evolve.py --nproc 8

    # Dry run (propose + validate syntax, no training):
    python3 autoevolve/evolve.py --nproc 1 --dry-run
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
AUTOEVOLVE_DIR = ROOT / "autoevolve"
PROMPTS_DIR = AUTOEVOLVE_DIR / "prompts"
SCRIPT_PATH = AUTOEVOLVE_DIR / "train_gpt.py"
BEST_SCRIPT_PATH = AUTOEVOLVE_DIR / "best_train_gpt.py"
FRONTIER_SCRIPT_PATH = AUTOEVOLVE_DIR / "frontier_train_gpt.py"
RESULTS_FILE = AUTOEVOLVE_DIR / "results.tsv"
DOSSIER_FILE = PROMPTS_DIR / "memory_dossier.md"
INCUMBENT_STATE_FILE = AUTOEVOLVE_DIR / "incumbent_state.json"
FRONTIER_STATE_FILE = AUTOEVOLVE_DIR / "frontier_state.json"
LOGS_DIR = AUTOEVOLVE_DIR / "logs"
ROOT_TRAIN_LOGS_DIR = ROOT / "logs"
RECORDS_DIR = ROOT / "records" / "track_10min_16mb"
ROOT_README = ROOT / "README.md"

DEFAULT_MODEL = "gpt-5.4"
DEFAULT_PUBLIC_SOTA_BPB = 1.1194
DEFAULT_BRANCH = "autoevolve/run"
ARTIFACT_LIMIT = 16_000_000
MAX_WALLCLOCK = 600  # 10 minutes — hard competition limit
MAX_CONSECUTIVE_FAILURES = 5
UPSTREAM_SYNC_EVERY = 5
REPEAT_FAMILY_THRESHOLD = 2
NEAR_MISS_MAX_GAP = 0.0040
FRONTIER_CONTINUATION_STEPS = 2

MODE_SCOUT = "proxy"
MODE_FULL = "final"

SCOUT_WALLCLOCK_SECONDS = 4_800
SCOUT_PROCESS_TIMEOUT_SECONDS = 14_400

FULL_WALLCLOCK_SECONDS = 600
FULL_PROCESS_TIMEOUT_SECONDS = 1500

LEGACY_RESULTS_COLUMNS = [
    "iteration",
    "timestamp",
    "val_bpb",
    "artifact_bytes",
    "status",
    "description",
    "reasoning",
]

RESULTS_COLUMNS = [
    "iteration",
    "timestamp",
    "proposal_family",
    "mode",
    "status",
    "val_bpb",
    "artifact_bytes",
    "total_bytes",
    "description",
    "reasoning",
    "llm_seconds",
    "prompt_tokens",
    "completion_tokens",
    "timeout_stage",
    "last_step",
    "last_train_time_ms",
    "last_eval_time_ms",
    "parse_status",
    "commit_sha",
]

FAMILY_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    (
        "quantization_compression",
        (
            "quant",
            "qat",
            "int4",
            "int5",
            "int6",
            "int8",
            "compression",
            "zstd",
            "zlib",
            "prun",
            "export align",
            "fake-quant",
            "fake quant",
            "proxquant",
            "block quant",
        ),
    ),
    (
        "activation_mlp",
        (
            "swiglu",
            "geglu",
            "relu",
            "activation",
            "mlp",
            "gated mlp",
            "smeargate",
        ),
    ),
    (
        "optimizer_schedule",
        (
            "warmdown",
            "warmup",
            "learning rate",
            "lr ",
            "adam",
            "muon",
            "weight decay",
            "swa",
            "grad clip",
            "schedule",
        ),
    ),
    (
        "attention_kv",
        (
            "attention",
            "kv head",
            "num_kv",
            "gqa",
            "mqa",
            "rope",
            "alibi",
            "fire positional",
            "qk",
            "head count",
        ),
    ),
    (
        "architecture_depth",
        (
            "layer",
            "depth",
            "recurrent",
            "recurrence",
            "looped",
            "width",
            "model_dim",
            "u-net",
            "skip",
            "transformer block",
        ),
    ),
    (
        "embedding_tokenizer",
        (
            "embedding",
            "vocab",
            "tokenizer",
            "bigram",
            "factorized embedding",
            "subword",
        ),
    ),
    (
        "evaluation_strategy",
        (
            "eval",
            "evaluation",
            "sliding window",
            "stride",
            "ttt",
            "test-time",
            "lora",
        ),
    ),
    (
        "systems_throughput",
        (
            "compile",
            "flash",
            "kernel",
            "throughput",
            "checkpointing",
            "fp8",
            "memory",
            "cuda",
        ),
    ),
    (
        "initialization_norm",
        (
            "init",
            "initialization",
            "orthogonal",
            "normalization",
            "rmsnorm",
            "norm",
            "gain",
        ),
    ),
]

RESEARCH_FAILURE_STATUSES = {"crash", "discard", "near_miss", "over_size", "parse_error"}
RESEARCH_NONKEEP_STATUSES = RESEARCH_FAILURE_STATUSES
INFRA_FAILURE_STATUSES = {"llm_error", "invalid"}
RESEARCH_SUCCESS_STATUSES = {"baseline", "keep"}
SCORABLE_STATUSES = {"baseline", "keep", "near_miss", "discard", "over_size", "crash", "parse_error"}
TRACK_RECORD_README_RE = re.compile(
    r"\|\s*[^|]+\|\s*(1\.\d{4})\s*\|[^|]*\|[^|]*\|[^|]*\|\s*\[info\]\((records/track_10min_16mb/[^)]+/README\.md)\)\s*\|"
)


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-evolve Parameter Golf training")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    p.add_argument("--nproc", type=int, default=1, help="GPUs for torchrun (default: 1)")
    p.add_argument("--dry-run", action="store_true", help="Propose changes only, skip training")
    p.add_argument(
        "--max-iters",
        type=int,
        default=0,
        help="Max iterations for this run (default: 0 = run until manually stopped)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def setup_openai():
    """Lazy-import openai and create client from .env or environment."""
    try:
        from dotenv import load_dotenv

        load_dotenv(AUTOEVOLVE_DIR / ".env")
    except ImportError:
        load_env_file(AUTOEVOLVE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "ERROR: openai package is not installed. "
            "Install it in the active environment with: python -m pip install openai"
        )
        sys.exit(1)

    return OpenAI(api_key=api_key)


def load_ranked_track_record_scripts() -> list[tuple[float, Path]]:
    """Return track_10min_16mb scripts ranked by the local README leaderboard."""
    if not ROOT_README.exists():
        return []
    candidates: list[tuple[float, Path]] = []
    for line in ROOT_README.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRACK_RECORD_README_RE.search(line)
        if not match:
            continue
        score = float(match.group(1))
        record_readme = ROOT / match.group(2)
        script = record_readme.parent / "train_gpt.py"
        if script.exists():
            candidates.append((score, script))
    candidates.sort(key=lambda item: (item[0], item[1].as_posix()))
    return candidates


def current_public_sota_bpb() -> float:
    ranked = load_ranked_track_record_scripts()
    if ranked:
        return ranked[0][0]
    return DEFAULT_PUBLIC_SOTA_BPB


def find_best_sota_script() -> Path | None:
    """Find the best local track record script, preferring the README leaderboard."""
    ranked = load_ranked_track_record_scripts()
    if ranked:
        return ranked[0][1]

    if not RECORDS_DIR.exists():
        return None
    records = sorted(RECORDS_DIR.iterdir(), reverse=True)
    for record in records:
        script = record / "train_gpt.py"
        if script.exists():
            return script
    return None


def read_json_file(path: Path, default: dict | list) -> dict | list:
    if not path.exists():
        return json.loads(json.dumps(default))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return json.loads(json.dumps(default))
    if isinstance(default, dict) and not isinstance(data, dict):
        return json.loads(json.dumps(default))
    if isinstance(default, list) and not isinstance(data, list):
        return json.loads(json.dumps(default))
    return data


def write_json_file(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_results_file() -> list[dict[str, str]]:
    """Create or migrate results.tsv to the current schema and return parsed rows."""
    if not RESULTS_FILE.exists():
        write_results_rows([])
        return []

    rows = load_results_rows()
    raw_header = read_results_header()
    if raw_header != RESULTS_COLUMNS:
        write_results_rows(rows)
    return rows


def setup_workspace(mode: str) -> None:
    """Create directories, ensure schema, and copy the best SOTA baseline as starting point."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not INCUMBENT_STATE_FILE.exists():
        write_json_file(INCUMBENT_STATE_FILE, {})
    if not FRONTIER_STATE_FILE.exists():
        write_json_file(FRONTIER_STATE_FILE, {"active": False})

    if not SCRIPT_PATH.exists():
        sota = find_best_sota_script()
        if sota:
            shutil.copy2(sota, SCRIPT_PATH)
            shutil.copy2(sota, BEST_SCRIPT_PATH)
            print(f"Copied SOTA baseline from {sota.parent.name}")
        else:
            print("ERROR: No SOTA script found in records/track_10min_16mb/")
            sys.exit(1)

    if not BEST_SCRIPT_PATH.exists() and SCRIPT_PATH.exists():
        shutil.copy2(SCRIPT_PATH, BEST_SCRIPT_PATH)

    if not FRONTIER_SCRIPT_PATH.exists() and BEST_SCRIPT_PATH.exists():
        shutil.copy2(BEST_SCRIPT_PATH, FRONTIER_SCRIPT_PATH)

    rows = ensure_results_file()
    if not rows:
        write_json_file(INCUMBENT_STATE_FILE, {})
        write_json_file(FRONTIER_STATE_FILE, {"active": False})
        if BEST_SCRIPT_PATH.exists():
            shutil.copy2(BEST_SCRIPT_PATH, FRONTIER_SCRIPT_PATH)
    elif load_frontier_state(mode) is None and BEST_SCRIPT_PATH.exists():
        shutil.copy2(BEST_SCRIPT_PATH, FRONTIER_SCRIPT_PATH)
    write_memory_dossier(rows, mode)


def setup_git(branch: str) -> None:
    """Create or switch to the autoevolve git branch, and configure push auth."""
    token = os.getenv("GITHUB_TOKEN")
    if token:
        url_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        origin_url = url_result.stdout.strip()
        if origin_url and "github.com" in origin_url and "@" not in origin_url:
            auth_url = origin_url.replace("https://", f"https://{token}@")
            subprocess.run(
                ["git", "remote", "set-url", "origin", auth_url],
                capture_output=True,
                cwd=ROOT,
            )
            print("  Git push auth configured via GITHUB_TOKEN.")
    else:
        print("  WARNING: GITHUB_TOKEN not set in .env — results won't be pushed to GitHub.")
        print("           Add GITHUB_TOKEN=ghp_... to autoevolve/.env to auto-save results.")

    cur = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    if cur != branch:
        checkout_result = subprocess.run(
            ["git", "checkout", "-b", branch],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if checkout_result.returncode != 0:
            checkout_result = subprocess.run(
                ["git", "checkout", branch],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
        if checkout_result.returncode != 0:
            details = clean_field(checkout_result.stderr or checkout_result.stdout)[:240]
            raise RuntimeError(f"Failed to switch to git branch '{branch}': {details}")
    final_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    if final_branch != branch:
        raise RuntimeError(f"Expected git branch '{branch}', but current branch is '{final_branch or '?'}'")
    print(f"Git branch: {branch}")


# ---------------------------------------------------------------------------
# Mode Configuration
# ---------------------------------------------------------------------------

def resolve_mode(nproc: int) -> str:
    return MODE_SCOUT if nproc == 1 else MODE_FULL


def mode_train_wallclock_seconds(mode: str) -> int:
    return SCOUT_WALLCLOCK_SECONDS if mode == MODE_SCOUT else FULL_WALLCLOCK_SECONDS


def mode_process_timeout_seconds(mode: str) -> int:
    return SCOUT_PROCESS_TIMEOUT_SECONDS if mode == MODE_SCOUT else FULL_PROCESS_TIMEOUT_SECONDS


def build_run_env(mode: str, iteration: int) -> tuple[dict[str, str], str]:
    env = os.environ.copy()
    env["DATA_PATH"] = str(ROOT / "data" / "datasets" / "fineweb10B_sp1024")
    env["TOKENIZER_PATH"] = str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")
    env["PYTHONUNBUFFERED"] = "1"
    env["MAX_WALLCLOCK_SECONDS"] = str(mode_train_wallclock_seconds(mode))
    run_id = f"autoevolve_{mode}_{iteration:04d}_{int(time.time())}"
    env["RUN_ID"] = run_id
    return env, run_id


def mode_prompt_guidance(mode: str) -> str:
    if mode == MODE_SCOUT:
        return (
            "Mode: PROXY (1xH100 long-horizon local proxy).\n"
            "- Optimize for transfer to the official 8xH100 / 600s competition setting, not for ultra-short-run hacks.\n"
            "- Preserve official artifact and evaluation behavior; do not rely on cheaper-proxy-only shortcuts.\n"
            "- Expect single-GPU throughput and eval timing to differ from the final hardware, so prefer robust wins over fragile timing tricks.\n"
            f"- The runner caps training at {SCOUT_WALLCLOCK_SECONDS}s on 1xH100 before final 8xH100 validation."
        )
    return (
        "Mode: FINAL (8xH100 competition-fidelity mode).\n"
        "- Optimize for final score under the official train/eval budget.\n"
        "- Higher-overhead ideas are allowed if you justify the throughput tradeoff."
    )


# ---------------------------------------------------------------------------
# Upstream Sync
# ---------------------------------------------------------------------------

def sync_upstream() -> str | None:
    """Fetch upstream and check for new SOTA records. Returns summary or None."""
    print("  Checking upstream for new records...")
    fetched = subprocess.run(
        ["git", "fetch", "https://github.com/openai/parameter-golf.git", "main", "--quiet"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    if fetched.returncode != 0:
        print(f"  Upstream fetch failed: {fetched.stderr.strip()[:100]}")
        return None

    diff = subprocess.run(
        ["git", "diff", "HEAD", "FETCH_HEAD", "--name-only", "--", "records/", "README.md"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    changed = diff.stdout.strip()
    if not changed:
        print("  No new upstream changes.")
        return None

    readme_diff = subprocess.run(
        ["git", "diff", "HEAD", "FETCH_HEAD", "--", "README.md"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )

    new_scores = []
    for line in readme_diff.stdout.splitlines():
        if line.startswith("+") and "| 1." in line:
            match = re.search(r"\|\s*(1\.\d{4})\s*\|", line)
            if match:
                new_scores.append(float(match.group(1)))

    if new_scores:
        best_new = min(new_scores)
        summary = f"Upstream has new records! Best new score: {best_new:.4f}"
        print(f"  {summary}")
        local_sota = current_public_sota_bpb()
        if best_new < local_sota:
            print(f"  WARNING: New public SOTA ({best_new:.4f}) beats our baseline ({local_sota:.4f})!")
            print("  Consider pulling latest: git pull https://github.com/openai/parameter-golf.git main")
        return summary

    summary = f"Upstream has changes in: {changed[:200]}"
    print(f"  {summary}")
    return summary


# ---------------------------------------------------------------------------
# Results Schema Helpers
# ---------------------------------------------------------------------------

def read_results_header() -> list[str]:
    if not RESULTS_FILE.exists():
        return RESULTS_COLUMNS
    with open(RESULTS_FILE, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if not rows:
        return RESULTS_COLUMNS
    return rows[0]


def empty_result_row() -> dict[str, str]:
    return {column: "" for column in RESULTS_COLUMNS}


def clean_field(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "N/A":
        return ""
    return text.replace("\t", " ").replace("\n", " ").strip()


def clean_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def clean_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(int(value))


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = clean_field(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = clean_field(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_proposal_family(raw_family: str | None, fallback_text: str = "") -> str:
    text = clean_field(raw_family).lower()
    if not text:
        text = fallback_text.lower()
    text = text.replace("/", " ").replace("-", " ").replace("_", " ")
    canonical = {
        "infrastructure": "infrastructure",
        "quantization compression": "quantization_compression",
        "quantization": "quantization_compression",
        "compression": "quantization_compression",
        "optimizer schedule": "optimizer_schedule",
        "optimizer": "optimizer_schedule",
        "activation mlp": "activation_mlp",
        "activation": "activation_mlp",
        "mlp": "activation_mlp",
        "attention kv": "attention_kv",
        "attention": "attention_kv",
        "architecture depth": "architecture_depth",
        "architecture": "architecture_depth",
        "depth": "architecture_depth",
        "embedding tokenizer": "embedding_tokenizer",
        "embedding": "embedding_tokenizer",
        "tokenizer": "embedding_tokenizer",
        "evaluation strategy": "evaluation_strategy",
        "evaluation": "evaluation_strategy",
        "systems throughput": "systems_throughput",
        "systems": "systems_throughput",
        "throughput": "systems_throughput",
        "initialization norm": "initialization_norm",
        "initialization": "initialization_norm",
        "norm": "initialization_norm",
    }
    if text in canonical:
        return canonical[text]
    for family, keywords in FAMILY_ALIASES:
        if any(keyword in text for keyword in keywords):
            return family
    return "unknown"


def family_label(family: str) -> str:
    labels = {
        "quantization_compression": "Quantization / Compression",
        "activation_mlp": "Activation / MLP",
        "optimizer_schedule": "Optimizer / Schedule",
        "attention_kv": "Attention / KV",
        "architecture_depth": "Architecture / Depth",
        "embedding_tokenizer": "Embedding / Tokenizer",
        "evaluation_strategy": "Evaluation Strategy",
        "systems_throughput": "Systems / Throughput",
        "initialization_norm": "Initialization / Norm",
        "infrastructure": "Infrastructure",
        "unknown": "Unknown",
    }
    return labels.get(family, family.replace("_", " ").title())


def infer_timeout_stage_from_reasoning(reasoning: str) -> str:
    text = reasoning.lower()
    if "stage=compile" in text or " compile " in f" {text} ":
        return "compile"
    if "stage=post_export" in text or "serialization complete" in text:
        return "post_export"
    if "stage=eval" in text or "final eval" in text or "eval_time" in text:
        return "eval"
    if "stage=train" in text or "last_step=" in text or "train_time" in text or "timeout" in text:
        return "train"
    return ""


def load_results_rows() -> list[dict[str, str]]:
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE, newline="", encoding="utf-8") as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    if not table:
        return []

    header = table[0]
    raw_rows = table[1:]
    rows: list[dict[str, str]] = []

    if header == LEGACY_RESULTS_COLUMNS:
        for raw in raw_rows:
            raw = raw + [""] * (len(header) - len(raw))
            old_row = dict(zip(header, raw))
            row = empty_result_row()
            row["iteration"] = clean_field(old_row.get("iteration"))
            row["timestamp"] = clean_field(old_row.get("timestamp"))
            row["proposal_family"] = normalize_proposal_family(
                None, f"{old_row.get('description', '')} {old_row.get('reasoning', '')}"
            )
            row["mode"] = "legacy"
            row["status"] = clean_field(old_row.get("status"))
            row["val_bpb"] = clean_field(old_row.get("val_bpb"))
            row["artifact_bytes"] = clean_field(old_row.get("artifact_bytes"))
            row["description"] = clean_field(old_row.get("description"))
            row["reasoning"] = clean_field(old_row.get("reasoning"))
            row["timeout_stage"] = ""
            row["parse_status"] = "legacy_import"
            rows.append(row)
        return rows

    for raw in raw_rows:
        padded = raw + [""] * (len(header) - len(raw))
        row_in = dict(zip(header, padded))
        row = empty_result_row()
        for column in RESULTS_COLUMNS:
            if column in row_in:
                row[column] = clean_field(row_in.get(column))
        row["proposal_family"] = normalize_proposal_family(
            row.get("proposal_family"),
            f"{row.get('description', '')} {row.get('reasoning', '')}",
        )
        if not row["mode"]:
            row["mode"] = "legacy"
        if row["mode"] == "legacy" and row.get("parse_status") == "legacy_import":
            row["timeout_stage"] = ""
        rows.append(row)
    return rows


def write_results_rows(rows: list[dict[str, str]]) -> None:
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULTS_COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            normalized = empty_result_row()
            for column in RESULTS_COLUMNS:
                normalized[column] = clean_field(row.get(column, ""))
            writer.writerow(normalized)


def append_result_row(row: dict[str, str]) -> list[dict[str, str]]:
    rows = load_results_rows()
    normalized = empty_result_row()
    for column in RESULTS_COLUMNS:
        normalized[column] = clean_field(row.get(column, ""))
    rows.append(normalized)
    write_results_rows(rows)
    return rows


def update_last_result_fields(**fields: str) -> list[dict[str, str]]:
    rows = load_results_rows()
    if not rows:
        return rows
    for key, value in fields.items():
        if key in RESULTS_COLUMNS:
            rows[-1][key] = clean_field(value)
    write_results_rows(rows)
    return rows


def get_best_bpb(rows: list[dict[str, str]] | None = None, mode: str | None = None) -> float:
    if mode is not None:
        return current_mode_best_bpb(mode)
    best = current_public_sota_bpb()
    if rows is None:
        rows = load_results_rows()
    for row in rows:
        if row.get("status") not in {"baseline", "keep"}:
            continue
        value = parse_float(row.get("val_bpb"))
        if value is not None:
            best = min(best, value)
    return best


def get_iteration_count(rows: list[dict[str, str]] | None = None) -> int:
    if rows is None:
        rows = load_results_rows()
    if not rows:
        return 0
    iterations = [parse_int(row.get("iteration")) for row in rows]
    iterations = [value for value in iterations if value is not None]
    return max(iterations) if iterations else len(rows)


def load_incumbent_states() -> dict[str, dict[str, str]]:
    data = read_json_file(INCUMBENT_STATE_FILE, {})
    return data if isinstance(data, dict) else {}


def get_mode_incumbent(mode: str) -> dict[str, str] | None:
    states = load_incumbent_states()
    raw = states.get(mode)
    if not isinstance(raw, dict):
        return None
    value = parse_float(str(raw.get("val_bpb", "")))
    if value is None:
        return None
    incumbent = {key: clean_field(val) for key, val in raw.items()}
    incumbent["val_bpb"] = clean_float(value)
    return incumbent


def save_mode_incumbent(
    mode: str,
    iteration: int,
    val_bpb: float,
    description: str,
    proposal_family: str,
    source_status: str,
    metrics: dict[str, object] | None = None,
) -> dict[str, str]:
    metrics = metrics or {}
    states = load_incumbent_states()
    states[mode] = {
        "iteration": clean_int(iteration),
        "val_bpb": clean_float(val_bpb),
        "description": clean_field(description),
        "proposal_family": normalize_proposal_family(proposal_family, description),
        "source_status": clean_field(source_status),
        "last_train_time_ms": clean_int(parse_int(clean_field(metrics.get("last_train_time_ms")))),
        "last_eval_time_ms": clean_int(parse_int(clean_field(metrics.get("last_eval_time_ms")))),
        "updated_at": datetime.now().isoformat(),
    }
    write_json_file(INCUMBENT_STATE_FILE, states)
    return states[mode]


def clear_mode_incumbent(mode: str) -> None:
    states = load_incumbent_states()
    if mode in states:
        states.pop(mode, None)
        write_json_file(INCUMBENT_STATE_FILE, states)


def load_frontier_state(mode: str | None = None) -> dict[str, str] | None:
    raw = read_json_file(FRONTIER_STATE_FILE, {"active": False})
    if not isinstance(raw, dict) or not raw.get("active"):
        return None
    frontier = {key: clean_field(val) for key, val in raw.items()}
    if mode and frontier.get("mode") != mode:
        return None
    remaining = parse_int(frontier.get("remaining_steps"))
    if remaining is None or remaining <= 0:
        return None
    if not FRONTIER_SCRIPT_PATH.exists():
        return None
    frontier["remaining_steps"] = clean_int(remaining)
    frontier["branch_depth"] = clean_int(parse_int(frontier.get("branch_depth")) or 0)
    frontier["val_bpb"] = clean_float(parse_float(frontier.get("val_bpb")))
    frontier["gap_to_incumbent"] = clean_float(parse_float(frontier.get("gap_to_incumbent")))
    return frontier


def save_frontier_state(
    *,
    mode: str,
    iteration: int,
    val_bpb: float,
    incumbent_bpb: float,
    proposal_family: str,
    description: str,
    base_kind: str,
    remaining_steps: int,
    branch_depth: int,
) -> dict[str, str]:
    state = {
        "active": True,
        "mode": mode,
        "source_iteration": clean_int(iteration),
        "val_bpb": clean_float(val_bpb),
        "gap_to_incumbent": clean_float(val_bpb - incumbent_bpb),
        "proposal_family": normalize_proposal_family(proposal_family, description),
        "description": clean_field(description),
        "base_kind": clean_field(base_kind),
        "remaining_steps": clean_int(max(remaining_steps, 0)),
        "branch_depth": clean_int(max(branch_depth, 0)),
        "updated_at": datetime.now().isoformat(),
    }
    write_json_file(FRONTIER_STATE_FILE, state)
    shutil.copy2(SCRIPT_PATH, FRONTIER_SCRIPT_PATH)
    return {key: clean_field(val) for key, val in state.items()}


def clear_frontier_state() -> None:
    write_json_file(
        FRONTIER_STATE_FILE,
        {
            "active": False,
            "updated_at": datetime.now().isoformat(),
        },
    )
    if BEST_SCRIPT_PATH.exists():
        shutil.copy2(BEST_SCRIPT_PATH, FRONTIER_SCRIPT_PATH)


def select_base_state(mode: str) -> tuple[Path, str, dict[str, str] | None]:
    frontier = load_frontier_state(mode)
    if frontier is not None:
        return FRONTIER_SCRIPT_PATH, "frontier", frontier
    return BEST_SCRIPT_PATH, "incumbent", None


def sync_script_to_active_base(mode: str) -> None:
    base_path, _, _ = select_base_state(mode)
    if base_path.exists():
        shutil.copy2(base_path, SCRIPT_PATH)


def current_mode_best_bpb(mode: str) -> float:
    incumbent = get_mode_incumbent(mode)
    if incumbent is not None:
        value = parse_float(incumbent.get("val_bpb"))
        if value is not None:
            return value
    return current_public_sota_bpb()


def current_mode_anchor_text(mode: str) -> str:
    incumbent = get_mode_incumbent(mode)
    if incumbent is not None:
        value = incumbent.get("val_bpb", "?")
        return f"{value} (local incumbent)"
    return f"{current_public_sota_bpb():.4f} (public SOTA anchor; local incumbent not yet benchmarked)"


def is_near_miss(val_bpb: float, incumbent_bpb: float) -> bool:
    gap = val_bpb - incumbent_bpb
    return 0.0 < gap <= NEAR_MISS_MAX_GAP


def frontier_follow_on_allowed(
    *,
    base_kind: str,
    frontier_state: dict[str, str] | None,
    val_bpb: float,
    incumbent_bpb: float,
) -> bool:
    if not is_near_miss(val_bpb, incumbent_bpb):
        return False
    if base_kind != "frontier" or frontier_state is None:
        return True
    prior_frontier_bpb = parse_float(frontier_state.get("val_bpb"))
    if prior_frontier_bpb is None:
        return True
    return val_bpb < prior_frontier_bpb


def build_search_state(mode: str, base_kind: str, incumbent: dict[str, str] | None, frontier: dict[str, str] | None) -> str:
    lines = [f"- Current run mode: `{mode}`"]
    if incumbent is not None:
        lines.append(
            f"- Local incumbent benchmark for `{mode}`: {incumbent.get('val_bpb','?')} "
            f"(iteration #{incumbent.get('iteration','?')}, status={incumbent.get('source_status','?')})."
        )
    else:
        lines.append("- Local incumbent benchmark for this mode is not established yet.")

    if base_kind == "frontier" and frontier is not None:
        lines.append(
            f"- Base script for this iteration: exploratory frontier from iteration #{frontier.get('source_iteration','?')} "
            f"with val_bpb={frontier.get('val_bpb','?')} "
            f"(gap +{frontier.get('gap_to_incumbent','?')} vs local incumbent, "
            f"remaining continuation steps={frontier.get('remaining_steps','?')}, "
            f"branch_depth={frontier.get('branch_depth','?')})."
        )
        lines.append("- Frontier rule: either find a coherent follow-on that reduces the gap, or the loop will fall back to the incumbent.")
    else:
        lines.append("- Base script for this iteration: incumbent best-known script.")
        lines.append("- Exploration rule: a near-miss may earn a short continuation budget, but the incumbent remains the safety anchor.")
    return "\n".join(lines)


def is_infrastructure_row(row: dict[str, str]) -> bool:
    return row.get("status", "") in INFRA_FAILURE_STATUSES


def is_research_row(row: dict[str, str]) -> bool:
    return row.get("status", "") in (RESEARCH_FAILURE_STATUSES | RESEARCH_SUCCESS_STATUSES)


def detect_repeat_family_guard(rows: list[dict[str, str]], mode: str) -> dict[str, str] | None:
    mode_research = [row for row in rows if row.get("mode") == mode and is_research_row(row)]
    if not mode_research:
        return None

    contiguous_tail: list[dict[str, str]] = []
    for row in reversed(mode_research):
        status = row.get("status")
        if status in RESEARCH_SUCCESS_STATUSES:
            break
        if status in RESEARCH_NONKEEP_STATUSES:
            contiguous_tail.append(row)

    if len(contiguous_tail) < REPEAT_FAMILY_THRESHOLD:
        return None

    tail = list(reversed(contiguous_tail[:REPEAT_FAMILY_THRESHOLD]))
    families = {row.get("proposal_family", "unknown") for row in tail}
    families.discard("")
    if len(families) != 1:
        return None
    family = next(iter(families))
    if family == "unknown":
        return None
    iterations = ", ".join(f"#{row.get('iteration', '?')}" for row in tail)
    return {
        "family": family,
        "label": family_label(family),
        "iterations": iterations,
        "message": (
            f"Active repeat guard: the last {REPEAT_FAMILY_THRESHOLD} non-kept research outcomes "
            f"in `{mode}` mode since the last keep/baseline ({iterations}) were all "
            f"{family_label(family)}. The next proposal must use a "
            "different proposal_family unless it includes repeat_exemption explaining a new "
            "mechanism and why the prior failures do not apply."
        ),
    }


def build_memory_dossier(rows: list[dict[str, str]], mode: str) -> str:
    total = len(rows)
    mode_rows = [row for row in rows if row.get("mode") == mode]
    infra_rows = [row for row in mode_rows if is_infrastructure_row(row)]
    research_rows = [row for row in mode_rows if is_research_row(row)]
    repeat_guard = detect_repeat_family_guard(rows, mode)
    incumbent = get_mode_incumbent(mode)
    frontier = load_frontier_state(mode)

    lines = [
        "# AutoEvolve Memory Dossier",
        "",
        f"- Mode: `{mode}`",
        f"- Committed experiment rows (all modes): {total}",
        f"- Committed experiment rows for `{mode}`: {len(mode_rows)}",
        f"- Infrastructure failures for `{mode}`: {len(infra_rows)}",
        f"- Research outcomes for `{mode}`: {len(research_rows)}",
        "",
        "## Read This First",
    ]

    if incumbent is not None:
        lines.append(f"- Local incumbent val_bpb for this mode: {incumbent.get('val_bpb','?')}")
        lines.append(
            f"- The current local incumbent for `{mode}` is from iteration #{incumbent.get('iteration','?')} "
            f"with val_bpb={incumbent.get('val_bpb','?')} ({incumbent.get('source_status','?')})."
        )
    else:
        lines.append("- Local incumbent val_bpb for this mode: not yet benchmarked")
        lines.append("- There is no local incumbent benchmark for this mode yet.")

    if frontier is not None:
        lines.append(
            f"- There is an active exploration frontier from iteration #{frontier.get('source_iteration','?')} "
            f"with val_bpb={frontier.get('val_bpb','?')} and gap +{frontier.get('gap_to_incumbent','?')} "
            f"vs incumbent; remaining continuation steps={frontier.get('remaining_steps','?')}."
        )
    else:
        lines.append("- There is no active exploration frontier right now.")

    other_mode_rows = total - len(mode_rows)
    if other_mode_rows > 0:
        lines.append(
            f"- There are {other_mode_rows} committed rows from other run modes; treat them as secondary context, not the primary anchor for this mode."
        )

    if infra_rows:
        infra_counts = Counter(row.get("status", "unknown") for row in infra_rows)
        lines.append(
            "- Most historical rows were infrastructure/protocol failures. Do not treat them as model evidence."
        )
        lines.append(
            "- Infrastructure status counts: "
            + ", ".join(f"{status}={count}" for status, count in sorted(infra_counts.items()))
        )
    else:
        lines.append("- There are no infrastructure-only failures in the committed history.")

    if research_rows:
        lines.append(
            f"- Research memory for `{mode}` should focus on family, timing, throughput risk, and failure stage."
        )
        if any(
            row.get("mode") == "legacy" and row.get("parse_status") == "legacy_import"
            for row in research_rows
        ):
            lines.append(
                "- Legacy imported crash rows came from timeout stubs without partial stdout, so timeout stage and timing telemetry are genuinely unknown."
            )
    else:
        lines.append("- There are no committed research outcomes yet.")

    lines.extend(["", "## Research Family Summary"])
    if research_rows:
        family_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in research_rows:
            family_groups[row.get("proposal_family", "unknown")].append(row)
        for family, family_rows in sorted(family_groups.items()):
            latest = family_rows[-1]
            statuses = Counter(row.get("status", "unknown") for row in family_rows)
            timing = latest.get("last_train_time_ms") or ""
            stage = latest.get("timeout_stage") or ""
            suffix = []
            if timing:
                suffix.append(f"last_train={timing}ms")
            if stage:
                suffix.append(f"stage={stage}")
            suffix_text = f" ({', '.join(suffix)})" if suffix else ""
            lines.append(
                f"- {family_label(family)}: {len(family_rows)} rows; "
                + ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
                + suffix_text
            )
    else:
        lines.append("- No research families recorded yet.")

    lines.extend(["", "## Recent Research Outcomes"])
    recent_research = research_rows[-8:]
    if recent_research:
        for row in recent_research:
            val_bpb = row.get("val_bpb") or "n/a"
            total_bytes = row.get("total_bytes") or row.get("artifact_bytes") or "n/a"
            stage = row.get("timeout_stage") or "-"
            lines.append(
                f"- #{row.get('iteration', '?')} [{row.get('status', '?')}] "
                f"{family_label(row.get('proposal_family', 'unknown'))} | "
                f"bpb={val_bpb} | total_bytes={total_bytes} | stage={stage} | "
                f"{row.get('description', '')[:140]}"
            )
    else:
        lines.append("- No research outcomes yet.")

    lines.extend(["", "## Infrastructure Failures"])
    recent_infra = infra_rows[-5:]
    if recent_infra:
        for row in recent_infra:
            lines.append(
                f"- #{row.get('iteration', '?')} [{row.get('status', '?')}] "
                f"{row.get('description', '')[:160]}"
            )
    else:
        lines.append("- None.")

    lines.extend(["", "## Active Guard"])
    if repeat_guard:
        lines.append(f"- {repeat_guard['message']}")
    else:
        lines.append("- No active repeat-family guard.")

    lines.extend(["", "## Mode Heuristics"])
    if mode == MODE_SCOUT:
        lines.append(
            "- Proxy mode should rank ideas by likely transfer to the official 8xH100 / 600s setting, not by cheap single-GPU shortcuts."
        )
        lines.append(
            "- Preserve official-like evaluation behavior in proxy runs so local wins are more meaningful when promoted."
        )
        lines.append(
            "- Proxy mode is still single-GPU, so some throughput and eval timing noise remains; promote meaningful local wins to final validation."
        )
    else:
        lines.append("- Final mode should confirm only the strongest proxy-mode ideas under the exact competition budget.")

    return "\n".join(lines).strip() + "\n"


def write_memory_dossier(rows: list[dict[str, str]], mode: str) -> None:
    DOSSIER_FILE.write_text(build_memory_dossier(rows, mode), encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM Prompting
# ---------------------------------------------------------------------------

def load_prompt_template() -> str:
    agent_file = PROMPTS_DIR / "agent.md"
    if agent_file.exists():
        return agent_file.read_text(encoding="utf-8")
    return (
        "You are an ML researcher competing in the OpenAI Parameter Golf challenge.\n"
        "Current local incumbent val_bpb for this run mode: **{{best_bpb}}**\n"
        "Leaderboard SOTA: **{{leaderboard_sota}}**\n\n"
        "## RUN MODE\n{{run_mode}}\n\n"
        "## SEARCH STATE\n{{search_state}}\n\n"
        "## CURRENT TRAINING SCRIPT\n```python\n{{current_code}}\n```\n\n"
        "## MEMORY DOSSIER\n{{history}}\n\n{{repeat_guard}}\n\n## DOMAIN KNOWLEDGE\n{{program}}\n\n"
        "Return JSON with diagnosis, hypothesis, proposal_family, expected_delta, risk_assessment, "
        "description, optional repeat_exemption, and changes."
    )


def load_program() -> str:
    program_file = PROMPTS_DIR / "program.md"
    if program_file.exists():
        return program_file.read_text(encoding="utf-8")
    return ""


def build_prompt(
    current_code: str,
    dossier: str,
    best_bpb: float,
    iteration: int,
    mode: str,
    repeat_guard: dict[str, str] | None,
    search_state: str,
) -> str:
    template = load_prompt_template()
    program = load_program()
    repeat_guard_text = repeat_guard["message"] if repeat_guard else "No active repeat-family guard."
    leaderboard_sota = current_public_sota_bpb()
    return (
        template.replace("{{best_bpb}}", f"{best_bpb:.4f}")
        .replace("{{leaderboard_sota}}", f"{leaderboard_sota:.4f}")
        .replace("{{current_code}}", current_code)
        .replace("{{history}}", dossier)
        .replace("{{program}}", program)
        .replace("{{iteration}}", str(iteration))
        .replace("{{run_mode}}", mode_prompt_guidance(mode))
        .replace("{{search_state}}", search_state)
        .replace("{{repeat_guard}}", repeat_guard_text)
    )


def propose_modification(client, model: str, prompt: str) -> tuple[dict, dict[str, int | float]]:
    """Call the LLM to get a proposed modification plus usage metadata."""
    print(f"  Calling {model} for proposal...")
    t0 = time.time()

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=[{"role": "user", "content": prompt}],
        text={"format": {"type": "json_object"}},
        max_output_tokens=32000,
        truncation="auto",
    )

    elapsed = time.time() - t0
    text = response.output_text
    usage = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    if usage:
        prompt_tokens = int(getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0)) or 0)
        completion_tokens = int(getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0)) or 0)
        print(f"  LLM responded in {elapsed:.1f}s (input={prompt_tokens}, output={completion_tokens} tokens)")
    else:
        print(f"  LLM responded in {elapsed:.1f}s")

    try:
        proposal = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse JSON from LLM response (first 500 chars): {text[:500]}")
        proposal = json.loads(match.group(0))

    llm_meta = {
        "llm_seconds": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    return proposal, llm_meta


# ---------------------------------------------------------------------------
# Code Validation & Application
# ---------------------------------------------------------------------------

def apply_search_replace(code: str, search: str, replace: str) -> tuple[str | None, str]:
    """Apply a single search/replace block. Returns (new_code, error)."""
    if search in code:
        count = code.count(search)
        if count > 1:
            return None, f"Search string matched {count} times (must be unique)"
        return code.replace(search, replace, 1), ""

    def strip_trailing(text: str) -> str:
        return "\n".join(line.rstrip() for line in text.split("\n"))

    code_stripped = strip_trailing(code)
    search_stripped = strip_trailing(search)
    if search_stripped in code_stripped:
        return code_stripped.replace(search_stripped, strip_trailing(replace), 1), ""

    first_line = search.strip().split("\n")[0][:80]
    return None, f"Search string not found. First line: '{first_line}'"


def validate_competition_constraints(code: str) -> tuple[bool, str]:
    """Verify that the modified script still respects competition rules."""
    errors: list[str] = []

    match = re.search(r"max_wallclock_seconds\s*=\s*.*?(\d+\.?\d*)", code)
    if not match:
        errors.append("max_wallclock_seconds not found — training time is unbounded")
    elif float(match.group(1)) > MAX_WALLCLOCK:
        errors.append(f"max_wallclock_seconds={match.group(1)} exceeds {MAX_WALLCLOCK}s limit")

    if "final_int8_zlib_roundtrip" not in code:
        errors.append("Missing 'final_int8_zlib_roundtrip' output — cannot parse val_bpb")
    if "Total submission size" not in code:
        errors.append("Missing 'Total submission size' output — cannot parse artifact bytes")
    if all(marker not in code for marker in ("zlib.compress", "zstandard", "ZstdCompressor", "lzma.compress")):
        errors.append("No compression path (zlib/zstd/lzma) found — artifact won't be compressed")
    if "val_bpb" not in code:
        errors.append("No val_bpb computation found — cannot evaluate")
    if "Code size:" not in code or "Total submission size" not in code:
        errors.append("Artifact size reporting removed — cannot verify 16MB limit")

    if errors:
        return False, "; ".join(errors)
    return True, ""


def apply_proposal(current_code: str, proposal: dict) -> tuple[str | None, str]:
    """Apply proposed changes to the current code. Returns (new_code, error)."""
    changes = proposal.get("changes", [])
    if not changes:
        return None, "No changes provided in proposal"

    code = current_code
    for i, change in enumerate(changes):
        search = change.get("search", "")
        replace = change.get("replace", "")
        if not search:
            return None, f"Change {i + 1}: empty search string"

        result, err = apply_search_replace(code, search, replace)
        if result is None:
            return None, f"Change {i + 1}/{len(changes)}: {err}"
        code = result

    try:
        ast.parse(code)
    except SyntaxError as err:
        return None, f"SyntaxError after applying changes at line {err.lineno}: {err.msg}"

    for token in ("def main()", "class Hyperparameters"):
        if token not in code:
            return None, f"Missing required component after changes: {token}"

    ok, err = validate_competition_constraints(code)
    if not ok:
        return None, f"Competition constraint violation: {err}"

    return code, ""


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def build_log_file_path(iteration: int, suffix: str = ".log") -> Path:
    return LOGS_DIR / f"exp_{iteration:04d}{suffix}"


def read_text_maybe_bytes(blob: str | bytes | None) -> str:
    if blob is None:
        return ""
    if isinstance(blob, bytes):
        return blob.decode("utf-8", errors="replace")
    return blob


def run_experiment(
    script_path: Path,
    nproc: int,
    mode: str,
    iteration: int,
) -> dict[str, object]:
    env, run_id = build_run_env(mode, iteration)
    process_timeout = mode_process_timeout_seconds(mode)
    train_log_path = ROOT_TRAIN_LOGS_DIR / f"{run_id}.txt"

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(script_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    print(
        "  Runner config: "
        f"mode={mode}, train_wallclock={env['MAX_WALLCLOCK_SECONDS']}s, "
        f"process_timeout={process_timeout}s, run_id={run_id}"
    )
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=process_timeout,
            env=env,
            cwd=ROOT,
        )
        elapsed = time.time() - t0
        output = read_text_maybe_bytes(result.stdout)
        print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")
        return {
            "output": output,
            "exit_code": result.returncode,
            "elapsed_s": elapsed,
            "timed_out": False,
            "timeout_s": process_timeout,
            "mode": mode,
            "run_id": run_id,
            "train_log_path": train_log_path if train_log_path.exists() else None,
        }
    except subprocess.TimeoutExpired as err:
        elapsed = time.time() - t0
        partial = read_text_maybe_bytes(err.stdout)
        timeout_msg = (
            f"\nTIMEOUT: Experiment exceeded {process_timeout}s outer limit "
            f"(runner mode: {mode}, train wallclock cap: {env['MAX_WALLCLOCK_SECONDS']}s)"
        )
        return {
            "output": (partial or "") + timeout_msg,
            "exit_code": 124,
            "elapsed_s": elapsed,
            "timed_out": True,
            "timeout_s": process_timeout,
            "mode": mode,
            "run_id": run_id,
            "train_log_path": train_log_path if train_log_path.exists() else None,
        }


def load_run_analysis_output(run_result: dict[str, object]) -> tuple[str, str]:
    """Return (analysis_output, captured_output). Prefer the train log when present."""
    captured_output = read_text_maybe_bytes(run_result.get("output"))
    analysis_chunks: list[str] = []

    train_log_path = run_result.get("train_log_path")
    if isinstance(train_log_path, Path) and train_log_path.exists():
        train_log_text = train_log_path.read_text(encoding="utf-8", errors="replace")
        analysis_chunks.append(train_log_text)

    if captured_output:
        analysis_chunks.append(captured_output)

    analysis_output = "\n\n--- RUNNER OUTPUT ---\n\n".join(chunk for chunk in analysis_chunks if chunk)
    return analysis_output, captured_output


def save_experiment_logs(iteration: int, run_result: dict[str, object], analysis_output: str) -> None:
    log_path = build_log_file_path(iteration)
    fallback_output = read_text_maybe_bytes(run_result.get("output"))
    log_path.write_text(analysis_output or fallback_output, encoding="utf-8")

    train_log_path = run_result.get("train_log_path")
    if isinstance(train_log_path, Path) and train_log_path.exists():
        mirrored = build_log_file_path(iteration, "_train.txt")
        shutil.copy2(train_log_path, mirrored)


def infer_timeout_stage(output: str, metrics: dict[str, object]) -> str:
    if metrics.get("final_metric_present"):
        return "complete"
    if metrics.get("saw_final_eval_mode") or metrics.get("saw_final_eval"):
        return "eval"
    if metrics.get("saw_serialization"):
        return "post_export"
    if metrics.get("last_step") is not None or "warmup_step:" in output:
        return "train"
    if metrics.get("saw_setup"):
        return "compile"
    return "launch"


def parse_experiment_output(output: str) -> dict[str, object]:
    """Parse metrics from training output and preserve partial progress."""
    metrics: dict[str, object] = {
        "val_bpb": None,
        "val_loss": None,
        "last_val_bpb": None,
        "last_val_loss": None,
        "artifact_bytes": None,
        "total_bytes": None,
        "last_train_time_ms": None,
        "last_eval_time_ms": None,
        "peak_memory_mib": None,
        "last_step": None,
        "last_total_steps": None,
        "final_metric_present": False,
        "saw_setup": False,
        "saw_serialization": False,
        "saw_final_eval": False,
        "saw_final_eval_mode": False,
    }
    final_candidates: dict[str, dict[str, object]] = {}

    def record_final_candidate(kind: str, line: str) -> None:
        loss_match = re.search(r"val_loss:([0-9.]+)", line)
        bpb_match = re.search(r"val_bpb:([0-9.]+)", line)
        if not loss_match and not bpb_match:
            return

        precision_rank = 1 if "_exact" in line else 0
        previous = final_candidates.get(kind)
        previous_precision = int(previous.get("precision_rank", -1)) if previous else -1
        if precision_rank < previous_precision:
            return

        candidate = dict(previous or {})
        candidate["precision_rank"] = precision_rank
        if loss_match:
            candidate["val_loss"] = float(loss_match.group(1))
        if bpb_match:
            candidate["val_bpb"] = float(bpb_match.group(1))
        eval_match = re.search(r"eval_time:(\d+)ms", line)
        if eval_match:
            candidate["eval_time_ms"] = int(eval_match.group(1))
        final_candidates[kind] = candidate

    for line in output.splitlines():
        if not line:
            continue

        if any(
            marker in line
            for marker in (
                "Running Python",
                "Running PyTorch",
                "model_params:",
                "train_loader:",
                "val_loader:",
                "attention_mode:",
            )
        ):
            metrics["saw_setup"] = True

        step_match = re.search(r"step:(\d+)/(\d+)", line)
        if step_match:
            metrics["last_step"] = int(step_match.group(1))
            metrics["last_total_steps"] = int(step_match.group(2))

        if "final_eval_mode:" in line:
            metrics["saw_final_eval_mode"] = True

        if "Serialized model" in line or "Total submission size" in line or "peak memory allocated:" in line:
            metrics["saw_serialization"] = True

        if any(
            marker in line
            for marker in (
                "DIAGNOSTIC post_ema",
                "final_int6_roundtrip",
                "final_int6_sliding_window",
                "final_int8_zlib_roundtrip",
                "legal_ttt",
                "ttt_sliding:start",
                "ttt_sliding:done",
            )
        ):
            metrics["saw_final_eval"] = True

        if line.startswith("final_int8_zlib_roundtrip"):
            record_final_candidate("roundtrip", line)

        if line.startswith("legal_ttt"):
            record_final_candidate("legal_ttt", line)

        if "val_loss:" in line and "val_bpb:" in line:
            match = re.search(r"val_bpb:([0-9.]+)", line)
            if match:
                metrics["last_val_bpb"] = float(match.group(1))
            match = re.search(r"val_loss:([0-9.]+)", line)
            if match:
                metrics["last_val_loss"] = float(match.group(1))

        if "Total submission size" in line:
            match = re.search(r"(\d+)\s*bytes", line)
            if match:
                metrics["total_bytes"] = int(match.group(1))

        if "Serialized model" in line and "bytes" in line:
            match = re.search(r"(\d+)\s*bytes", line)
            if match:
                metrics["artifact_bytes"] = int(match.group(1))

        if "train_time:" in line and "step:" in line:
            match = re.search(r"train_time:(\d+)ms", line)
            if match:
                metrics["last_train_time_ms"] = int(match.group(1))

        if "peak memory allocated:" in line:
            match = re.search(r"(\d+)\s*MiB", line)
            if match:
                metrics["peak_memory_mib"] = int(match.group(1))

    preferred_final = final_candidates.get("legal_ttt") or final_candidates.get("roundtrip")
    if preferred_final:
        metrics["final_metric_present"] = True
        metrics["val_bpb"] = preferred_final.get("val_bpb")
        metrics["val_loss"] = preferred_final.get("val_loss")
        eval_time_ms = preferred_final.get("eval_time_ms")
        if isinstance(eval_time_ms, int):
            metrics["last_eval_time_ms"] = eval_time_ms

    metrics["timeout_stage"] = infer_timeout_stage(output, metrics)
    return metrics


# ---------------------------------------------------------------------------
# Git Helpers
# ---------------------------------------------------------------------------

def git_current_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def git_commit(msg: str) -> str:
    """Stage all result files and commit. Returns HEAD after commit (or current HEAD if no changes)."""
    log_paths = sorted(str(path) for path in LOGS_DIR.glob("exp_*"))
    files_to_stage = [
        str(SCRIPT_PATH),
        str(BEST_SCRIPT_PATH),
        str(FRONTIER_SCRIPT_PATH),
        str(RESULTS_FILE),
        str(DOSSIER_FILE),
        str(INCUMBENT_STATE_FILE),
        str(FRONTIER_STATE_FILE),
    ] + log_paths
    add_result = subprocess.run(
        ["git", "add"] + files_to_stage,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if add_result.returncode != 0:
        details = clean_field(add_result.stderr or add_result.stdout)[:240]
        raise RuntimeError(f"git add failed: {details}")

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
    )
    if staged.returncode == 0:
        return git_current_head()

    commit_result = subprocess.run(
        ["git", "commit", "-m", msg],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if commit_result.returncode != 0:
        details = clean_field(commit_result.stderr or commit_result.stdout)[:240]
        raise RuntimeError(f"git commit failed: {details}")
    return git_current_head()


def git_push(branch: str) -> None:
    r = subprocess.run(
        ["git", "push", "origin", branch, "--set-upstream", "--quiet"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
    )
    if r.returncode == 0:
        print("  Pushed to origin.")
    else:
        err = (r.stderr or r.stdout).strip()[:120]
        print(f"  Push failed (non-fatal): {err}")


def persist_outcome(
    branch: str,
    mode: str,
    commit_message: str,
    annotation_message: str,
) -> str:
    """Persist result files, then write the primary outcome commit SHA back into the last row."""
    rows = load_results_rows()
    write_memory_dossier(rows, mode)
    outcome_sha = git_commit(commit_message)

    rows = update_last_result_fields(commit_sha=outcome_sha)
    write_memory_dossier(rows, mode)
    final_sha = git_commit(annotation_message)
    git_push(branch)
    return final_sha or outcome_sha


# ---------------------------------------------------------------------------
# Outcome Row Construction
# ---------------------------------------------------------------------------

def build_result_row(
    *,
    iteration: int,
    mode: str,
    proposal_family: str,
    status: str,
    description: str,
    reasoning: str,
    llm_meta: dict[str, int | float] | None = None,
    metrics: dict[str, object] | None = None,
    parse_status: str = "",
    timeout_stage: str = "",
) -> dict[str, str]:
    llm_meta = llm_meta or {}
    metrics = metrics or {}
    row = empty_result_row()
    row["iteration"] = str(iteration)
    row["timestamp"] = datetime.now().isoformat()
    row["proposal_family"] = normalize_proposal_family(proposal_family, f"{description} {reasoning}")
    row["mode"] = mode
    row["status"] = status

    final_bpb = parse_float(clean_field(metrics.get("val_bpb")))
    partial_bpb = parse_float(clean_field(metrics.get("last_val_bpb")))
    row["val_bpb"] = clean_float(final_bpb if final_bpb is not None else partial_bpb)
    row["artifact_bytes"] = clean_int(parse_int(clean_field(metrics.get("artifact_bytes"))))
    row["total_bytes"] = clean_int(parse_int(clean_field(metrics.get("total_bytes"))))
    row["description"] = clean_field(description)
    row["reasoning"] = clean_field(reasoning)
    row["llm_seconds"] = clean_float(llm_meta.get("llm_seconds"), digits=1)
    row["prompt_tokens"] = clean_int(parse_int(clean_field(llm_meta.get("prompt_tokens"))))
    row["completion_tokens"] = clean_int(parse_int(clean_field(llm_meta.get("completion_tokens"))))
    row["timeout_stage"] = clean_field(timeout_stage or clean_field(metrics.get("timeout_stage")))
    row["last_step"] = clean_int(parse_int(clean_field(metrics.get("last_step"))))
    row["last_train_time_ms"] = clean_int(parse_int(clean_field(metrics.get("last_train_time_ms"))))
    row["last_eval_time_ms"] = clean_int(parse_int(clean_field(metrics.get("last_eval_time_ms"))))
    row["parse_status"] = clean_field(parse_status)
    row["commit_sha"] = ""
    return row


def summarize_failure_reason(
    base_reasoning: str,
    run_result: dict[str, object] | None,
    metrics: dict[str, object] | None,
    extra: str = "",
) -> str:
    metrics = metrics or {}
    parts = [clean_field(base_reasoning)]
    if run_result is not None:
        if run_result.get("timed_out"):
            parts.append(
                f"TIMEOUT stage={metrics.get('timeout_stage', '')} "
                f"elapsed={int(float(run_result.get('elapsed_s', 0.0)))}s"
            )
        elif run_result.get("exit_code", 0) != 0:
            parts.append(
                f"EXIT_CODE {run_result.get('exit_code')} stage={metrics.get('timeout_stage', '')}"
            )
    if metrics.get("last_step") is not None:
        parts.append(f"last_step={metrics['last_step']}")
    if metrics.get("last_train_time_ms") is not None:
        parts.append(f"last_train_time_ms={metrics['last_train_time_ms']}")
    if metrics.get("last_val_bpb") is not None:
        parts.append(f"last_val_bpb={metrics['last_val_bpb']:.4f}")
    if extra:
        parts.append(extra)
    return " | ".join(part for part in parts if part)


def benchmark_local_incumbent(mode: str, nproc: int, branch: str) -> dict[str, str]:
    incumbent = get_mode_incumbent(mode)
    if incumbent is not None:
        return incumbent

    rows = load_results_rows()
    baseline_iteration = get_iteration_count(rows) + 1
    description = f"Baseline incumbent benchmark for {mode} mode"
    reasoning = (
        "Measure the incumbent script under the current run mode before any mutations so "
        "future keep/near-miss/discard decisions compare against a real local anchor."
    )

    print(f"  No local incumbent benchmark for {mode} mode. Running baseline benchmark first...")
    shutil.copy2(BEST_SCRIPT_PATH, SCRIPT_PATH)
    run_result = run_experiment(BEST_SCRIPT_PATH, nproc, mode, baseline_iteration)
    analysis_output, _ = load_run_analysis_output(run_result)
    save_experiment_logs(baseline_iteration, run_result, analysis_output)
    metrics = parse_experiment_output(analysis_output)

    if run_result.get("timed_out") or int(run_result.get("exit_code", 0)) != 0:
        row = build_result_row(
            iteration=baseline_iteration,
            mode=mode,
            proposal_family="unknown",
            status="crash",
            description=description,
            reasoning=summarize_failure_reason(reasoning, run_result, metrics),
            metrics=metrics,
            parse_status="baseline_timeout" if run_result.get("timed_out") else "baseline_nonzero_exit",
            timeout_stage=clean_field(metrics.get("timeout_stage")),
        )
        append_result_row(row)
        persist_outcome(
            branch,
            mode,
            f"Iter {baseline_iteration}: baseline benchmark crash [{mode}]",
            f"Annotate iter {baseline_iteration} baseline crash metadata",
        )
        raise RuntimeError("Baseline incumbent benchmark failed before the search loop could start.")

    val_bpb = parse_float(clean_field(metrics.get("val_bpb")))
    if val_bpb is None:
        row = build_result_row(
            iteration=baseline_iteration,
            mode=mode,
            proposal_family="unknown",
            status="parse_error",
            description=description,
            reasoning=summarize_failure_reason(reasoning, run_result, metrics),
            metrics=metrics,
            parse_status="baseline_missing_final_metric",
            timeout_stage=clean_field(metrics.get("timeout_stage")),
        )
        append_result_row(row)
        persist_outcome(
            branch,
            mode,
            f"Iter {baseline_iteration}: baseline benchmark parse_error [{mode}]",
            f"Annotate iter {baseline_iteration} baseline parse metadata",
        )
        raise RuntimeError("Baseline incumbent benchmark completed without a parseable val_bpb.")

    total_bytes = parse_int(clean_field(metrics.get("total_bytes")))
    if total_bytes is not None and total_bytes > ARTIFACT_LIMIT:
        overage = total_bytes - ARTIFACT_LIMIT
        row = build_result_row(
            iteration=baseline_iteration,
            mode=mode,
            proposal_family="unknown",
            status="over_size",
            description=description,
            reasoning=f"{reasoning} | OVER_SIZE: {total_bytes} bytes (+{overage})",
            metrics=metrics,
            parse_status="baseline_over_size",
        )
        append_result_row(row)
        persist_outcome(
            branch,
            mode,
            f"Iter {baseline_iteration}: baseline benchmark over_size [{mode}]",
            f"Annotate iter {baseline_iteration} baseline over_size metadata",
        )
        raise RuntimeError("Baseline incumbent benchmark exceeded the artifact limit.")

    row = build_result_row(
        iteration=baseline_iteration,
        mode=mode,
        proposal_family="unknown",
        status="baseline",
        description=description,
        reasoning=reasoning,
        metrics=metrics,
        parse_status="baseline_seed",
    )
    append_result_row(row)
    save_mode_incumbent(
        mode,
        baseline_iteration,
        val_bpb,
        description,
        "unknown",
        "baseline",
        metrics=metrics,
    )
    clear_frontier_state()
    sync_script_to_active_base(mode)
    persist_outcome(
        branch,
        mode,
        f"Iter {baseline_iteration}: baseline benchmark [{mode}]",
        f"Annotate iter {baseline_iteration} baseline metadata",
    )
    incumbent = get_mode_incumbent(mode)
    if incumbent is None:
        raise RuntimeError("Failed to persist local incumbent benchmark state.")
    return incumbent


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    resolved_mode = resolve_mode(args.nproc)
    train_wallclock = mode_train_wallclock_seconds(resolved_mode)
    process_timeout = mode_process_timeout_seconds(resolved_mode)

    def record_row(row: dict[str, str]) -> list[dict[str, str]]:
        if args.dry_run:
            return load_results_rows()
        return append_result_row(row)

    print("=" * 70)
    print("  Parameter Golf Auto-Evolve Agent")
    print(
        f"  Model: {args.model} | GPUs: {args.nproc} | Dry-run: {args.dry_run} | "
        f"Mode: {resolved_mode}"
    )
    print(
        f"  Constraints: code must preserve <= {MAX_WALLCLOCK}s competition limit | "
        f"runner train wallclock={train_wallclock}s | outer timeout={process_timeout}s"
    )
    print("=" * 70)

    client = setup_openai()
    setup_workspace(resolved_mode)
    if not args.dry_run:
        setup_git(DEFAULT_BRANCH)
    branch = DEFAULT_BRANCH

    rows = load_results_rows()
    if not args.dry_run:
        try:
            benchmark_local_incumbent(resolved_mode, args.nproc, branch)
        except RuntimeError as err:
            print(f"ERROR: {err}")
            sys.exit(1)
        rows = load_results_rows()
    best_bpb = current_mode_best_bpb(resolved_mode)
    iteration = get_iteration_count(rows)
    consecutive_failures = 0
    iters_this_session = 0

    print(f"Starting from iteration {iteration + 1}, anchor BPB: {current_mode_anchor_text(resolved_mode)}\n")

    while True:
        if args.max_iters > 0 and iters_this_session >= args.max_iters:
            print(f"\nReached max iterations ({args.max_iters}). Stopping.")
            break
        iteration += 1
        iters_this_session += 1

        print(f"\n{'=' * 70}")
        print(f"  ITERATION {iteration}  |  Anchor BPB: {current_mode_anchor_text(resolved_mode)}  |  Mode: {resolved_mode}")
        print(f"{'=' * 70}")

        if iteration % UPSTREAM_SYNC_EVERY == 1:
            try:
                sync_upstream()
            except Exception as err:
                print(f"  Sync error (non-fatal): {err}")

        rows = load_results_rows()
        write_memory_dossier(rows, resolved_mode)
        repeat_guard = detect_repeat_family_guard(rows, resolved_mode)
        best_bpb = current_mode_best_bpb(resolved_mode)
        incumbent_state = get_mode_incumbent(resolved_mode)
        base_path, base_kind, frontier_state = select_base_state(resolved_mode)
        if base_path.exists():
            shutil.copy2(base_path, SCRIPT_PATH)

        current_code = SCRIPT_PATH.read_text(encoding="utf-8")
        dossier = DOSSIER_FILE.read_text(encoding="utf-8")
        search_state = build_search_state(resolved_mode, base_kind, incumbent_state, frontier_state)
        prompt = build_prompt(current_code, dossier, best_bpb, iteration, resolved_mode, repeat_guard, search_state)

        if base_kind == "frontier" and frontier_state is not None:
            print(
                "  Base:       frontier "
                f"(iter #{frontier_state.get('source_iteration','?')}, "
                f"bpb={frontier_state.get('val_bpb','?')}, "
                f"remaining={frontier_state.get('remaining_steps','?')})"
            )
        else:
            print("  Base:       incumbent")

        try:
            proposal, llm_meta = propose_modification(client, args.model, prompt)
        except Exception as err:
            print(f"  LLM ERROR: {err}")
            traceback.print_exc()
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family="infrastructure",
                status="llm_error",
                description=str(err)[:200],
                reasoning="LLM request failed before any code mutation.",
                parse_status="llm_error",
            )
            record_row(row)
            if not args.dry_run:
                persist_outcome(
                    branch,
                    resolved_mode,
                    f"Iter {iteration}: llm_error",
                    f"Annotate iter {iteration} llm_error metadata",
                )
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            time.sleep(5)
            continue

        description = clean_field(proposal.get("description") or "Unknown change")[:200]
        diagnosis = clean_field(proposal.get("diagnosis") or "")[:400]
        hypothesis = clean_field(proposal.get("hypothesis") or "")[:400]
        expected = clean_field(proposal.get("expected_delta") or "?")[:200]
        risk = clean_field(proposal.get("risk_assessment") or proposal.get("risk") or "?")[:300]
        repeat_exemption = clean_field(proposal.get("repeat_exemption") or "")
        proposal_family = normalize_proposal_family(
            proposal.get("proposal_family"),
            " ".join(
                [
                    description,
                    diagnosis,
                    hypothesis,
                    json.dumps(proposal.get("changes", []))[:1200],
                ]
            ),
        )
        reasoning = (
            f"Family: {family_label(proposal_family)} | Diagnosis: {diagnosis} | "
            f"Hypothesis: {hypothesis} | Expected: {expected}"
        )
        reasoning += f" | Base: {base_kind}"
        if repeat_exemption:
            reasoning += f" | Repeat exemption: {repeat_exemption}"

        print(f"  Proposed:   {description}")
        print(f"  Family:     {family_label(proposal_family)}")
        if diagnosis:
            print(f"  Diagnosis:  {diagnosis}")
        if hypothesis:
            print(f"  Hypothesis: {hypothesis}")
        print(f"  Expected:   {expected}")
        print(f"  Risk:       {risk}")

        if repeat_guard and proposal_family == repeat_guard["family"] and len(repeat_exemption) < 60:
            err = (
                f"Blocked repeated family {family_label(proposal_family)} without a concrete repeat_exemption. "
                f"{repeat_guard['message']}"
            )
            print(f"  INVALID: {err}")
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="invalid",
                description=f"{description} | repeat-family guard",
                reasoning=f"{reasoning} | {err}",
                llm_meta=llm_meta,
                parse_status="repeat_guard_blocked",
            )
            record_row(row)
            if not args.dry_run:
                persist_outcome(
                    branch,
                    resolved_mode,
                    f"Iter {iteration}: invalid repeated family [{proposal_family}]",
                    f"Annotate iter {iteration} invalid metadata",
                )
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            continue

        new_code, err = apply_proposal(current_code, proposal)
        if new_code is None:
            print(f"  INVALID: {err}")
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="invalid",
                description=f"{description} | {err}",
                reasoning=reasoning,
                llm_meta=llm_meta,
                parse_status="proposal_invalid",
            )
            record_row(row)
            if not args.dry_run:
                persist_outcome(
                    branch,
                    resolved_mode,
                    f"Iter {iteration}: invalid proposal [{proposal_family}]",
                    f"Annotate iter {iteration} invalid metadata",
                )
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            continue

        backup_code = current_code
        SCRIPT_PATH.write_text(new_code, encoding="utf-8")
        n_lines = len(new_code.strip().split("\n"))
        print(f"  Applied ({len(new_code):,} bytes, {n_lines} lines)")

        if args.dry_run:
            print("  [DRY RUN] Skipping training. Change saved for review.")
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="dry_run",
                description=description,
                reasoning=reasoning,
                llm_meta=llm_meta,
                parse_status="dry_run",
            )
            SCRIPT_PATH.write_text(backup_code, encoding="utf-8")
            print("  [DRY RUN] No persistent state was updated.")
            consecutive_failures = 0
            continue

        print("  Running training experiment...")
        run_result = run_experiment(
            SCRIPT_PATH,
            args.nproc,
            resolved_mode,
            iteration,
        )
        analysis_output, captured_output = load_run_analysis_output(run_result)
        save_experiment_logs(iteration, run_result, analysis_output)
        print(f"  Log saved to {build_log_file_path(iteration)}")

        metrics = parse_experiment_output(analysis_output)

        if run_result.get("timed_out") or int(run_result.get("exit_code", 0)) != 0:
            print(f"  CRASH (exit code {run_result.get('exit_code')})")
            tail = analysis_output.strip().splitlines()[-15:]
            for line in tail:
                print(f"    {line}")
            if base_kind == "frontier":
                clear_frontier_state()
            sync_script_to_active_base(resolved_mode)

            failure_reason = summarize_failure_reason(
                reasoning,
                run_result,
                metrics,
                extra=f"runner_output_tail={clean_field(' | '.join(tail[-5:]))[:240]}",
            )
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="crash",
                description=description,
                reasoning=failure_reason,
                llm_meta=llm_meta,
                metrics=metrics,
                parse_status="timeout" if run_result.get("timed_out") else "nonzero_exit",
                timeout_stage=clean_field(metrics.get("timeout_stage")),
            )
            record_row(row)
            persist_outcome(
                branch,
                resolved_mode,
                f"Iter {iteration}: crash [{proposal_family}] {description[:72]}",
                f"Annotate iter {iteration} crash metadata",
            )
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            continue

        val_bpb = metrics.get("val_bpb")
        total_bytes = metrics.get("total_bytes")

        if val_bpb is None:
            print("  Could not parse final val_bpb from output")
            if base_kind == "frontier":
                clear_frontier_state()
            sync_script_to_active_base(resolved_mode)
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="parse_error",
                description=description,
                reasoning=summarize_failure_reason(reasoning, run_result, metrics),
                llm_meta=llm_meta,
                metrics=metrics,
                parse_status="missing_final_metric",
                timeout_stage=clean_field(metrics.get("timeout_stage")),
            )
            record_row(row)
            persist_outcome(
                branch,
                resolved_mode,
                f"Iter {iteration}: parse_error [{proposal_family}]",
                f"Annotate iter {iteration} parse_error metadata",
            )
            consecutive_failures += 1
            continue

        print(f"  val_bpb:  {float(val_bpb):.4f}")
        print(f"  total:    {int(total_bytes):,} bytes" if total_bytes else "  total:    unknown")
        if metrics.get("last_train_time_ms") is not None:
            train_s = int(metrics["last_train_time_ms"]) / 1000
            print(f"  train:    {train_s:.1f}s")
        if metrics.get("last_eval_time_ms") is not None:
            eval_s = int(metrics["last_eval_time_ms"]) / 1000
            print(f"  eval:     {eval_s:.1f}s")
        if metrics.get("peak_memory_mib") is not None:
            print(f"  memory:   {metrics['peak_memory_mib']} MiB")

        if total_bytes and int(total_bytes) > ARTIFACT_LIMIT:
            overage = int(total_bytes) - ARTIFACT_LIMIT
            print(f"  OVER SIZE ({int(total_bytes):,} > {ARTIFACT_LIMIT:,}, +{overage:,} bytes)")
            if base_kind == "frontier":
                clear_frontier_state()
            sync_script_to_active_base(resolved_mode)
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="over_size",
                description=description,
                reasoning=f"{reasoning} | OVER_SIZE: {total_bytes} bytes (+{overage})",
                llm_meta=llm_meta,
                metrics=metrics,
                parse_status="scored_over_size",
            )
            record_row(row)
            persist_outcome(
                branch,
                resolved_mode,
                f"Iter {iteration}: over_size [{proposal_family}]",
                f"Annotate iter {iteration} over_size metadata",
            )
            consecutive_failures += 1
            continue

        incumbent_state = get_mode_incumbent(resolved_mode)
        incumbent_bpb = best_bpb if incumbent_state is None else parse_float(incumbent_state.get("val_bpb")) or best_bpb

        if float(val_bpb) < incumbent_bpb:
            delta = incumbent_bpb - float(val_bpb)
            print(
                f"  >>> INCUMBENT IMPROVEMENT: {incumbent_bpb:.4f} -> {float(val_bpb):.4f} "
                f"(delta: -{delta:.4f}) <<<"
            )
            shutil.copy2(SCRIPT_PATH, BEST_SCRIPT_PATH)
            save_mode_incumbent(
                resolved_mode,
                iteration,
                float(val_bpb),
                description,
                proposal_family,
                "keep",
                metrics=metrics,
            )
            clear_frontier_state()
            sync_script_to_active_base(resolved_mode)
            best_bpb = current_mode_best_bpb(resolved_mode)
            row = build_result_row(
                iteration=iteration,
                mode=resolved_mode,
                proposal_family=proposal_family,
                status="keep",
                description=description,
                reasoning=f"{reasoning} | incumbent_delta=-{delta:.4f}",
                llm_meta=llm_meta,
                metrics=metrics,
                parse_status="scored_keep",
            )
            record_row(row)
            persist_outcome(
                branch,
                resolved_mode,
                f"Iter {iteration}: keep [{proposal_family}] {description[:72]}",
                f"Annotate iter {iteration} keep metadata",
            )
            consecutive_failures = 0
        else:
            gap = float(val_bpb) - incumbent_bpb
            near_miss_allowed = frontier_follow_on_allowed(
                base_kind=base_kind,
                frontier_state=frontier_state,
                val_bpb=float(val_bpb),
                incumbent_bpb=incumbent_bpb,
            )
            if near_miss_allowed:
                if base_kind == "frontier" and frontier_state is not None:
                    remaining_steps = max((parse_int(frontier_state.get("remaining_steps")) or 1) - 1, 0)
                    branch_depth = (parse_int(frontier_state.get("branch_depth")) or 0) + 1
                else:
                    remaining_steps = FRONTIER_CONTINUATION_STEPS
                    branch_depth = 1
                frontier_reasoning = f"{reasoning} | near_miss_gap=+{gap:.4f}"
                if remaining_steps > 0:
                    save_frontier_state(
                        mode=resolved_mode,
                        iteration=iteration,
                        val_bpb=float(val_bpb),
                        incumbent_bpb=incumbent_bpb,
                        proposal_family=proposal_family,
                        description=description,
                        base_kind=base_kind,
                        remaining_steps=remaining_steps,
                        branch_depth=branch_depth,
                    )
                    print(
                        f"  Near miss (+{gap:.4f} vs incumbent); "
                        f"saved as active frontier with {remaining_steps} continuation steps."
                    )
                else:
                    clear_frontier_state()
                    print(
                        f"  Near miss (+{gap:.4f} vs incumbent), but frontier budget is exhausted. "
                        "Returning to incumbent."
                    )
                sync_script_to_active_base(resolved_mode)
                row = build_result_row(
                    iteration=iteration,
                    mode=resolved_mode,
                    proposal_family=proposal_family,
                    status="near_miss",
                    description=description,
                    reasoning=frontier_reasoning,
                    llm_meta=llm_meta,
                    metrics=metrics,
                    parse_status="scored_near_miss",
                )
                record_row(row)
                persist_outcome(
                    branch,
                    resolved_mode,
                    f"Iter {iteration}: near_miss [{proposal_family}] {description[:72]}",
                    f"Annotate iter {iteration} near_miss metadata",
                )
                consecutive_failures = 0
            else:
                print(f"  No incumbent improvement ({float(val_bpb):.4f} >= {incumbent_bpb:.4f})")
                if base_kind == "frontier":
                    clear_frontier_state()
                sync_script_to_active_base(resolved_mode)
                row = build_result_row(
                    iteration=iteration,
                    mode=resolved_mode,
                    proposal_family=proposal_family,
                    status="discard",
                    description=description,
                    reasoning=f"{reasoning} | incumbent_gap=+{gap:.4f}",
                    llm_meta=llm_meta,
                    metrics=metrics,
                    parse_status="scored_discard",
                )
                record_row(row)
                persist_outcome(
                    branch,
                    resolved_mode,
                    f"Iter {iteration}: discard [{proposal_family}] {description[:72]}",
                    f"Annotate iter {iteration} discard metadata",
                )
                consecutive_failures = 0

        time.sleep(2)

    print(f"\n{'=' * 70}")
    print(f"  Auto-evolve finished after {iteration} iterations")
    print(f"  Anchor BPB:    {current_mode_anchor_text(resolved_mode)}")
    print(f"  Best script:  {BEST_SCRIPT_PATH}")
    print(f"  Results log:  {RESULTS_FILE}")
    print(f"  Dossier:      {DOSSIER_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
