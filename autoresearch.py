"""
Autoresearch: autonomous ML research loop for parameter-golf.

Uses Claude Code (`claude -p`) as the research agent — it reads train_gpt.py,
makes surgical edits, and returns a description of the change. Then we run
training, evaluate, and keep or revert.

Pipelined: while training experiment N, speculatively proposes experiment N+1.
If N is reverted (most common), the speculative proposal is already ready.
If N is kept, the speculative proposal is discarded and re-proposed.

Usage:
    # Download data first (one-time):
    python3 data/cached_challenge_fineweb.py --variant sp1024

    # Run the loop (single GPU, 3-min experiments):
    python3 autoresearch.py

    # Customize:
    EXPERIMENT_SECONDS=120 MAX_EXPERIMENTS=50 GPUS=1 python3 autoresearch.py

    # Multi-GPU:
    GPUS=8 EXPERIMENT_SECONDS=600 python3 autoresearch.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# ---------------------
# Configuration
# ---------------------

TRAIN_SCRIPT = Path("train_gpt.py")
PROGRAM_FILE = Path("program.md")

EXPERIMENT_SECONDS = int(os.environ.get("EXPERIMENT_SECONDS", "180"))
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "100"))
GPUS = int(os.environ.get("GPUS", "1"))
CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "high")
LANE = os.environ.get("AUTORESEARCH_LANE", "core").strip().lower()
STAGE = os.environ.get("AUTORESEARCH_STAGE", "discovery").strip().lower()
NAMESPACE = os.environ.get("AUTORESEARCH_NAMESPACE", "").strip()
PROPOSAL_TIMEOUT_SECONDS = int(
    os.environ.get("PROPOSAL_TIMEOUT_SECONDS", "240" if STAGE == "discovery" else "420")
)
TRAIN_TIMEOUT_PADDING_SECONDS = int(
    os.environ.get("TRAIN_TIMEOUT_PADDING_SECONDS", "300" if STAGE == "discovery" else "600")
)

# When iterating on fewer GPUs or shorter times, reduce iterations
# so the model doesn't waste time in warmdown too early
EXPERIMENT_ITERATIONS = int(os.environ.get("EXPERIMENT_ITERATIONS", "20000"))
VAL_LOSS_EVERY = int(os.environ.get("VAL_LOSS_EVERY", "0"))  # 0 = only at end

# Baseline BPB to seed the first comparison (from the official baseline run)
BASELINE_BPB = float(os.environ.get("BASELINE_BPB", "0"))
MAX_ARTIFACT_BYTES = int(os.environ.get("MAX_ARTIFACT_BYTES", "16000000"))
MAX_EVAL_TIME_MS = int(os.environ.get("MAX_EVAL_TIME_MS", "120000"))
MAX_EVAL_MEMORY_MIB = int(os.environ.get("MAX_EVAL_MEMORY_MIB", "0"))
MAX_QUANTIZATION_GAP = float(os.environ.get("MAX_QUANTIZATION_GAP", "0.08"))
STORAGE_MAX_REGRESSION = float(os.environ.get("STORAGE_MAX_REGRESSION", "0.003"))
STORAGE_MIN_SIZE_IMPROVEMENT = int(os.environ.get("STORAGE_MIN_SIZE_IMPROVEMENT", "250000"))
REPRESENTATION_VERIFIED = bool(int(os.environ.get("REPRESENTATION_VERIFIED", "0")))

AUTORESEARCH_DIR = Path("autoresearch") / NAMESPACE if NAMESPACE else Path("autoresearch")
TRAIN_SCRIPT_BEST = AUTORESEARCH_DIR / "train_gpt.best.py"
HISTORY_FILE = AUTORESEARCH_DIR / "history.jsonl"
LOGS_DIR = AUTORESEARCH_DIR / "logs"
EXPERIMENTS_DIR = AUTORESEARCH_DIR / "experiments"


@dataclass(frozen=True)
class LanePolicy:
    name: str
    objective: str
    stage_desc: str
    prompt_guidance: tuple[str, ...]


LANE_POLICIES: dict[str, LanePolicy] = {
    "core": LanePolicy(
        name="core",
        objective="Improve post-quantization val_bpb under the artifact cap without sacrificing training throughput.",
        stage_desc="Use the 1xH100 short-horizon proxy for discovery, then promote promising configs to longer and wider runs.",
        prompt_guidance=(
            "This lane is for core model, optimizer, and schedule changes close to the current training stack.",
            "Prefer changes that improve early convergence in the first ~300 steps and do not materially slow step time.",
            "The validated frontier is 12×448 dim, mlp_mult=2, matrix_lr=0.08, warmdown_iters=600 at 1.3508 BPB (15.8MB). Refine LOCALLY around this base.",
            "PRIORITY ORDER: (1) warmdown sweep around 600 (try 500/550/700), (2) narrow matrix_lr sweep around 0.08, (3) width nudges around 448, (4) KV-head reduction, (5) depth/width rebalance only if under cap.",
            "DO NOT TRY: scalar_lr changes, matrix_lr>=0.09, generic activation swaps, large multi-change jumps, or naive depth increases without size-recovery plan.",
            "Artifact headroom is very tight (~0.2MB). Any param increase risks going over 16MB. Prefer zero-size-risk schedule changes first.",
        ),
    ),
    "eval_time": LanePolicy(
        name="eval_time",
        objective="Trade evaluation compute for lower val_bpb while staying within a strict evaluation-time budget.",
        stage_desc="Treat evaluation latency and memory as first-class constraints, not just the training metric.",
        prompt_guidance=(
            "Post-quantization calibration is the TOP PRIORITY. Softcap calibration gave a 0.068 BPB win — this is the strongest eval-time lever found so far.",
            "PRIORITY ORDER: (1) skip_weight scale calibration, (2) per-head temperature or attention scaling, (3) LM-head / final-logit calibration variants, (4) confirm reproducibility with a control run.",
            "Track which calibration gains are ADDITIVE vs OVERLAPPING. If two calibrations target the same distortion, do not assume they stack.",
            "Do NOT make core-model architecture changes in this lane. Focus exclusively on eval-time compensation and calibration.",
            "Keep eval time well under the gate. Current best uses ~44.5s of 60s budget.",
        ),
    ),
    "representation": LanePolicy(
        name="representation",
        objective="Change tokenization or segmentation only when accounting and reproducibility remain exact.",
        stage_desc="Correctness comes before performance in this lane; invalid accounting disqualifies the result.",
        prompt_guidance=(
            "This lane is for tokenizer, segmentation, and byte/latent representation work.",
            "Do not chase tiny metric wins through unclear accounting. Exact byte accounting and dataset/tokenizer consistency are mandatory.",
            "Favor narrow, testable representation changes over wholesale pipeline rewrites.",
        ),
    ),
    "storage": LanePolicy(
        name="storage",
        objective="Optimize final compressed artifact bytes and post-export val_bpb together.",
        stage_desc="A discovery win can come from either lower post-export bpb or materially smaller compressed artifacts at near-equal bpb.",
        prompt_guidance=(
            "This lane is for quantization-aware training, codebooks, tying, and zlib/export-friendly parameterization.",
            "Focus on post-export behavior, not just raw training loss.",
            "Prefer changes that reduce quantization gap or shrink final bytes without hurting bpb much.",
        ),
    ),
}


def ensure_dirs():
    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    EXPERIMENTS_DIR.mkdir(exist_ok=True)


def save_experiment_snapshot(
    experiment_id: int,
    description: str,
    entry: dict,
    code: str,
    training_output: str | None = None,
):
    """Save a versioned snapshot of each experiment: code, rationale, and results."""
    exp_dir = EXPERIMENTS_DIR / f"{experiment_id:04d}"
    exp_dir.mkdir(exist_ok=True)

    # Save the code that was tested
    (exp_dir / "train_gpt.py").write_text(code)

    # Save training log if available
    if training_output:
        (exp_dir / "train.log").write_text(training_output)

    # Write a rationale/results markdown file
    kept_str = "KEPT" if entry.get("kept") else "REVERTED"
    bpb = entry.get("val_bpb")
    bpb_str = f"{bpb:.4f}" if bpb is not None else "N/A (failed)"
    size = entry.get("artifact_bytes")
    size_str = f"{size:,}" if size else "N/A"
    model_params = entry.get("model_params")
    prequant_bpb = entry.get("prequant_val_bpb")
    q_gap = entry.get("quantization_gap")
    last_step = entry.get("last_step")
    error = entry.get("error", "")

    md = f"""# Experiment {experiment_id}

**Date:** {entry.get('timestamp', 'unknown')}
**Lane/Stage:** {entry.get('lane', LANE)}/{entry.get('stage', STAGE)}
**Result:** {kept_str}
**val_bpb:** {bpb_str}
**Artifact size:** {size_str} bytes
**Model params:** {model_params if model_params else 'N/A'}
**Last step:** {last_step if last_step is not None else 'N/A'}
**Pre-quant val_bpb:** {f'{prequant_bpb:.4f}' if prequant_bpb is not None else 'N/A'}
**Quantization gap:** {f'{q_gap:.4f}' if q_gap is not None else 'N/A'}
**Eval time:** {entry.get('eval_time_ms', 'N/A')} ms
**Peak memory:** {entry.get('peak_memory_mib', 'N/A')} MiB
**Gate reason:** {entry.get('gate_reason', 'N/A')}
**Propose time:** {entry.get('propose_seconds', '?')}s
**Train time:** {entry.get('train_seconds', '?')}s
"""
    if error:
        md += f"**Error:** {error}\n"

    md += f"""
## Change
{description}

## Diff from previous best
"""
    # Include a simple diff summary
    if TRAIN_SCRIPT_BEST.exists():
        best_code = TRAIN_SCRIPT_BEST.read_text()
        if code != best_code:
            best_lines = set(best_code.splitlines())
            new_lines = set(code.splitlines())
            added = len(new_lines - best_lines)
            removed = len(best_lines - new_lines)
            md += f"+{added} lines / -{removed} lines (vs current best)\n"
        else:
            md += "Identical to current best\n"
    else:
        md += "(no best to compare against yet)\n"

    (exp_dir / "README.md").write_text(md)


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def append_history(entry: dict):
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def parse_val_bpb(output: str) -> float | None:
    """Extract the final val_bpb from training output."""
    m = re.search(r"final_int8_zlib_roundtrip_exact val_loss:\S+ val_bpb:(\d+\.\d+)", output)
    if m:
        return float(m.group(1))
    matches = re.findall(r"val_bpb:(\d+\.\d+)", output)
    if matches:
        return float(matches[-1])
    return None


def parse_artifact_size(output: str) -> int | None:
    """Extract total submission size from output."""
    m = re.search(r"Total submission size int8\+zlib:\s*(\d+)\s*bytes", output)
    if m:
        return int(m.group(1))
    return None


def parse_model_params(output: str) -> int | None:
    """Extract model parameter count from training output."""
    m = re.search(r"model_params:(\d+)", output)
    if m:
        return int(m.group(1))
    return None


def parse_last_step_val_bpb(output: str) -> float | None:
    """Extract the final pre-quantization val_bpb from step logs."""
    matches = re.findall(r"step:\d+/\d+ val_loss:\S+ val_bpb:(\d+\.\d+)", output)
    if matches:
        return float(matches[-1])
    return None


def parse_last_step_index(output: str) -> int | None:
    """Extract the last completed training step from output."""
    matches = re.findall(r"step:(\d+)/\d+", output)
    if matches:
        return int(matches[-1])
    return None


def parse_eval_time_ms(output: str) -> int | None:
    """Extract final evaluation time in milliseconds."""
    m = re.search(r"final_int8_zlib_roundtrip val_loss:\S+ val_bpb:\S+ eval_time:(\d+)ms", output)
    if m:
        return int(m.group(1))
    return None


def parse_peak_memory_mib(output: str) -> int | None:
    """Extract peak memory usage in MiB."""
    m = re.search(r"peak memory allocated:\s*(\d+)\s*MiB", output)
    if m:
        return int(m.group(1))
    return None


def lane_policy() -> LanePolicy:
    if LANE not in LANE_POLICIES:
        valid = ", ".join(sorted(LANE_POLICIES))
        raise ValueError(f"Unknown AUTORESEARCH_LANE={LANE!r}. Expected one of: {valid}")
    return LANE_POLICIES[LANE]


def stage_config() -> str:
    return f"{LANE}/{STAGE}"


def kept_entries(history: list[dict]) -> list[dict]:
    entries = []
    for h in history:
        entry_lane = h.get("lane")
        lane_matches = entry_lane == LANE or (entry_lane is None and LANE == "core" and not NAMESPACE)
        if lane_matches and h.get("kept") and h.get("val_bpb") is not None:
            entries.append(h)
    return entries


def best_entry_for_lane(history: list[dict]) -> dict | None:
    entries = kept_entries(history)
    if not entries:
        return None
    return min(entries, key=lambda h: h["val_bpb"])


def keep_decision(
    *,
    val_bpb: float | None,
    artifact_bytes: int | None,
    eval_time_ms: int | None,
    peak_memory_mib: int | None,
    quantization_gap: float | None,
    best_entry: dict | None,
    over_budget: bool,
) -> tuple[bool, str]:
    """Return whether to keep a result under the active lane policy."""
    if over_budget:
        return False, f"artifact_over_budget ({artifact_bytes} > {MAX_ARTIFACT_BYTES})"
    if val_bpb is None:
        return False, "missing_val_bpb"

    best_bpb = best_entry["val_bpb"] if best_entry is not None else None

    if LANE == "representation" and not REPRESENTATION_VERIFIED:
        return False, "representation_requires_verification"

    if LANE == "eval_time":
        if eval_time_ms is None:
            return False, "missing_eval_time"
        if eval_time_ms > MAX_EVAL_TIME_MS:
            return False, f"eval_time_exceeded ({eval_time_ms}ms > {MAX_EVAL_TIME_MS}ms)"
        if MAX_EVAL_MEMORY_MIB > 0 and peak_memory_mib is not None and peak_memory_mib > MAX_EVAL_MEMORY_MIB:
            return False, f"eval_memory_exceeded ({peak_memory_mib}MiB > {MAX_EVAL_MEMORY_MIB}MiB)"

    if LANE == "storage":
        if quantization_gap is not None and quantization_gap > MAX_QUANTIZATION_GAP:
            return False, f"quantization_gap_exceeded ({quantization_gap:.4f} > {MAX_QUANTIZATION_GAP:.4f})"
        if best_entry is None:
            return True, "first_successful_storage_result"
        best_size = best_entry.get("artifact_bytes")
        if val_bpb < best_bpb:
            return True, f"improved_val_bpb ({best_bpb:.4f} -> {val_bpb:.4f})"
        if (
            best_size is not None
            and artifact_bytes is not None
            and val_bpb <= best_bpb + STORAGE_MAX_REGRESSION
            and artifact_bytes <= best_size - STORAGE_MIN_SIZE_IMPROVEMENT
        ):
            return True, (
                "accepted_storage_tradeoff "
                f"(bpb {best_bpb:.4f}->{val_bpb:.4f}, size {best_size}->{artifact_bytes})"
            )
        return False, "no_storage_improvement"

    if best_bpb is None or val_bpb < best_bpb:
        return True, f"improved_val_bpb ({best_bpb if best_bpb is not None else 'none'} -> {val_bpb:.4f})"
    return False, f"no_val_bpb_improvement (best={best_bpb:.4f}, got={val_bpb:.4f})"


def summarize_current_config(code: str) -> str:
    """Summarize the current default config from train_gpt.py for the prompt."""
    def extract_int(name: str) -> str:
        m = re.search(rf'{name} = int\(os\.environ\.get\("[^"]+", (\d+)\)\)', code)
        return m.group(1) if m else "?"

    def extract_float(name: str) -> str:
        m = re.search(rf'{name} = float\(os\.environ\.get\("[^"]+", ([0-9.]+)\)\)', code)
        return m.group(1) if m else "?"

    return (
        f"layers={extract_int('num_layers')}, dim={extract_int('model_dim')}, "
        f"heads={extract_int('num_heads')}, kv_heads={extract_int('num_kv_heads')}, "
        f"mlp_mult={extract_int('mlp_mult')}, matrix_lr={extract_float('matrix_lr')}, "
        f"scalar_lr={extract_float('scalar_lr')}"
    )


def format_history_for_prompt(history: list[dict], max_entries: int = 30) -> str:
    lane_history = [
        h for h in history
        if h.get("lane") == LANE or (h.get("lane") is None and LANE == "core" and not NAMESPACE)
    ]
    if not lane_history:
        return "No experiments run yet."

    recent = lane_history[-max_entries:]
    lines = []
    for h in recent:
        status = "KEPT" if h.get("kept") else "REVERTED"
        bpb = h.get("val_bpb")
        bpb_str = f"{bpb:.4f}" if bpb is not None else "FAILED"
        size = h.get("artifact_bytes")
        size_str = f" size={size}" if size else ""
        params = h.get("model_params")
        params_str = f" params={params}" if params else ""
        q_gap = h.get("quantization_gap")
        q_gap_str = f" qgap={q_gap:.4f}" if q_gap is not None else ""
        error = h.get("error", "")
        error_str = f" error={error}" if error else ""
        lines.append(
            f"  #{h['id']:3d} [{status:8s}] bpb={bpb_str}{size_str}{params_str}{q_gap_str}{error_str} | {h['description']}"
        )

    best_kept = kept_entries(history)
    best_bpb = min(h["val_bpb"] for h in best_kept) if best_kept else None

    summary = f"Experiments so far: {len(lane_history)} total, {len(best_kept)} kept\n"
    if best_bpb is not None:
        summary += f"Current best val_bpb: {best_bpb:.4f}\n"
    summary += "\nRecent experiments:\n" + "\n".join(lines)
    return summary


def build_proposal_prompt(program: str, history: list[dict], best_bpb: float | None) -> str:
    history_block = format_history_for_prompt(history)
    best_str = f"{best_bpb:.4f}" if best_bpb is not None else "unknown (no successful run yet)"
    current_config = summarize_current_config(TRAIN_SCRIPT.read_text())
    policy = lane_policy()
    lane_guidance = "\n".join(f"- {line}" for line in policy.prompt_guidance)

    return f"""You are an autonomous ML researcher running experiments for the parameter-golf challenge.
Your job: make ONE specific modification to train_gpt.py to try to improve validation BPB.

## Research Program
{program}

## Experiment History
{history_block}

Lane: {policy.name}
Stage: {STAGE}
Lane objective: {policy.objective}
Stage intent: {policy.stage_desc}
Current best val_bpb: {best_str}
Current default config in train_gpt.py: {current_config}
Experiment time budget: {EXPERIMENT_SECONDS}s on {GPUS} GPU(s)

## Instructions
1. Read train_gpt.py to understand the current state
2. Decide on ONE specific change to try (guided by the research program and history)
3. Edit train_gpt.py to make that change — use surgical edits, not full rewrites
4. After editing, read the changed lines back to VERIFY your edit was actually applied
5. Print a single line starting with "DESCRIPTION:" summarizing what you changed and why

Guidelines:
- Make exactly ONE conceptual change per experiment so we can isolate what helps
- Consider what has and hasn't worked in the experiment history
- The script must remain functional — don't break imports, class structure, etc.
- The final metric is post-int8-quantization roundtrip val_bpb (lower is better)
- Total artifact (compressed weights + code) must stay under {MAX_ARTIFACT_BYTES:,} bytes
- Be creative but grounded — vary your approaches, don't repeat failed ideas
- This is a 1xH100 short-horizon proxy. Prefer changes that improve convergence in the first ~300 steps and do not materially slow step time
- The current best is already close to the size ceiling, so avoid parameter increases unless you also have a concrete compression hypothesis
- Favor small shape/hyperparameter/training-schedule changes over speculative full-module rewrites
- Focus on changes likely to improve BPB: architecture, hyperparameters, training tricks, compression

Lane-specific guidance:
{lane_guidance}"""


def run_claude_proposal(prompt: str) -> str | None:
    """Run claude -p and return the description. Claude edits train_gpt.py in place."""
    try:
        result = subprocess.run(
            [
                "claude",
                "-p", prompt,
                "--model", CLAUDE_MODEL,
                "--effort", CLAUDE_EFFORT,
                "--allowedTools", "Read,Edit,Glob,Grep",
                "--output-format", "text",
            ],
            capture_output=True,
            text=True,
            timeout=PROPOSAL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None

    output = result.stdout + "\n" + result.stderr

    desc_match = re.search(r"DESCRIPTION:\s*(.+)", output)
    if not desc_match:
        return None

    desc = desc_match.group(1).strip()
    lowered = desc.lower()
    banned_fragments = [
        "manually restore",
        "/root/.claude",
        "backup",
        "auth login",
        "you can manually restore",
        "cp \"",
        "cp '",
    ]
    if any(fragment in lowered for fragment in banned_fragments):
        return None
    if len(desc) < 20:
        return None
    return desc


def run_training(experiment_id: int) -> tuple[str, int]:
    """Run train_gpt.py and return (output, returncode)."""
    env = os.environ.copy()
    env["MAX_WALLCLOCK_SECONDS"] = str(EXPERIMENT_SECONDS)
    env["ITERATIONS"] = str(EXPERIMENT_ITERATIONS)
    env["RUN_ID"] = f"autoresearch_{experiment_id}"
    if VAL_LOSS_EVERY > 0:
        env["VAL_LOSS_EVERY"] = str(VAL_LOSS_EVERY)

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={GPUS}",
        str(TRAIN_SCRIPT),
    ]

    timeout = EXPERIMENT_SECONDS + TRAIN_TIMEOUT_PADDING_SECONDS

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Time budget: {EXPERIMENT_SECONDS}s, timeout: {timeout}s")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
        )
        output = result.stdout + "\n" + result.stderr
        return output, result.returncode
    except subprocess.TimeoutExpired as e:
        output = (e.stdout or "") + "\n" + (e.stderr or "")
        return output, -1


def evaluate_and_record(
    experiment_id: int,
    description: str,
    new_code: str,
    prev_code: str,
    output: str,
    returncode: int,
    propose_time: float,
    train_time: float,
    timestamp: str,
    best_bpb: float | None,
    history: list[dict],
) -> tuple[bool, float | None]:
    """Evaluate training results, keep or revert, return (kept, best_bpb)."""

    log_file = LOGS_DIR / f"experiment_{experiment_id:04d}.log"
    log_file.write_text(output)

    if returncode != 0 and returncode != -1:
        print(f"  FAILED — see {log_file}")
        lines = output.strip().split("\n")
        for line in lines[-10:]:
            print(f"    {line}")
        TRAIN_SCRIPT.write_text(prev_code)
        entry = {
            "id": experiment_id, "description": description,
            "val_bpb": None, "artifact_bytes": None, "kept": False,
            "lane": LANE, "stage": STAGE, "gate_reason": f"exit_{returncode}",
            "error": f"exit_{returncode}",
            "propose_seconds": round(propose_time, 1),
            "train_seconds": round(train_time, 1), "timestamp": timestamp,
        }
        append_history(entry)
        history.append(entry)
        save_experiment_snapshot(experiment_id, description, entry, new_code, output)
        return False, best_bpb

    if returncode == -1:
        print(f"  TIMED OUT — see {log_file}")
        TRAIN_SCRIPT.write_text(prev_code)
        entry = {
            "id": experiment_id, "description": description,
            "val_bpb": None, "artifact_bytes": None, "kept": False,
            "lane": LANE, "stage": STAGE, "gate_reason": "timeout",
            "error": "timeout",
            "propose_seconds": round(propose_time, 1),
            "train_seconds": round(train_time, 1), "timestamp": timestamp,
        }
        append_history(entry)
        history.append(entry)
        save_experiment_snapshot(experiment_id, description, entry, new_code, output)
        return False, best_bpb

    val_bpb = parse_val_bpb(output)
    artifact_bytes = parse_artifact_size(output)
    model_params = parse_model_params(output)
    prequant_val_bpb = parse_last_step_val_bpb(output)
    last_step = parse_last_step_index(output)
    eval_time_ms = parse_eval_time_ms(output)
    peak_memory_mib = parse_peak_memory_mib(output)
    quantization_gap = (
        val_bpb - prequant_val_bpb
        if val_bpb is not None and prequant_val_bpb is not None
        else None
    )

    print(f"  val_bpb:       {val_bpb}")
    print(f"  artifact size: {artifact_bytes}")
    print(f"  model params:  {model_params}")
    print(f"  last step:     {last_step}")
    print(f"  prequant_bpb:  {prequant_val_bpb}")
    print(f"  quant gap:     {quantization_gap}")
    print(f"  eval time ms:  {eval_time_ms}")
    print(f"  peak mem MiB:  {peak_memory_mib}")

    kept = False
    over_budget = artifact_bytes is not None and artifact_bytes > MAX_ARTIFACT_BYTES
    best_entry = best_entry_for_lane(history)
    decision_keep, gate_reason = keep_decision(
        val_bpb=val_bpb,
        artifact_bytes=artifact_bytes,
        eval_time_ms=eval_time_ms,
        peak_memory_mib=peak_memory_mib,
        quantization_gap=quantization_gap,
        best_entry=best_entry,
        over_budget=over_budget,
    )

    if decision_keep:
        prior_best = best_entry["val_bpb"] if best_entry is not None else None
        improvement = (prior_best - val_bpb) if prior_best is not None and val_bpb is not None else 0.0
        print(f"  KEPT by lane policy: {gate_reason}")
        if prior_best is not None and val_bpb is not None:
            print(f"  Improvement: {prior_best:.4f} -> {val_bpb:.4f} (Δ = {improvement:.4f})")
        best_bpb = val_bpb if val_bpb is not None else best_bpb
        shutil.copy2(TRAIN_SCRIPT, TRAIN_SCRIPT_BEST)
        kept = True
    else:
        print(f"  Reverting by lane policy: {gate_reason}")
        TRAIN_SCRIPT.write_text(prev_code)

    entry = {
        "id": experiment_id, "description": description,
        "val_bpb": val_bpb, "artifact_bytes": artifact_bytes,
        "model_params": model_params,
        "prequant_val_bpb": prequant_val_bpb,
        "quantization_gap": quantization_gap,
        "last_step": last_step,
        "eval_time_ms": eval_time_ms,
        "peak_memory_mib": peak_memory_mib,
        "lane": LANE,
        "stage": STAGE,
        "gate_reason": gate_reason,
        "kept": kept, "over_budget": over_budget,
        "propose_seconds": round(propose_time, 1),
        "train_seconds": round(train_time, 1), "timestamp": timestamp,
    }
    append_history(entry)
    history.append(entry)
    save_experiment_snapshot(experiment_id, description, entry, new_code, output)

    print(f"\n  Current best: {best_bpb:.4f}")
    return kept, best_bpb


def main():
    ensure_dirs()

    if not TRAIN_SCRIPT.exists():
        print(f"Error: {TRAIN_SCRIPT} not found. Run from the parameter-golf directory.")
        sys.exit(1)

    if not PROGRAM_FILE.exists():
        print(f"Error: {PROGRAM_FILE} not found.")
        sys.exit(1)

    try:
        subprocess.run(["claude", "--version"], capture_output=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: 'claude' CLI not found. Install Claude Code first:")
        print("  npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    program = PROGRAM_FILE.read_text()
    history = load_history()

    if not TRAIN_SCRIPT_BEST.exists():
        shutil.copy2(TRAIN_SCRIPT, TRAIN_SCRIPT_BEST)
    else:
        # On restarts, resume from the seeded/best code rather than whatever
        # happens to be at repo root. Claude always edits TRAIN_SCRIPT.
        best_code = TRAIN_SCRIPT_BEST.read_text()
        if TRAIN_SCRIPT.read_text() != best_code:
            TRAIN_SCRIPT.write_text(best_code)

    best = best_entry_for_lane(history)
    best_bpb: float | None = best["val_bpb"] if best is not None else None

    if best_bpb is None and BASELINE_BPB > 0:
        best_bpb = BASELINE_BPB

    experiment_id = max((h["id"] for h in history), default=0)

    print("=" * 70)
    print("AUTORESEARCH — Parameter Golf (Claude Code, pipelined)")
    print("=" * 70)
    print(f"  Lane/stage:      {stage_config()}")
    print(f"  Namespace dir:   {AUTORESEARCH_DIR}")
    print(f"  Best BPB so far:  {f'{best_bpb:.4f}' if best_bpb else 'none (starting fresh)'}")
    print(f"  Experiments done: {len(history)}")
    print(f"  Time per run:     {EXPERIMENT_SECONDS}s on {GPUS} GPU(s)")
    print(f"  Claude model:     {CLAUDE_MODEL} (effort: {CLAUDE_EFFORT})")
    print(f"  Max experiments:  {MAX_EXPERIMENTS}")
    print(f"  Mode:             pipelined (speculative proposals)")
    print("=" * 70)

    # Speculative proposal state
    speculative_description: str | None = None
    speculative_code: str | None = None  # the modified train_gpt.py from speculative proposal
    speculative_base_bpb: float | None = None  # best_bpb at time of speculation

    for i in range(MAX_EXPERIMENTS):
        experiment_id += 1
        timestamp = datetime.now(timezone.utc).isoformat()

        print(f"\n{'─' * 70}")
        print(f"EXPERIMENT {experiment_id}  ({i + 1}/{MAX_EXPERIMENTS})")
        print(f"{'─' * 70}")

        prev_code = TRAIN_SCRIPT.read_text()

        # --- PROPOSAL PHASE ---
        # Check if we have a valid speculative proposal
        use_speculative = (
            speculative_description is not None
            and speculative_code is not None
            and speculative_base_bpb == best_bpb  # still valid (best didn't change)
        )

        if use_speculative:
            description = speculative_description
            TRAIN_SCRIPT.write_text(speculative_code)
            propose_time = 0.0
            print(f"  Using speculative proposal (prepared during last training)")
            print(f"  → {description}")
            speculative_description = None
            speculative_code = None
            speculative_base_bpb = None
        else:
            if speculative_description is not None:
                print(f"  Discarding stale speculative proposal (best_bpb changed)")
                speculative_description = None
                speculative_code = None
                speculative_base_bpb = None

            print("Proposing modification (claude -p)...")
            t_propose = time.time()
            description = run_claude_proposal(
                build_proposal_prompt(program, history, best_bpb)
            )
            propose_time = time.time() - t_propose
            print(f"  Proposal took {propose_time:.1f}s")

            if description is None:
                print("  Failed to get modification. Skipping.")
                entry = {
                    "id": experiment_id, "description": "Failed to propose",
                    "val_bpb": None, "artifact_bytes": None, "kept": False,
                    "lane": LANE, "stage": STAGE, "gate_reason": "invalid_proposal",
                    "error": "invalid_proposal", "timestamp": timestamp,
                }
                append_history(entry)
                history.append(entry)
                continue

            print(f"  → {description}")

        # Check if the file actually changed
        new_code = TRAIN_SCRIPT.read_text()
        if new_code == prev_code:
            print("  No changes made to train_gpt.py. Skipping.")
            entry = {
                "id": experiment_id,
                "description": f"{description} (NO CHANGES)",
                "val_bpb": None, "artifact_bytes": None, "kept": False,
                "lane": LANE, "stage": STAGE, "gate_reason": "no_changes",
                "error": "no_changes", "timestamp": timestamp,
            }
            append_history(entry)
            history.append(entry)
            continue

        # --- TRAINING PHASE (with speculative proposal in parallel) ---
        print("Training...")
        t_train = time.time()

        # Start training as a subprocess (it loads train_gpt.py at startup)
        env = os.environ.copy()
        env["MAX_WALLCLOCK_SECONDS"] = str(EXPERIMENT_SECONDS)
        env["ITERATIONS"] = str(EXPERIMENT_ITERATIONS)
        env["RUN_ID"] = f"autoresearch_{experiment_id}"
        if VAL_LOSS_EVERY > 0:
            env["VAL_LOSS_EVERY"] = str(VAL_LOSS_EVERY)

        cmd = ["torchrun", "--standalone", f"--nproc_per_node={GPUS}", str(TRAIN_SCRIPT)]
        train_timeout = EXPERIMENT_SECONDS + TRAIN_TIMEOUT_PADDING_SECONDS
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Time budget: {EXPERIMENT_SECONDS}s, timeout: {train_timeout}s")

        train_proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env,
        )

        # Once training has started and loaded the file, revert train_gpt.py
        # to the current best so Claude can work on it for the next proposal
        time.sleep(5)  # brief pause to ensure torchrun has loaded the script
        TRAIN_SCRIPT.write_text(prev_code)

        # Speculatively propose the next experiment while training runs
        print("  Speculatively proposing next experiment while training...")
        spec_prompt = build_proposal_prompt(program, history, best_bpb)

        def speculative_propose():
            return run_claude_proposal(spec_prompt)

        spec_thread_result: list[str | None] = [None]
        def spec_worker():
            spec_thread_result[0] = speculative_propose()

        spec_thread = threading.Thread(target=spec_worker)
        spec_thread.start()

        # Wait for training to complete
        try:
            stdout, stderr = train_proc.communicate(timeout=train_timeout)
            output = stdout + "\n" + stderr
            returncode = train_proc.returncode
        except subprocess.TimeoutExpired:
            train_proc.kill()
            stdout, stderr = train_proc.communicate()
            output = (stdout or "") + "\n" + (stderr or "")
            returncode = -1

        train_time = time.time() - t_train
        print(f"  Training took {train_time:.1f}s (exit code {returncode})")

        # Wait for speculative proposal to finish too
        spec_thread.join(timeout=max(600 - train_time, 30))
        spec_desc = spec_thread_result[0]
        if spec_desc:
            # Save the speculative proposal's code (Claude edited train_gpt.py)
            speculative_code = TRAIN_SCRIPT.read_text()
            speculative_description = spec_desc
            speculative_base_bpb = best_bpb
            # Revert train_gpt.py again so evaluate_and_record can work cleanly
            TRAIN_SCRIPT.write_text(prev_code)
            print(f"  Speculative proposal ready: {spec_desc[:80]}...")
        else:
            speculative_code = None
            speculative_description = None
            speculative_base_bpb = None
            # Make sure train_gpt.py is in the right state
            TRAIN_SCRIPT.write_text(prev_code)
            print("  Speculative proposal failed (will propose fresh next iteration)")

        # Now put the experiment's code back for evaluation
        TRAIN_SCRIPT.write_text(new_code)

        # --- EVALUATION PHASE ---
        kept, best_bpb = evaluate_and_record(
            experiment_id, description, new_code, prev_code,
            output, returncode, propose_time, train_time,
            timestamp, best_bpb, history,
        )

        # If the experiment was kept, the speculative proposal is stale
        # (it was based on old best_bpb / old code)
        if kept and speculative_description is not None:
            print("  (speculative proposal invalidated by this improvement)")
            speculative_description = None
            speculative_code = None
            speculative_base_bpb = None

    # Final summary
    print("\n" + "=" * 70)
    print("AUTORESEARCH COMPLETE")
    print("=" * 70)
    kept_count = len(kept_entries(history))
    lane_total = sum(
        1 for h in history if h.get("lane") == LANE or (h.get("lane") is None and LANE == "core" and not NAMESPACE)
    )
    print(f"  Lane/stage:        {stage_config()}")
    print(f"  Namespace dir:     {AUTORESEARCH_DIR}")
    print(f"  Total experiments: {lane_total}")
    print(f"  Kept:              {kept_count}")
    print(f"  Best val_bpb:      {best_bpb}")
    print(f"  Best code saved:   {TRAIN_SCRIPT_BEST}")
    print("=" * 70)


if __name__ == "__main__":
    main()
