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
from datetime import datetime, timezone
from pathlib import Path

# ---------------------
# Configuration
# ---------------------

TRAIN_SCRIPT = Path("train_gpt.py")
TRAIN_SCRIPT_BEST = Path("autoresearch/train_gpt.best.py")
PROGRAM_FILE = Path("program.md")
HISTORY_FILE = Path("autoresearch/history.jsonl")
LOGS_DIR = Path("autoresearch/logs")
EXPERIMENTS_DIR = Path("autoresearch/experiments")

EXPERIMENT_SECONDS = int(os.environ.get("EXPERIMENT_SECONDS", "180"))
MAX_EXPERIMENTS = int(os.environ.get("MAX_EXPERIMENTS", "100"))
GPUS = int(os.environ.get("GPUS", "1"))
CLAUDE_MODEL = os.environ.get("AUTORESEARCH_MODEL", "opus")
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "high")

# When iterating on fewer GPUs or shorter times, reduce iterations
# so the model doesn't waste time in warmdown too early
EXPERIMENT_ITERATIONS = int(os.environ.get("EXPERIMENT_ITERATIONS", "20000"))
VAL_LOSS_EVERY = int(os.environ.get("VAL_LOSS_EVERY", "0"))  # 0 = only at end

# Baseline BPB to seed the first comparison (from the official baseline run)
BASELINE_BPB = float(os.environ.get("BASELINE_BPB", "0"))


def ensure_dirs():
    Path("autoresearch").mkdir(exist_ok=True)
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
    error = entry.get("error", "")

    md = f"""# Experiment {experiment_id}

**Date:** {entry.get('timestamp', 'unknown')}
**Result:** {kept_str}
**val_bpb:** {bpb_str}
**Artifact size:** {size_str} bytes
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


def format_history_for_prompt(history: list[dict], max_entries: int = 30) -> str:
    if not history:
        return "No experiments run yet."

    recent = history[-max_entries:]
    lines = []
    for h in recent:
        status = "KEPT" if h.get("kept") else "REVERTED"
        bpb = h.get("val_bpb")
        bpb_str = f"{bpb:.4f}" if bpb is not None else "FAILED"
        size = h.get("artifact_bytes")
        size_str = f" size={size}" if size else ""
        error = h.get("error", "")
        error_str = f" error={error}" if error else ""
        lines.append(
            f"  #{h['id']:3d} [{status:8s}] bpb={bpb_str}{size_str}{error_str} | {h['description']}"
        )

    best_kept = [h for h in history if h.get("kept") and h.get("val_bpb") is not None]
    best_bpb = min(h["val_bpb"] for h in best_kept) if best_kept else None

    summary = f"Experiments so far: {len(history)} total, {len(best_kept)} kept\n"
    if best_bpb is not None:
        summary += f"Current best val_bpb: {best_bpb:.4f}\n"
    summary += "\nRecent experiments:\n" + "\n".join(lines)
    return summary


def build_proposal_prompt(program: str, history: list[dict], best_bpb: float | None) -> str:
    history_block = format_history_for_prompt(history)
    best_str = f"{best_bpb:.4f}" if best_bpb is not None else "unknown (no successful run yet)"

    return f"""You are an autonomous ML researcher running experiments for the parameter-golf challenge.
Your job: make ONE specific modification to train_gpt.py to try to improve validation BPB.

## Research Program
{program}

## Experiment History
{history_block}

Current best val_bpb: {best_str}
Experiment time budget: {EXPERIMENT_SECONDS}s on {GPUS} GPU(s)

## Instructions
1. Read train_gpt.py to understand the current state
2. Decide on ONE specific change to try (guided by the research program and history)
3. Edit train_gpt.py to make that change — use surgical edits, not full rewrites
4. Print a single line starting with "DESCRIPTION:" summarizing what you changed and why

Guidelines:
- Make exactly ONE conceptual change per experiment so we can isolate what helps
- Consider what has and hasn't worked in the experiment history
- The script must remain functional — don't break imports, class structure, etc.
- The final metric is post-int8-quantization roundtrip val_bpb (lower is better)
- Total artifact (compressed weights + code) must stay under 16,000,000 bytes
- Be creative but grounded — vary your approaches, don't repeat failed ideas
- Focus on changes likely to improve BPB: architecture, hyperparameters, training tricks, compression"""


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
                "--max-turns", "10",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return None

    output = result.stdout + "\n" + result.stderr

    desc_match = re.search(r"DESCRIPTION:\s*(.+)", output)
    if desc_match:
        return desc_match.group(1).strip()

    lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1][:200]

    return None


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

    timeout = EXPERIMENT_SECONDS + 600

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

    print(f"  val_bpb:       {val_bpb}")
    print(f"  artifact size: {artifact_bytes}")

    kept = False
    over_budget = artifact_bytes is not None and artifact_bytes > 16_000_000

    if over_budget:
        print(f"  OVER SIZE BUDGET ({artifact_bytes} > 16,000,000). Reverting.")
        TRAIN_SCRIPT.write_text(prev_code)
    elif val_bpb is not None:
        if best_bpb is None or val_bpb < best_bpb:
            improvement = (best_bpb - val_bpb) if best_bpb else 0
            print(f"  IMPROVEMENT! {best_bpb} → {val_bpb} (Δ = {improvement:.4f})")
            best_bpb = val_bpb
            shutil.copy2(TRAIN_SCRIPT, TRAIN_SCRIPT_BEST)
            kept = True
        else:
            delta = val_bpb - best_bpb
            print(f"  No improvement (Δ = +{delta:.4f}). Reverting.")
            TRAIN_SCRIPT.write_text(prev_code)
    else:
        print("  Could not parse val_bpb. Reverting.")
        TRAIN_SCRIPT.write_text(prev_code)

    entry = {
        "id": experiment_id, "description": description,
        "val_bpb": val_bpb, "artifact_bytes": artifact_bytes,
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

    best_bpb: float | None = None
    for h in history:
        if h.get("kept") and h.get("val_bpb") is not None:
            if best_bpb is None or h["val_bpb"] < best_bpb:
                best_bpb = h["val_bpb"]

    if best_bpb is None and BASELINE_BPB > 0:
        best_bpb = BASELINE_BPB

    experiment_id = max((h["id"] for h in history), default=0)

    print("=" * 70)
    print("AUTORESEARCH — Parameter Golf (Claude Code, pipelined)")
    print("=" * 70)
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
        train_timeout = EXPERIMENT_SECONDS + 600
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
    kept_count = sum(1 for h in history if h.get("kept"))
    print(f"  Total experiments: {len(history)}")
    print(f"  Kept:              {kept_count}")
    print(f"  Best val_bpb:      {best_bpb}")
    print(f"  Best code saved:   {TRAIN_SCRIPT_BEST}")
    print("=" * 70)


if __name__ == "__main__":
    main()
