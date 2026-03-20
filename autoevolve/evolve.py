#!/usr/bin/env python3
"""
Auto-evolving research agent for the OpenAI Parameter Golf challenge.

Inspired by Andrej Karpathy's autoresearch project. Uses an LLM to autonomously
propose, test, and iterate on modifications to the training script.

Usage:
    # On 8xH100 (competition setup):
    python3 autoevolve/evolve.py --nproc 8

    # On 1xH100 (cheaper iteration):
    python3 autoevolve/evolve.py --nproc 1

    # Dry run (propose + validate syntax, no training):
    python3 autoevolve/evolve.py --dry-run

    # Use a cheaper/faster model:
    python3 autoevolve/evolve.py --model gpt-4o --nproc 1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
AUTOEVOLVE_DIR = ROOT / "autoevolve"
SCRIPT_PATH = AUTOEVOLVE_DIR / "train_gpt.py"
BEST_SCRIPT_PATH = AUTOEVOLVE_DIR / "best_train_gpt.py"
RESULTS_FILE = AUTOEVOLVE_DIR / "results.tsv"
LOGS_DIR = AUTOEVOLVE_DIR / "logs"
PROGRAM_FILE = AUTOEVOLVE_DIR / "program.md"
SOTA_SCRIPT = (
    ROOT
    / "records"
    / "track_10min_16mb"
    / "2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit"
    / "train_gpt.py"
)

DEFAULT_MODEL = "gpt-5.4"
BEST_KNOWN_BPB = 1.1748
ARTIFACT_LIMIT = 16_000_000
MAX_CONSECUTIVE_FAILURES = 5

# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-evolve Parameter Golf training")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model (default: gpt-5.4)")
    p.add_argument("--nproc", type=int, default=8, help="GPUs for torchrun (default: 8)")
    p.add_argument("--dry-run", action="store_true", help="Propose changes only, skip training")
    p.add_argument("--max-iters", type=int, default=0, help="Max iterations (0 = unlimited)")
    p.add_argument("--branch", default="autoevolve/run", help="Git branch name")
    p.add_argument("--resume", action="store_true", help="Resume from existing results.tsv")
    p.add_argument("--seed", type=int, default=None, help="Override training seed")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_openai():
    """Lazy-import openai and create client from .env or environment."""
    try:
        from dotenv import load_dotenv
        load_dotenv(AUTOEVOLVE_DIR / ".env")
    except ImportError:
        pass  # dotenv optional if env is already set

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found. Set it in .env or environment.")
        sys.exit(1)

    from openai import OpenAI
    return OpenAI(api_key=api_key)


def setup_workspace() -> None:
    """Create directories and copy the SOTA baseline as starting point."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not SCRIPT_PATH.exists():
        if SOTA_SCRIPT.exists():
            shutil.copy2(SOTA_SCRIPT, SCRIPT_PATH)
            shutil.copy2(SOTA_SCRIPT, BEST_SCRIPT_PATH)
            print(f"Copied SOTA baseline to {SCRIPT_PATH}")
        else:
            print(f"ERROR: SOTA script not found at {SOTA_SCRIPT}")
            sys.exit(1)
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "iteration\ttimestamp\tval_bpb\tartifact_bytes\tstatus\tdescription\treasoning\n"
        )


def setup_git(branch: str) -> None:
    """Create or switch to the autoevolve git branch."""
    cur = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout.strip()
    if cur != branch:
        r = subprocess.run(
            ["git", "checkout", "-b", branch],
            capture_output=True, text=True, cwd=ROOT,
        )
        if r.returncode != 0:
            subprocess.run(
                ["git", "checkout", branch],
                capture_output=True, text=True, cwd=ROOT,
            )
    print(f"Git branch: {branch}")


# ---------------------------------------------------------------------------
# Results Tracking
# ---------------------------------------------------------------------------

def get_best_bpb() -> float:
    """Return the best BPB achieved so far (from results.tsv)."""
    best = BEST_KNOWN_BPB
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().strip().split("\n")[1:]:
            parts = line.split("\t")
            if len(parts) >= 5 and parts[4] == "keep":
                try:
                    best = min(best, float(parts[2]))
                except ValueError:
                    pass
    return best


def get_iteration_count() -> int:
    """Return number of experiments already recorded."""
    if not RESULTS_FILE.exists():
        return 0
    lines = RESULTS_FILE.read_text().strip().split("\n")
    return max(0, len(lines) - 1)


def read_results_history() -> str:
    """Read and return recent results history for the LLM context."""
    if not RESULTS_FILE.exists():
        return "No experiments run yet. This is the first iteration."
    content = RESULTS_FILE.read_text()
    lines = content.strip().split("\n")
    if len(lines) <= 1:
        return "No experiments run yet. This is the first iteration."
    # Keep header + last 50 entries to fit context
    if len(lines) > 51:
        return "\n".join(lines[:1] + lines[-50:])
    return content


def log_result(
    iteration: int,
    val_bpb: float | None,
    artifact_bytes: int | None,
    status: str,
    description: str,
    reasoning: str = "",
) -> None:
    """Append an experiment result to results.tsv."""
    ts = datetime.now().isoformat()
    bpb = f"{val_bpb:.4f}" if val_bpb is not None else "N/A"
    nbytes = str(artifact_bytes) if artifact_bytes is not None else "N/A"
    # Escape tabs/newlines in description and reasoning for TSV safety
    desc_safe = description.replace("\t", " ").replace("\n", " ")
    reason_safe = reasoning.replace("\t", " ").replace("\n", " ")
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{iteration}\t{ts}\t{bpb}\t{nbytes}\t{status}\t{desc_safe}\t{reason_safe}\n")


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

def build_prompt(
    current_code: str, history: str, best_bpb: float, iteration: int, program: str,
) -> str:
    return f"""You are a world-class ML researcher autonomously competing in the OpenAI Parameter Golf challenge.

## THE CHALLENGE
Train the best possible language model under these hard constraints:
- Artifact ≤ 16,000,000 bytes (code + zlib-compressed INT8 model weights)
- Training time ≤ 10 minutes on 8×H100 SXM GPUs
- Evaluation time ≤ 10 minutes (separate from training)
- Metric: val_bpb (bits per byte) on FineWeb validation set — LOWER IS BETTER
- No network calls during evaluation. Fully self-contained.

Current best val_bpb: **{best_bpb:.4f}**
Leaderboard SOTA: **{BEST_KNOWN_BPB}**

## CURRENT TRAINING SCRIPT
You have COMPLETE FREEDOM to change ANYTHING in this script: the architecture,
optimizer, training loop, quantization, evaluation, initialization, activations,
normalization, attention mechanism, or any other aspect.

```python
{current_code}
```

## EXPERIMENT HISTORY
{history}

## DOMAIN KNOWLEDGE & IDEAS
{program}

## YOUR TASK — ITERATION #{iteration}

You are spending real GPU money on every experiment. Think DEEPLY before proposing.
Your proposal must be your single highest-confidence idea for improving val_bpb.

### REASONING PROCESS:

**Step 1 — Diagnose**: What is the current bottleneck? Capacity? Optimization?
Quantization? Where are bits being wasted?

**Step 2 — Hypothesize**: What single change addresses that bottleneck? Draw on
your knowledge of the full ML literature (SSMs, MoE, recurrent depth, QAT,
SwiGLU, advanced optimizers, curriculum learning, TTT, distillation, etc.)

**Step 3 — Estimate**: Expected BPB improvement? If < 0.001, pick something bolder.

**Step 4 — Learn from history**: What failed before and WHY? Don't repeat mistakes.

**Step 5 — Implement**: Provide SEARCH/REPLACE blocks to edit the script.

### RETURN FORMAT
Return a JSON object:
{{
  "diagnosis": "What is the current bottleneck? (2-3 sentences)",
  "hypothesis": "What change addresses it and why? (2-3 sentences)",
  "expected_delta": "Estimated BPB change (e.g. -0.008) with justification",
  "risk_assessment": "What could go wrong?",
  "description": "Concise 1-sentence summary",
  "changes": [
    {{
      "explanation": "What this specific edit does",
      "search": "EXACT lines from the current script to find (include enough context for unique match, at least 3-5 lines)",
      "replace": "New lines to replace with"
    }}
  ]
}}

### RULES FOR SEARCH/REPLACE BLOCKS
- "search" must be an EXACT substring of the current script (whitespace-sensitive!)
- Include enough surrounding context (3-5+ lines) so the match is UNIQUE in the file
- You may include multiple change blocks — they are applied in order
- To ADD new code, use a search block that matches the insertion point and include
  the original lines plus the new lines in "replace"
- To DELETE code, use a search block and set "replace" to the remaining lines
- You CAN make sweeping changes (rewrite entire classes, add new modules, etc.)
  — just provide enough search/replace blocks to cover it

### HARD CONSTRAINTS
- Result must be valid Python, runnable via: torchrun --standalone --nproc_per_node=8 train_gpt.py
- Must stay under 16MB artifact (code + compressed model)
- Must train in under 10 minutes on 8×H100
- Must keep the data loading interface (reads fineweb binary shards)
- Must keep the val_bpb evaluation and INT8+zlib serialization output format
- Keep under 1500 lines

### PHILOSOPHY
- You are a researcher, not a hyperparameter tuner.
- Bold architectural changes with strong theoretical backing beat safe tweaks.
- Every GPU-minute wasted on a low-confidence idea is money burned.
- Think about what top ML labs would try if they were competing."""


def propose_modification(client, model: str, prompt: str) -> dict:
    """Call the LLM to get a proposed modification."""
    print(f"  Calling {model} for proposal...")
    t0 = time.time()

    response = client.chat.completions.create(
        model=model,
        reasoning={"effort": "high"},
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_completion_tokens=32000,  # Full script is ~13k tokens; give headroom
    )

    elapsed = time.time() - t0
    text = response.choices[0].message.content
    usage = response.usage
    print(
        f"  LLM responded in {elapsed:.1f}s "
        f"(input={usage.prompt_tokens}, output={usage.completion_tokens} tokens)"
    )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from LLM response (first 500 chars): {text[:500]}")


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

    # Try stripping trailing whitespace on each line for fuzzy match
    def strip_trailing(s: str) -> str:
        return "\n".join(line.rstrip() for line in s.split("\n"))

    code_stripped = strip_trailing(code)
    search_stripped = strip_trailing(search)
    if search_stripped in code_stripped:
        # Find the position in the stripped version and apply
        return code_stripped.replace(search_stripped, strip_trailing(replace), 1), ""

    # Show context for debugging
    first_line = search.strip().split("\n")[0][:80]
    return None, f"Search string not found. First line: '{first_line}'"


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
            return None, f"Change {i+1}: empty search string"

        result, err = apply_search_replace(code, search, replace)
        if result is None:
            return None, f"Change {i+1}/{len(changes)}: {err}"
        code = result

    # Validate the final result
    try:
        ast.parse(code)
    except SyntaxError as e:
        return None, f"SyntaxError after applying changes at line {e.lineno}: {e.msg}"

    lines = code.strip().split("\n")
    if len(lines) > 1500:
        return None, f"Script too long after changes: {len(lines)} lines (max 1500)"

    # Structural sanity checks
    required = ["def main()", "class Hyperparameters", "class GPT("]
    for token in required:
        if token not in code:
            return None, f"Missing required component after changes: {token}"

    return code, ""


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_experiment(
    script_path: Path, nproc: int, seed: int | None = None,
) -> tuple[str, int]:
    """Run training via torchrun and return (output, exit_code)."""
    env = os.environ.copy()
    env["DATA_PATH"] = str(ROOT / "data" / "datasets" / "fineweb10B_sp1024")
    env["TOKENIZER_PATH"] = str(ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model")
    env["VOCAB_SIZE"] = "1024"
    if seed is not None:
        env["SEED"] = str(seed)

    cmd = [
        "torchrun", "--standalone",
        f"--nproc_per_node={nproc}",
        str(script_path),
    ]

    print(f"  Running: {' '.join(cmd)}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=1200,  # 20-minute hard timeout (train + eval)
            env=env, cwd=ROOT,
        )
        elapsed = time.time() - t0
        output = result.stdout + "\n--- STDERR ---\n" + result.stderr
        print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Experiment exceeded 20-minute hard limit", 1


def parse_experiment_output(output: str) -> dict:
    """Parse metrics from training output."""
    metrics = {
        "val_bpb": None,
        "val_loss": None,
        "artifact_bytes": None,
        "total_bytes": None,
        "train_time_ms": None,
        "peak_memory_mib": None,
    }

    for line in output.split("\n"):
        # Post-quantization final results (the definitive score)
        if "final_int8_zlib_roundtrip " in line and "exact" not in line:
            m = re.search(r"val_bpb:([0-9.]+)", line)
            if m:
                metrics["val_bpb"] = float(m.group(1))
            m = re.search(r"val_loss:([0-9.]+)", line)
            if m:
                metrics["val_loss"] = float(m.group(1))

        # Artifact sizes
        if "Total submission size int8+zlib:" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                metrics["total_bytes"] = int(m.group(1))

        if "Serialized model int8+zlib:" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                metrics["artifact_bytes"] = int(m.group(1))

        # Training time (last match wins = final step)
        if "train_time:" in line and "step:" in line:
            m = re.search(r"train_time:(\d+)ms", line)
            if m:
                metrics["train_time_ms"] = int(m.group(1))

        # Peak memory
        if "peak memory allocated:" in line:
            m = re.search(r"(\d+)\s*MiB", line)
            if m:
                metrics["peak_memory_mib"] = int(m.group(1))

    return metrics


# ---------------------------------------------------------------------------
# Git Helpers
# ---------------------------------------------------------------------------

def git_commit(msg: str) -> None:
    subprocess.run(["git", "add", str(SCRIPT_PATH)], capture_output=True, cwd=ROOT)
    subprocess.run(
        ["git", "commit", "-m", msg],
        capture_output=True, cwd=ROOT,
    )


def git_revert(backup_code: str, msg: str) -> None:
    SCRIPT_PATH.write_text(backup_code)
    git_commit(msg)


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  Parameter Golf Auto-Evolve Agent")
    print(f"  Model: {args.model} | GPUs: {args.nproc} | Dry-run: {args.dry_run}")
    print("=" * 70)

    client = setup_openai()
    setup_workspace()
    if not args.dry_run:
        setup_git(args.branch)

    program = PROGRAM_FILE.read_text() if PROGRAM_FILE.exists() else ""
    best_bpb = get_best_bpb()
    iteration = get_iteration_count() if args.resume else get_iteration_count()
    consecutive_failures = 0

    print(f"Starting from iteration {iteration + 1}, best BPB: {best_bpb:.4f}\n")

    while True:
        iteration += 1
        if 0 < args.max_iters < iteration:
            print(f"\nReached max iterations ({args.max_iters}). Stopping.")
            break

        print(f"\n{'=' * 70}")
        print(f"  ITERATION {iteration}  |  Best BPB: {best_bpb:.4f}")
        print(f"{'=' * 70}")

        # ---- 1. Read current state ----
        current_code = SCRIPT_PATH.read_text()
        history = read_results_history()

        # ---- 2. Propose modification ----
        prompt = build_prompt(current_code, history, best_bpb, iteration, program)
        try:
            proposal = propose_modification(client, args.model, prompt)
        except Exception as e:
            print(f"  LLM ERROR: {e}")
            traceback.print_exc()
            log_result(iteration, None, None, "llm_error", str(e)[:200])
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            time.sleep(5)
            continue

        description = proposal.get("description", "Unknown change")[:200]
        diagnosis = proposal.get("diagnosis", "")[:300]
        hypothesis = proposal.get("hypothesis", "")[:300]
        expected = proposal.get("expected_delta", "?")
        risk = proposal.get("risk_assessment", proposal.get("risk", "?"))[:200]
        reasoning = f"Diagnosis: {diagnosis} | Hypothesis: {hypothesis} | Expected: {expected}"
        print(f"  Proposed:   {description}")
        if diagnosis:
            print(f"  Diagnosis:  {diagnosis}")
        if hypothesis:
            print(f"  Hypothesis: {hypothesis}")
        print(f"  Expected:   {expected}")
        print(f"  Risk:       {risk}")

        # ---- 3. Apply and validate proposed changes ----
        new_code, err = apply_proposal(current_code, proposal)
        if new_code is None:
            print(f"  INVALID: {err}")
            log_result(iteration, None, None, "invalid", f"{description} | {err}", reasoning)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            continue

        # ---- 4. Apply change ----
        backup_code = current_code
        SCRIPT_PATH.write_text(new_code)
        n_lines = len(new_code.strip().split("\n"))
        print(f"  Applied ({len(new_code):,} bytes, {n_lines} lines)")

        if args.dry_run:
            print("  [DRY RUN] Skipping training. Change saved for review.")
            log_result(iteration, None, None, "dry_run", description, reasoning)
            # In dry-run, revert so next iteration proposes from the same base
            SCRIPT_PATH.write_text(backup_code)
            consecutive_failures = 0
            continue

        # ---- 5. Git commit ----
        git_commit(f"Experiment {iteration}: {description}")

        # ---- 6. Run training ----
        print("  Running training experiment...")
        output, exit_code = run_experiment(SCRIPT_PATH, args.nproc, args.seed)

        # Save full log
        log_file = LOGS_DIR / f"exp_{iteration:04d}.log"
        log_file.write_text(output)
        print(f"  Log saved to {log_file}")

        if exit_code != 0:
            print(f"  CRASH (exit code {exit_code})")
            tail = output.strip().split("\n")[-15:]
            for ln in tail:
                print(f"    {ln}")
            git_revert(backup_code, f"Revert experiment {iteration} (crash)")
            log_result(iteration, None, None, "crash", description, reasoning)
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {MAX_CONSECUTIVE_FAILURES} consecutive failures. Stopping.")
                break
            continue

        # ---- 7. Parse results ----
        metrics = parse_experiment_output(output)
        val_bpb = metrics["val_bpb"]
        total_bytes = metrics["total_bytes"]

        if val_bpb is None:
            print("  Could not parse val_bpb from output")
            git_revert(backup_code, f"Revert experiment {iteration} (parse error)")
            log_result(iteration, None, total_bytes, "parse_error", description, reasoning)
            consecutive_failures += 1
            continue

        print(f"  val_bpb:  {val_bpb:.4f}")
        print(f"  total:    {total_bytes:,} bytes" if total_bytes else "  total:    unknown")
        if metrics["train_time_ms"]:
            print(f"  time:     {metrics['train_time_ms'] / 1000:.1f}s")
        if metrics["peak_memory_mib"]:
            print(f"  memory:   {metrics['peak_memory_mib']} MiB")

        # ---- 8. Decide: keep or discard ----
        if total_bytes and total_bytes > ARTIFACT_LIMIT:
            print(f"  OVER SIZE ({total_bytes:,} > {ARTIFACT_LIMIT:,})")
            git_revert(backup_code, f"Revert experiment {iteration} (over size)")
            log_result(iteration, val_bpb, total_bytes, "over_size", description, reasoning)
            consecutive_failures += 1
            continue

        if val_bpb < best_bpb:
            delta = best_bpb - val_bpb
            print(f"  >>> IMPROVEMENT: {best_bpb:.4f} -> {val_bpb:.4f} (delta: -{delta:.4f}) <<<")
            best_bpb = val_bpb
            shutil.copy2(SCRIPT_PATH, BEST_SCRIPT_PATH)
            log_result(iteration, val_bpb, total_bytes, "keep", description, reasoning)
            consecutive_failures = 0
        else:
            print(f"  No improvement ({val_bpb:.4f} >= {best_bpb:.4f})")
            git_revert(backup_code, f"Revert experiment {iteration} (no improvement)")
            log_result(iteration, val_bpb, total_bytes, "discard", description, reasoning)
            consecutive_failures = 0  # Experiment ran OK, just no gain

        time.sleep(2)

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"  Auto-evolve finished after {iteration} iterations")
    print(f"  Best val_bpb: {best_bpb:.4f}")
    print(f"  Best script:  {BEST_SCRIPT_PATH}")
    print(f"  Results log:  {RESULTS_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
