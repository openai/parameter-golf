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

    # Use a specific model:
    python3 autoevolve/evolve.py --model gpt-5.4 --nproc 1
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
PROMPTS_DIR = AUTOEVOLVE_DIR / "prompts"
SCRIPT_PATH = AUTOEVOLVE_DIR / "train_gpt.py"
BEST_SCRIPT_PATH = AUTOEVOLVE_DIR / "best_train_gpt.py"
RESULTS_FILE = AUTOEVOLVE_DIR / "results.tsv"
LOGS_DIR = AUTOEVOLVE_DIR / "logs"
RECORDS_DIR = ROOT / "records" / "track_10min_16mb"
NON_RECORD_DIR = ROOT / "records" / "track_non_record_16mb"

DEFAULT_MODEL = "gpt-5.4"
BEST_KNOWN_BPB = 1.1428
ARTIFACT_LIMIT = 16_000_000
MAX_WALLCLOCK = 600  # 10 minutes — hard competition limit
MAX_CONSECUTIVE_FAILURES = 5
UPSTREAM_SYNC_EVERY = 5  # check for new SOTA every N iterations


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-evolve Parameter Golf training")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
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


def find_best_sota_script() -> Path | None:
    """Find the best record script from track_10min_16mb by parsing folder names."""
    if not RECORDS_DIR.exists():
        return None
    # Records are sorted by date, pick the latest (which should be the best)
    records = sorted(RECORDS_DIR.iterdir(), reverse=True)
    for r in records:
        script = r / "train_gpt.py"
        if script.exists():
            return script
    return None


def setup_workspace() -> None:
    """Create directories and copy the best SOTA baseline as starting point."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SCRIPT_PATH.exists():
        sota = find_best_sota_script()
        if sota:
            shutil.copy2(sota, SCRIPT_PATH)
            shutil.copy2(sota, BEST_SCRIPT_PATH)
            print(f"Copied SOTA baseline from {sota.parent.name}")
        else:
            print("ERROR: No SOTA script found in records/track_10min_16mb/")
            sys.exit(1)

    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(
            "iteration\ttimestamp\tval_bpb\tartifact_bytes\tstatus\tdescription\treasoning\n"
        )


def setup_git(branch: str) -> None:
    """Create or switch to the autoevolve git branch, and configure push auth."""
    # Configure token-based push auth so results survive pod death
    token = os.getenv("GITHUB_TOKEN")
    if token:
        # Inject token into origin remote URL for password-free push
        url_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, cwd=ROOT,
        )
        origin_url = url_result.stdout.strip()
        if origin_url and "github.com" in origin_url and "@" not in origin_url:
            # e.g. https://github.com/user/repo.git → https://TOKEN@github.com/user/repo.git
            auth_url = origin_url.replace("https://", f"https://{token}@")
            subprocess.run(
                ["git", "remote", "set-url", "origin", auth_url],
                capture_output=True, cwd=ROOT,
            )
            print("  Git push auth configured via GITHUB_TOKEN.")
    else:
        print("  WARNING: GITHUB_TOKEN not set in .env — results won't be pushed to GitHub.")
        print("           Add GITHUB_TOKEN=ghp_... to autoevolve/.env to auto-save results.")

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
# Upstream Sync
# ---------------------------------------------------------------------------

def sync_upstream() -> str | None:
    """Fetch upstream and check for new SOTA records. Returns summary or None."""
    print("  Checking upstream for new records...")
    r = subprocess.run(
        ["git", "fetch", "https://github.com/openai/parameter-golf.git", "main", "--quiet"],
        capture_output=True, text=True, cwd=ROOT, timeout=30,
    )
    if r.returncode != 0:
        print(f"  Upstream fetch failed: {r.stderr.strip()[:100]}")
        return None

    # Check if upstream has new records
    diff = subprocess.run(
        ["git", "diff", "HEAD", "FETCH_HEAD", "--name-only", "--", "records/", "README.md"],
        capture_output=True, text=True, cwd=ROOT,
    )
    changed = diff.stdout.strip()
    if not changed:
        print("  No new upstream changes.")
        return None

    # Pull the README to check leaderboard
    readme_diff = subprocess.run(
        ["git", "diff", "HEAD", "FETCH_HEAD", "--", "README.md"],
        capture_output=True, text=True, cwd=ROOT,
    )

    # Look for new BPB scores in the diff
    new_scores = []
    for line in readme_diff.stdout.split("\n"):
        if line.startswith("+") and "| 1." in line:
            m = re.search(r"\|\s*(1\.\d{4})\s*\|", line)
            if m:
                new_scores.append(float(m.group(1)))

    if new_scores:
        best_new = min(new_scores)
        summary = f"Upstream has new records! Best new score: {best_new:.4f}"
        print(f"  {summary}")
        if best_new < BEST_KNOWN_BPB:
            print(f"  WARNING: New public SOTA ({best_new:.4f}) beats our baseline ({BEST_KNOWN_BPB})!")
            print(f"  Consider pulling latest: git pull https://github.com/openai/parameter-golf.git main")
        return summary

    summary = f"Upstream has changes in: {changed[:200]}"
    print(f"  {summary}")
    return summary


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
    """Read and return recent results history for the LLM context, formatted for readability."""
    if not RESULTS_FILE.exists():
        return "No experiments run yet. This is the first iteration."
    content = RESULTS_FILE.read_text()
    lines = content.strip().split("\n")
    if len(lines) <= 1:
        return "No experiments run yet. This is the first iteration."

    # Parse into readable format so the model can easily learn from past results
    entries = []
    for line in lines[1:][-30:]:  # Last 30 entries
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        itr = parts[0]
        bpb = parts[2]
        status = parts[4]
        desc = parts[5][:120]
        reasoning = parts[6][:200] if len(parts) > 6 else ""
        entries.append(f"  #{itr} [{status:12}] bpb={bpb:>8}  {desc}")
        if status in ("crash", "invalid", "over_size", "parse_error") and reasoning:
            entries.append(f"      ERROR DETAIL: {reasoning}")

    if not entries:
        return "No experiments run yet. This is the first iteration."

    return "Recent experiments (newest last):\n" + "\n".join(entries)


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
    desc_safe = description.replace("\t", " ").replace("\n", " ")
    reason_safe = reasoning.replace("\t", " ").replace("\n", " ")
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{iteration}\t{ts}\t{bpb}\t{nbytes}\t{status}\t{desc_safe}\t{reason_safe}\n")


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

def load_prompt_template() -> str:
    """Load the agent prompt template from prompts/agent.md."""
    agent_file = PROMPTS_DIR / "agent.md"
    if agent_file.exists():
        return agent_file.read_text()
    # Fallback: minimal inline prompt
    return (
        "You are an ML researcher competing in the OpenAI Parameter Golf challenge.\n"
        "Current best val_bpb: **{{best_bpb}}**\nLeaderboard SOTA: **{{leaderboard_sota}}**\n\n"
        "## CURRENT TRAINING SCRIPT\n```python\n{{current_code}}\n```\n\n"
        "## EXPERIMENT HISTORY\n{{history}}\n\n## DOMAIN KNOWLEDGE\n{{program}}\n\n"
        "Return a JSON object with: diagnosis, hypothesis, expected_delta, risk_assessment, "
        "description, and changes (array of {explanation, search, replace})."
    )


def load_program() -> str:
    """Load program.md from prompts/ folder."""
    program_file = PROMPTS_DIR / "program.md"
    if program_file.exists():
        return program_file.read_text()
    return ""


def build_prompt(
    current_code: str, history: str, best_bpb: float, iteration: int,
) -> str:
    template = load_prompt_template()
    program = load_program()
    return (
        template
        .replace("{{best_bpb}}", f"{best_bpb:.4f}")
        .replace("{{leaderboard_sota}}", f"{BEST_KNOWN_BPB}")
        .replace("{{current_code}}", current_code)
        .replace("{{history}}", history)
        .replace("{{program}}", program)
        .replace("{{iteration}}", str(iteration))
    )


def propose_modification(client, model: str, prompt: str) -> dict:
    """Call the LLM to get a proposed modification."""
    print(f"  Calling {model} for proposal...")
    t0 = time.time()

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_output_tokens=32000,
    )

    elapsed = time.time() - t0
    text = response.output_text
    usage = getattr(response, "usage", None)
    if usage:
        in_toks = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", 0))
        out_toks = getattr(usage, "output_tokens", getattr(usage, "completion_tokens", 0))
        print(f"  LLM responded in {elapsed:.1f}s (input={in_toks}, output={out_toks} tokens)")
    else:
        print(f"  LLM responded in {elapsed:.1f}s")

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
        return code_stripped.replace(search_stripped, strip_trailing(replace), 1), ""

    first_line = search.strip().split("\n")[0][:80]
    return None, f"Search string not found. First line: '{first_line}'"


def validate_competition_constraints(code: str) -> tuple[bool, str]:
    """
    Verify that the modified script still respects competition rules.
    This prevents the LLM from "cheating" by removing limits.
    """
    errors = []

    # 1. max_wallclock_seconds must exist and be <= 600
    m = re.search(r'max_wallclock_seconds\s*=\s*.*?(\d+\.?\d*)', code)
    if not m:
        errors.append("max_wallclock_seconds not found — training time is unbounded")
    elif float(m.group(1)) > MAX_WALLCLOCK:
        errors.append(f"max_wallclock_seconds={m.group(1)} exceeds {MAX_WALLCLOCK}s limit")

    # 2. Must keep the output format lines (we need to parse results)
    if "final_int8_zlib_roundtrip" not in code:
        errors.append("Missing 'final_int8_zlib_roundtrip' output — cannot parse val_bpb")
    if "Total submission size" not in code:
        errors.append("Missing 'Total submission size' output — cannot parse artifact bytes")

    # 3. Must have quantization + serialization (artifact must be compressed)
    if "zlib.compress" not in code and "zstandard" not in code and "ZstdCompressor" not in code:
        errors.append("No compression (zlib/zstd) found — artifact won't be compressed")

    # 4. Must have val_bpb evaluation
    if "val_bpb" not in code:
        errors.append("No val_bpb computation found — cannot evaluate")

    # 5. Artifact size constraint: check for code that computes total bytes
    if "code_bytes" not in code or "quant_file_bytes" not in code:
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
            return None, f"Change {i+1}: empty search string"

        result, err = apply_search_replace(code, search, replace)
        if result is None:
            return None, f"Change {i+1}/{len(changes)}: {err}"
        code = result

    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return None, f"SyntaxError after applying changes at line {e.lineno}: {e.msg}"

    # Line count check
    lines = code.strip().split("\n")
    if len(lines) > 1500:
        return None, f"Script too long after changes: {len(lines)} lines (max 1500)"

    # Structural sanity checks
    required = ["def main()", "class Hyperparameters"]
    for token in required:
        if token not in code:
            return None, f"Missing required component after changes: {token}"

    # Competition constraint checks
    ok, err = validate_competition_constraints(code)
    if not ok:
        return None, f"Competition constraint violation: {err}"

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
    # Force wallclock to competition limit (safety net)
    env["MAX_WALLCLOCK_SECONDS"] = str(MAX_WALLCLOCK)
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
            timeout=1500,  # 25-min hard timeout (10 train + 10 eval + 5 overhead)
            env=env, cwd=ROOT,
        )
        elapsed = time.time() - t0
        output = result.stdout + "\n--- STDERR ---\n" + result.stderr
        print(f"  Completed in {elapsed:.0f}s (exit code {result.returncode})")
        return output, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT: Experiment exceeded 25-minute hard limit (10 train + 10 eval + 5 buffer)", 1


def parse_experiment_output(output: str) -> dict:
    """Parse metrics from training output. Handles various quant/compression combos."""
    metrics = {
        "val_bpb": None,
        "val_loss": None,
        "artifact_bytes": None,
        "total_bytes": None,
        "train_time_ms": None,
        "eval_time_ms": None,
        "peak_memory_mib": None,
    }

    for line in output.split("\n"):
        # Post-quantization final results (the definitive score)
        # Matches: "final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXXms"
        if "final_int8_zlib_roundtrip " in line and "exact" not in line:
            m = re.search(r"val_bpb:([0-9.]+)", line)
            if m:
                metrics["val_bpb"] = float(m.group(1))
            m = re.search(r"val_loss:([0-9.]+)", line)
            if m:
                metrics["val_loss"] = float(m.group(1))
            m = re.search(r"eval_time:(\d+)ms", line)
            if m:
                metrics["eval_time_ms"] = int(m.group(1))

        # Total artifact size — flexible matching for various labels
        # Matches: "Total submission size int8+zlib: NNNN bytes"
        #          "Total submission size int6+zstd: NNNN bytes"
        #          "Total submission size: NNNN bytes"
        if "Total submission size" in line:
            m = re.search(r"(\d+)\s*bytes", line)
            if m:
                metrics["total_bytes"] = int(m.group(1))

        # Serialized model size — flexible matching
        # Matches: "Serialized model int6+zstd: NNNN bytes"
        #          "Serialized model int8+zlib: NNNN bytes"
        #          "Serialized model: NNNN bytes"
        if "Serialized model" in line and "bytes" in line:
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
    """Stage all result files and commit."""
    files_to_stage = [
        str(SCRIPT_PATH),       # current train_gpt.py
        str(BEST_SCRIPT_PATH),  # best train_gpt.py
        str(RESULTS_FILE),      # results.tsv
        str(LOGS_DIR),          # autoevolve/logs/ (tracked, not gitignored)
    ]
    subprocess.run(["git", "add"] + files_to_stage, capture_output=True, cwd=ROOT)
    subprocess.run(["git", "commit", "-m", msg], capture_output=True, cwd=ROOT)


def git_push(branch: str) -> None:
    """Push current branch to origin so results survive pod shutdown."""
    r = subprocess.run(
        ["git", "push", "origin", branch, "--set-upstream", "--quiet"],
        capture_output=True, text=True, cwd=ROOT, timeout=60,
    )
    if r.returncode == 0:
        print("  Pushed to origin.")
    else:
        err = (r.stderr or r.stdout).strip()[:120]
        print(f"  Push failed (non-fatal): {err}")


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
    print(f"  Constraints: {MAX_WALLCLOCK}s wallclock, {ARTIFACT_LIMIT:,} byte artifact")
    print("=" * 70)

    client = setup_openai()
    setup_workspace()
    if not args.dry_run:
        setup_git(args.branch)
    branch = args.branch

    best_bpb = get_best_bpb()
    iteration = get_iteration_count()
    consecutive_failures = 0
    iters_this_session = 0

    print(f"Starting from iteration {iteration + 1}, best BPB: {best_bpb:.4f}\n")

    while True:
        iteration += 1
        iters_this_session += 1
        if 0 < args.max_iters < iters_this_session:
            print(f"\nReached max iterations ({args.max_iters}). Stopping.")
            break

        print(f"\n{'=' * 70}")
        print(f"  ITERATION {iteration}  |  Best BPB: {best_bpb:.4f}")
        print(f"{'=' * 70}")

        # ---- 0. Periodic upstream sync ----
        if iteration % UPSTREAM_SYNC_EVERY == 1:
            try:
                sync_upstream()
            except Exception as e:
                print(f"  Sync error (non-fatal): {e}")

        # ---- 1. Read current state ----
        current_code = SCRIPT_PATH.read_text()
        history = read_results_history()

        # ---- 2. Propose modification ----
        prompt = build_prompt(current_code, history, best_bpb, iteration)
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
            # Include crash details so model can learn from the failure
            crash_tail = " | ".join(ln.strip() for ln in tail if ln.strip())[-300:]
            crash_reasoning = f"{reasoning} | CRASH: {crash_tail}"
            log_result(iteration, None, None, "crash", description, crash_reasoning)
            git_push(branch)
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
            git_push(branch)
            consecutive_failures += 1
            continue

        print(f"  val_bpb:  {val_bpb:.4f}")
        print(f"  total:    {total_bytes:,} bytes" if total_bytes else "  total:    unknown")
        if metrics["train_time_ms"]:
            train_s = metrics["train_time_ms"] / 1000
            print(f"  train:    {train_s:.1f}s", end="")
            if train_s > MAX_WALLCLOCK:
                print(f"  WARNING: exceeded {MAX_WALLCLOCK}s limit!")
            else:
                print()
        if metrics["eval_time_ms"]:
            eval_s = metrics["eval_time_ms"] / 1000
            print(f"  eval:     {eval_s:.1f}s", end="")
            if eval_s > MAX_WALLCLOCK:
                print(f"  WARNING: eval exceeded {MAX_WALLCLOCK}s limit!")
            else:
                print()
        if metrics["peak_memory_mib"]:
            print(f"  memory:   {metrics['peak_memory_mib']} MiB")

        # ---- 8. Decide: keep or discard ----
        if total_bytes and total_bytes > ARTIFACT_LIMIT:
            overage = total_bytes - ARTIFACT_LIMIT
            print(f"  OVER SIZE ({total_bytes:,} > {ARTIFACT_LIMIT:,}, +{overage:,} bytes)")
            git_revert(backup_code, f"Revert experiment {iteration} (over size)")
            size_reasoning = f"{reasoning} | OVER_SIZE: {total_bytes} bytes, +{overage} over limit"
            log_result(iteration, val_bpb, total_bytes, "over_size", description, size_reasoning)
            git_push(branch)
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

        # ---- 9. Push to origin so results survive pod shutdown ----
        git_push(branch)
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
