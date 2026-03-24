"""Gradient attribution: per-layer gradient norm logging via source instrumentation.

Creates a patched copy of train_gpt_mlx.py that logs per-parameter L2 gradient
norms at each training step. The original script is never modified.

CLI:
  python scripts/causal/gradient_attribution.py \
    --script train_gpt_mlx.py \
    --output results/causal/diagnostics/gradient_attribution.json \
    [--val-every 100]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Pure functions (testable without MLX or subprocess)
# ---------------------------------------------------------------------------


def find_last_accumulate_flat_grads(source: str) -> tuple[int, str]:
    """Find the LAST occurrence of accumulate_flat_grads call in source.

    Returns (line_index, line_text) where line_index is 0-based.
    Skips the function definition (lines starting with 'def ').
    """
    lines = source.splitlines()
    last_idx = -1
    last_line = ""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "accumulate_flat_grads" in stripped and not stripped.startswith("def "):
            last_idx = i
            last_line = line
    if last_idx < 0:
        raise ValueError("Could not find any accumulate_flat_grads call site in source")
    return last_idx, last_line


def validate_sentinel(source: str, target_line_idx: int) -> None:
    """Validate dual sentinel: both markers must appear within +/-10 lines.

    Sentinels:
      1. 'train_loss = train_loss + loss' (substring)
      2. 'lr_mul' (substring)

    The design specifies +/-5 lines but the actual file structure places
    lr_mul ~8 lines before the target. We use +/-10 for robustness.

    Raises ValueError if validation fails.
    """
    lines = source.splitlines()
    window_start = max(0, target_line_idx - 10)
    window_end = min(len(lines), target_line_idx + 11)
    window = "\n".join(lines[window_start:window_end])

    has_train_loss = "train_loss = train_loss + loss" in window
    has_lr_mul = "lr_mul" in window

    if not (has_train_loss and has_lr_mul):
        missing = []
        if not has_train_loss:
            missing.append("'train_loss = train_loss + loss'")
        if not has_lr_mul:
            missing.append("'lr_mul'")
        raise ValueError(
            f"Sentinel validation failed at line {target_line_idx}: "
            f"missing {', '.join(missing)} within +/-5 lines. "
            f"The patch site may have drifted."
        )


def instrument_source(source: str, gradient_log_path: str) -> str:
    """Insert gradient norm logging code after the last accumulate_flat_grads call.

    The logging block is inserted after the for-loop that contains the
    accumulate_flat_grads call (i.e., after the grad accumulation loop ends,
    before grads = tree_unflatten).

    Returns the patched source string.
    """
    line_idx, _line_text = find_last_accumulate_flat_grads(source)
    validate_sentinel(source, line_idx)

    lines = source.splitlines()

    # Find the end of the for-loop block that contains the accumulate_flat_grads call.
    # We look for the next line at the same or lesser indentation level as the
    # for-loop that contains our target line.
    # First, find the for-loop that contains the target line
    target_indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())

    # Walk forward from target to find where the for-loop body ends.
    # The for-loop body has indentation >= target_indent.
    # We want to insert AFTER the last line of the for-loop body.
    insert_idx = line_idx + 1
    for i in range(line_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == "":
            # Empty line - continue looking
            insert_idx = i + 1
            continue
        line_indent = len(lines[i]) - len(lines[i].lstrip())
        if line_indent < target_indent:
            # We've exited the loop body
            insert_idx = i
            break
        insert_idx = i + 1

    # Determine indentation for the inserted block (same as the for-loop's parent)
    # The for-loop body is at target_indent, so the for statement is one level up
    # We want to insert at the for statement's level (the parent block)
    parent_indent = target_indent - 4
    if parent_indent < 0:
        parent_indent = 0

    # Check: we want to insert right before 'grads = tree_unflatten' if it exists
    # at or near insert_idx
    for i in range(max(0, insert_idx - 2), min(len(lines), insert_idx + 3)):
        if "tree_unflatten" in lines[i]:
            insert_idx = i
            parent_indent = len(lines[i]) - len(lines[i].lstrip())
            break

    indent = " " * parent_indent
    escaped_path = gradient_log_path.replace("\\", "\\\\").replace("'", "\\'")

    logging_block = f"""{indent}# --- GRADIENT ATTRIBUTION LOGGING ---
{indent}import json as _json_log
{indent}_gradient_log_path = '{escaped_path}'
{indent}_grad_norms = {{}}
{indent}for _k, _v in accum.items():
{indent}    _grad_norms[_k] = float(mx.sqrt(mx.sum(_v * _v)).item())
{indent}with open(_gradient_log_path, 'a') as _f:
{indent}    _f.write(_json_log.dumps({{"step": step, "elapsed_ms": approx_train_time_ms if 'approx_train_time_ms' in dir() else 0.0, "val_loss": float(train_loss.item()), "lr_mul": float(lr_mul), "layer_norms": _grad_norms}}) + '\\n')
{indent}# --- END GRADIENT ATTRIBUTION LOGGING ---
"""

    # Insert the block
    result_lines = lines[:insert_idx] + logging_block.splitlines() + lines[insert_idx:]
    return "\n".join(result_lines)


def parse_jsonlines(content: str) -> list[dict]:
    """Parse JSON-lines content into a list of dicts."""
    records = []
    for line in content.strip().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def detect_phase_boundaries(records: list[dict]) -> dict:
    """Detect phase boundaries from lr_mul transitions.

    Warmup end: first step where lr_mul reaches 1.0.
    Warmdown start: first step where lr_mul drops below 1.0 after being at 1.0.
    """
    warmup_end_step = None
    warmdown_start_step = None
    reached_full_lr = False

    for rec in records:
        lr = rec.get("lr_mul", 0.0)
        step = rec.get("step", 0)

        if not reached_full_lr and abs(lr - 1.0) < 1e-6:
            reached_full_lr = True
            warmup_end_step = step
        elif reached_full_lr and lr < 1.0 - 1e-6:
            warmdown_start_step = step
            break

    return {
        "warmup_end_step": warmup_end_step,
        "warmdown_start_step": warmdown_start_step,
        "warmdown_mode": "step_based",  # default; can be refined with wallclock info
    }


def validate_patched_source(patched: str) -> None:
    """Validate the patched source is valid Python via ast.parse()."""
    try:
        ast.parse(patched)
    except SyntaxError as e:
        raise ValueError(f"Patched source has syntax errors: {e}") from e


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Gradient attribution: per-layer norm logging")
    parser.add_argument("--script", required=True, help="Path to train_gpt_mlx.py")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--val-every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--gradient-log", default=None, help="Path for JSON-lines gradient log")
    args = parser.parse_args()

    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"Error: script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    source = script_path.read_text(encoding="utf-8")

    # Find last accumulate_flat_grads
    line_idx, line_text = find_last_accumulate_flat_grads(source)
    print(f"Found last accumulate_flat_grads at line {line_idx + 1}: {line_text.strip()}")

    # Validate sentinel
    validate_sentinel(source, line_idx)
    print("Sentinel validation passed.")

    # Determine gradient log path
    grad_log_path = args.gradient_log or str(
        Path(args.output).parent / "gradient_norms.jsonl"
    )

    # Instrument source
    patched = instrument_source(source, gradient_log_path=grad_log_path)

    # Validate patched source
    validate_patched_source(patched)
    print("Patched source is valid Python.")

    # Write instrumented copy
    instrumented_path = script_path.parent / "train_gpt_mlx_instrumented.py"
    instrumented_path.write_text(patched, encoding="utf-8")
    print(f"Instrumented script written to {instrumented_path}")

    # If gradient log exists, parse it and produce output
    output = _build_output_from_log(grad_log_path)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Output written to {args.output}")


def _build_output_from_log(grad_log_path: str) -> dict:
    """Build output JSON from gradient log file, or return empty structure."""
    log_path = Path(grad_log_path)
    if not log_path.exists():
        return {
            "phase_boundaries": {
                "warmup_end_step": None,
                "warmdown_start_step": None,
                "warmdown_mode": "step_based",
            },
            "per_step_norms": [],
            "phase_correlations": {},
        }

    content = log_path.read_text(encoding="utf-8")
    records = parse_jsonlines(content)
    boundaries = detect_phase_boundaries(records)

    # Compute per-phase correlations between layer norms and val_loss
    phase_correlations = _compute_phase_correlations(records, boundaries)

    return {
        "phase_boundaries": boundaries,
        "per_step_norms": records,
        "phase_correlations": phase_correlations,
    }


def _compute_phase_correlations(records, boundaries):
    """Correlate per-layer gradient norms with val_loss within each phase."""
    import numpy as np

    if not records or not any(r.get("layer_norms") for r in records):
        return {}

    warmup_end = boundaries.get("warmup_end_step", 20)
    warmdown_start = boundaries.get("warmdown_start_step", len(records))

    phases = {
        "warmup": [r for r in records if r["step"] <= warmup_end and r.get("layer_norms")],
        "main": [r for r in records if warmup_end < r["step"] < warmdown_start and r.get("layer_norms")],
        "warmdown": [r for r in records if r["step"] >= warmdown_start and r.get("layer_norms")],
    }

    # Collect all param names from first record with norms
    param_names = set()
    for r in records:
        if r.get("layer_norms"):
            param_names = set(r["layer_norms"].keys())
            break

    correlations = {}
    for phase_name, phase_records in phases.items():
        if len(phase_records) < 3:  # Need ≥3 points for correlation
            correlations[phase_name] = {}
            continue
        val_losses = np.array([r.get("val_loss", 0) for r in phase_records])
        phase_corr = {}
        for param in param_names:
            norms = np.array([r["layer_norms"].get(param, 0) for r in phase_records])
            if np.std(norms) < 1e-10 or np.std(val_losses) < 1e-10:
                phase_corr[param] = {"correlation": 0.0, "p_value": 1.0}
                continue
            from scipy.stats import pearsonr
            corr, p_val = pearsonr(norms, val_losses)
            phase_corr[param] = {"correlation": float(corr), "p_value": float(p_val)}
        correlations[phase_name] = phase_corr

    return correlations


if __name__ == "__main__":
    main()
