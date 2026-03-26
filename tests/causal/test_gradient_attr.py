"""Tests for scripts/causal/gradient_attribution.py."""
from __future__ import annotations

import ast
import json
import os
import textwrap

import pytest

from scripts.causal import gradient_attribution as ga


# ---------------------------------------------------------------------------
# T18 Tests
# ---------------------------------------------------------------------------


MOCK_SOURCE_MULTI = textwrap.dedent("""\
    def accumulate_flat_grads(accum, grads, scale):
        pass

    # warmup loop
    accum = accumulate_flat_grads(accum, grads, grad_scale)
    warmup_loss = warmup_loss + loss

    # main training loop
    lr_mul = args.lr_mul(step, train_time_ms)
    accum = accumulate_flat_grads(accum, grads, grad_scale)
    train_loss = train_loss + loss.astype(mx.float32) * grad_scale
""")


def test_last_occurrence_targeting():
    """Mock source with multiple accumulate_flat_grads, finds LAST."""
    line_idx, line_text = ga.find_last_accumulate_flat_grads(MOCK_SOURCE_MULTI)

    # The LAST occurrence is in the main training loop section
    assert "accumulate_flat_grads" in line_text
    # Count occurrences - there are 3 total (def, warmup call, main call)
    lines = MOCK_SOURCE_MULTI.splitlines()
    all_indices = [
        i for i, l in enumerate(lines)
        if "accumulate_flat_grads" in l and not l.strip().startswith("def ")
    ]
    # The returned index should be the last call site (not the def)
    assert line_idx == all_indices[-1]


def test_sentinel_validation_passes():
    """Current train_gpt_mlx.py passes sentinel check."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "train_gpt_mlx.py"
    )
    source_path = os.path.normpath(source_path)
    source = open(source_path, encoding="utf-8").read()

    line_idx, _line_text = ga.find_last_accumulate_flat_grads(source)
    # Should not raise
    ga.validate_sentinel(source, line_idx)


def test_sentinel_validation_fails():
    """Modified source fails sentinel validation."""
    # Source with accumulate_flat_grads but without the sentinel markers nearby
    bad_source = textwrap.dedent("""\
        something_unrelated = True
        accum = accumulate_flat_grads(accum, grads, grad_scale)
        another_unrelated = False
    """)
    line_idx, _ = ga.find_last_accumulate_flat_grads(bad_source)
    with pytest.raises(ValueError, match="(?i)sentinel"):
        ga.validate_sentinel(bad_source, line_idx)


def test_patched_file_syntax():
    """ast.parse() on instrumented output produces valid Python."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "train_gpt_mlx.py"
    )
    source_path = os.path.normpath(source_path)
    source = open(source_path, encoding="utf-8").read()

    patched = ga.instrument_source(source, gradient_log_path="/tmp/test_grad.jsonl")

    # Must be valid Python
    tree = ast.parse(patched)
    assert tree is not None

    # Must contain the logging sentinel comments
    assert "# --- GRADIENT ATTRIBUTION LOGGING ---" in patched
    assert "# --- END GRADIENT ATTRIBUTION LOGGING ---" in patched


def test_jsonlines_parsing():
    """Mock JSON-lines, verify parsing."""
    lines = [
        json.dumps({"step": 1, "elapsed_ms": 100.0, "val_loss": 3.5, "lr_mul": 0.5, "layer_norms": {"a": 0.1}}),
        json.dumps({"step": 2, "elapsed_ms": 200.0, "val_loss": 3.3, "lr_mul": 1.0, "layer_norms": {"a": 0.2}}),
        json.dumps({"step": 3, "elapsed_ms": 300.0, "val_loss": 3.1, "lr_mul": 1.0, "layer_norms": {"a": 0.15}}),
        json.dumps({"step": 4, "elapsed_ms": 400.0, "val_loss": 2.9, "lr_mul": 0.8, "layer_norms": {"a": 0.12}}),
    ]
    content = "\n".join(lines)

    records = ga.parse_jsonlines(content)
    assert len(records) == 4
    assert records[0]["step"] == 1
    assert records[3]["lr_mul"] == 0.8


def test_phase_boundary_detection():
    """Detect warmdown onset from lr_mul transitions."""
    records = [
        {"step": 1, "lr_mul": 0.5},
        {"step": 2, "lr_mul": 1.0},
        {"step": 3, "lr_mul": 1.0},
        {"step": 4, "lr_mul": 0.9},
        {"step": 5, "lr_mul": 0.7},
    ]
    boundaries = ga.detect_phase_boundaries(records)
    # Warmdown starts at step 4 (first transition from 1.0 to <1.0)
    assert boundaries["warmdown_start_step"] == 4


def test_instrument_inserts_after_accumulation():
    """Verify the instrumentation code is inserted after the accumulation loop."""
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "train_gpt_mlx.py"
    )
    source_path = os.path.normpath(source_path)
    source = open(source_path, encoding="utf-8").read()

    patched = ga.instrument_source(source, gradient_log_path="/tmp/test.jsonl")
    patched_lines = patched.splitlines()

    # Find the logging block
    logging_start = None
    for i, line in enumerate(patched_lines):
        if "# --- GRADIENT ATTRIBUTION LOGGING ---" in line:
            logging_start = i
            break

    assert logging_start is not None

    # The accumulate_flat_grads call should appear before the logging block
    # Look backwards from logging_start for accumulate_flat_grads
    found_accum = False
    for i in range(logging_start - 1, max(0, logging_start - 20), -1):
        if "accumulate_flat_grads" in patched_lines[i]:
            found_accum = True
            break
    assert found_accum, "accumulate_flat_grads should appear before the logging block"
