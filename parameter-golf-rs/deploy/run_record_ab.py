"""Step-time A/B harness for record-shaped runs.

Phase 0 deliverable from the record-readiness plan: drive two `run_detached.py`
invocations (a baseline and a candidate) with the same shape, parse the
`run_timing_json={...}` line that `pg-train run` now emits, and report a
per-stage delta table along with median/p90 step-time.

The script is opinionated about what an A/B run looks like:

  * each side is run `--repeats` times (default 3), warm-up steps are required
    to be discarded inside the binary (`PG_GPU_BACKWARD_STAGE_TIMING_ENABLED=1`
    plus the spec's normal warm-up handle that, the script just collects what
    `pg-train` already exposes per run),
  * each invocation produces one `run_timing_json={...}` line, which we read
    from stdout (or from `--result-json` if you pass one through `--extra-flag`
    on each side),
  * the candidate "wins" only if median total step time is lower AND p90 of
    the candidate is at most p90 of the baseline. Per-stage deltas are always
    printed for transparency.

Usage:

    python deploy/run_record_ab.py \\
        --baseline-flags "run --spec specs/frontier_1855_merged_target.toml" \\
        --candidate-flags "run --spec specs/frontier_1855_merged_target.toml --enable-bf16-backward-chain" \\
        --repeats 3

The flags strings are fed verbatim to `cargo run --release -p pg-train -- ...`
(or, with `--remote`, to the Modal `run_detached.py` entrypoint). Everything
after `pg-train` is yours to set: spec path, env overrides, extra toggles.

This script does NOT itself add `PG_GPU_BACKWARD_STAGE_TIMING_ENABLED=1`; pass
it via the env-flag macros that `run_detached.py` already understands, or set
it in your shell before running. The harness is intentionally thin so it works
both locally (cargo run) and on Modal.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass

# Stage rows we always show in the delta table, in display order. Anything in
# `run_timing_json` that ends in `_per_step` is preserved if it's listed here;
# everything else is summarised by the totals row only.
STAGE_ROWS_PER_STEP: tuple[str, ...] = (
    "timing_train_step_ms_per_step",
    "timing_cuda_h2d_ms_per_step",
    "timing_cuda_zero_grads_ms_per_step",
    "timing_cuda_backward_ms_per_step",
    "timing_cuda_backward_forward_ms_per_step",
    "timing_cuda_backward_forward_logits_ms_per_step",
    "timing_cuda_backward_block_recompute_ms_per_step",
    "timing_cuda_backward_block_attn_out_ms_per_step",
    "timing_cuda_backward_block_attention_ms_per_step",
    "timing_cuda_backward_block_attention_sdpa_ms_per_step",
    "timing_cuda_backward_block_attention_xsa_accum_ms_per_step",
    "timing_cuda_backward_block_qkv_ms_per_step",
    "timing_cuda_backward_block_qkv_rope_ms_per_step",
    "timing_cuda_backward_block_qkv_proj_ms_per_step",
    "timing_cuda_backward_block_qkv_norm_resid_ms_per_step",
    "timing_cuda_backward_block_mlp_ms_per_step",
    "timing_cuda_backward_block_mlp_residual_ms_per_step",
    "timing_cuda_backward_block_mlp_down_ms_per_step",
    "timing_cuda_backward_block_mlp_act_ms_per_step",
    "timing_cuda_backward_block_mlp_up_ms_per_step",
    "timing_cuda_backward_block_mlp_norm_ms_per_step",
    "timing_cuda_backward_output_ms_per_step",
    "timing_cuda_non_bank_sync_ms_per_step",
    "timing_cuda_bank_update_ms_per_step",
    "timing_cuda_non_bank_update_ms_per_step",
    "timing_post_train_sync_ms_per_step",
    "timing_eval_ms_per_step",
)

AUDIT_ROWS: tuple[str, ...] = (
    "timing_recurrent_active_steps",
    "timing_recurrent_inactive_steps",
    "cuda_backward_graph_launches_active",
    "cuda_backward_graph_launches_inactive",
    "cuda_backward_graph_captures_active",
    "cuda_backward_graph_captures_inactive",
    "cuda_backward_graph_warmups_active",
    "cuda_backward_graph_warmups_inactive",
    "cuda_backward_graph_resets",
    "backward_nccl_bucket_overlap_windows",
    "backward_nccl_bucket_overlap_confirmed",
    "backward_nccl_bucket_overlap_max_window_ms",
)

RUN_TIMING_PREFIX = "run_timing_json="


@dataclass
class RunOutcome:
    label: str
    flags: str
    json_blob: dict
    returncode: int


def _resolve_modal_bin(modal_bin: str | None) -> str:
    if modal_bin:
        return modal_bin
    env_bin = os.environ.get("MODAL_BIN")
    if env_bin:
        return env_bin
    for candidate in (".venv/bin/modal", "../.venv/bin/modal", "modal"):
        if os.path.exists(candidate) or shutil.which(candidate):
            return candidate
    return "modal"


def _build_command(
    flags: str,
    *,
    remote: bool,
    cargo_target: str,
    modal_bin: str | None,
) -> list[str]:
    if remote:
        return [
            _resolve_modal_bin(modal_bin),
            "run",
            "deploy/run_detached.py",
            *shlex.split(flags),
        ]
    return [
        "cargo",
        "run",
        "--release",
        "-p",
        "pg-train",
        "--target-dir",
        cargo_target,
        "--",
        *shlex.split(flags),
    ]


def _extract_timing_json(stdout: str) -> dict | None:
    # Walk lines from the bottom — the timing line is emitted right before the
    # subcommand returns, so we usually want the last match. There can be only
    # one per `run` invocation, but tests / sweeps may interleave noise.
    last: dict | None = None
    for line in stdout.splitlines():
        if " result:" in line:
            continue
        idx = line.find(RUN_TIMING_PREFIX)
        if idx < 0:
            continue
        payload = line[idx + len(RUN_TIMING_PREFIX):].strip()
        try:
            last = json.loads(payload)
        except json.JSONDecodeError as err:
            print(
                f"warning: could not parse run_timing_json line: {err}\n  line={line!r}",
                file=sys.stderr,
            )
    return last


def _extract_modal_result_metrics(stdout: str) -> dict | None:
    """Fallback for Modal runs if stdout swallowed the raw timing line."""
    for line in reversed(stdout.splitlines()):
        marker = "result:"
        idx = line.find(marker)
        if idx < 0:
            continue
        payload = line[idx + len(marker):].strip()
        try:
            value = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return None


def _run_one(
    label: str,
    flags: str,
    *,
    remote: bool,
    cargo_target: str,
    modal_bin: str | None,
    timing_json_dir: str | None,
    warmup_steps: int,
) -> RunOutcome:
    result_json_path: Path | None = None
    effective_flags = flags
    if timing_json_dir:
        safe_label = label.replace("#", "_").replace("/", "_")
        if remote:
            effective_flags = f"{flags} --result-json /output/record_ab/{shlex.quote(safe_label)}.json"
        else:
            result_dir = Path(timing_json_dir)
            result_dir.mkdir(parents=True, exist_ok=True)
            result_json_path = result_dir / f"{safe_label}.json"
            effective_flags = f"{flags} --result-json {shlex.quote(str(result_json_path))}"
    if remote:
        tokens = shlex.split(effective_flags)
        if "--modal-wait" not in tokens:
            effective_flags = f"--modal-wait {effective_flags}"
            tokens = shlex.split(effective_flags)
        if "--record-timing-skip-steps" not in tokens:
            effective_flags = f"{effective_flags} --record-timing-skip-steps {max(0, warmup_steps)}"
        if (
            "--record-shaped-proxy-max-steps" not in tokens
            and "--mode" in tokens
            and tokens[tokens.index("--mode") + 1 : tokens.index("--mode") + 2]
            == ["record-shaped-proxy"]
        ):
            proxy_steps = max(8, warmup_steps + 20)
            effective_flags = f"{effective_flags} --record-shaped-proxy-max-steps {proxy_steps}"
    cmd = _build_command(
        effective_flags,
        remote=remote,
        cargo_target=cargo_target,
        modal_bin=modal_bin,
    )
    print(f"[{label}] running: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    env = os.environ.copy()
    env["PG_RECORD_TIMING_SKIP_STEPS"] = str(max(0, warmup_steps))
    if remote:
        env["PG_WAIT"] = "1"
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    timing = None
    if result_json_path and result_json_path.exists():
        try:
            timing = json.loads(result_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as err:
            print(f"warning: could not parse {result_json_path}: {err}", file=sys.stderr)
    if timing is None:
        timing = _extract_timing_json(proc.stdout)
    if timing is None and remote:
        timing = _extract_modal_result_metrics(proc.stdout)
    if timing is None:
        raise RuntimeError(
            f"[{label}] run did not emit run_timing_json= line "
            f"(returncode={proc.returncode}); did pg-train build with the latest changes?"
        )
    return RunOutcome(
        label=label,
        flags=flags,
        json_blob=timing,
        returncode=proc.returncode,
    )


def _stat(values: list[float]) -> tuple[float, float, float]:
    """Return (median, p90, mean). For len(values) < 2 p90 == max."""
    if not values:
        return 0.0, 0.0, 0.0
    median = statistics.median(values)
    if len(values) == 1:
        return median, values[0], values[0]
    sorted_v = sorted(values)
    rank = max(0, min(len(sorted_v) - 1, int(round(0.9 * (len(sorted_v) - 1)))))
    return median, sorted_v[rank], statistics.fmean(values)


def _gather(samples: list[RunOutcome], key: str) -> list[float]:
    out: list[float] = []
    for sample in samples:
        v = sample.json_blob.get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _format_row(name: str, base: list[float], cand: list[float]) -> str:
    b_med, b_p90, _ = _stat(base)
    c_med, c_p90, _ = _stat(cand)
    delta_med = c_med - b_med
    delta_med_pct = (delta_med / b_med * 100.0) if b_med else 0.0
    sign = "+" if delta_med > 0 else ""
    return (
        f"{name:<55}  "
        f"{b_med:>9.3f} / {b_p90:>9.3f}  "
        f"{c_med:>9.3f} / {c_p90:>9.3f}  "
        f"{sign}{delta_med:>+8.3f}  ({sign}{delta_med_pct:>+6.2f}%)"
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run baseline and candidate pg-train configs and print step-time deltas",
    )
    p.add_argument(
        "--baseline-flags",
        required=True,
        help="Flag string passed to `pg-train`. Example: 'run --spec specs/frontier_1855_merged_target.toml'",
    )
    p.add_argument(
        "--candidate-flags",
        required=True,
        help="Flag string for the candidate run.",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=30,
        help="Set PG_RECORD_TIMING_SKIP_STEPS for each child run. Default: 30.",
    )
    p.add_argument(
        "--remote",
        action="store_true",
        help="Drive deploy/run_detached.py instead of cargo run (for Modal H100 A/Bs).",
    )
    p.add_argument(
        "--modal-bin",
        help="Modal CLI path for --remote. Defaults to MODAL_BIN, .venv/bin/modal, ../.venv/bin/modal, then modal.",
    )
    p.add_argument(
        "--cargo-target",
        default="target/ab",
        help="Cargo --target-dir to use for local builds. Default keeps A/B builds out of the main target/ tree.",
    )
    p.add_argument(
        "--summary-json",
        help="If set, write the median/p90 deltas as JSON to this path.",
    )
    p.add_argument(
        "--timing-json-dir",
        help="If set, ask each pg-train run to write its run_timing_json to this directory.",
    )
    p.add_argument(
        "--require-candidate-win",
        action="store_true",
        help="Exit nonzero if candidate median/p90 does not beat baseline.",
    )
    args = p.parse_args()

    if args.repeats < 1:
        print("--repeats must be >= 1", file=sys.stderr)
        return 2

    baseline_runs: list[RunOutcome] = []
    candidate_runs: list[RunOutcome] = []
    for i in range(args.repeats):
        baseline_runs.append(
            _run_one(
                f"baseline#{i + 1}",
                args.baseline_flags,
                remote=args.remote,
                cargo_target=args.cargo_target,
                modal_bin=args.modal_bin,
                timing_json_dir=args.timing_json_dir,
                warmup_steps=args.warmup_steps,
            )
        )
        candidate_runs.append(
            _run_one(
                f"candidate#{i + 1}",
                args.candidate_flags,
                remote=args.remote,
                cargo_target=args.cargo_target,
                modal_bin=args.modal_bin,
                timing_json_dir=args.timing_json_dir,
                warmup_steps=args.warmup_steps,
            )
        )

    print()
    print("=" * 120)
    print(
        f"{'stage':<55}  {'baseline med / p90':>21}  {'candidate med / p90':>21}  {'delta':>9}  {'pct':>9}"
    )
    print("-" * 120)

    summary: dict[str, dict] = {}
    for stage in STAGE_ROWS_PER_STEP:
        b = _gather(baseline_runs, stage)
        c = _gather(candidate_runs, stage)
        if not b and not c:
            continue
        print(_format_row(stage, b, c))
        b_med, b_p90, b_mean = _stat(b)
        c_med, c_p90, c_mean = _stat(c)
        summary[stage] = {
            "baseline": {"median": b_med, "p90": b_p90, "mean": b_mean, "n": len(b)},
            "candidate": {"median": c_med, "p90": c_p90, "mean": c_mean, "n": len(c)},
            "delta_median": c_med - b_med,
        }
    for field in AUDIT_ROWS:
        b = _gather(baseline_runs, field)
        c = _gather(candidate_runs, field)
        if not b and not c:
            continue
        print(_format_row(field, b, c))
        b_med, b_p90, b_mean = _stat(b)
        c_med, c_p90, c_mean = _stat(c)
        summary[field] = {
            "baseline": {"median": b_med, "p90": b_p90, "mean": b_mean, "n": len(b)},
            "candidate": {"median": c_med, "p90": c_p90, "mean": c_mean, "n": len(c)},
            "delta_median": c_med - b_med,
        }
    print("=" * 120)

    headline = "timing_train_step_ms_per_step"
    base_total = _gather(baseline_runs, headline)
    cand_total = _gather(candidate_runs, headline)
    b_med, b_p90, _ = _stat(base_total)
    c_med, c_p90, _ = _stat(cand_total)
    won = c_med < b_med and c_p90 <= b_p90
    print(
        f"\nverdict: candidate {'WINS' if won else 'does not win'}  "
        f"(median {b_med:.3f} -> {c_med:.3f} ms, p90 {b_p90:.3f} -> {c_p90:.3f} ms)"
    )

    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "baseline_flags": args.baseline_flags,
                    "candidate_flags": args.candidate_flags,
                    "repeats": args.repeats,
                    "warmup_steps": args.warmup_steps,
                    "headline_step_time_ms": {
                        "baseline_median": b_med,
                        "baseline_p90": b_p90,
                        "candidate_median": c_med,
                        "candidate_p90": c_p90,
                    },
                    "candidate_wins": won,
                    "stages": summary,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

    return 0 if won or not args.require_candidate_win else 1


if __name__ == "__main__":
    raise SystemExit(main())
