#!/usr/bin/env python3
"""Phase 4: Path A PPM-D C++ backend CPU benchmark CLI.

Measures per-second probe throughput of `_ppmd_cpp.score_path_a_arrays` on a
synthetic workload (or, when fed a `--prefix-slice-path`, a real cached val
slice) so we can project the full non-record evaluation wallclock on a SLURM
`cpu_short`-routed node.

Throughput definition
---------------------
A "probe" is one trie terminal visit (one candidate-token prefix-probability
contribution to the position's normalization Z). Since each scoring position
walks the full vocab trie, we approximate the visited-terminal count per
position as `vocab_size`; the Phase 3 trie shares prefixes, so the true visit
count is <= `positions * vocab_size`. We use this upper bound as the probe
count:

    probes_per_second_estimate = (positions * vocab_size) / wallclock_seconds
    projected_full_eval_seconds = full_eval_probe_budget / probes_per_second_estimate

Projection-direction note (NOT a one-sided bound): because
`positions * vocab_size` is an UPPER bound on true probe visits, the per-second
estimate is an UPPER bound on the true rate, and the projected wallclock would
be a LOWER bound (i.e. optimistic) IF the budget denominator were a true count.
However, the default `full_eval_probe_budget` of 7.43e12 is itself derived from
the same `positions * vocab_size` upper-bound convention (per the cost
estimate in plans/path-a-ppmd-cpp-backend-plan.md). The numerator and
denominator share that convention, so the bias cancels for proportionally
scaling workloads and the projection acts as a SELF-CONSISTENT LINEAR
EXTRAPOLATION rather than a true upper or lower bound. To obtain real
one-sided bounds, future work should expose true terminal-visit counts from
the C++ scorer and supply a matching real-count budget via
`--full-eval-probe-budget`.

Usage (CPU-only login-node smoke):
    .venv-smoke/bin/python scripts/ppmd_cpp/bench_cpu.py \
        --mode synthetic --positions 64 --vocab 256

The synthetic mode does NOT require any cached data and runs on the login node.
The prefix-slice mode is gated behind a clean error (exit 3) if the file is
absent, since real cached prefix slices are not part of this phase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BUILD_DIR = _REPO_ROOT / "scripts" / "ppmd_cpp"
if str(_BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILD_DIR))

# Phase 4 default probe-budget anchor for projecting full-eval wallclock; see
# module docstring. Overridable per-invocation via --full-eval-probe-budget.
_DEFAULT_FULL_EVAL_PROBE_BUDGET = 7.43e12


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bench_cpu.py",
        description="Path A PPM-D C++ backend CPU benchmark.",
    )
    p.add_argument("--mode", required=True, choices=["synthetic", "prefix-slice"],
                   help="synthetic: random vocab; prefix-slice: load a cached .npz")
    p.add_argument("--positions", type=int, default=1024,
                   help="number of scoring positions (default 1024)")
    p.add_argument("--vocab", type=int, default=8192,
                   help="synthetic vocab size (default 8192)")
    p.add_argument("--avg-bytes-per-token", type=float, default=3.7,
                   help="synthetic mean token byte length (default 3.7)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                   help="OpenMP thread count (default os.cpu_count())")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0)")
    p.add_argument("--results-dir", type=str,
                   default=str(_REPO_ROOT / "results" / "ppmd_cpp_bench"),
                   help="results JSON directory")
    p.add_argument("--results-name", type=str, default=None,
                   help="results JSON basename (without .json); "
                        "defaults to <mode>_<utc_timestamp>")
    p.add_argument("--no-write", action="store_true",
                   help="skip writing the results JSON file")
    p.add_argument("--prefix-slice-path", type=str, default=None,
                   help="prefix-slice mode: path to .npz with target_ids/prev_ids/"
                        "nll_nats/vocab tables")
    p.add_argument("--full-eval-probe-budget", type=float,
                   default=_DEFAULT_FULL_EVAL_PROBE_BUDGET,
                   help="probe-count budget used as projection numerator "
                        f"(default {_DEFAULT_FULL_EVAL_PROBE_BUDGET:g})")
    return p.parse_args(argv)


def _build_synthetic_vocab(rng: np.random.Generator, vocab_size: int,
                           avg_bytes_per_token: float) -> tuple[
                               np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a deterministic random vocab with byte lengths Poisson(mean=avg).

    Returns (boundary_bytes, boundary_offsets, emittable, is_boundary).
    Lengths are clipped to [1, 16] so the trie depth is bounded for the bench.
    The same byte string is reused for both boundary and non-boundary tables
    (the bench does not exercise leading-space markers). The C++ scorer in
    scripts/ppmd_cpp/src/scorer.cpp::score_path_a_arrays treats the two table
    pointers as opaque inputs and does NOT short-circuit on identity, so this
    sharing does not bias the throughput measurement.
    """
    raw_lens = rng.poisson(lam=max(avg_bytes_per_token, 0.1), size=vocab_size)
    lens = np.clip(raw_lens, 1, 16).astype(np.int32)
    total = int(lens.sum())
    flat = rng.integers(0, 256, size=total, dtype=np.uint8)
    offsets = np.zeros(vocab_size + 1, dtype=np.int32)
    np.cumsum(lens, out=offsets[1:])
    emittable = np.ones(vocab_size, dtype=np.uint8)
    is_boundary = (rng.random(vocab_size) < 0.3).astype(np.uint8)
    return flat, offsets, emittable, is_boundary


def _run_synthetic(args: argparse.Namespace) -> dict:
    try:
        import _ppmd_cpp  # type: ignore
    except ImportError as e:
        print(f"ERROR: _ppmd_cpp extension not built: {e}", file=sys.stderr)
        sys.exit(2)

    if hasattr(_ppmd_cpp, "set_num_threads"):
        _ppmd_cpp.set_num_threads(int(args.threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(int(args.threads)))

    rng = np.random.default_rng(args.seed)
    flat, offsets, emittable, is_boundary = _build_synthetic_vocab(
        rng, args.vocab, args.avg_bytes_per_token)

    target_ids = rng.integers(0, args.vocab, size=args.positions, dtype=np.int32)
    prev_ids = np.empty(args.positions, dtype=np.int32)
    prev_ids[0] = -1
    if args.positions > 1:
        prev_ids[1:] = target_ids[:-1]
    # NLL nats roughly in [0.1, 6.0] so exp(-nll) stays well above 0.
    nll_nats = rng.uniform(0.1, 6.0, size=args.positions).astype(np.float64)

    hyperparams = {
        "order": 5,
        "lambda_hi": 0.9,
        "lambda_lo": 0.05,
        "conf_threshold": 0.9,
        "update_after_score": True,
    }

    t0 = time.perf_counter()
    result = _ppmd_cpp.score_path_a_arrays(
        target_ids, prev_ids, nll_nats,
        flat, offsets,            # boundary table
        flat, offsets,            # non-boundary table (same)
        emittable, is_boundary,
        hyperparams,
    )
    wallclock = time.perf_counter() - t0

    probe_count = float(args.positions) * float(args.vocab)
    pps = probe_count / wallclock if wallclock > 0.0 else float("inf")
    budget = float(args.full_eval_probe_budget)
    projected = (budget / pps) if pps > 0.0 else float("inf")

    return {
        "mode": "synthetic",
        "positions": int(args.positions),
        "vocab": int(args.vocab),
        "avg_bytes_per_token": float(args.avg_bytes_per_token),
        "threads": int(args.threads),
        "seed": int(args.seed),
        "wallclock_seconds": float(wallclock),
        "total_bits": float(result["total_bits"]),
        "total_bytes": int(result["total_bytes"]),
        "bpb": float(result["bpb"]),
        "probes_per_second_estimate": float(pps),
        "projected_full_eval_seconds": float(projected),
        "full_eval_probe_budget": float(budget),
        "start_state_digest": str(result.get("start_state_digest", "")),
        "end_state_digest": str(result.get("end_state_digest", "")),
    }


def _run_prefix_slice(args: argparse.Namespace) -> dict:
    if not args.prefix_slice_path:
        print("ERROR: --prefix-slice-path required for --mode prefix-slice",
              file=sys.stderr)
        sys.exit(3)
    if not Path(args.prefix_slice_path).exists():
        print(f"ERROR: prefix-slice file not found: {args.prefix_slice_path}",
              file=sys.stderr)
        sys.exit(3)
    try:
        import _ppmd_cpp  # noqa: F401  # type: ignore
    except ImportError as e:
        print(f"ERROR: _ppmd_cpp extension not built: {e}", file=sys.stderr)
        sys.exit(2)
    # Real prefix-slice loader is intentionally out of scope for Phase 4; the
    # SLURM bench script defaults to --mode synthetic. Bail with a clear error.
    print("ERROR: prefix-slice loader not implemented in Phase 4; use --mode synthetic",
          file=sys.stderr)
    sys.exit(3)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "synthetic":
        payload = _run_synthetic(args)
    else:
        payload = _run_prefix_slice(args)

    line = json.dumps(payload, sort_keys=True)
    print(line)

    if not args.no_write:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        name = args.results_name
        if not name:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = f"{args.mode}_{stamp}"
        out_path = results_dir / f"{name}.json"
        out_path.write_text(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
