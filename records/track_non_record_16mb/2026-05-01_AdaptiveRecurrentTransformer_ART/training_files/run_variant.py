from __future__ import annotations

import datetime
import os
import runpy
import sys
import uuid
from pathlib import Path

COMMON_DEFAULTS = {
    "TTT_ENABLED": "0",
    "SLIDING_WINDOW_ENABLED": "0",
    "ART_QUANT_DIAG_ENABLED": "0",
    "RAW_ROUNDTRIP_CHECK_ENABLED": "0",
    "ART_EVAL_ROUTE_STATS": "1",
    "RECURRENCE_PROBE_EVERY": "10",
    "TRAIN_LOG_EVERY": "100",
}


VARIANTS = {
    "baseline_prettt": {
        "notes": "Exact RoPE-fix triple ART score path through quantized eval, with TTT disabled.",
        "env": {},
    },
    "seed2_prettt": {
        "notes": "Same as baseline with a second seed to estimate 1x variance before paying for 8x.",
        "env": {"SEED": "7331"},
    },
    "eval_argmax_prettt": {
        "notes": "Train with sampled ART, evaluate with deterministic argmax routes to estimate route-sampling dependence.",
        "env": {"ART_EVAL_SAMPLE_ROUTES": "0"},
    },
    "eval_threshold055_prettt": {
        "notes": "Deterministic eval with a stricter continue threshold, testing whether marginal routes are hurting quantized score.",
        "env": {
            "ART_EVAL_SAMPLE_ROUTES": "0",
            "ART_EVAL_ROUTE_THRESHOLD": "0.6",
        },
    },
    "penalty_low_prettt": {
        "notes": "Lower cycle penalty; tests whether 8x compute should spend more recurrence for bpb.",
        "env": {"ART_CYCLE_PENALTY": "0.0001"},
    },
    "penalty_high_prettt": {
        "notes": "Higher cycle penalty; tests whether current avg_cycles can be reduced without losing pre-TTT score.",
        "env": {"ART_CYCLE_PENALTY": "0.003"},
    },
    "entropy_low_prettt": {
        "notes": "Lower router entropy reward; targets the observed high/noisy route entropy late in training.",
        "env": {
            "ART_ROUTER_ENTROPY_START": "0.02",
            "ART_ROUTER_ENTROPY_END": "0.0",
        },
    },
    "entropy_high_prettt": {
        "notes": "Higher router entropy reward; tests whether the late entropy rebound is beneficial exploration rather than noise.",
        "env": {
            "ART_ROUTER_ENTROPY_START": "0.10",
            "ART_ROUTER_ENTROPY_END": "0.01",
        },
    },
    "art_early_prettt": {
        "notes": "Enable ART earlier so routers train longer under the wallclock schedule.",
        "env": {"ART_HALT_ENABLE_AT": "0.45"},
    },
    "art_late_prettt": {
        "notes": "Enable ART later so the fixed recurrent backbone stabilizes longer before routing.",
        "env": {"ART_HALT_ENABLE_AT": "0.65"},
    },
    "batch_small_prettt": {
        "notes": "Smaller global train batch; probes whether the 1x result depends on high update frequency/noisier gradients.",
        "env": {"TRAIN_BATCH_TOKENS": "524288"},
    },
    "batch_large_prettt": {
        "notes": "Larger global train batch; proxy for whether 8x should spend extra throughput on batch size rather than steps.",
        "env": {"TRAIN_BATCH_TOKENS": "1048576"},
    },
    "ema_low_prettt": {
        "notes": "Lower EMA decay; tests whether post-reset EMA is lagging too much for short 1x runs.",
        "env": {"EMA_DECAY": "0.995"},
    },
    "ema_high_prettt": {
        "notes": "Higher EMA decay; proxy for 8x where more optimizer steps occur inside the same wallclock.",
        "env": {"EMA_DECAY": "0.9985"},
    },
    "routed_gptq_prettt": {
        "notes": "Use routed ART during GPTQ calibration; tests whether calibration should follow adaptive eval routes.",
        "env": {"ART_GPTQ_CALIBRATE_ROUTED": "1"},
    },
    "full_ttt_baseline": {
        "notes": "Baseline plus post-quant TTT, for final confirmation only; expensive.",
        "env": {"TTT_ENABLED": "1"},
    },
}


def _usage() -> None:
    print("Usage: python run_variant.py <variant>")
    print("")
    print("Variants:")
    for name, spec in VARIANTS.items():
        print(f"  {name:24s} {spec['notes']}")


def run_variant(name: str) -> None:
    if name not in VARIANTS:
        _usage()
        raise SystemExit(f"Unknown variant: {name}")

    spec = VARIANTS[name]
    defaults = dict(COMMON_DEFAULTS)
    defaults.update(spec["env"])
    for key, value in defaults.items():
        os.environ.setdefault(key, str(value))

    os.environ.setdefault("EXPERIMENT_VARIANT", name)
    os.environ.setdefault("EXPERIMENT_NOTES", spec["notes"])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ.setdefault("RUN_ID", f"{ts}_{name}_{uuid.uuid4().hex[:8]}")

    base = Path(__file__).with_name("train_gpt_base_prettt.py")
    sys.argv = [str(base), *sys.argv[2:]]
    runpy.run_path(str(base), run_name="__main__")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "list", "--list"}:
        _usage()
        return
    run_variant(sys.argv[1])


if __name__ == "__main__":
    main()
