#!/usr/bin/env python3
"""
auto_experiment.py — Local CPU-only random search over v10 hyperconfigs.

Runs quick smoke tests (200 iterations, tiny batch) to validate config combos
before launching expensive GPU runs. Results logged to experiments.jsonl.

Usage:
    python3 auto_experiment.py [--n_configs 20] [--iterations 200] [--script train_gpt_v10_safe.py]
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent

# ── Config search space ────────────────────────────────────────────────────────

SAFE_SPACE = {
    "hedge_eta": [0.05, 0.1, 0.2, 0.3],
    "ngram_delta": [0.001, 0.01, 0.05, 0.1],
    "ngram_alpha_center": [0.5, 0.6, 0.7, 0.8],
    "ngram_max_order": [8, 10, 11],
    "swa_every": [30, 50, 80],
    "ema_decay": [0.995, 0.997, 0.999],
}

MOONSHOT_SPACE = {
    **SAFE_SPACE,
    "model_dim": [576, 608, 640],
    "mlp_mult": [2.5, 3.0, 3.5],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_config(space: dict, seed: int) -> dict:
    rng = random.Random(seed)
    return {k: rng.choice(v) for k, v in space.items()}


def config_to_env(cfg: dict) -> dict:
    """Convert config dict to environment variable overrides."""
    env_map = {
        "hedge_eta": "HEDGE_ETA",
        "ngram_delta": "NGRAM_DELTA",
        "ngram_alpha_center": "NGRAM_ALPHA_CENTER",
        "ngram_max_order": "NGRAM_MAX_ORDER",
        "swa_every": "SWA_EVERY",
        "ema_decay": "EMA_DECAY",
        "model_dim": "MODEL_DIM",
        "mlp_mult": "MLP_MULT",
    }
    return {env_map[k]: str(v) for k, v in cfg.items() if k in env_map}


def run_smoke_test(script: Path, cfg: dict, iterations: int, seed: int) -> dict:
    """Run a quick smoke test and return result dict."""
    env = os.environ.copy()
    env.update(config_to_env(cfg))
    env.update({
        "ITERATIONS": str(iterations),
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_LOSS_EVERY": "0",        # skip val during smoke
        "VAL_BATCH_SIZE": "4096",
        "SEED": str(seed),
        "RUN_ID": f"auto_{seed}",
        "DISABLE_TTT": "1",           # smoke tests never use TTT
        "SKIP_QUANTIZE": "1",         # skip quant for speed
        "DEVICE": "cpu",
    })

    t0 = time.time()
    result = {
        "script": script.name,
        "seed": seed,
        "config": cfg,
        "timestamp": datetime.utcnow().isoformat(),
        "iterations": iterations,
        "status": "unknown",
        "train_loss": None,
        "wall_seconds": None,
        "error": None,
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - t0
        result["wall_seconds"] = round(elapsed, 1)

        if proc.returncode != 0:
            result["status"] = "error"
            result["error"] = proc.stderr[-2000:] if proc.stderr else "(no stderr)"
        else:
            # Parse final train loss from stdout
            train_loss = _parse_train_loss(proc.stdout)
            result["status"] = "ok"
            result["train_loss"] = train_loss
            result["stdout_tail"] = proc.stdout[-1000:]

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Exceeded 300s timeout"
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = str(exc)

    return result


def _parse_train_loss(stdout: str) -> float | None:
    """Extract last reported train loss from script output."""
    loss = None
    for line in stdout.splitlines():
        # Look for patterns like "step 199 | loss 2.3456" or "train_loss=2.3456"
        for token in line.split():
            if token.startswith("loss") and "=" in token:
                try:
                    loss = float(token.split("=")[-1])
                except ValueError:
                    pass
            try:
                if loss is None and "loss" in line.lower():
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "loss" in p.lower() and i + 1 < len(parts):
                            try:
                                loss = float(parts[i + 1].rstrip(","))
                            except ValueError:
                                pass
            except Exception:
                pass
    return loss


def rank_results(results: list[dict]) -> list[dict]:
    """Sort by train_loss ascending (lower is better), errors last."""
    ok = [r for r in results if r["status"] == "ok" and r["train_loss"] is not None]
    bad = [r for r in results if r not in ok]
    ok.sort(key=lambda r: r["train_loss"])
    return ok + bad


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto random search for v10 configs")
    parser.add_argument("--n_configs", type=int, default=20, help="Number of random configs to try")
    parser.add_argument("--iterations", type=int, default=200, help="Training steps per smoke test")
    parser.add_argument(
        "--script",
        type=str,
        default="train_gpt_v10_safe.py",
        choices=["train_gpt_v10_safe.py", "train_gpt_v10_moonshot.py"],
        help="Which training script to test",
    )
    parser.add_argument("--seed_offset", type=int, default=42, help="Base seed for config sampling")
    parser.add_argument("--dry_run", action="store_true", help="Print configs without running")
    args = parser.parse_args()

    script = HERE / args.script
    if not script.exists():
        print(f"ERROR: Script not found: {script}")
        sys.exit(1)

    space = MOONSHOT_SPACE if "moonshot" in args.script else SAFE_SPACE
    log_path = HERE / "experiments.jsonl"

    print(f"Auto experiment: {args.n_configs} configs × {args.iterations} iters")
    print(f"Script: {args.script}")
    print(f"Log: {log_path}")
    print()

    results = []
    for i in range(args.n_configs):
        seed = args.seed_offset + i
        cfg = sample_config(space, seed)

        print(f"[{i+1:2d}/{args.n_configs}] seed={seed} config={cfg}")

        if args.dry_run:
            print("  (dry run — skipping)")
            continue

        result = run_smoke_test(script, cfg, args.iterations, seed)
        results.append(result)

        status_str = result["status"]
        if result["train_loss"] is not None:
            status_str += f" loss={result['train_loss']:.4f}"
        if result["wall_seconds"] is not None:
            status_str += f" ({result['wall_seconds']}s)"
        print(f"  → {status_str}")

        # Append to log immediately (so partial results survive crashes)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    if args.dry_run or not results:
        return

    # Summary
    ranked = rank_results(results)
    print()
    print("=" * 60)
    print("TOP 5 CONFIGS BY TRAIN LOSS:")
    print("=" * 60)
    for r in ranked[:5]:
        print(f"  seed={r['seed']} loss={r['train_loss']} config={r['config']}")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    print()
    print(f"Summary: {n_ok} OK, {n_err} errors. Full log: {log_path}")

    # Save ranked summary
    summary_path = HERE / "auto_experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"ranked": ranked, "args": vars(args)}, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
