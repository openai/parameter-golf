#!/usr/bin/env python3
"""TTT/LoRA parameter sweep on a fixed quantized artifact.

Runs multiple TTT eval variants on the same INT6 GPTQ artifact produced by
a 4-hour long-train run. Each variant uses TTT_EVAL_ONLY=1 to skip training
and run only the phased score-first TTT evaluation.

Usage:
    # Dry-run: show all variant commands
    python scripts/run_longtrain_ttt_sweep.py --dry-run --artifact /path/to/final_model.int6.ptz

    # Run sweep locally (8 GPU)
    python scripts/run_longtrain_ttt_sweep.py --artifact /path/to/final_model.int6.ptz --output-dir ./sweep_results

    # Run specific variants only
    python scripts/run_longtrain_ttt_sweep.py --variants v0_control_pr1979,v2_rank128_lr3e4

    # Generate on-pod command (for RunPod launcher integration)
    python scripts/run_longtrain_ttt_sweep.py --emit-pod-command --artifact /root/rehearsal_out/seed42/final_model.int6.ptz

    # Set timeout per variant
    python scripts/run_longtrain_ttt_sweep.py --max-minutes-per-variant 20

    # Include optional variants
    python scripts/run_longtrain_ttt_sweep.py --include-optional --artifact /path/to/model.ptz
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Fixed environment variables applied to every variant
# ---------------------------------------------------------------------------
FIXED_TTT_ENV = {
    "TTT_WEIGHT_DECAY": "1.0",
    "TTT_BETA1": "0",
    "TTT_BETA2": "0.999",
    "TTT_K_LORA": "1",
    "TTT_MLP_LORA": "1",
    "TTT_O_LORA": "1",
    "TTT_OPTIMIZER": "adam",
    "TTT_WARM_START_A": "1",
    "FUSED_CE_ENABLED": "1",
    "GLOBAL_TTT_LR": "0.001",
    "TTT_ENABLED": "1",
    "TTT_EVAL_ONLY": "1",
}

# ---------------------------------------------------------------------------
# Sweep variants — per-variant overrides layered on top of FIXED_TTT_ENV
# ---------------------------------------------------------------------------
VARIANTS = {
    "v0_control_pr1979": {
        "description": "PR #1950/1979 baseline control",
        "env": {
            "TTT_LORA_RANK": "96",
            "TTT_LORA_ALPHA": "144",
            "TTT_LORA_LR": "0.0001",
            "TTT_BATCH_SIZE": "64",
            "TTT_CHUNK_SIZE": "48",
            "GLOBAL_TTT_EPOCHS": "1",
            "GLOBAL_TTT_CHUNK_TOKENS": "32768",
            "GLOBAL_TTT_BATCH_SEQS": "32",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0",
            "GLOBAL_TTT_WARMUP_CHUNKS": "0",
            "PHASED_TTT_PREFIX_DOCS": "2000",
            "PHASED_TTT_NUM_PHASES": "3",
            "TTT_WARM_START_A": "1",
        },
    },
    "v1_rank128_alpha192": {
        "description": "Higher LoRA rank and alpha",
        "env": {
            "TTT_LORA_RANK": "128",
            "TTT_LORA_ALPHA": "192",
            "TTT_LORA_LR": "0.0001",
            "TTT_BATCH_SIZE": "64",
            "TTT_CHUNK_SIZE": "48",
            "GLOBAL_TTT_EPOCHS": "1",
            "GLOBAL_TTT_CHUNK_TOKENS": "32768",
            "GLOBAL_TTT_BATCH_SEQS": "32",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0",
            "GLOBAL_TTT_WARMUP_CHUNKS": "0",
            "PHASED_TTT_PREFIX_DOCS": "2000",
            "PHASED_TTT_NUM_PHASES": "3",
            "TTT_WARM_START_A": "1",
        },
    },
    "v2_rank128_lr3e4": {
        "description": "Rank 128 + higher LR",
        "env": {
            "TTT_LORA_RANK": "128",
            "TTT_LORA_ALPHA": "192",
            "TTT_LORA_LR": "0.0003",
            "TTT_BATCH_SIZE": "64",
            "TTT_CHUNK_SIZE": "48",
            "GLOBAL_TTT_EPOCHS": "1",
            "GLOBAL_TTT_CHUNK_TOKENS": "32768",
            "GLOBAL_TTT_BATCH_SEQS": "32",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0",
            "GLOBAL_TTT_WARMUP_CHUNKS": "0",
            "PHASED_TTT_PREFIX_DOCS": "2000",
            "PHASED_TTT_NUM_PHASES": "3",
            "TTT_WARM_START_A": "1",
        },
    },
    "v3_local_batch_chunk": {
        "description": "Rank 128 + LR 3e-4 + larger local batch/chunk",
        "env": {
            "TTT_LORA_RANK": "128",
            "TTT_LORA_ALPHA": "192",
            "TTT_LORA_LR": "0.0003",
            "TTT_BATCH_SIZE": "128",
            "TTT_CHUNK_SIZE": "64",
            "GLOBAL_TTT_EPOCHS": "1",
            "GLOBAL_TTT_CHUNK_TOKENS": "32768",
            "GLOBAL_TTT_BATCH_SEQS": "32",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0",
            "GLOBAL_TTT_WARMUP_CHUNKS": "0",
            "PHASED_TTT_PREFIX_DOCS": "2000",
            "PHASED_TTT_NUM_PHASES": "3",
            "TTT_WARM_START_A": "1",
        },
    },
    "v4_global2_largechunk": {
        "description": "Full sweep: rank128 + lr3e-4 + batch128 + 2 global epochs + large global chunks",
        "env": {
            "TTT_LORA_RANK": "128",
            "TTT_LORA_ALPHA": "192",
            "TTT_LORA_LR": "0.0003",
            "TTT_BATCH_SIZE": "128",
            "TTT_CHUNK_SIZE": "64",
            "GLOBAL_TTT_EPOCHS": "2",
            "GLOBAL_TTT_CHUNK_TOKENS": "65536",
            "GLOBAL_TTT_BATCH_SEQS": "64",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0001",
            "GLOBAL_TTT_WARMUP_CHUNKS": "2",
            "PHASED_TTT_PREFIX_DOCS": "2000",
            "PHASED_TTT_NUM_PHASES": "3",
            "TTT_WARM_START_A": "1",
        },
    },
    "v5_prefix3000": {
        "description": "v4 + more prefix documents",
        "env": {
            "TTT_LORA_RANK": "128",
            "TTT_LORA_ALPHA": "192",
            "TTT_LORA_LR": "0.0003",
            "TTT_BATCH_SIZE": "128",
            "TTT_CHUNK_SIZE": "64",
            "GLOBAL_TTT_EPOCHS": "2",
            "GLOBAL_TTT_CHUNK_TOKENS": "65536",
            "GLOBAL_TTT_BATCH_SEQS": "64",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0001",
            "GLOBAL_TTT_WARMUP_CHUNKS": "2",
            "PHASED_TTT_PREFIX_DOCS": "3000",
            "PHASED_TTT_NUM_PHASES": "3",
            "TTT_WARM_START_A": "1",
        },
    },
    "v6_prefix3000_phase4_optional": {
        "description": "v5 + 4 phases (exploratory)",
        "optional": True,
        "env": {
            "TTT_LORA_RANK": "128",
            "TTT_LORA_ALPHA": "192",
            "TTT_LORA_LR": "0.0003",
            "TTT_BATCH_SIZE": "128",
            "TTT_CHUNK_SIZE": "64",
            "GLOBAL_TTT_EPOCHS": "2",
            "GLOBAL_TTT_CHUNK_TOKENS": "65536",
            "GLOBAL_TTT_BATCH_SEQS": "64",
            "GLOBAL_TTT_WARMUP_START_LR": "0.0001",
            "GLOBAL_TTT_WARMUP_CHUNKS": "2",
            "PHASED_TTT_PREFIX_DOCS": "3000",
            "PHASED_TTT_NUM_PHASES": "4",
            "TTT_WARM_START_A": "1",
        },
    },
}

# Keys expected in every variant result JSON
RESULT_FIELDS = [
    "variant_id", "description", "quantized_bpb_fixed", "post_ttt_bpb",
    "ttt_gain_bpb", "eval_seconds", "total_wallclock_seconds",
    "docs_evaluated", "tokens_evaluated", "prefix_docs", "phases",
    "peak_memory_mib", "status", "error",
]

DEFAULT_NGPUS = 8
DEFAULT_TIMEOUT_MINUTES = 20
DEFAULT_DATA_PATH = "/root/data"
DEFAULT_TOKENIZER_PATH = "/root/data/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
DEFAULT_TRAIN_SCRIPT = "train_gpt.py"


def select_variants(variant_filter, include_optional):
    """Return ordered list of (variant_id, variant_config) to run."""
    if variant_filter:
        requested = [v.strip() for v in variant_filter.split(",")]
        for vid in requested:
            if vid not in VARIANTS:
                print("ERROR: unknown variant '%s'. Available: %s" %
                      (vid, ", ".join(VARIANTS.keys())), file=sys.stderr)
                sys.exit(1)
        return [(vid, VARIANTS[vid]) for vid in requested]

    result = []
    for vid, cfg in VARIANTS.items():
        if cfg.get("optional") and not include_optional:
            continue
        result.append((vid, cfg))
    return result


def build_variant_env(variant_id, variant_config, artifact_path,
                      output_dir, train_script_path, data_path, tok_path):
    """Build complete environment dict for one TTT eval variant.

    Merges: os.environ (inherit) + FIXED_TTT_ENV + variant overrides + paths.
    """
    env = dict(os.environ)
    env.update(FIXED_TTT_ENV)
    env.update(variant_config["env"])

    variant_out = os.path.join(output_dir, variant_id)
    env["LOAD_QUANTIZED_MODEL_PATH"] = str(artifact_path)
    env["TTT_EVAL_OUTPUT_JSON"] = os.path.join(variant_out, "ttt_eval_summary.json")
    env["OUTPUT_DIR"] = variant_out

    if data_path:
        env["DATA_DIR"] = data_path
    if tok_path:
        env["TOKENIZER_PATH"] = tok_path

    return env


def generate_variant_manifest(variants_to_run, artifact_path, output_dir):
    """Write ttt_sweep_manifest.json with all variant configs."""
    manifest = {
        "artifact_path": str(artifact_path),
        "output_dir": str(output_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fixed_env": FIXED_TTT_ENV,
        "variants": {},
    }
    for vid, cfg in variants_to_run:
        manifest["variants"][vid] = {
            "description": cfg.get("description", ""),
            "optional": cfg.get("optional", False),
            "env_overrides": cfg["env"],
        }

    manifest_path = os.path.join(output_dir, "ttt_sweep_manifest.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("Wrote manifest: %s" % manifest_path)
    return manifest_path


def run_variant(variant_id, variant_config, env, train_script, ngpus,
                timeout_minutes, output_dir):
    """Run one TTT variant via torchrun. Returns result dict."""
    variant_out = os.path.join(output_dir, variant_id)
    os.makedirs(variant_out, exist_ok=True)

    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=%d" % ngpus,
        train_script,
    ]

    log_path = os.path.join(variant_out, "eval.log")
    summary_json_path = os.path.join(variant_out, "ttt_eval_summary.json")

    result = {
        "variant_id": variant_id,
        "description": variant_config.get("description", ""),
        "env_overrides": variant_config["env"],
        "quantized_bpb_fixed": None,
        "post_ttt_bpb": None,
        "ttt_gain_bpb": None,
        "eval_seconds": None,
        "total_wallclock_seconds": None,
        "docs_evaluated": None,
        "tokens_evaluated": None,
        "prefix_docs": int(variant_config["env"].get("PHASED_TTT_PREFIX_DOCS", 0)),
        "phases": int(variant_config["env"].get("PHASED_TTT_NUM_PHASES", 1)),
        "peak_memory_mib": None,
        "status": "pending",
        "error": None,
    }

    print("\n" + "=" * 72)
    print("VARIANT: %s — %s" % (variant_id, variant_config.get("description", "")))
    print("Command: %s" % " ".join(cmd))
    print("Log:     %s" % log_path)
    print("Timeout: %d min" % timeout_minutes)
    print("=" * 72)

    t0 = time.time()
    try:
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
            )
            timeout_sec = timeout_minutes * 60
            proc.wait(timeout=timeout_sec)
            elapsed = time.time() - t0
            result["total_wallclock_seconds"] = round(elapsed, 1)

            if proc.returncode != 0:
                result["status"] = "error"
                result["error"] = "exit code %d" % proc.returncode
                # Try to extract last few lines for diagnostics
                try:
                    with open(log_path, "r") as rf:
                        lines = rf.readlines()
                        tail = "".join(lines[-10:])
                        result["error"] += " | tail: " + tail.strip()[:500]
                except Exception:
                    pass
            else:
                result["status"] = "success"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result["total_wallclock_seconds"] = round(elapsed, 1)
        result["status"] = "timeout"
        result["error"] = "exceeded %d min timeout" % timeout_minutes
        proc.kill()
        proc.wait()
    except Exception as exc:
        elapsed = time.time() - t0
        result["total_wallclock_seconds"] = round(elapsed, 1)
        result["status"] = "error"
        result["error"] = str(exc)

    # Try to read the machine-readable summary produced by train_gpt.py
    if os.path.exists(summary_json_path):
        try:
            with open(summary_json_path, "r") as f:
                summary = json.load(f)
            for key in ("quantized_bpb_fixed", "post_ttt_bpb", "ttt_gain_bpb",
                        "eval_seconds", "docs_evaluated", "tokens_evaluated",
                        "peak_memory_mib"):
                if key in summary:
                    result[key] = summary[key]
        except Exception as exc:
            if result["error"]:
                result["error"] += " | json parse: " + str(exc)
            else:
                result["error"] = "json parse: " + str(exc)

    # Write per-variant result
    result_path = os.path.join(variant_out, "variant_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print("  -> status=%s  bpb=%s  wallclock=%.0fs" % (
        result["status"], result.get("post_ttt_bpb", "N/A"),
        result.get("total_wallclock_seconds", 0)))

    return result


def aggregate_results(output_dir, results):
    """Write ttt_sweep_results.csv and ttt_sweep_summary.json from results."""
    csv_path = os.path.join(output_dir, "ttt_sweep_results.csv")
    summary_path = os.path.join(output_dir, "ttt_sweep_summary.json")

    # CSV
    fieldnames = RESULT_FIELDS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print("\nWrote CSV: %s" % csv_path)

    # Summary JSON
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_variants": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "timed_out": sum(1 for r in results if r["status"] == "timeout"),
        "best_variant": None,
        "results": results,
    }

    # Find best variant by post_ttt_bpb
    successful = [r for r in results
                  if r["status"] == "success" and r.get("post_ttt_bpb") is not None]
    if successful:
        best = min(successful, key=lambda r: r["post_ttt_bpb"])
        summary["best_variant"] = {
            "variant_id": best["variant_id"],
            "post_ttt_bpb": best["post_ttt_bpb"],
            "ttt_gain_bpb": best.get("ttt_gain_bpb"),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote summary: %s" % summary_path)

    return csv_path, summary_path


def aggregate_results_from_disk(output_dir, variants_run):
    """Read per-variant result JSONs from disk and aggregate.

    Useful for re-aggregation after partial runs.
    """
    results = []
    for vid, _ in variants_run:
        result_path = os.path.join(output_dir, vid, "variant_result.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                results.append(json.load(f))
    return aggregate_results(output_dir, results)


def dry_run(variants_to_run, artifact_path, output_dir, ngpus, timeout_minutes,
            train_script, data_path, tok_path):
    """Print all variant commands without executing."""
    print("=" * 72)
    print("DRY RUN — %d variants" % len(variants_to_run))
    print("Artifact: %s" % artifact_path)
    print("Output:   %s" % output_dir)
    print("GPUs:     %d" % ngpus)
    print("Timeout:  %d min/variant" % timeout_minutes)
    print("=" * 72)

    for i, (vid, cfg) in enumerate(variants_to_run):
        env = build_variant_env(vid, cfg, artifact_path, output_dir,
                                train_script, data_path, tok_path)
        # Show only the TTT-specific env vars
        ttt_keys = sorted(set(list(FIXED_TTT_ENV.keys()) + list(cfg["env"].keys())
                              + ["LOAD_QUANTIZED_MODEL_PATH", "TTT_EVAL_OUTPUT_JSON",
                                 "OUTPUT_DIR"]))
        env_str = " \\\n    ".join(
            "%s=%s" % (k, env[k]) for k in ttt_keys if k in env
        )

        optional_tag = " [OPTIONAL]" if cfg.get("optional") else ""
        print("\n--- Variant %d/%d: %s%s ---" % (i + 1, len(variants_to_run),
                                                  vid, optional_tag))
        print("Description: %s" % cfg.get("description", ""))
        print("Command:")
        print("  %s \\\n    torchrun --standalone --nproc_per_node=%d %s" % (
            env_str, ngpus, train_script))
        print()


def emit_pod_command(variants_to_run, artifact_path, output_dir, ngpus,
                     timeout_minutes, train_script, data_path, tok_path):
    """Generate a single shell script for running sweep on a RunPod pod."""
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# TTT/LoRA sweep — generated by run_longtrain_ttt_sweep.py",
        "# Variants: %d" % len(variants_to_run),
        "",
        "ARTIFACT=%s" % _shell_quote(str(artifact_path)),
        "OUTPUT_DIR=%s" % _shell_quote(str(output_dir)),
        "TRAIN_SCRIPT=%s" % _shell_quote(str(train_script)),
        "NGPUS=%d" % ngpus,
        "",
        "mkdir -p $OUTPUT_DIR",
        "",
    ]

    for vid, cfg in variants_to_run:
        env_exports = []
        merged = dict(FIXED_TTT_ENV)
        merged.update(cfg["env"])
        merged["LOAD_QUANTIZED_MODEL_PATH"] = "$ARTIFACT"
        merged["OUTPUT_DIR"] = "$OUTPUT_DIR/%s" % vid
        merged["TTT_EVAL_OUTPUT_JSON"] = "$OUTPUT_DIR/%s/ttt_eval_summary.json" % vid

        lines.append("# --- %s: %s ---" % (vid, cfg.get("description", "")))
        lines.append("echo '=== Starting variant: %s ==='" % vid)
        lines.append("mkdir -p $OUTPUT_DIR/%s" % vid)

        for k in sorted(merged.keys()):
            lines.append("export %s=%s" % (k, _shell_quote(merged[k])
                                           if "$" not in merged[k]
                                           else merged[k]))

        lines.append(
            "timeout %dm torchrun --standalone --nproc_per_node=$NGPUS"
            " $TRAIN_SCRIPT > $OUTPUT_DIR/%s/eval.log 2>&1 || "
            "echo 'VARIANT %s exited with code '$?" % (timeout_minutes, vid, vid)
        )
        lines.append("echo '=== Finished variant: %s ==='\\n" % vid)
        lines.append("")

    lines.append("echo 'Sweep complete.'")
    return "\n".join(lines)


def _shell_quote(s):
    """Simple POSIX shell quoting."""
    return "'" + s.replace("'", "'\\''") + "'"


def main():
    parser = argparse.ArgumentParser(
        description="TTT/LoRA parameter sweep on a fixed quantized artifact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--artifact", type=str,
        default=os.environ.get("LOAD_QUANTIZED_MODEL_PATH", ""),
        help="Path to quantized .ptz artifact (or set LOAD_QUANTIZED_MODEL_PATH).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./sweep_results",
        help="Root directory for sweep outputs (default: ./sweep_results).",
    )
    parser.add_argument(
        "--variants", type=str, default=None,
        help="Comma-separated list of variant IDs to run (default: all non-optional).",
    )
    parser.add_argument(
        "--include-optional", action="store_true",
        help="Include variants marked as optional.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--emit-pod-command", action="store_true",
        help="Emit a shell script for running on a RunPod pod.",
    )
    parser.add_argument(
        "--ngpus", type=int, default=DEFAULT_NGPUS,
        help="Number of GPUs for torchrun (default: %d)." % DEFAULT_NGPUS,
    )
    parser.add_argument(
        "--max-minutes-per-variant", type=int, default=DEFAULT_TIMEOUT_MINUTES,
        help="Per-variant timeout in minutes (default: %d)." % DEFAULT_TIMEOUT_MINUTES,
    )
    parser.add_argument(
        "--train-script", type=str, default=DEFAULT_TRAIN_SCRIPT,
        help="Path to train_gpt.py (default: %s)." % DEFAULT_TRAIN_SCRIPT,
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Override DATA_DIR path.",
    )
    parser.add_argument(
        "--tokenizer-path", type=str, default=None,
        help="Override TOKENIZER_PATH.",
    )
    parser.add_argument(
        "--reaggregate", action="store_true",
        help="Re-aggregate results from existing per-variant JSONs (no execution).",
    )

    args = parser.parse_args()

    variants_to_run = select_variants(args.variants, args.include_optional)
    if not variants_to_run:
        print("ERROR: no variants selected.", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)

    # --- Re-aggregate mode ---
    if args.reaggregate:
        csv_path, summary_path = aggregate_results_from_disk(output_dir, variants_to_run)
        print("Re-aggregation complete.")
        return

    # --- Validate artifact path ---
    if not args.artifact:
        print("ERROR: --artifact path required (or set LOAD_QUANTIZED_MODEL_PATH).",
              file=sys.stderr)
        sys.exit(1)

    artifact_path = os.path.abspath(args.artifact)

    # --- Dry-run mode ---
    if args.dry_run:
        dry_run(variants_to_run, artifact_path, output_dir, args.ngpus,
                args.max_minutes_per_variant, args.train_script,
                args.data_path, args.tokenizer_path)
        generate_variant_manifest(variants_to_run, artifact_path, output_dir)
        return

    # --- Emit pod command mode ---
    if args.emit_pod_command:
        script = emit_pod_command(
            variants_to_run, artifact_path, output_dir, args.ngpus,
            args.max_minutes_per_variant, args.train_script,
            args.data_path, args.tokenizer_path)
        print(script)
        return

    # --- Live execution mode ---
    if not os.path.exists(artifact_path):
        print("WARNING: artifact not found at %s — proceeding anyway "
              "(may fail at runtime)." % artifact_path, file=sys.stderr)

    os.makedirs(output_dir, exist_ok=True)
    generate_variant_manifest(variants_to_run, artifact_path, output_dir)

    results = []
    sweep_t0 = time.time()

    for i, (vid, cfg) in enumerate(variants_to_run):
        print("\n[%d/%d] Running variant: %s" % (i + 1, len(variants_to_run), vid))
        env = build_variant_env(vid, cfg, artifact_path, output_dir,
                                args.train_script, args.data_path,
                                args.tokenizer_path)
        result = run_variant(vid, cfg, env, args.train_script, args.ngpus,
                             args.max_minutes_per_variant, output_dir)
        results.append(result)

    sweep_elapsed = time.time() - sweep_t0
    print("\n" + "=" * 72)
    print("SWEEP COMPLETE — %d variants in %.0f seconds (%.1f min)" % (
        len(results), sweep_elapsed, sweep_elapsed / 60))
    print("=" * 72)

    aggregate_results(output_dir, results)

    # Print quick comparison table
    print("\n%-35s  %-8s  %-10s  %-10s  %s" % (
        "VARIANT", "STATUS", "POST_BPB", "GAIN_BPB", "WALLCLOCK"))
    print("-" * 85)
    for r in results:
        post_bpb = "%.5f" % r["post_ttt_bpb"] if r.get("post_ttt_bpb") is not None else "N/A"
        gain = "%.5f" % r["ttt_gain_bpb"] if r.get("ttt_gain_bpb") is not None else "N/A"
        wc = "%.0fs" % r["total_wallclock_seconds"] if r.get("total_wallclock_seconds") else "N/A"
        print("%-35s  %-8s  %-10s  %-10s  %s" % (
            r["variant_id"], r["status"], post_bpb, gain, wc))


if __name__ == "__main__":
    main()
