#!/usr/bin/env python3
"""Modal-based parallel hyperparameter search for Parameter Golf.

Dispatches multiple training trials to Modal GPU containers in parallel,
collects results, and feeds them back into the local autoresearch system.

Usage:
    # One-time setup: upload data to Modal Volume
    modal run autoresearch/modal_search.py::upload_data

    # Run parallel sweep across all presets on A10G
    modal run autoresearch/modal_search.py --mode preset --trials 8

    # Run parallel sweep on H100 (matches submission hardware)
    modal run autoresearch/modal_search.py --mode preset --trials 8 --gpu h100

    # Evolution search with 6 parallel trials
    modal run autoresearch/modal_search.py --mode evolution --trials 6

    # Random search
    modal run autoresearch/modal_search.py --mode random --trials 10
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path, PosixPath
from typing import Any, Optional

import modal

# ---------------------------------------------------------------------------
# Modal app & infrastructure
# ---------------------------------------------------------------------------

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App("parameter-golf-search")

volume = modal.Volume.from_name("parameter-golf-data", create_if_missing=True)
results_volume = modal.Volume.from_name("parameter-golf-results", create_if_missing=True)
VOLUME_PATH = PosixPath("/vol/data")
RESULTS_PATH = PosixPath("/vol/results")

# Container image with all dependencies pre-installed
cuda_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.10",
        "numpy",
        "sentencepiece",
        "tiktoken",
        "tqdm",
        "typing-extensions==4.15.0",
        "datasets",
        "huggingface-hub",
        "setuptools",
        "kernels",
    )
    .add_local_file(
        str(Path(__file__).resolve().parents[1] / "train_gpt.py"),
        remote_path="/root/train_gpt.py",
    )
    .add_local_dir(
        str(Path(__file__).resolve().parents[1] / "data"),
        remote_path="/root/data",
    )
)

# ---------------------------------------------------------------------------
# Data upload (one-time setup)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=30 * MINUTES,
)
def upload_data():
    """Download FineWeb data into the Modal Volume (run once)."""
    import subprocess
    import sys

    data_dir = VOLUME_PATH / "datasets" / "fineweb10B_sp1024"
    tok_dir = VOLUME_PATH / "tokenizers"

    if (data_dir / "fineweb_val_000000.bin").exists():
        print(f"Data already exists at {data_dir}")
        volume.commit()
        return

    # Download using the project's script
    subprocess.run(
        [sys.executable, "/root/data/cached_challenge_fineweb.py",
         "--variant", "sp1024", "--train-shards", "10"],
        check=True,
        env={**os.environ, "DATA_DIR": str(VOLUME_PATH)},
    )
    volume.commit()
    print("Data uploaded successfully")


# ---------------------------------------------------------------------------
# Single training trial on GPU
# ---------------------------------------------------------------------------

@app.function(
    image=cuda_image,
    volumes={VOLUME_PATH: volume, RESULTS_PATH: results_volume},
    gpu="A10G",  # overridden at call site via .with_options()
    timeout=1 * HOURS,
    retries=modal.Retries(max_retries=1, initial_delay=0.0),
)
def run_gpu_trial(
    run_id: str,
    config: dict[str, str],
    nproc: int = 1,
    mode: str = "preset",
    preset: str = "",
    code_mutation: str = "",
    parents: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Run a single training trial on a Modal GPU container.

    Returns a dict with trial results (val_bpb, val_loss, total_bytes, etc.)
    compatible with the local autoresearch TrialResult format.
    """
    import re
    import subprocess
    import sys

    volume.reload()

    # Point data paths to Modal Volume
    env = {**os.environ, **config}
    env["DATA_PATH"] = str(VOLUME_PATH / "datasets" / "fineweb10B_sp1024")
    env["TOKENIZER_PATH"] = str(VOLUME_PATH / "tokenizers" / "fineweb_1024_bpe.model")
    env["RUN_ID"] = run_id
    env["PYTHONUNBUFFERED"] = "1"
    env["OUT_DIR"] = str(RESULTS_PATH / "logs")

    # Ensure output dirs exist
    (RESULTS_PATH / "logs").mkdir(parents=True, exist_ok=True)
    (RESULTS_PATH / "logs" / "autoresearch").mkdir(parents=True, exist_ok=True)

    log_path = RESULTS_PATH / "logs" / "autoresearch" / f"{run_id}.log"
    script_path = "/root/train_gpt.py"

    started = time.time()

    # Build command
    if nproc > 1:
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", script_path]
    else:
        cmd = [sys.executable, "-u", script_path]

    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True,
        )

    elapsed = time.time() - started
    log_text = log_path.read_text(encoding="utf-8")
    status = "ok" if proc.returncode == 0 else "crash"

    # Parse metrics from log
    val_bpb = 0.0
    val_loss = 0.0
    total_bytes = 0
    quantized_model_bytes = 0
    model_params = 0

    VAL_RE = re.compile(
        r"final_int8_zlib_roundtrip_exact.*?val_loss:\s*(?P<val_loss>[-+0-9.eE]+).*?val_bpb:\s*(?P<val_bpb>[-+0-9.eE]+)",
        flags=re.IGNORECASE,
    )
    TOTAL_SIZE_RE = re.compile(r"total\s+submission\s+size\s+int8\+zlib:\s*(?P<bytes>\d+)\s*bytes", flags=re.IGNORECASE)
    MODEL_SIZE_RE = re.compile(r"serialized\s+model\s+int8\+zlib:\s*(?P<bytes>\d+)\s*bytes", flags=re.IGNORECASE)
    PARAM_RE = re.compile(r"model_params:\s*(?P<params>\d+)", flags=re.IGNORECASE)

    if status == "ok":
        val_matches = list(VAL_RE.finditer(log_text))
        if val_matches:
            val_loss = float(val_matches[-1].group("val_loss"))
            val_bpb = float(val_matches[-1].group("val_bpb"))

        total_matches = list(TOTAL_SIZE_RE.finditer(log_text))
        if total_matches:
            total_bytes = int(total_matches[-1].group("bytes"))

        model_matches = list(MODEL_SIZE_RE.finditer(log_text))
        if model_matches:
            quantized_model_bytes = int(model_matches[-1].group("bytes"))

        param_matches = list(PARAM_RE.finditer(log_text))
        if param_matches:
            model_params = int(param_matches[-1].group("params"))

        if val_bpb == 0.0:
            status = "parse_error:no_val_metrics"

    results_volume.commit()

    return {
        "run_id": run_id,
        "backend": "cuda",
        "mode": mode,
        "status": status,
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "total_bytes": total_bytes,
        "train_script_path": "train_gpt.py",
        "train_script_bytes": 0,
        "quantized_model_bytes": quantized_model_bytes,
        "model_params": model_params,
        "elapsed_seconds": elapsed,
        "log_path": str(log_path),
        "preset": preset,
        "code_mutation": code_mutation,
        "parents": parents or [],
        "config": config,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Parallel sweep orchestrator (runs locally, dispatches to Modal)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    mode: str = "preset",
    trials: int = 8,
    seed: int = 1337,
    gpu: str = "A10G",
    nproc: int = 1,
    preset: Optional[str] = None,
    population: int = 6,
    baseline_first: bool = False,
):
    """Run parallel hyperparameter search on Modal GPUs.

    Generates trial configs locally using the same logic as run_search.py,
    then dispatches all trials to Modal in parallel via starmap.
    """
    # Import the local autoresearch module for config generation
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from autoresearch import run_search

    rng = random.Random(seed)
    backend = "cuda"
    configs: list[tuple[dict[str, str], str, str, str, list[str]]] = []

    if baseline_first:
        cfg = run_search.normalize_config(backend, run_search.PRESETS[backend]["baseline"])
        configs.append((cfg, "preset", "baseline", "preset:baseline", []))

    if mode == "preset":
        preset_names = [preset] if preset else list(run_search.PRESETS[backend].keys())
        for i in range(trials):
            name = preset_names[i % len(preset_names)] if preset else rng.choice(preset_names)
            cfg = run_search.normalize_config(backend, run_search.PRESETS[backend][name])
            if run_search.estimated_total_bytes(cfg, run_search.script_for_backend(backend)) < run_search.ARTIFACT_LIMIT_BYTES:
                configs.append((cfg, "preset", name, f"preset:{name}", []))

    elif mode == "random":
        base_cfg = run_search.normalize_config(backend, run_search.PRESETS[backend]["baseline"])
        for _ in range(trials):
            candidate = run_search.mutate_config(base_cfg, backend, rng, intensity=rng.randint(2, 5))
            if run_search.estimated_total_bytes(candidate, run_search.script_for_backend(backend)) < run_search.ARTIFACT_LIMIT_BYTES:
                desc = run_search.describe_delta(base_cfg, candidate)
                configs.append((candidate, "random", "", desc, []))

    elif mode == "evolution":
        # Load existing trial results for seeding
        best = run_search.load_best()
        population_pool = run_search.load_population(backend)[:population]

        base_cfg = run_search.normalize_config(backend, best.config) if best else run_search.normalize_config(backend, run_search.PRESETS[backend]["baseline"])

        for _ in range(trials):
            if len(population_pool) >= 2 and rng.random() < 0.6:
                parents = rng.sample(population_pool, 2)
                candidate = run_search.crossover_config(
                    parents[0].config, parents[1].config, backend, rng,
                )
                parent_ids = [p.run_id for p in parents]
            else:
                candidate = run_search.mutate_config(base_cfg, backend, rng, intensity=rng.randint(2, 5))
                parent_ids = [best.run_id] if best else []

            if run_search.estimated_total_bytes(candidate, run_search.script_for_backend(backend)) < run_search.ARTIFACT_LIMIT_BYTES:
                desc = run_search.describe_delta(base_cfg, candidate)
                configs.append((candidate, "evolution", "", desc, parent_ids))

    if not configs:
        print("No valid configs generated")
        return

    print(f"\n{'='*60}")
    print(f"Parameter Golf Modal Search")
    print(f"{'='*60}")
    print(f"Mode: {mode} | Trials: {len(configs)} | GPU: {gpu} | nproc: {nproc}")
    print(f"{'='*60}\n")

    # Generate run IDs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    trial_inputs = []
    for i, (cfg, trial_mode, trial_preset, desc, parents) in enumerate(configs):
        run_id = f"ar_modal_{trial_mode}_{timestamp}_{i:03d}"
        trial_inputs.append((
            run_id, cfg, nproc, trial_mode, trial_preset, "", parents, desc,
        ))
        print(f"  [{i}] {run_id}: {desc[:80]}")

    print(f"\nDispatching {len(trial_inputs)} trials to Modal ({gpu})...\n")

    # Configure GPU at call site
    gpu_fn = run_gpu_trial.with_options(gpu=gpu)

    # Launch all trials in parallel via starmap
    results = []
    for result in gpu_fn.starmap(trial_inputs, order_outputs=False):
        results.append(result)
        status = result["status"]
        bpb = result["val_bpb"]
        rid = result["run_id"]
        if status == "ok" and bpb > 0:
            print(f"  ✓ {rid}: val_bpb={bpb:.6f} bytes={result['total_bytes']}")
        else:
            print(f"  ✗ {rid}: status={status}")

    # Find best result
    ok_results = [r for r in results if r["status"] == "ok" and r["val_bpb"] > 0]
    if ok_results:
        best = min(ok_results, key=lambda r: r["val_bpb"])
        print(f"\n{'='*60}")
        print(f"Best: val_bpb={best['val_bpb']:.6f} ({best['description']})")
        print(f"{'='*60}")

        # Save results locally for the autoresearch system
        local_results_dir = Path(__file__).resolve().parents[1] / "logs" / "autoresearch"
        local_results_dir.mkdir(parents=True, exist_ok=True)

        # Append to results.tsv
        tsv_path = local_results_dir / "results.tsv"
        for r in results:
            with tsv_path.open("a", encoding="utf-8") as f:
                f.write("\t".join([
                    r["run_id"], r["backend"], r["mode"],
                    f"{r['val_bpb']:.8f}", f"{r['val_loss']:.8f}",
                    str(r["total_bytes"]), r["status"],
                    r["preset"], r["code_mutation"],
                    ",".join(r["parents"]),
                    r["train_script_path"], r["description"],
                ]) + "\n")

        # Save per-trial JSON
        trials_dir = local_results_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            trial_path = trials_dir / f"{r['run_id']}.json"
            trial_path.write_text(json.dumps(r, indent=2), encoding="utf-8")

        # Update best_config.json if this is better
        best_json_path = local_results_dir / "best_config.json"
        if best_json_path.exists():
            existing = json.loads(best_json_path.read_text(encoding="utf-8"))
            if best["val_bpb"] < existing.get("val_bpb", float("inf")):
                best_json_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
                print(f"New overall best! Updated best_config.json")
        else:
            best_json_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

        print(f"\nResults saved to {local_results_dir}")
    else:
        print("\nNo successful trials")

    return results
