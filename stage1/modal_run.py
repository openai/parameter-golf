#!/usr/bin/env python3
"""Modal launcher for Parameter Golf Stage 1 family runs.

Usage:
    # Run a single slot on 1xH100:
    uv run modal run pgolf/parameter-golf/stage1/modal_run.py --slot P0

    # Run a single slot on 1xA100-80GB:
    uv run modal run pgolf/parameter-golf/stage1/modal_run.py --slot P0 --gpu a100-80gb

    # Run all slots sequentially:
    uv run modal run pgolf/parameter-golf/stage1/modal_run.py --all

    # Dry run (print configs):
    uv run modal run pgolf/parameter-golf/stage1/modal_run.py --slot P1 --dry-run
"""
from __future__ import annotations

import json
import modal

app = modal.App("pgolf-stage1")

# Persistent volume for caching downloaded dataset across runs.
data_vol = modal.Volume.from_name("pgolf-data", create_if_missing=True)

DATA_VOL_PATH = "/data"
REPO_URL = "https://github.com/openai/parameter-golf.git"
WORK_DIR = "/workspace/parameter-golf"

VENV = "/opt/venv"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        f'export PATH="$HOME/.local/bin:$PATH" && uv venv {VENV}',
        f'export PATH="$HOME/.local/bin:$PATH" && '
        f"VIRTUAL_ENV={VENV} uv pip install "
        "torch numpy sentencepiece huggingface-hub datasets tqdm triton",
    )
    .env({"VIRTUAL_ENV": VENV, "PATH": f"{VENV}/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(f"git clone {REPO_URL} {WORK_DIR}")
)

# ── Slot configs (mirrored from run_configs.json) ──────────────────────────
DEFAULTS = {
    "MAX_WALLCLOCK_SECONDS": "600",
    "TRAIN_LOG_EVERY": "50",
    "VAL_LOSS_EVERY": "200",
    "VOCAB_SIZE": "1024",
    "NUM_HEADS": "8",
    "TIE_EMBEDDINGS": "1",
}

SLOTS = {
    "P0": {"name": "baseline", "hypothesis": "none", "family": "baseline", "env": {}},
    "P1": {
        "name": "m02_seq2048_train",
        "hypothesis": "long_context_training_geometry",
        "family": "M02",
        "env": {
            "TRAIN_SEQ_LEN": "2048",
            "TIED_EMBED_LR": "0.04",
            "MATRIX_LR": "0.032",
            "SCALAR_LR": "0.032",
        },
    },
    "P2": {
        "name": "m09_fp16_embed_export",
        "hypothesis": "export_aware_quantization",
        "family": "M09",
        "env": {"INT8_KEEP_TOK_EMB_FP16": "1", "MLP_HIDDEN": "992"},
    },
    "P3": {
        "name": "m06_adamw_partition",
        "hypothesis": "optimizer_partitioning",
        "family": "M06",
        "env": {
            "TOKEN_OPTIMIZER": "adamw",
            "SCALAR_OPTIMIZER": "adamw",
            "TOKEN_WEIGHT_DECAY": "0.01",
            "SCALAR_WEIGHT_DECAY": "0.01",
        },
    },
    "P4": {
        "name": "m11_sliding_eval64",
        "hypothesis": "evaluation_time_context",
        "family": "M11",
        "env": {"EVAL_STRIDE": "64", "EVAL_BATCH_SEQS": "256"},
    },
    "P5": {
        "name": "m08_always_decay",
        "hypothesis": "wallclock_aware_schedule",
        "family": "M08",
        "env": {
            "WARMDOWN_ITERS": "20000",
            "MATRIX_LR": "0.06",
            "TIED_EMBED_LR": "0.07",
            "SCALAR_LR": "0.06",
            "GRAD_CLIP_NORM": "1.0",
        },
    },
    "P6": {
        "name": "m01_depth10_width480",
        "hypothesis": "matched_budget_architecture",
        "family": "M01",
        "env": {"NUM_LAYERS": "10", "MODEL_DIM": "480", "NUM_KV_HEADS": "4", "MLP_MULT": "2"},
    },
    "P7": {
        "name": "m07_adaptive_muon",
        "hypothesis": "adaptive_muon_compute",
        "family": "M07",
        "env": {
            "ADAPTIVE_MUON_BACKEND": "1",
            "MUON_BACKEND_STEPS": "5",
            "MUON_BACKEND_STEPS_EARLY": "3",
            "MUON_BACKEND_WARMUP_STEPS": "2000",
        },
    },
}


@app.function(
    image=image,
    gpu="a100-80gb",
    timeout=1200,
    volumes={DATA_VOL_PATH: data_vol},
)
def run_slot(slot_id: str) -> dict:
    """Run a single stage1 slot and return the final results."""
    import os
    import subprocess

    slot = SLOTS[slot_id]

    # ── Download data if not already cached in the volume ──
    data_dir = f"{DATA_VOL_PATH}/fineweb10B_sp1024"
    tokenizer_path = f"{DATA_VOL_PATH}/fineweb_1024_bpe.model"

    if not os.path.exists(tokenizer_path):
        print("=== Downloading dataset (first run, will be cached) ===", flush=True)
        subprocess.run(
            [
                "python3", f"{WORK_DIR}/data/cached_challenge_fineweb.py",
                "--variant", "sp1024",
                "--train-shards", "80",
            ],
            cwd=WORK_DIR,
            check=True,
        )
        # Copy into the volume for caching
        subprocess.run(
            f"cp -r {WORK_DIR}/data/datasets/fineweb10B_sp1024 {DATA_VOL_PATH}/fineweb10B_sp1024",
            shell=True, check=True,
        )
        subprocess.run(
            f"cp {WORK_DIR}/data/tokenizers/fineweb_1024_bpe.model {tokenizer_path}",
            shell=True, check=True,
        )
        data_vol.commit()
        print("=== Dataset cached to volume ===", flush=True)

    # ── Build env ──
    env = os.environ.copy()
    env.update(DEFAULTS)
    env.update(slot["env"])
    env["RUN_ID"] = f"stage1_{slot_id}_{slot['name']}"
    env["DATA_PATH"] = data_dir
    env["TOKENIZER_PATH"] = tokenizer_path

    print(f"\n=== Stage1 {slot_id}: {slot['name']} ({slot['family']}) ===", flush=True)
    print(f"Hypothesis: {slot['hypothesis']}", flush=True)
    print(f"Env overrides: {json.dumps(slot['env'], indent=2)}", flush=True)

    # ── Launch training via torchrun (1 GPU), stream output live ──
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        f"{WORK_DIR}/train_gpt.py",
    ]
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    output_lines = []
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        output_lines.append(line)
    returncode = proc.wait()

    # ── Parse final val_bpb from output ──
    final_bpb = None
    final_line = None
    for line in output_lines:
        if "final_int8_zlib_roundtrip_exact" in line:
            final_line = line
            for part in line.split():
                if part.startswith("val_bpb:"):
                    final_bpb = float(part.split(":")[1])

    summary = {
        "slot": slot_id,
        "name": slot["name"],
        "hypothesis": slot["hypothesis"],
        "family": slot["family"],
        "val_bpb": final_bpb,
        "final_line": final_line,
        "returncode": returncode,
    }
    print(f"\n=== RESULT: {json.dumps(summary, indent=2)} ===", flush=True)
    return summary


@app.local_entrypoint()
def main(
    slot: str = "",
    all: bool = False,
    dry_run: bool = False,
):
    if dry_run:
        slots_to_run = list(SLOTS.keys()) if all else [slot]
        for s in slots_to_run:
            cfg = SLOTS[s]
            env = {**DEFAULTS, **cfg["env"]}
            print(json.dumps({"slot": s, **cfg, "resolved_env": env}, indent=2))
        return

    if all:
        slots_to_run = list(SLOTS.keys())
    elif slot:
        if slot not in SLOTS:
            raise SystemExit(f"Unknown slot '{slot}'. Available: {', '.join(sorted(SLOTS))}")
        slots_to_run = [slot]
    else:
        raise SystemExit("Specify --slot <P0..P7> or --all")

    # GPU type is set in the @app.function decorator above.
    # Change it there to switch between a100-80gb / h100.
    results = []
    for s in slots_to_run:
        print(f"\n{'='*60}")
        print(f"Launching {s}: {SLOTS[s]['name']}")
        print(f"{'='*60}")
        r = run_slot.remote(s)
        results.append(r)

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("STAGE 1 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Slot':<6} {'Name':<25} {'Family':<12} {'val_bpb':<12} {'Status'}")
    print("-" * 70)
    for r in results:
        bpb = f"{r['val_bpb']:.8f}" if r["val_bpb"] else "FAILED"
        status = "OK" if r["returncode"] == 0 else f"EXIT {r['returncode']}"
        print(f"{r['slot']:<6} {r['name']:<25} {r['family']:<12} {bpb:<12} {status}")
