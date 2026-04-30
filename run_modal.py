"""
Parameter Golf Modal Runner.

Runs training on 8xH100 SXM GPUs via Modal cloud.

Setup (one-time):
    modal volume create pg-data
    # Upload dataset:
    modal volume put pg-data ./data/datasets/fineweb10B_sp1024/ /datasets/fineweb10B_sp1024/
    modal volume put pg-data ./data/tokenizers/ /tokenizers/

Usage:
    modal run run_modal.py --script train_gpt_v24.py
    modal run run_modal.py --script train_gpt_v24.py --env "NUM_LAYERS=11,MUON_WD=0.04"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import modal

app = modal.App("yahya-eval-ncu")

data_volume = modal.Volume.from_name("pg-data", create_if_missing=True)
DATA_MOUNT = "/data"

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.01-py3",
    )
    .pip_install(
        "sentencepiece",
        "zstandard",
        "numpy",
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation",
    )
)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=1800,  # 30 min max (10 train + 10 eval + buffer)
    volumes={DATA_MOUNT: data_volume},
    memory=128 * 1024,  # 128 GB RAM
)
def run_training(
    script_content: str,
    script_name: str = "train_gpt.py",
    env_overrides: dict[str, str] | None = None,
    seed: int = 1337,
) -> str:
    """Run training on 8xH100 and return the log."""
    import subprocess
    import tempfile

    # Write script to temp file
    work_dir = tempfile.mkdtemp()
    script_path = os.path.join(work_dir, script_name)
    with open(script_path, "w") as f:
        f.write(script_content)

    # Check data is mounted
    data_dir = os.path.join(DATA_MOUNT, "datasets", "fineweb10B_sp1024")
    tok_path = os.path.join(DATA_MOUNT, "tokenizers", "fineweb_1024_bpe.model")

    if not os.path.exists(data_dir):
        # Try flat structure
        data_dir = os.path.join(DATA_MOUNT, "fineweb10B_sp1024")
    if not os.path.exists(tok_path):
        tok_path = os.path.join(DATA_MOUNT, "fineweb_1024_bpe.model")

    print(f"Data dir: {data_dir} (exists: {os.path.exists(data_dir)})")
    print(f"Tokenizer: {tok_path} (exists: {os.path.exists(tok_path)})")

    if not os.path.exists(data_dir):
        # List what's in the volume
        for root, dirs, files in os.walk(DATA_MOUNT):
            level = root.replace(DATA_MOUNT, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 2:
                for f_name in files[:5]:
                    print(f"{indent}  {f_name}")
        raise RuntimeError(f"Dataset not found at {data_dir}")

    # Build environment
    env = os.environ.copy()
    env.update({
        "DATA_PATH": data_dir,
        "TOKENIZER_PATH": tok_path,
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "600",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "200",
        "EVAL_STRIDE": "64",
        "SEED": str(seed),
    })

    # Apply user overrides
    if env_overrides:
        env.update(env_overrides)

    # Run torchrun
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=8",
        script_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    print(f"Env overrides: {env_overrides}")

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,  # 25 min hard timeout
    )

    output = result.stdout + "\n" + result.stderr

    # Print to Modal logs
    print(output[-5000:])  # Last 5K chars

    if result.returncode != 0:
        print(f"FAILED with return code {result.returncode}")

    return output


@app.local_entrypoint()
def main(
    script: str = "train_gpt_v21.py",
    env: str = "",
    seed: int = 1337,
    run_id: str = "",
):
    """
    Run parameter-golf training on Modal.

    Args:
        script: Path to training script
        env: Comma-separated env overrides, e.g. "NUM_LAYERS=11,MUON_WD=0.04"
        seed: Random seed
        run_id: Run identifier for logging
    """
    # Read script
    script_path = Path(script)
    if not script_path.exists():
        script_path = Path(__file__).parent / script
    if not script_path.exists():
        print(f"Script not found: {script}")
        sys.exit(1)

    script_content = script_path.read_text()
    print(f"Script: {script_path} ({len(script_content)} bytes)")

    # Parse env overrides
    env_overrides = {}
    if run_id:
        env_overrides["RUN_ID"] = run_id
    if env:
        for pair in env.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                env_overrides[k.strip()] = v.strip()

    print(f"Seed: {seed}")
    print(f"Env overrides: {env_overrides}")
    print("Launching on Modal 8xH100...")

    output = run_training.remote(
        script_content=script_content,
        script_name=script_path.name,
        env_overrides=env_overrides,
        seed=seed,
    )

    # Save log locally
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_name = f"modal_{script_path.stem}_seed{seed}.txt"
    log_path = log_dir / log_name
    log_path.write_text(output)
    print(f"\nLog saved to: {log_path}")

    # Extract final score
    for line in output.split("\n"):
        if "final_sliding_window_exact" in line:
            print(f"\n*** RESULT: {line.strip()}")
        if "final_int8_zlib_roundtrip_exact" in line:
            print(f"*** POST-QUANT: {line.strip()}")
        if "Total submission size" in line:
            print(f"*** SIZE: {line.strip()}")
