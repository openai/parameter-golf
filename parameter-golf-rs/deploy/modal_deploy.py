"""Modal deployment for Parameter Golf competition.

Usage:
    modal run deploy/modal_deploy.py              # default: train
    modal run deploy/modal_deploy.py::smoke       # GPU smoke test
    modal run deploy/modal_deploy.py::gemm_bench  # GEMM shape sweep
    modal run deploy/modal_deploy.py::nccl_bench  # NCCL collective bandwidth

Runs on 8xH100 GPUs with 600s wall-clock limit.
"""

import modal

app = modal.App("parameter-golf-rs")

# Build the Rust binaries inside the container
image = (
    modal.Image.from_dockerfile("deploy/Dockerfile", context_dir=".", add_python="3.12")
    .pip_install("huggingface_hub")  # for data download
)

# Data volume for training shards
data_volume = modal.Volume.from_name("pg-data", create_if_missing=True)

# Output volume for artifacts
output_volume = modal.Volume.from_name("pg-output", create_if_missing=True)


# --- Benchmarks ---


@app.function(
    image=image,
    gpu="H100:8",
    timeout=120,
    startup_timeout=900,
)
def smoke():
    """GPU smoke test: enumerate devices, H2D/D2H sanity, alloc test."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    result = subprocess.run(
        ["pg-smoke"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=60,
    )
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Smoke test failed with code {result.returncode}")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=300,
    startup_timeout=900,
)
def gemm_bench():
    """GEMM TFLOPS sweep across competition-relevant shapes."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    result = subprocess.run(
        ["pg-gemm-bench"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=240,
    )
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"GEMM bench failed with code {result.returncode}")


@app.function(
    image=image,
    gpu="H100:8",
    timeout=300,
    startup_timeout=900,
)
def nccl_bench():
    """NCCL collective bandwidth benchmark (all_reduce, reduce_scatter, all_gather)."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    result = subprocess.run(
        ["pg-nccl-bench"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=240,
    )
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"NCCL bench failed with code {result.returncode}")


@app.function(
    image=image,
    gpu="H100:1",
    timeout=300,
    startup_timeout=900,
)
def parity_forward():
    """Run parity-forward on the baseline_sp8192 spec."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    result = subprocess.run(
        ["pg-parity-forward", "--spec", "/specs/baseline_sp8192.toml"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=240,
    )
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Parity-forward failed with code {result.returncode}")


# --- Training & Evaluation ---


@app.function(
    image=image,
    gpu="H100:8",
    timeout=700,  # 600s training + 100s overhead
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def train():
    """Run the full training + eval pipeline."""
    import subprocess
    import os
    import time

    os.environ["RUST_LOG"] = "info"
    os.environ["DATA_DIR"] = "/data/datasets/fineweb10B_sp1024"
    os.environ["OUTPUT_DIR"] = "/output"

    start = time.time()

    result = subprocess.run(
        ["pg-train"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=650,
    )

    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")
    print(f"stdout: {result.stdout[-2000:]}")
    print(f"stderr: {result.stderr[-2000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Commit output volume
    output_volume.commit()


@app.function(
    image=image,
    gpu="H100:8",
    timeout=300,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def evaluate():
    """Run evaluation with TTT on the saved artifact."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    os.environ["ARTIFACT_PATH"] = "/output/artifact.pgrs"
    os.environ["DATA_DIR"] = "/data/datasets/fineweb10B_sp1024"

    result = subprocess.run(
        ["pg-train", "--eval-only"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=250,
    )

    print(f"stdout: {result.stdout[-2000:]}")
    print(f"stderr: {result.stderr[-2000:]}")


@app.local_entrypoint()
def main():
    train.remote()
