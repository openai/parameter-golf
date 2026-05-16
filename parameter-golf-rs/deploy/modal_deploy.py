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


@app.function(
    image=image,
    gpu="H100:1",
    timeout=300,
    startup_timeout=900,
)
def parity_kernels():
    """Run kernel-level parity checks (parity-kernels)."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    result = subprocess.run(
        ["pg-parity-kernels"],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=240,
    )
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Parity-kernels failed with code {result.returncode}")


@app.function(
    image=image,
    gpu="H100:1",
    timeout=300,
    startup_timeout=900,
)
def parity_step():
    """Run the one-step backward parity harness on the baseline spec."""
    import subprocess
    import os

    os.environ["RUST_LOG"] = "info"
    result = subprocess.run(
        [
            "pg-parity-step",
            "--spec",
            "/specs/baseline_sp8192.toml",
            "--backend",
            "cuda-single",
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=240,
    )
    print(result.stdout)
    if result.stderr:
        print(f"stderr: {result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Parity-step failed with code {result.returncode}")


# --- Training & Evaluation ---


@app.function(
    image=image,
    gpu="H100:1",
    timeout=700,
    startup_timeout=900,
    volumes={
        "/data": data_volume,
        "/output": output_volume,
    },
)
def train_single():
    """Run the training CLI on one H100. Defaults to cuda-single proxy mode."""
    import subprocess
    import os
    import time

    os.environ["RUST_LOG"] = "info"
    train_glob = os.environ.get("PG_TRAIN_GLOB")
    val_glob = os.environ.get("PG_VAL_GLOB")
    tokenizer_vocab = os.environ.get("PG_TOKENIZER_VOCAB")
    artifact_path = os.environ.get("PG_ARTIFACT_PATH", "/output/artifact_single.pgrs")
    builtin = os.environ.get("PG_VARIANT", "baseline_sp8192")
    mode = os.environ.get("PG_MODE", "proxy")
    backend = os.environ.get("PG_BACKEND", "cuda-single")
    world_size = os.environ.get("PG_WORLD_SIZE", "1")
    rank = os.environ.get("PG_RANK", "0")

    start = time.time()

    cmd = [
        "pg-train",
        "run",
        "--builtin",
        builtin,
        "--mode",
        mode,
        "--backend",
        backend,
        "--artifact",
        artifact_path,
        "--world-size",
        world_size,
        "--rank",
        rank,
    ]
    if train_glob:
        cmd.extend(["--train-data", train_glob])
    if val_glob:
        cmd.extend(["--val-data", val_glob])
    if tokenizer_vocab:
        cmd.extend(["--tokenizer-vocab", tokenizer_vocab])

    result = subprocess.run(
        cmd,
        env=os.environ,
        capture_output=True,
        text=True,
        timeout=650,
    )

    elapsed = time.time() - start
    print(f"Training completed in {elapsed:.1f}s")
    print(f"stdout: {result.stdout[-4000:]}")
    print(f"stderr: {result.stderr[-4000:]}")

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    output_volume.commit()


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
    train_glob = os.environ.get("PG_TRAIN_GLOB")
    val_glob = os.environ.get("PG_VAL_GLOB")
    tokenizer_vocab = os.environ.get("PG_TOKENIZER_VOCAB")
    artifact_path = os.environ.get("PG_ARTIFACT_PATH", "/output/artifact.pgrs")
    builtin = os.environ.get("PG_VARIANT", "baseline_sp8192")
    mode = os.environ.get("PG_MODE", "record")
    backend = os.environ.get("PG_BACKEND", "cuda-distributed")
    world_size = os.environ.get("PG_WORLD_SIZE", "8")
    rank = os.environ.get("PG_RANK", "0")

    start = time.time()

    cmd = [
        "pg-train",
        "run",
        "--builtin",
        builtin,
        "--mode",
        mode,
        "--backend",
        backend,
        "--artifact",
        artifact_path,
        "--world-size",
        world_size,
        "--rank",
        rank,
    ]
    if train_glob:
        cmd.extend(["--train-data", train_glob])
    if val_glob:
        cmd.extend(["--val-data", val_glob])
    if tokenizer_vocab:
        cmd.extend(["--tokenizer-vocab", tokenizer_vocab])

    result = subprocess.run(
        cmd,
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
    artifact_path = os.environ.get("PG_ARTIFACT_PATH", "/output/artifact.pgrs")
    val_glob = os.environ.get("PG_VAL_GLOB")
    tokenizer_vocab = os.environ.get("PG_TOKENIZER_VOCAB")
    builtin = os.environ.get("PG_VARIANT", "baseline_sp8192")

    cmd = [
        "pg-eval",
        "run",
        "--builtin",
        builtin,
        "--artifact",
        artifact_path,
    ]
    if val_glob:
        cmd.extend(["--val-data", val_glob])
    if tokenizer_vocab:
        cmd.extend(["--tokenizer-vocab", tokenizer_vocab])

    result = subprocess.run(cmd, env=os.environ, capture_output=True, text=True, timeout=250)

    print(f"stdout: {result.stdout[-2000:]}")
    print(f"stderr: {result.stderr[-2000:]}")


@app.local_entrypoint()
def main():
    train.remote()
