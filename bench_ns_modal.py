"""Benchmark Newton-Schulz variants at our exact matrix sizes on H100."""
import modal

app = modal.App("ns-benchmark")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "numpy",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu126",
    )
    .pip_install("psutil", "packaging", "ninja")
    .pip_install("triton")
    .run_commands(
        "pip install git+https://github.com/Dao-AILab/gram-newton-schulz.git --no-build-isolation",
        "git clone https://github.com/Dao-AILab/gram-newton-schulz.git /opt/gns",
    )
)


@app.function(image=image, gpu="H100:1", timeout=900)
def bench():
    import subprocess
    import os

    os.chdir("/opt/gns")

    # Benchmark all 4 bank shapes from the parameter-golf model:
    # qo_bank shard:       ~3 x 512 x 512    (square)
    # kv_bank shard:       ~3 x 256 x 512    (1:2)
    # mlp_up_bank shard:   ~2 x 512 x 1536   (1:3, after transpose)
    # mlp_down_bank shard: ~2 x 512 x 1536   (1:3)
    configs = [
        ("qo_bank (square)",    512, 512,  3),
        ("kv_bank (1:2)",       256, 512,  3),
        ("mlp_up_bank (1:3)",   512, 1536, 2),
        ("mlp_down_bank (1:3)", 512, 1536, 2),
        # Also test larger sizes where Gram NS should clearly win
        ("large (1:3)",         2048, 6144, 4),
        ("large (1:8)",         512, 4096, 4),
    ]

    for name, M, N, batch in configs:
        print(f"\n{'='*60}")
        print(f"  {name}: M={M}, N={N}, batch={batch}")
        print(f"{'='*60}", flush=True)

        result = subprocess.run(
            ["python", "benchmarks/benchmark_newton_schulz.py",
             "--M", str(M), "--N", str(N), "--batch-size", str(batch),
             "--warmup", "10", "--repeats", "50"],
            capture_output=True, text=True,
        )
        print(result.stdout, flush=True)
        if result.stderr:
            # Filter out just warnings/errors, not the full stderr
            for line in result.stderr.split("\n"):
                if "error" in line.lower() or "warning" in line.lower():
                    print(f"  STDERR: {line}", flush=True)


@app.local_entrypoint()
def main():
    bench.remote()
