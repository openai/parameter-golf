import os
import re
import subprocess
import modal

APP_NAME = "parameter-golf-repro-longcontext-8h100"
REPO_REMOTE_PATH = "/workspace/parameter-golf"
TARGET_SCRIPT = "records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py"

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub",
        "kernels",
        "setuptools",
        "typing-extensions==4.15.0",
        "datasets",
        "tiktoken",
        "sentencepiece",
    )
    .add_local_dir(".", remote_path=REPO_REMOTE_PATH)
)

@app.function(image=image, gpu="H100:8", timeout=90 * 60, cpu=32, memory=196608)
def run():
    os.chdir(REPO_REMOTE_PATH)

    subprocess.run(["python", "data/cached_challenge_fineweb.py", "--variant", "sp1024"], check=True)

    env = os.environ.copy()
    env.update(
        {
            "RUN_ID": "modal_longcontext_8h100_repro",
            "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
            "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": "600",
            "NCCL_IB_DISABLE": "1",
        }
    )

    cmd = ["torchrun", "--standalone", "--nproc_per_node=8", TARGET_SCRIPT]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Training failed with exit code {rc}")

    log = "".join(lines)
    exact = re.search(r"final_int8_zlib_roundtrip_exact\\s+val_loss:([0-9.]+)\\s+val_bpb:([0-9.]+)", log)
    size = re.search(r"Total submission size int8\\+zlib:\\s*([0-9]+)\\s*bytes", log)
    step = re.search(r"stopping_early: wallclock_cap .* step:([0-9]+)/([0-9]+)", log)

    out = {
        "val_loss": float(exact.group(1)) if exact else None,
        "val_bpb": float(exact.group(2)) if exact else None,
        "bytes_total_int8_zlib": int(size.group(1)) if size else None,
        "steps_done": int(step.group(1)) if step else None,
        "steps_target": int(step.group(2)) if step else None,
    }
    print("\\n=== REPRO SUMMARY ===")
    print(out)
    return out

@app.local_entrypoint()
def main():
    print(run.remote())
