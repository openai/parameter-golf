"""Parameter Golf Smoke Test — verify model trains without NaN"""
import os
import sys
import subprocess

# Clone repo
os.system("git clone https://github.com/nickferrantelive/parameter-golf.git /tmp/pg")
os.chdir("/tmp/pg")

# Download small data subset
os.system("pip install -q sentencepiece huggingface-hub datasets")
os.system("python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    
    if props.major < 7:
        print("⚠️ GPU too old (need sm_70+). Installing compatible PyTorch...")
        os.system("pip install -q torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118")
        import importlib
        importlib.reload(torch)
        print(f"PyTorch after reinstall: {torch.__version__}")

# Run baseline for 50 steps with short wallclock
print("\n=== TESTING BASELINE (50 steps, 2 min cap) ===")
env = os.environ.copy()
env.update({
    "RUN_ID": "smoke_baseline",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "MAX_WALLCLOCK_SECONDS": "120",
    "TRAIN_SEQ_LEN": "512",
    "TRAIN_BATCH_TOKENS": "65536",
})

result = subprocess.run(
    ["python3", "train_gpt.py"],
    env=env,
    capture_output=True,
    text=True,
    timeout=300
)

print("STDOUT (last 20 lines):")
for line in result.stdout.strip().split("\n")[-20:]:
    print(line)

if result.returncode != 0:
    print(f"\nSTDERR (last 10 lines):")
    for line in result.stderr.strip().split("\n")[-10:]:
        print(line)

if "loss=nan" in result.stdout.lower() or "loss:nan" in result.stdout.lower():
    print("\n❌ BASELINE HAS NaN")
elif "val_bpb" in result.stdout:
    print("\n✅ BASELINE TRAINED OK")
else:
    print(f"\n⚠️ Unclear result (exit code {result.returncode})")
