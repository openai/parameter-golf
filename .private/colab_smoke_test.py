# Parameter Golf - Kitchen Sink Smoke Test (Colab AIO Cell)
# Depth Recurrence (5x4 loops, dim=576) + MLP3x + SmearGate + BigramHash + Int6+zstd + SWA + TTT
# Run this as a single cell in Google Colab with GPU runtime

import subprocess, os, sys

# --- Setup ---
os.chdir("/content")
if not os.path.exists("parameter-golf"):
    subprocess.run(["git", "clone", "https://github.com/anthony-maio/parameter-golf.git"], check=True)
os.chdir("parameter-golf")
subprocess.run(["git", "checkout", "submission/depth-recurrence-kitchen-sink"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "zstandard", "huggingface-hub", "datasets"], check=True)

# Download minimal dataset (1 shard for smoke test)
subprocess.run([sys.executable, "data/cached_challenge_fineweb.py", "--variant", "sp1024", "--train-shards", "1"], check=True)

# --- Run training (short smoke test: 100 iterations, no wallclock cap) ---
env = os.environ.copy()
env.update({
    "RUN_ID": "colab_smoke",
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE": "1024",
    "ITERATIONS": "100",
    "MAX_WALLCLOCK_SECONDS": "0",
    "VAL_LOSS_EVERY": "50",
    "TRAIN_LOG_EVERY": "10",
    "TRAIN_BATCH_TOKENS": "131072",  # smaller batch for single GPU
})

subprocess.run(
    ["torchrun", "--standalone", "--nproc_per_node=1",
     "records/track_10min_16mb/2026-03-20_DepthRecurrence_Int6_MLP3x_SmearGate_BigramHash_TTT/train_gpt.py"],
    env=env, check=True,
)
