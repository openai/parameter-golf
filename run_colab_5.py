# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_5 Training Run
# Runs train_gpt_sota_5.py — Conservative improvements on sota_1 baseline:
# BigramHash 3072×112 (matching #1 record), Brotli-11, Soft-Round QAT from step 3000,
# Sigmoid-Gated Skips. No risky experiments (no VRL, no gated_attn, no rope_base changes).

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Clone repo

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
import glob
import os

REPO_URL = "https://github.com/angela231005/parameter-golf"
REPO_DIR = "parameter-golf"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} pull")

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Install dependencies

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system("pip install -q sentencepiece zstandard brotli")
os.system('python3 -c "import sentencepiece, zstandard, brotli; print(\'deps OK\')"')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Download & prepare data

# %% [code] {"jupyter":{"outputs_hidden":false}}
# os.system("python3 data/download_hf_docs_and_tokenize.py")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42          # change per run: 314, 42, 999
NPROC = 1           # 1 for Colab/single H100, 8 for full node
TARGET_MB = 15.9

# --- Paths (set to your existing dataset/tokenizer locations) ---
DATA_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

# --- Fixed SOTA-5 settings (conservative, proven-only techniques) ---
BIGRAM_VOCAB_SIZE = 3072   # #1 record uses 3072
BIGRAM_DIM = 112           # #1 record uses 112
QAT_START_STEP = 3000      # Enable QAT early in warmdown (warmdown starts ~2927)
WARMDOWN_ITERS = 4000
ITERATIONS = 6927

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"BIGRAM_VOCAB_SIZE={BIGRAM_VOCAB_SIZE}",
    f"BIGRAM_DIM={BIGRAM_DIM}",
    f"QAT_START_STEP={QAT_START_STEP}",
    f"WARMDOWN_ITERS={WARMDOWN_ITERS}",
    f"ITERATIONS={ITERATIONS}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_5.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
