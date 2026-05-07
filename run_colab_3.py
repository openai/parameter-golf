# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_3 Training Run
# Run each cell in order. Runs train_gpt_sota_3.py (DiffAttn, MTP=3 decayed, val-set GPTQ calib).

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
# Install dependencies (flash_attn_3 removed — using PyTorch built-in SDPA)
os.system("pip install -q sentencepiece zstandard")
os.system('python3 -c "import sentencepiece, zstandard; print(\'deps OK\')"')

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
# folder with fineweb_train_*.bin & fineweb_val_*.bin
DATA_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

# --- Fixed SOTA settings ---
BIGRAM_VOCAB_SIZE = 3072
BIGRAM_DIM = 112
WARMDOWN_ITERS = 4000
ITERATIONS = 6927    # step-based stopping (equivalent to 600s on 8×H100)

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"BIGRAM_VOCAB_SIZE={BIGRAM_VOCAB_SIZE}",
    f"BIGRAM_DIM={BIGRAM_DIM}",
    f"WARMDOWN_ITERS={WARMDOWN_ITERS}",
    f"ITERATIONS={ITERATIONS}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
    # sota_3 new flags (all default to 1/on)
    "DIFF_ATTN=1",
    "GPTQ_USE_VAL=1",
    "MTP_NUM_HEADS=3",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_3.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
