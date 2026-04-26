# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_7 Training Run
# Record-matching base + innovations: MTP OFF, Trigram OFF, VE "9,10",
# warmdown 3500, GPTQ block 128 (all matching record), plus:
# - Brotli-11 compression (better than record's lzma-9)
# - SWA fix (record collects but never applies — we evaluate & pick best)
# - late_qat_threshold 0.20 (record uses 0.15 — more QAT adaptation steps)

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
NPROC = 1           # 1 for Colab/single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths (set to your existing dataset/tokenizer locations) ---
DATA_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

# --- SOTA-7: Record-matching + innovations ---
ITERATIONS = 6927

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"ITERATIONS={ITERATIONS}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_7.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
