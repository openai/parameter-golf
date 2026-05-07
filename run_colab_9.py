# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_9 Training Run
# Built on sota_7 base with 3 architectural innovations from PR #1204:
# - QK_GAIN=4.0 (PR #1125 sweep: -0.006 BPB)
# - Parallel Residuals from physical layer 7+ (dual lane: attn reads lane0, MLP reads lane1)
# - Mini Depth Recurrence: repeat layers 4,5 after encoder (activated at step 3000)
# - Bigram dim reduced from 128 to 112 to fit budget

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
# ## 3. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false}}
# --- Tune these ---
SEED = 42          # change per run: 314, 42, 999
NPROC = 1           # 1 for single GPU, 8 for full node
TARGET_MB = 15.9

# --- Paths ---
DATA_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/datasets/fineweb10B_sp1024"
TOKENIZER_PATH = "/kaggle/input/datasets/haphmph/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

ITERATIONS = 6927

env = " ".join([
    f"SEED={SEED}",
    f"DATA_PATH={DATA_PATH}",
    f"TOKENIZER_PATH={TOKENIZER_PATH}",
    f"ITERATIONS={ITERATIONS}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
    # --- SOTA_9 innovations ---
    f"QK_GAIN_INIT=4.0",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=7",
    f"RECUR_LAYERS=4,5",
    f"RECUR_START_STEP=3000",
    f"BIGRAM_DIM=112",
    # Inductor workaround for FusedMixOrderReductions assertion
    f"TORCHINDUCTOR_COMBO_KERNELS=0",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_9.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
