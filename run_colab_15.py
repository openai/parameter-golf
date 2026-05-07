# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_15 Training Run
# Built on sota_12 with 2 key changes:
# 1. **Dynamic Tanh (DyT)** replaces RMSNorm everywhere
#    `DyT(x) = tanh(alpha * x)` — purely elementwise, no reduction kernel
# 2. **JEPA latent prediction** — auxiliary training objective
#    At each position t, predict h[t+1] from h[t] in embedding space (not token space)
#    A tiny 2-layer MLP (512→64→512) learns to align representations temporally
#    Loss: `1 - cosine_sim(pred, stop_grad(h[t+1]))` — collapse prevented by main CE loss
#    Weight: JEPA_WEIGHT=0.1 (additive to main loss)
# Expected: sota_12 baseline (1.1147 BPB) + DyT stability + JEPA representation gain

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
    # --- Architecture (sota_12 base) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    # --- Depth Recurrence (4 layers, starts earlier) ---
    f"RECUR_LAYERS=2,3,4,5",
    f"RECUR_START_STEP=1500",
    f"RECUR_PASSES=1",
    # --- MTP auxiliary ---
    f"MTP_NUM_HEADS=2",
    f"MTP_LOSS_WEIGHT=0.1",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=5500",
    # --- Trigram + VE ---
    f"TRIGRAM=1",
    f"VE_LAYERS=8,9,10",
    # --- Legal Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.001",
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    # --- JEPA latent prediction (NEW) ---
    f"JEPA_NECK=64",     # bottleneck dim of predictor MLP
    f"JEPA_WEIGHT=0.1",  # aux loss weight
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_15.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
