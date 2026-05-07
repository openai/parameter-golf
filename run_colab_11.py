# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_11 Training Run
# Built on sota_10 with upgrades targeting sub-1.1 BPB:
# - MTP 2 heads enabled by default (-0.002 to -0.004 BPB auxiliary training signal)
# - Trigram ON by default — reuses bigram table at near-zero cost (-0.001 BPB)
# - VE on 3 layers (8,9,10) instead of 2 (-0.001 BPB)
# - recur_layers "2,3,4,5" → 4-layer deep recurrence (was 3 layers)
# - recur_start_step 1500 (was 3000) — more training in recurrence mode
# - warmdown_iters 5500 (was 4200) — longer final convergence
# - gptq_ar_seqs 64 (was 32) — better Hessian estimation for GPTQ
# - recur_passes configurable (set to 2 for even deeper iterations at cost of fewer steps)

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
    # --- Base architecture (from sota_9) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    # --- Depth Recurrence (sota_11: 4 layers, starts earlier) ---
    f"RECUR_LAYERS=2,3,4,5",       # was 3,4,5 in sota_10
    f"RECUR_START_STEP=1500",      # was 3000 — more training in recurrence mode
    f"RECUR_PASSES=1",             # set to 2 for deeper recurrence (slower steps)
    # --- MTP auxiliary head (NEW in sota_11) ---
    f"MTP_NUM_HEADS=2",
    f"MTP_LOSS_WEIGHT=0.1",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=5500",        # was 4200
    # --- Trigram + VE (sota_11 defaults already set, explicit here for clarity) ---
    f"TRIGRAM=1",                  # reuses bigram table at near-zero cost
    f"VE_LAYERS=8,9,10",           # was 9,10 — one more VE injection layer
    # --- GPTQ (65 seqs for better Hessian) ---
    # gptq_ar_seqs default is now 64 in sota_11
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_11.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
