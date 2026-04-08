# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_16 Training Run
# Built on sota_15 (DyT + JEPA + sota_12 base) with 2 eval-time enhancements:
#
# 1. **N-gram Tilt** (from PR #1437, achieved 1.08091 BPB on leaderboard)
#    At each TTT scoring position, boost the bigram-predicted next token by exp(beta).
#    `p_tilt = p_model * exp(beta * 1[t == bigram_hint]) / Z`
#    Bigram table updated AFTER scoring (score-first, fully causal).
#    Expected gain: ~0.010–0.015 BPB
#
# 2. **Eval-Time Hash Embedding** (from PR #1460, achieved 1.08269 BPB)
#    Zero-init nn.Embedding(16384, 512) created at eval, trained via TTT SGD.
#    `h = (prev_token * 2039 + curr_token) % 16384`, residual added before backbone.
#    Expected gain: ~0.0004 BPB
#
# Both are purely eval-time, no training change. Fully legal under Issue #1017.
# Expected: 1.105 → ~1.09 BPB (N-gram Tilt is the key unlock)

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
    f"TTT_LR=0.005",  # matched to PR #1460 (was 0.001)
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    # --- JEPA latent prediction ---
    f"JEPA_NECK=64",
    f"JEPA_WEIGHT=0.1",
    # --- N-gram Tilt (NEW: ~0.010-0.015 BPB gain) ---
    f"NGRAM_BETA=0.5",      # log-space tilt strength for bigram hint
    # --- Eval-Time Hash Embedding (NEW: ~0.0004 BPB gain) ---
    f"HASH_EMB_SIZE=16384",  # 0 to disable
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_16.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
