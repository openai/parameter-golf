# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_17 Training Run
# Built on sota_16 (sota_15 + N-gram Tilt + Hash Embedding) with:
#
# **nGPT: Normalized Transformer on the Hypersphere**
# (Loshchilov et al., 2024 — "nGPT: Normalized Transformer with Representation Learning on the Hypersphere")
#
# Key changes vs sota_16:
# - **No DyT prenorm**: removed from all blocks. Input to each sublayer is already on the sphere.
# - **L2-normalize input embeddings**: `x = F.normalize(x + bigram, dim=-1)` before blocks.
# - **Sphere-walk residual updates**: `x = normalize(x + α * f(x), dim=-1)` instead of `x = x + scale * f(norm(x))`
# - **Per-dim alpha vectors**: `attn_scale` and `mlp_scale` init to `1/sqrt(D) ≈ 0.044` per dim.
# - **Identity final_norm**: last block output is already on the sphere.
#
# Motivation: isotropic embeddings → better representation geometry → faster convergence
# (nGPT paper reports 4–10× fewer steps to same loss as standard transformer)
#
# User observation: future SOTA will use fewer steps but each step will be more compute-efficient
# (stronger loss decrease per step → nGPT naturally fits this pattern)

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
    f"TTT_LR=0.005",
    f"TTT_EPOCHS=3",
    f"TTT_CHUNK_SIZE=32768",
    # --- JEPA latent prediction ---
    f"JEPA_NECK=64",
    f"JEPA_WEIGHT=0.1",
    # --- N-gram Tilt ---
    f"NGRAM_BETA=0.5",
    # --- Eval-Time Hash Embedding ---
    f"HASH_EMB_SIZE=16384",
    # --- nGPT: hypersphere normalization (NEW) ---
    f"NGPT=1",              # 1 = enabled (default), 0 = disable (fallback to DyT as in sota_16)
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_17.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
