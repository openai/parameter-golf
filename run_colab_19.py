# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_19 Training Run
# Built on sota_10 (clean, stable baseline) with three proven eval-time enhancements:
# - Legal Score-First TTT (PR #549: -0.0025 BPB)
#   chunk=32768, SGD lr=0.002 global cosine, 3 epochs, all blocks unfrozen
# - N-gram Tilt (PR #1437): boost bigram-predicted token by exp(0.5) at inference
# - Eval-Time Hash Embedding (PR #1460): zero-init lookup adapts via TTT at 10× LR
#
# All enhancements are strictly causal / score-first — fully legal per challenge rules.

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
    # --- Architecture (from sota_10) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    f"RECUR_LAYERS=3,4,5",
    f"RECUR_START_STEP=3000",
    f"WARMDOWN_ITERS=5500",      # extended from sota_10's 4200
    f"GPTQ_AR_SEQS=64",          # PR #1019: 64 seqs is optimal (was 32)
    # --- Legal Score-First TTT (PR #549 recipe) ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.002",             # PR #549: 0.002 cosine global
    f"TTT_EPOCHS=3",             # PR #549: 3 epochs per chunk
    f"TTT_CHUNK_SIZE=32768",     # PR #549: 32768 tokens per chunk
    f"TTT_FREEZE_BLOCKS=0",      # PR #549: all blocks adapt (freeze=0 is best)
    # --- N-gram tilt (PR #1437) ---
    f"NGRAM_BETA=0.5",           # boost bigram hint by exp(0.5)
    # --- Eval-time hash embedding (PR #1460) ---
    f"HASH_EMB_SIZE=16384",      # zero-init, trained at 10× TTT LR
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_19.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
