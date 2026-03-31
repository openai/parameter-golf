# %% [markdown]
# # Parameter Golf — SOTA Training Run
# Run each cell in order. Works on 1×H100 (Colab) or 8×H100.

# %% [markdown]
# ## 1. Clone repo

# %%
import os

REPO_URL = "https://github.com/angela231005/parameter-golf"
REPO_DIR = "/content/parameter-golf"

if not os.path.exists(REPO_DIR):
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} pull")

os.chdir(REPO_DIR)
print("cwd:", os.getcwd())

# %% [markdown]
# ## 2. Install dependencies

# %%
# Flash Attention 3 (Hopper / H100 required)
os.system("pip install -q flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291")
os.system("pip install -q sentencepiece zstandard")

# Verify
os.system('python3 -c "from flash_attn_interface import flash_attn_func; import sentencepiece, zstandard; print(\'deps OK\')"')

# %% [markdown]
# ## 3. Download & prepare data

# %%
os.system("python3 data/download_hf_docs_and_tokenize.py")

# %% [markdown]
# ## 4. Set hyperparameters

# %%
# --- Tune these ---
SEED          = 42          # change per run: 314, 42, 999
NPROC         = 1           # 1 for Colab/single H100, 8 for full node
TARGET_MB     = 15.9

# --- Fixed SOTA settings ---
BIGRAM_VOCAB_SIZE = 3072
BIGRAM_DIM        = 112
WARMDOWN_ITERS    = 4000
ITERATIONS        = 6927    # step-based stopping (equivalent to 600s on 8×H100)

env = " ".join([
    f"SEED={SEED}",
    f"BIGRAM_VOCAB_SIZE={BIGRAM_VOCAB_SIZE}",
    f"BIGRAM_DIM={BIGRAM_DIM}",
    f"WARMDOWN_ITERS={WARMDOWN_ITERS}",
    f"ITERATIONS={ITERATIONS}",
    f"MAX_WALLCLOCK_SECONDS=0",
    f"TARGET_MB={TARGET_MB}",
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota.py"
print("Command:")
print(cmd)

# %% [markdown]
# ## 5. Train!

# %%
os.system(cmd)
