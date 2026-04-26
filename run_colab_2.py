# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_2 Training Run
# Run each cell in order. Runs train_gpt_sota_2.py (VRL, gated_attn, rope50k, longer_qat, swa30, ve7layers).

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Clone repo

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-31T13:06:21.086153Z","iopub.execute_input":"2026-03-31T13:06:21.086529Z","iopub.status.idle":"2026-03-31T13:06:22.463952Z","shell.execute_reply.started":"2026-03-31T13:06:21.086495Z","shell.execute_reply":"2026-03-31T13:06:22.462941Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-31T13:06:22.465853Z","iopub.execute_input":"2026-03-31T13:06:22.466603Z","iopub.status.idle":"2026-03-31T13:07:05.632461Z","shell.execute_reply.started":"2026-03-31T13:06:22.466566Z","shell.execute_reply":"2026-03-31T13:07:05.631110Z"}}
# Install dependencies (flash_attn_3 removed — using PyTorch built-in SDPA)
os.system("pip install -q sentencepiece zstandard")
os.system('python3 -c "import sentencepiece, zstandard; print(\'deps OK\')"')

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Download & prepare data

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-31T13:07:05.634149Z","iopub.execute_input":"2026-03-31T13:07:05.634706Z","iopub.status.idle":"2026-03-31T13:07:05.640102Z","shell.execute_reply.started":"2026-03-31T13:07:05.634660Z","shell.execute_reply":"2026-03-31T13:07:05.639029Z"}}
# os.system("python3 data/download_hf_docs_and_tokenize.py")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Set hyperparameters

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-31T13:07:05.641493Z","iopub.execute_input":"2026-03-31T13:07:05.641952Z","iopub.status.idle":"2026-03-31T13:07:06.559607Z","shell.execute_reply.started":"2026-03-31T13:07:05.641901Z","shell.execute_reply":"2026-03-31T13:07:06.558547Z"}}
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
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_2.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Train!

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-31T13:07:06.562506Z","iopub.execute_input":"2026-03-31T13:07:06.563290Z","iopub.status.idle":"2026-03-31T13:07:12.700341Z","shell.execute_reply.started":"2026-03-31T13:07:06.563239Z","shell.execute_reply":"2026-03-31T13:07:12.699356Z"}}
os.system(cmd)
