# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Parameter Golf — SOTA_13 Training Run
# Adds on top of sota_12:
# 1. 4-gram hash embedding (extends bigram+trigram → 4-gram, zero new params, ~0.001 BPB)
# 2. Cautious Weight Decay (CWD) in Muon: WD only applied where it opposes param magnitude
# 3. GPTQ_DAMP lowered 0.01→0.005 for better Hessian conditioning
# 4. More GPTQ AR calibration seqs: 64→96 for better Hessian estimation
# 5. TTT chunk size: 32768→16384 (2× more TTT adaptation steps)
# 6. TTT epochs: 3→4 (more adaptation per chunk)
# 7. recur_passes=2 (2× deeper recurrence at eval, 0 training cost)
# Expected: -0.005 to -0.012 BPB → target ~1.097-1.104 BPB

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
    # --- Inductor / Triton: prevent register-limit OOM on H100 ---
    f"TORCHINDUCTOR_COMBO_KERNELS=0",
    f"TORCHINDUCTOR_PERSISTENT_REDUCTIONS=0",  # use split reductions (fewer registers)
    f"TORCHINDUCTOR_MAX_FUSION_SIZE=32",        # cap fusion depth (default 64)
    # --- Architecture (sota_9 base) ---
    f"QK_GAIN_INIT=4.0",
    f"BIGRAM_DIM=112",
    # --- N-gram hash embeddings (NEW: 4-gram added) ---
    f"TRIGRAM=1",
    f"FOURGRAM=1",
    # --- Parallel Residuals ---
    f"PARALLEL_RESIDUAL=1",
    f"PARALLEL_START_LAYER=5",
    # --- Depth Recurrence (sota_11+: 4 layers, starts earlier) ---
    f"RECUR_LAYERS=2,3,4,5",
    f"RECUR_START_STEP=1500",
    f"RECUR_PASSES=1",       # 1 during training (recur_passes=2 backward OOMs Triton)
    f"EVAL_RECUR_PASSES=2",  # 2× deeper recurrence for TTT scoring (inference_mode, no backward)
    # --- MTP auxiliary ---
    f"MTP_NUM_HEADS=2",
    f"MTP_LOSS_WEIGHT=0.1",
    # --- Training schedule ---
    f"WARMDOWN_ITERS=5500",
    # --- VE layers ---
    f"VE_LAYERS=8,9,10",
    # --- Optimizer ---
    f"CAUTIOUS_WD=1",        # NEW: Cautious Weight Decay in Muon
    f"MUON_WD=0.04",
    # --- GPTQ (improved) ---
    f"GPTQ_AR_SEQS=96",      # NEW: more calibration seqs (was 64)
    f"GPTQ_DAMP=0.005",      # NEW: lower damping (was 0.01) for better Hessian fit
    # --- Legal Score-First TTT ---
    f"TTT_ENABLED=1",
    f"TTT_LR=0.001",         # SGD lr for adaptation
    f"TTT_EPOCHS=4",         # NEW: 4 epochs (was 3)
    f"TTT_CHUNK_SIZE=16384", # NEW: smaller=more TTT steps (was 32768)
])

cmd = f"{env} torchrun --standalone --nproc_per_node={NPROC} train_gpt_sota_13.py"
print("Command:")
print(cmd)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Train!

# %% [code] {"jupyter":{"outputs_hidden":false}}
os.system(cmd)
