# SR-CM-P2Loss

A constraint-optimized GPT leveraging difficulty-aware training (P2 loss) and wallclock-aligned optimization to maximize compression efficiency.

**val_bpb: 1.0577** | **~15.06 MB** | 8×H100

---

## Results (8×H100, PyTorch, 10-min cap)

- Final roundtrip (int8+zlib):
  - **val_loss: 1.78588484**
  - **val_bpb: 1.05770160**
- Training stopped at **step 7821 / 20000** due to wallclock cap
- Train time: **~603s**
- Step time: **~77ms**
- Peak memory: **~12.5GB**
- Total submission size: **~15.06MB**

---

## Key Innovations

### 1. P2 Loss (Primary Driver)

Loss scaled by:

\[
(1 - p)^2
\]

- Focuses gradient on low-confidence tokens  
- Accelerates early learning  
- Improves final compression (BPB)  

> This was the dominant performance driver.

---

### 2. Wallclock-Aware LR Warmdown (Critical)

- Learning rate decays over the **final 35% of wallclock time**  
- Aligns optimization with the **hard 10-minute constraint**  

> Step-based schedules underperformed.

---

### 3. Residual Mixing + Scaling (SR)

- Learned residual mixing between current state and initial embedding  
- Per-channel scaling of attention and MLP outputs  

- Improves stability  
- Enables deeper signal propagation  

---

### 4. Conv Token Mixer (CM)

- Depthwise 1D convolution before transformer blocks  
- Lightweight local context mixing  

- Small but consistent efficiency gain  

---

### 5. Compression-Aware Training + QAT

- **Int6 quantization target (USE_INT6=1)**  
- **Late-stage QAT (LATE_QAT=1)** when LR decays  
- Encourages weights to align with quantization constraints during training  

---

### 6. Compression Strategy

- Per-row quantization (large tensors)  
- Small/control tensors retained in higher precision  
- zstd compression (fallback zlib)

- Final:
  - ~82MB → **~15MB (~5.4× reduction overall, ~3.9× tensor payload)**  
  - Minimal degradation after roundtrip  

---

## Architecture

| Component | Setting |
|----------|--------|
| Model | GPT-style decoder |
| Layers | 11 |
| Dim | 512 |
| Heads | 8 |
| KV Heads | 4 (GQA) |
| MLP | PReLU + squared activation |
| MLP Mult | 2 |
| Embeddings | Tied |
| Token Mixer | Depthwise Conv1D |
| Residual | Learned mixing + scaling |

---

## Optimization

- **Hybrid optimizer split:**
  - Matrix params → **Muon (orthogonalized updates)**
  - Scalar/control params → **Adam**
  - Embeddings → separate Adam group  

- Weight decay:
  - Muon: 0.012  
  - Adam: 0.012  

- Large batch:
  - **524k tokens/step**
  - Distributed across 8 GPUs  

---

## Training Schedule

- Warmup  
- **Wallclock-aware warmdown (35% of run)**  
- Early stop at time cap (~603s)

---

## Output Shaping

- Logit softcap: **15.0**  
- Logit sharpening: **1.10**

---

## What Mattered

- **P2 Loss → primary gain**
- **Wallclock-aware warmdown → critical**
- **Residual mixing → stability**
- **GQA → parameter efficiency**
- **QAT + compression-aware training → size/perf balance**

---

## What Didn’t Matter

- EMA  
- Recurrence  
- Top-k masking  
- Tokenizer changes  
- Most normalization tweaks  

---

## Key Insight (Why This Works)

Under tight parameter and time constraints, performance is driven by:

- **Gradient allocation (P2 loss)**
- **Training schedule aligned to wallclock limits**
- **Stability of signal propagation**
- **Co-design of training and compression**

---

## Run Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=FINAL_SUBMISSION \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_FRAC=0.35 \
TIED_EMBED_LR=0.01 \
MATRIX_LR=0.04 \
SCALAR_LR=0.035 \
MUON_WEIGHT_DECAY=0.012 \
ADAM_WEIGHT_DECAY=0.012 \
LOGIT_SOFTCAP=15.0 \
LOGIT_SHARPEN=1.10 \
USE_INT6=1 \
USE_ZSTD=1 \
LATE_QAT=1 \
LATE_QAT_THRESHOLD=0.05 \
INT8_KEEP_FLOAT_MAX_NUMEL=8192 \
TRAIN_LOG_EVERY=500 \
VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py