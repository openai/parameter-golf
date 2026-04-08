# Notable Non-Record Submission: 1.1239 BPB — 106.2 Asymmetric Binary U-Net Transformer

**1-bit Quantisation + 15L (7 Encoder - 8 Decoder) + NeoMuon + 4x relu² MLP + SmearGate + Factored Tied Embedding + Poly5 Softcap + YaRN 2048 + 8192 BPE + FP8 QAT + LZMA + Stride-16 Sliding Eval**

**val_bpb: 1.1239** (sliding, seed=42) | **15.67 MB** artifact | 8×H100 SXM, 50k steps (~2.15h)

> **This is a **non-record submission** — training exceeds the 10-minute wallclock constraint (50,000 steps / ~2.15 hours). Submitted to demonstrate the compression frontier: 106.2 parameters in 15.67MB via 1-bit quantisation. Over 120M possible with FP4 (implemented) with a worse bpb. Full experiment log: [RESULTS.md](RESULTS.md). Complete training logs: [logs/](https://github.com/CiprianFlorin-Ifrim/openai-parameter-golf-submission/tree/main/logs/cuda).**

## Results (seed=42, 8×H100 SXM)

| Metric | Value |
|--------|-------|
| Sliding BPB (s16) | **1.1239** |
| val_bpb | 1.1497 |
| RT bpb | 1.1516 |
| Steps | 50,000 |
| ms/step | 155.3 |
| Training time | 7,763s (~2.15h) |
| optimal_T | 0.90 |
| Artifact | 15,670,651 bytes (15.67MB) |
| Parameters | 106,154,616 |

### Comparison to Ternary Submission

Binary reaches better absolute quality but requires circa 13x more training time. Within the 10-minute budget, binary's best fitting run (14L, 4,820 steps) scores 1.1824 sliding — 0.025 bpb worse than ternary (my previous record PR). The zero state is worth more at convergence than the 60% parameter density advantage.

The results document linked here and in my repo showcases all methods and sweeps applied to both Binary and Ternary Bitnets, which unfortunately are incompatible with many methods, such as Tversky Layers, EMA, Muon WD, LM Logit Head ranking and many more.

## Architecture

- 15 transformer layers, dim=768, 8 heads, 4 KV heads (GQA), head_dim=96
- Binary quantisation: weights {-1, +1}, 1 bit/param, per-group (128) absmean scaling
- 4x MLP expansion (hidden=3072) with **relu²** activation, fused gate+up projection
- U-Net encoder/decoder with learned skip weights (ones-init) and per-block residual mix from input embedding
- **SmearGate:** causal cumulative mean blending with learned tanh gate, zero-init for safe residual start
- Factored tied embedding: 8192×254 bottleneck with learned projections
- Polynomial softcap (degree 5, cap=10) with Z-loss regularisation (1e-4)
- YaRN positional encoding (max_len=2048, ROPE_BASE=5000)
- Fused QKV projection
- FlashAttention-3 (Hopper native kernels)
- 106.2M parameters, 15.67MB artifact (97.3M binary + 2.5M fp8 + 70KB code)

## Key Techniques

### Architecture
- **Binary quantisation:** 1 bit/param packs 60% more parameters per MB than ternary (1.6 bits/param), allowing 15 layers vs 10 within similar budget
- **4x relu² MLP:* relu² strictly dominates relu; 4x width outperforms 3x even with fewer layers at matched budget
- **SmearGate:** blends each position with causal cumulative mean; adds 22ms/step overhead but provides -0.007 bpb at scale. Viable here because the run is not wallclock-constrained

### Training
- **NeoMuon** with 3 Newton-Schulz steps optimizer
- **50,000 steps unconstrained:** binary converges slower than ternary (my other #640, at 4,000 steps (the 10-minute equivalent) binary lags by 0.025 bpb. Extended training closes the gap and surpasses ternary, showcasing with "unlimited compute" the models can be quite powerful.
- **524k batch tokens:**

### Evaluation
- **Temperature scaling (T=0.90):** auto-calibrated grid
- **Sliding window (stride=16):** evaluation protocol

### Compression
- **Bit-packing + LZMA (preset=9):** binary weights pack at exactly 1 bit/param before LZMA entropy coding
- **FP8 QAT (e4m3):** for non-binary parameters. Clean roundtrip, binary has no zero state, so `mean(|Q|)=1.0` always; no shrinkage correction needed
- **No EMA:** despite clean binary roundtrip math, EMA still hurts quality by 0.03 bpb in practice

## Setup and Run

```bash
# Environment setup (conda + Python 3.13 + PyTorch + FlashAttention-3 + Triton + dataset)
bash setup.sh

# Activate and run
conda activate golf
SEED=42 bash run_cuda_binary.sh
```

<details>
<summary>Full run command</summary>

```bash
RUN_ID=binary_run \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
ATTN_PROJ_TYPE=standard \
LOGIT_HEAD_TYPE=standard \
TVERSKY_MEMBERSHIP=sigmoid \
TVERSKY_NUM_FEATURES=0 \
TVERSKY_FEATURE_POOLS=0 \
VOCAB_SIZE=8192 \
BITNET_GROUP_SIZE=128 \
BIGRAM_HASH=0 \
EMBED_DIM=254 \
TRAINING_DEPTH_RECURRENCE=0 \
EVAL_DEPTH_RECURRENCE=0 \
NUM_LAYERS=15 \
MODEL_DIM=768 \
NUM_KV_HEADS=4 \
NUM_HEADS=8 \
DIFF_ATTN=0 \
MLP_MULT=4 \
MLP_GROUPS=0 \
MATRIX_OPTIMIZER=muon \
ADAM_LR=0.05 \
ADAM_WD=0.05 \
MUON_BACKEND_STEPS=3 \
MUON_MOMENTUM=0.95 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
MUON_WD=0.0 \
MATRIX_LR=0.04 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.02 \
WARMDOWN_FRACTION=0.2 \
LOGIT_SOFTCAP=10 \
QK_GAIN_INIT=2.25 \
ROPE_TYPE=yarn \
YARN_MAX_LEN=2048 \
ROPE_BASE=5000 \
BATCH_TOKENS_START=0 \
BATCH_SCHEDULE_FRACTION=0.33 \
TRAIN_BATCH_TOKENS=524288 \
SEQ_LEN_START=0 \
SEQ_SCHEDULE_FRACTION=0.0 \
TRAIN_SEQ_LEN=1024 \
SMEAR=1 \
ITERATIONS=50000 \
WARMUP_STEPS=5 \
MAX_WALLCLOCK_SECONDS=0 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=500 \
CHURN_LOG_EVERY=1000 \
VAL_MAX_TOKENS=0 \
TIE_EMBEDDINGS=1 \
UNTIE_AT_FRACTION=0.00 \
HEAD_LR=0.02 \
CORR_WEIGHT_LR=0.02 \
ACTIVATION=relu2 \
SOFTCAP_TYPE=poly \
MTP_HEADS=0 \
REFINER=0 \
REFINER_KERNEL=3 \
SLIDING_EVAL=1 \
SLIDING_EVAL_STRIDE=16 \
SLIDING_BATCH_SIZE=256 \
TEMP_SCALING=1 \
FP_STORAGE=FP8 \
EMA=0 \
EMA_DECAY=0.995 \
EMA_START_FRACTION=0.5 \
SEED=42 \
COMPILE_MODE=default \
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train_gpt_cuda_binary.py
```

</details>

## Compliance

- [x] Artifact <=16,000,000 bytes (15,670,651)
- [x] Sliding window eval stride=16
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute
- [x] Train time: **non-record submission** (7,763s/ 2.2h / 50,000 steps)
