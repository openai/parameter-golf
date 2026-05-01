# Record: 1.1570 BPB — 73.7M Ternary U-Net Transformer

**BitNet b1.58 + 10L + NeoMuon + 4x relu² MLP + Factored Tied Embedding + Poly5 Softcap + YaRN 2048 + 8192 BPE + FP8 QAT + Base-3 LZMA + Stride-16 Sliding Eval**

**val_bpb: 1.1570** (3-seed mean sliding, std 0.0007) | **15.99 MB** max artifact | 8×H100 SXM, 599s

> **Full experiment log covering 250+ runs, ablations, and decision rationale, that could help anyone else: [RESULTS.md](RESULTS.md). Complete training logs in my personal repo: [logs/](https://github.com/CiprianFlorin-Ifrim/openai-parameter-golf-submission/tree/main/logs/cuda).**

The results document linked here and in my repo showcases all methods and sweeps applied to both Binary and Ternary Bitnets, which unfortunately are incompatible with many methods, such as Tversky Layers, EMA, Muon WD, LM Logit Head ranking and many more. Scaling ratios and applicable/rejected techniques can be useful for other submissions too.

## Results (3 seeds, 8×H100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s16) | val_bpb | RT bpb | Artifact |
|------|-------|---------|-------------------|---------|--------|----------|
| 42 | 6,530 | 91.7 | **1.1565** | 1.1816 | 1.1837 | 15,993,853 bytes |
| 1337 | 6,520 | 91.9 | 1.1568 | 1.1825 | 1.1839 | 15,995,705 bytes |
| 7 | 6,530 | 91.8 | 1.1578 | 1.1823 | 1.1850 | 15,992,753 bytes |
| **Mean** | **6,527** | **91.8** | **1.1570** | **1.1821** | **1.1842** | **15,994,104 bytes** |
| **Std** | **5** | **0.1** | **0.0007** | **0.0005** | **0.0007** | **1,498 bytes** |

## Architecture

- 10 transformer layers, dim=768, 8 heads, 4 KV heads (GQA), head_dim=96
- BitNet b1.58 ternary quantisation: weights {-1, 0, +1}, ~1.6 bits/param, per-group (128) absmean scaling
- 4x MLP expansion (hidden=3072) with **relu²** activation, fused gate+up projection
- U-Net encoder/decoder with learned skip weights (ones-init) and per-block residual mix from input embedding
- Factored tied embedding: 8192×254 bottleneck with learned 254-to-768 and 768-to-254 projections
- Polynomial softcap (degree 5, cap=10) with Z-loss regularisation (1e-4)
- YaRN positional encoding (max_len=2048, ROPE_BASE=5000)
- Fused QKV projection (single TernaryLinear)
- FlashAttention-3 (Hopper native kernels)
- 73.7M parameters, 15.92MB artifact (64.9M ternary + 2.5M fp8 + 70KB code)

## Key Techniques

### Architecture
- **Width over depth:** 768d/10L outperforms 512d/25L — faster steps (91ms vs 127ms) yield 6,530 vs 4,720 steps in 600s
- **4x relu² MLP:** relu² is -0.024 bpb over relu at zero cost; 4x width adds -0.008 bpb over 3x at same step budget
- **EMBED_DIM=254:** frees ~4MB for wider MLP; 254 = 256-2 to fit code within the byte budget

### Training
- **NeoMuon** with 3 Newton-Schulz steps: compensates for ternary STE gradient attenuation; 3 steps equivalent to 5 at convergence (+190 free steps)
- **Fused QKV + fused relu²:** ~4-6ms/step saving (~180 extra training steps)
- **FlashAttention-3:** -9% step time (~380 free steps)
- **524k batch tokens:** optimal for ternary STE — 262k too noisy, 1M loses gradient updates

### Evaluation
- **Temperature scaling (T=0.90):** 5-point grid on training tokens; relu² logits slightly underconfident
- **Sliding window (stride=16):** full context per scored token, ~0.025 bpb over chunked eval

### Compression
- **Base-3 + LZMA (preset=9):** 5 trits/byte packing, 39% reduction over int8+zlib; auto-compared against bitmask per run
- **FP8 QAT (e4m3):** halves fp_params (~5MB to ~2.5MB), only 0.002 bpb RT penalty
- **Shrinkage fix:** corrects ternary zero-fraction scale mismatch, eliminating all roundtrip gaps

## Setup and Run

```bash
# Environment setup (conda + Python 3.13 + PyTorch + FlashAttention-3 + Triton + dataset)
bash setup.sh

# Activate and run
conda activate golf
SEED=42 bash run_cuda_ternary.sh
```

<details>
<summary>Full run command</summary>

```bash
RUN_ID=ternary_run \
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
NUM_LAYERS=10 \
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
SMEAR=0 \
ITERATIONS=10000 \
WARMUP_STEPS=5 \
MAX_WALLCLOCK_SECONDS=599 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=1000 \
CHURN_LOG_EVERY=0 \
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
SEED=42 \
COMPILE_MODE=default \
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train_gpt_cuda_ternary.py
```

</details>

## Compliance

- [x] 3 seeds run on 8×H100 SXM
- [x] All 3 seeds train in <=600s (max: 599.7s)
- [x] All 3 seeds artifact <=16,000,000 bytes (max: 15,995,705)
- [x] Sliding window eval stride=16, consistent (std=0.0007)
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute
