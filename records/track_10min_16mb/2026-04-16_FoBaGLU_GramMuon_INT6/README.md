# FoBa-GLU + GramMuon + INT6 QAT (Non-Record, Unlimited Compute)

## Results

| Metric | Value |
|--------|-------|
| `final_int8_zlib_roundtrip_exact val_bpb` | **2.63600050** |
| Artifact total | 9,161,900 B (9.2 MB) |
| Code | 29,024 B |
| Model (INT6+zlib) | 9,132,876 B |
| Limit | 16,000,000 B ✅ PASS |

Hardware: 1× H100 (Colab), ~87 min wall time.

## Approach

Three novel techniques grounded in sparse recovery theory (OMP/FoBa from compressed sensing):

### 1. FoBa-GLU

Replaces ReLU² MLP with a pursuit-gated activation. Each forward pass selects k=50% of hidden neurons by gate magnitude — analogous to the atom-selection step in Orthogonal Matching Pursuit. The gate is a learned linear projection; the up projection is masked to only the selected neurons.

Motivation: sparse activations → structured weight sparsity → better INT6+zlib compression.
Block-diagonal sparse initialization ensures compressibility from step 0.

### 2. GramMuon

Extension of the Muon optimizer with Frobenius-norm Gram correction in the Newton-Schulz orthogonalization:

```
X = X / (||X||_F + ε)   # instead of X / ||X||_2
```

Prevents NaN divergence for small-magnitude weight matrices (e.g. tied embeddings at std=0.005). Stabilizes training without any LR reduction.

### 3. INT6 QAT with straight-through estimator

Quantization-aware training using 64 quantization levels (vs 256 for INT8). The reduced dynamic range produces weights that compress more efficiently under zlib. Straight-through estimator passes gradients through the rounding operation.

## Architecture

- 11 transformer layers, d_model=512
- GQA: 8 query heads, 4 KV heads
- MLP multiplier: 3.0 (FoBa-GLU, k=0.5)
- Sequence length: 4096
- Tied embeddings, RMSNorm, RoPE
- 35,149,912 total parameters

## Training

```bash
RUN_ID=foba_glu_h100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=10000 \
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=524288 \
VAL_LOSS_EVERY=2000 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 \
  records/track_10min_16mb/2026-04-16_FoBaGLU_GramMuon_INT6/train_gpt.py
```

## Theoretical basis

FoBa-GLU is motivated by the Forward-Backward Pursuit algorithm (Zhang & Huang 2011) applied to neural activation selection. The top-k gate selection is equivalent to the OMP forward step: at each layer, select the k atoms (neurons) most correlated with the input signal. This connection to compressed sensing provides a theoretical framework for why sparse activations improve compression — the RIP condition guarantees stable recovery of sparse signals in random dictionaries.

GramMuon stabilizes Newton-Schulz iteration by replacing the spectral norm with the Frobenius norm, which is cheaper to compute and more numerically stable for nearly-zero matrices.
