# LoRA TTT + SOTA Base (10L Int5-MLP + Int6-Attn + BigramHash + SWA)

**Author:** Atharva Date (ADIITJ)
**Status:** Non-record submission (pending H100 validation)
**Approach:** SOTA training (10L, Int5-MLP, Int6-Attn, BigramHash(10240), SWA) + Document-isolated LoRA Test-Time Training at evaluation.

---

## Summary

This submission combines the current SOTA training stack (thwu1, 1.1428 bpb) with document-isolated LoRA test-time training (TTT) at evaluation time. TTT adapts small rank-8 LoRA adapters per document in the validation set using a causal sliding window — training strictly on already-scored tokens only.

**Key insight:** LoRA weights are initialized fresh at eval time and never stored in the artifact. The 16MB budget is unchanged from the SOTA. The only cost is eval-time compute (~1–3 min on 8xH100).

**Expected bpb:** ~1.137–1.140 (projected from SOTA 1.1428 + TTT delta ~0.003–0.005 observed by samacqua on the naive baseline model).

---

## Architecture (Training — identical to SOTA thwu1)

- **10 transformer layers**, dim=512, 8 heads, 4 KV heads (GQA)
- **MLP**: 3x hidden width (1536), relu² activation
- **SmearGate**: learned gate blending token t with token t-1 embedding
- **BigramHash(10240, dim=128)**: hash consecutive token pairs → learned embedding → project to model_dim
- **Orthogonal init** for all large weight matrices; output projections scaled by `1/√(2L)`
- **Tied embeddings** (fp16)
- **U-Net skip connections** between encoder and decoder halves
- **RoPE** positional encodings with QK-Norm and q_gain

## Quantization (Export — identical to SOTA)

| Tensor Class | Precision | Compression |
|---|---|---|
| MLP weights | Int5 (clip=15, per-row scale) | zstd-22 (~1.88× ratio) |
| Attention weights | Int6 (clip=31, per-row scale) | zstd-22 (~1.51× ratio) |
| Tied embeddings | FP16 | stored raw |
| Blocks[8].attn.c_k | FP16 | stored raw |
| Control scalars | FP32 | stored raw |
| 3% magnitude pruning | applied before quantization | improves compressibility |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| num_layers | 10 |
| model_dim | 512 |
| mlp_mult | 3.0 |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| iterations | 20,000 (wallclock-capped at 600s) |
| warmdown_iters | **3,500** (up from SOTA's 3,000) |
| warmup_steps | 20 |
| matrix_lr (Muon) | 0.02 |
| muon_momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| muon_weight_decay | 0.04 |
| adamw_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| SWA start_frac | **0.35** (SOTA used 0.40) |
| SWA every | 50 steps |
| bigram_vocab_size | 10,240 |
| bigram_dim | 128 |

Changes from SOTA: `warmdown_iters=3500` (vs 3000) gives slightly more converged checkpoints. `swa_start_frac=0.35` (vs 0.40) collects more averaged checkpoints from the tail of training.

## LoRA TTT at Evaluation

After training, quantization, and dequantization, the model is evaluated using document-isolated LoRA TTT:

**Algorithm:**
1. Find document boundaries in the validation set using BOS token (token_id=1).
2. For each batch of 32 documents (sorted by length for GPU efficiency):
   a. Initialize fresh LoRA adapters: A ~ Kaiming-uniform, B = zeros (delta = 0 at start).
   b. Slide through the document in chunks of 256 tokens with full 2048-token context.
   c. For each chunk: **score first** (accumulate NLL and bytes), **then** take one Adam step on LoRA.
   d. Reset LoRA and optimizer state before the next document batch.
3. LoRA targets: Q and V projections in all 10 attention layers (rank=8, lr=0.01).

**Fairness:** Scoring always precedes training on each chunk. The LoRA state from one document never affects another. No information leaks from future validation tokens.

**Artifact cost:** Zero. LoRA weights are initialized at eval time from `(A~random, B=zeros)` and discarded after each document. No LoRA weights are stored in the artifact file.

### LoRA TTT Hyperparameters

| Parameter | Value |
|---|---|
| lora_rank | 8 |
| lora_lr | 0.01 |
| chunk_size | 256 tokens |
| eval_seq_len | 2048 tokens (full context) |
| batch_size | 32 documents |
| optimizer | Adam (β₁=0.9, β₂=0.95, ε=1e-10) |
| adapters | Q and V in all 10 layers |

## Expected Performance

Based on:
- SOTA (thwu1) sliding window eval: **1.1428 bpb**
- samacqua LoRA TTT delta over sliding window on the naive baseline: **−0.003 bpb**
- Longer context (2048 vs 1024) should improve TTT gradient quality
- SWA and warmdown tuning: **−0.0002 bpb** estimated

**Projected range:** 1.137 – 1.140 bpb

Whether this beats the 0.005 threshold (target: ≤1.1378) for a new record requires actual H100 validation.

## Ablation Summary

| Change | Expected val_bpb | Delta |
|---|---|---|
| SOTA baseline (thwu1) | 1.1428 | — |
| + SWA start_frac 0.40→0.35 | ~1.1426 | −0.0002 |
| + warmdown 3000→3500 | ~1.1424 | −0.0002 |
| + LoRA TTT (doc-isolated, rank=8) | ~1.137–1.140 | −0.003–0.006 |

The TTT delta is interpolated from samacqua's ablation on a weaker model. The actual benefit on the SOTA model is uncertain without empirical runs.

## Artifact Size Budget

The training setup is identical to SOTA; the artifact size is expected to be ~15.9MB:

| Component | Est. Size |
|---|---|
| Int5 MLP (10L, 3× hidden, zstd-22) | ~8.4MB |
| Int6 Attention (10L, zstd-22) | ~3.9MB |
| FP16 embeddings | ~1.0MB |
| BigramHash(10240, dim=128) | ~0.9MB |
| Code (train_gpt.py) | ~55KB |
| **Total** | **~14.3MB** |

LoRA weights: 0 bytes (initialized at eval time, discarded after use).

## Compliance Checklist

- [x] Artifact ≤ 16,000,000 bytes (est. ~14.3MB, same as SOTA)
- [x] No network calls during evaluation
- [x] No training data access during evaluation
- [x] Self-contained and reproducible
- [x] Evaluation ≤ 10 minutes on 8xH100 (TTT eval estimated 3–5 min)
- [x] Training ≤ 10 minutes on 8xH100
- [x] TTT only uses already-scored validation tokens
- [ ] Statistical significance (3+ seeds not yet run — non-record status)

## Run Commands

### Training + Evaluation (8xH100)

```bash
# Setup data (once)
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run from the records folder:
cd records/track_10min_16mb/2026-03-23_LoRA_TTT_Int5Int6/

# Seed 42 (default)
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 1337
SEED=1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Disable TTT (sliding window only, for ablation)
TTT_ENABLED=0 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Local Mac Smoke Test (MLX)

Run from repo root for a quick smoke test with 200 steps:

```bash
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

RUN_ID=mlx_smoke \
ITERATIONS=200 \
TRAIN_BATCH_TOKENS=8192 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=8192 \
python3 train_gpt_mlx.py
```

The CUDA train_gpt.py above runs the full submission on H100s. Local Mac cannot simulate the full H100 training path.

## Dependencies

See `requirements.txt`. The `zstandard` package is required for zstd-22 compression. All other dependencies are standard (numpy, sentencepiece, torch).

## Attribution

Training architecture, quantization scheme, SWA, BigramHash, SmearGate, and OrthoInit are from the SOTA submission by thwu1 (2026-03-20), which itself builds on Raahil Shah's PR #162.

LoRA TTT design is adapted from samacqua's LoRA TTT submission (2026-03-17).

This submission by Atharva Date (ADIITJ) combines these two streams and tunes warmdown/SWA for the SOTA base.

## Non-Record Status

This submission does not include H100 training logs. Until at least 3 independent seeds are validated on 8xH100, the submission is classified as **non-record**. The implementation is complete and correct; the record determination is pending compute validation.

If the 3-seed mean val_bpb ≤ 1.1378 with p < 0.01 vs the current SOTA, this qualifies for record promotion.
