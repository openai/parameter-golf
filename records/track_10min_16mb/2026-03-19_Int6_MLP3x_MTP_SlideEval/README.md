# Int6 MLP3x + MTP + Sliding Window Eval

## Summary

This submission stacks seven orthogonal techniques to achieve **val_bpb 1.1605** (best seed) / **1.1625** (mean across 3 seeds) with a **15.28 MB** artifact (under 16 MB). The key innovation enabling the result is **int6 quantization with zstd-22 compression**, which saves ~25% artifact space versus the baseline's int8+zlib, allowing a **3× MLP expansion** (hidden=1536 vs baseline 1024) that provides significantly more model capacity.

## Techniques

### 1. Int6 Quantization + zstd-22 Compression

Large 2D weight matrices are quantized to 6-bit range [-31, 31] (stored in int8 bytes) with per-row fp16 scales. The 2 zero high bits in each byte compress extremely well under zstd level 22, reducing the compressed artifact by ~25% versus int8+zlib. This frees ~4MB of artifact space that is reinvested into model capacity.

Dequantization is scale-driven: `dequant = q * scale` where `scale = abs_max / 31` was precomputed during quantization. The formula is identical to int8 dequantization — only the scale denominator changes.

### 2. Wider MLP (MLP_HIDDEN=1536, 3× expansion)

With int6+zstd saving ~4MB, the MLP hidden dimension increases from the baseline's 1024 (2× expansion) to 1536 (3× expansion). This adds ~5M parameters (21.4M total vs 17M baseline), providing 50% more MLP capacity. The relu² activation is preserved.

### 3. Long-Context Training (TRAIN_SEQ_LEN=4096)

Training at 4096-token sequences (4× the baseline's 1024) gives each token significantly more context during training. This improves learned representations and naturally matches the sliding window evaluation length, avoiding RoPE position extrapolation artifacts.

### 4. Multi-Token Prediction (MTP) Auxiliary Head

One MTP head predicts token at position i+2 from hidden state at position i, with loss weight 0.01. The MTP head is a single CastedLinear(512, 1024, bias=False) — 524,288 parameters that are excluded from the exported artifact. The MTP head reuses hidden states already in GPU cache, adding <2% throughput overhead ("free FLOPs" on memory-bandwidth-bound H100s).

### 5. FP16 Tied Embedding Passthrough

The tied embedding matrix (tok_emb.weight) is kept in fp16 during export instead of being quantized to int8/int6. Since this matrix serves as both input lookup and output projection, quantization errors compound across both roles. FP16 passthrough eliminates this at a cost of ~523KB.

### 6. Sliding Window Evaluation (stride=512)

Each validation token is scored with at least 3584 tokens of preceding context (vs the baseline's average ~512). Windows of 4096 tokens advance by 512 tokens; only the last 512 positions per window are scored. Eval takes ~97 seconds on 8×H100.

### 7. Training Dynamics Co-optimization

| Parameter | Baseline | This Submission |
|-----------|----------|-----------------|
| MATRIX_LR | 0.04 | 0.02 |
| MUON_MOMENTUM | 0.95 | 0.99 |
| WARMDOWN_ITERS | 1200 | 3000 |
| TRAIN_BATCH_TOKENS | 524,288 | 393,216 |
| TRAIN_SEQ_LEN | 1024 | 4096 |

Lower learning rate and higher momentum produce smoother weight distributions that quantize better under int6. The smaller batch (393K) accommodates the 4× longer sequences while maintaining throughput.

## Results

### 3-Seed Validation

| Seed | Non-overlapping BPB | Sliding Window BPB | Artifact |
|------|--------------------|--------------------|----------|
| 1337 | 1.1725 | **1.1605** | 15.28 MB |
| 42 | 1.1765 | **1.1645** | 15.12 MB |
| 2024 | 1.1745 | **1.1625** | 15.10 MB |

- **Mean sliding window BPB: 1.1625** (baseline: 1.2244)
- **Mean improvement: 0.110 nats** (required: 0.005)
- **t-statistic: −56.84**, **p-value: 0.00015** (required: p < 0.01)

### Key Metrics (seed 1337)

```
model_params: 22,302,792 (21,778,504 base + 524,288 MTP, MTP excluded from artifact)
MLP_HIDDEN: 1536 (3× expansion)
train_batch_tokens: 393,216
train_seq_len: 4096
steps: 10,427 at 57.55ms/step
peak memory: 8,615 MiB

Pre-quant:  val_loss:1.9700 val_bpb:1.1667
Post-quant (non-overlapping): val_loss:1.9798 val_bpb:1.1725
Post-quant (sliding window stride=512): val_loss:1.9594 val_bpb:1.1605

Artifact: 15,281,626 bytes (15.28 MB)
```

## Reproduction

```bash
RUN_ID=int6_mlp3x_mtp \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=4096 \
TRAIN_BATCH_TOKENS=393216 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.01 \
QAT_FRACTION=0 \
EMA_ENABLED=0 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3000 \
FP16_EMBED_EXPORT=1 \
MLP_HIDDEN=1536 \
INT6_QUANT=1 \
PRUNE_FRACTION=0 \
GRAD_CLIP_NORM=1.0 \
EVAL_STRIDE=512 \
TRAIN_LOG_EVERY=200 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires `pip install zstandard` for zstd compression.

## Files

- `train_gpt.py` — self-contained training script with int6 quant, MTP, sliding window eval
- `train_seed1337.txt` — full training log from seed 1337 run
- `train_seed42.txt` — full training log from seed 42 run
- `train_seed2024.txt` — full training log from seed 2024 run
- `submission.json` — leaderboard metadata
- `README.md` — this file

## Methodology

This configuration emerged from an overnight research sprint of 40+ experiments on 8×H100 GPUs, systematically exploring the solution space:

1. **Architecture exploration** — tested SwiGLU MLP, looped/depth-recurrent transformers (3×3 at d=832, 4×2 at d=704), and width vs depth tradeoffs. SwiGLU reduced throughput 16%; looped transformers were 2× slower per step. Baseline relu² MLP retained for throughput.
2. **Quantization strategies** — tested per-group int8 (groups of 64, 128), fp16 embedding passthrough, magnitude pruning (5–50%), and int6+zstd. Per-group inflated artifacts past 16MB; 20% pruning compounded with int8 quant error; int6+zstd enabled wider MLP while fitting under 16MB.
3. **Training dynamics** — swept batch size (524K to 2M), learning rates, Muon momentum (0.95–0.99), warmdown schedules (1200–20000 iters), gradient clipping. Discovered 1.5× batch reduces quant gap; higher momentum + lower LR complement longer sequences.
4. **Auxiliary objectives** — tested MTP at weights 0.01–0.1 with 1–2 heads, external QAT, GPU-resident EMA. MTP at very low weight (0.01) provides free-FLOP regularization with <2% overhead.
5. **Evaluation methods** — tested eval@2048 with RoPE extrapolation (didn't help), sliding window at strides 64–512 (stride=512 is optimal cost-benefit), and NTK-aware scaling. Training at eval seq_len eliminates extrapolation artifacts.

Many ideas that seemed promising in theory were eliminated by empirical results: per-group quantization, looped transformers, aggressive pruning, higher learning rates with larger batches, and EMA all either hurt throughput, inflated artifacts, or compounded quantization errors.

## Attribution

This submission was developed by **Maestro** (an agentic AI system by [iGent AI](https://igent.ai)) working collaboratively with **Sean Ward**. Maestro autonomously designed experiments, implemented code changes (MTP heads, int6 quantization, sliding window evaluation, EMA, external QAT, magnitude pruning), ran training and evaluation on 8×H100 GPUs, analyzed results, and iterated toward the final configuration.