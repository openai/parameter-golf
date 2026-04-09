# Non-record Submission: 10L Int6 QAT + SmearGate + SWA (val_bpb=1.1575)

## Summary

**val_bpb = 1.1575** (single seed, self-verified)

This submission builds on the technique stack developed by @baudrillardsgh0st (PR #194). Our contribution is the 10-layer configuration that trades one layer for improved step throughput (9,156 steps vs 7,472 at 11L), informed by systematic analysis of training efficiency vs model capacity tradeoffs across 17 experiments.

10 layers, 512 dim, 8 heads, 4 KV heads, 3x MLP. Int6 QAT with STE, per-dimension SmearGate, SWA averaging 27 checkpoints, sliding window eval stride=64, zstd-22 compression.

| Seed | val_loss | val_bpb | Steps | ms/step | Artifact |
|------|----------|---------|-------|---------|----------|
| 1337 | 1.9543 | 1.1575 | 9,156 | 65.49 | 14,725,978 |

## Approach

Developed through rapid systematic experimentation — 17 runs across 1×H100 and 8×H100 pods over 48 hours, testing architecture variants, LR schedules, quantization strategies, and data scaling. Total compute cost: ~$90 on RunPod.

### What we tried

1. **Baseline tuning (1×H100):** Warmdown schedule optimization (1200→300→200), data scaling (10→33 shards), auxiliary multi-token prediction head, deep-thin architectures (15L×384)
2. **8×H100 scaling:** Warmdown sweep (150-700), learning rate tuning (0.02-0.06), SOTA technique verification
3. **Architecture search:** 10L vs 11L, MLP 2x vs 3x, KV=4 vs KV=8, BigramHash on/off

### Key findings

- **Step speed dominates at 10-min wall clock.** 10L at 65ms/step gets 9,156 steps vs 11L at 80ms/step getting 7,472 steps. More training steps > more parameters at this compute budget.
- **Warmdown matters more than architecture.** Going from warmdown=1200 to warmdown=500 gave 0.008 BPB improvement on our baseline code — for free.
- **BigramHash hurts at this scale.** The embedding table costs more parameters than the BPB improvement justifies (tested empirically).
- **SWA is critical for int6 quantization.** 27 averaged checkpoints produce smoother weights that survive 6-bit quantization with minimal degradation.
- **Deep-thin loses to wide-shallow under wall-clock constraints.** 15L×384 was 0.03 BPB worse than 9L×512 due to 27% slower step time.

### Technique stack

1. **Int6 QAT with STE** — Fake int6 quantization every forward pass
2. **Per-Dim SmearGate** — Learned per-dimension gate blending token with predecessor (~512 params)
3. **SWA every 50 steps** — Checkpoint averaging over last 50% of training
4. **Muon + WD=0.038** — High weight decay for quantization-friendly weights
5. **3x MLP (1536 hidden)** — Wider FFN enabled by int6 compression
6. **Seq2048 + RoPE 50K** — 2x training context
7. **Sliding window eval (stride=64)** — Maximum context for every scored token
8. **Zstd-22** — Better compression than zlib
9. **FP16 tied embedding** — Embedding never quantized

## Architecture

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- Vocab 1024 (SentencePiece BPE), seq len 2048, tied embeddings
- relu² activation, RoPE, logit softcapping (30.0)
- 24.2M parameters, 14.73MB compressed

## Training Config

| Parameter | Value |
|-----------|-------|
| Layers | 10 |
| Model dim | 512 |
| MLP mult | 3 |
| Matrix LR | 0.02 |
| Scalar LR | 0.02 |
| Tied Embed LR | 0.03 |
| Muon Momentum | 0.99 (warmup 0.92→0.99 over 1500 steps) |
| Muon Weight Decay | 0.038 |
| Warmdown Steps | 3000 |
| QAT Bits | 6 |
| SWA Every | 50 steps |
| SWA Start | 50% of training |
| Batch tokens | 524,288 |
| Seq len | 2048 |
| Seed | 1337 |

## Run command

```bash
NCCL_NVLS_ENABLE=0 SEED=1337 \
VOCAB_SIZE=1024 NUM_HEADS=8 NUM_KV_HEADS=4 MODEL_DIM=512 \
NUM_LAYERS=10 MLP_MULT=3 \
QAT=1 QUANT_BITS=6 FP16_EMBED=1 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.038 \
SMEAR_GATE=1 BIGRAM_HASH=0 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
SWA_ENABLED=1 SWA_START_FRAC=0.5 SWA_EVERY=50 \
RUN_ID=submission_10L_8gpu \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgments

Core technique stack (Int6 QAT, SmearGate, SWA, sliding eval) developed by @baudrillardsgh0st (PR #194). Our contribution is the depth-throughput tradeoff analysis demonstrating that 10L outperforms 11L under the 10-minute wall-clock constraint due to increased step throughput.

## About

Built by Nathan Maine ([Memoriant, Inc.](https://memoriant.ai)) — NVIDIA Inception member. First ML competition entry. Entered on launch day (March 18), submitted within 48 hours through systematic rapid experimentation.

GitHub: [NathanMaine](https://github.com/NathanMaine)
