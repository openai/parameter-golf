# Non-record: Blueprint Stack — Progressive Seq-Len + Multi-scale RoPE + Byte Embeddings (1xRTX 3080)

**val_bpb: 1.5568** (quantized int8+zstd roundtrip, single seed)

Non-record submission trained on a single NVIDIA RTX 3080 (12 GB). This serves as a single-GPU exploration of a combined technique stack, with systematic ablation results showing which leaderboard-proven techniques transfer to resource-constrained hardware.

## Architecture

Base: 10L, 512d, 8H/4KV GQA, 3x MLP (relu²), tied embeddings, U-Net skip connections, Muon optimizer (matrix params), Adam (embeddings/scalars), SWA weight averaging.

Additions over baseline `train_gpt.py`:

| Technique | Details |
|-----------|---------|
| Progressive sequence length | Schedule `0:512, 0.35:1024, 0.7:2048` — cheaper early steps on local statistics |
| Multi-scale RoPE by KV group | Bases `[1000, 10000, 100000, 1000000]` — each KV group sees a different context scale |
| Byte-level token embeddings | `dim=64` side channel from UTF-8 bytes of each SentencePiece token |
| Mixed-bit quantization export | int5 MLP, int6 attention/bigram/byte weights + zstd compression |

## Results

| Metric | Value |
|--------|-------|
| val_bpb (pre-quantization) | 1.5113 |
| val_bpb (int8+zstd roundtrip) | **1.5568** |
| Quantization degradation | +0.0455 |
| Training steps | 3,647 |
| ms/step | 164.5 |
| Peak memory | 1,150 MiB allocated / 1,814 MiB reserved |
| Artifact size | 15,900,102 bytes (under 16 MB cap) |
| Hardware | 1x NVIDIA RTX 3080 (12 GB) |
| Training time | 600 seconds (10 minutes) |

Loss was still dropping when the wallclock cap hit. On 8xH100 (challenge hardware), the same architecture would achieve approximately double the training steps.

## Ablation Summary

Surveyed 801 leaderboard submissions and deep-dived the top 5 merged entries. Tested 12 techniques across 3 phases on single RTX 3080:

**Phase 1** — 6 leaderboard techniques on simple baseline (131k tokens):
- XSA (last 4 layers): +0.020 bpb (FAIL — step overhead)
- Partial RoPE 16/64: +0.028 bpb (FAIL)
- LN Scale: +0.109 bpb (FAIL)
- EMA(0.997): +0.025 bpb (FAIL)
- LeakyReLU(0.5)²: **-0.003 bpb (PASS)**

**Phase 2** — 5 additive techniques on LeakyReLU baseline: all failed (progressive seq-len, byte embeddings, multi-scale RoPE, mixed-bit export, QAT).

**Phase 3** — LeakyReLU on blueprint stack: no benefit (-0.0095 worse). Additionally, `torch.compile(fullgraph=True)` generates 6x more memory and 26x slower kernels for `F.leaky_relu` vs `torch.relu`, making 786k batch tokens infeasible with leaky_relu.

**Key finding**: Most 8xH100-proven techniques hurt on single GPU because per-step overhead reduces total training steps at the 10-minute wallclock cap. On challenge hardware (~7,100 steps), techniques like XSA, EMA, and Partial RoPE should provide meaningful gains.

## Run

```bash
# 10-minute run on single GPU
RUN_ID=full_blueprint_seed42 \
TRAIN_LOG_NAME=full_blueprint_seed42.log \
NPROC_PER_NODE=1 \
MAX_WALLCLOCK_SECONDS=600 \
bash eval/eval.sh
```

## Hardware Note

This is a **non-record** submission — trained on 1xRTX 3080, not 8xH100. The score is not competitive with the leaderboard but demonstrates a viable technique stack and systematic ablation methodology. The architecture is designed to scale to challenge hardware where step-count-dependent techniques (XSA, EMA, Partial RoPE) become viable.
