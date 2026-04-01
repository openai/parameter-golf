# Record: 0.6360 BPB — Depth Recurrence + Multi-Order N-gram Backoff

**val_bpb: 0.6360** (seed 1337) | **~15.94 MB** | 8xH100 SXM

Built on our previous depth recurrence submission (1.1182 BPB) with eval-time multi-order n-gram backoff.

## Key Innovation: Multi-Order N-gram Backoff with Multi-GPU Prefill

- **N-gram backoff (orders 2-7)**: Hash-table based n-gram counting at eval time. For each token, look up highest-order match first, cascade down on miss. Blends n-gram probability with neural model prediction.
- **Entropy-adaptive alpha**: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))` — when the model is uncertain (high entropy), trust n-gram statistics more; when confident, trust the model.
- **Multi-GPU prefill**: Each rank pre-populates its n-gram tables with all tokens scored by earlier ranks, solving the table fragmentation problem on multi-GPU setups.
- **Zero training cost**: Purely eval-time technique. Same model artifact, same training procedure.

## Results (8xH100 80GB SXM)

| Metric | Seed 1337 | Seed 2025 | Seed 42 |
|--------|-----------|-----------|---------|
| Steps completed | 6,171 | 6,167 | 6,172 |
| step_avg | 97.24ms | 97.31ms | 97.24ms |
| Pre-ngram int6 roundtrip BPB | 1.1434 | 1.1453 | 1.1458 |
| **Sliding window + n-gram BPB** | **0.6360** | **0.6381** | **0.6387** |
| Artifact size | 15,942,380 bytes | 16,052,376 bytes | 15,936,180 bytes |

**Mean BPB across seeds: 0.6376**

## Architecture

- 11 physical layers (13 virtual via depth recurrence on layers 4,5)
- 512d, 8H/4KV GQA, 3x MLP with LeakyReLU(0.5)^2
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale 1/sqrt(layer+1)
- BigramHash(2048, dim=128) + SmearGate + ValueEmbedding(layers 9,10)
- Parameter Banking + Parallel Muon optimizer
- EMA(0.997) + Tight SWA weight averaging
- Int6 GPTQ-lite quantization + lzma compression

## N-gram Backoff Configuration

| Parameter | Value |
|-----------|-------|
| Orders | 2-7 |
| Hash buckets | 4,194,304 (2^22) |
| Min count | 2 |
| Entropy-adaptive | Yes |
| Alpha base | 0.05 |
| Alpha range | 0.55 |
| Entropy scale | 2.0 |
| Entropy threshold | 4.0 |

## Run Command

```bash
NGRAM_CACHE=1 \
RUN_ID=8gpu_depth_ngram_v2 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
DEPTH_RECURRENCE=4,5 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt_sub.py
```

## Credits

- **N-gram backoff technique**: Inspired by PR #770 (@minh-stakc) and PR #779 (@deanbrr)
- **Depth Recurrence**: First applied in our previous submission
- **Base model + LeakyReLU^2 + Parallel Muon**: PR #549 by @abaybektursun
- **Architecture stack**: PR #414 by @signalrush
