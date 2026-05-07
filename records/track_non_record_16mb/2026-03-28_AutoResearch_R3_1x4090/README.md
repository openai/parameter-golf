# Non-Record Submission: 1.1974 BPB — AutoResearch R3 (1×RTX 4090)

**val_bpb: 1.1974** | **100.9M params** | 1×RTX 4090 (24GB)

> **This is a non-record submission** — trained on a single consumer GPU (RTX 4090) rather than 8×H100. Submitted to demonstrate that automated hyperparameter search can achieve competitive scores on commodity hardware. The key insight — aggressive batch size reduction — is hardware-agnostic and may benefit H100 submissions as well.

## Results (1×RTX 4090, PyTorch 2.6.0+cu124)

| Metric | Value |
|--------|-------|
| val_bpb | **1.1974** |
| Steps | 404 |
| Training time | 300.2s (5 min budget) |
| Total time (incl. eval) | 406.2s |
| Parameters | 100.9M |
| Peak VRAM | 18.6 GB |
| MFU | 4.38% |
| tok/sec | ~87,000 |
| Total tokens | 26.5M |

## Key Insight: Batch Size Reduction

Reducing `TOTAL_BATCH_SIZE` from 2^19 (524K tokens, default) to 2^16 (64K tokens) increased training steps from ~63 to ~404 within the same 5-minute wallclock budget. The 6.4× increase in gradient updates dramatically improved convergence:

| Batch Size | Steps in 5 min | val_bpb |
|------------|----------------|---------|
| 2^19 (default) | ~63 | 1.476 |
| **2^16 (ours)** | **~404** | **1.197** |

This single change accounted for the majority of improvement. It suggests the default batch size over-batches for short training runs where step count matters more than per-step efficiency.

## Architecture

- 12 transformer layers, 1024 embedding dim, 8 heads, 8 KV heads (full attention, no GQA reduction)
- **Value Embeddings** (31.4M params): gated fusion on alternating layers, learned value representations per token
- Window pattern: `L` (all long-range attention — no sliding window)
- Vocab: 32768 BPE (SentencePiece)
- Sequence length: 2048
- MLP ratio: 4× (standard)
- Tied input/output embeddings

## Training

- **Optimizer:** AdamW (embeddings/scalars) + Muon (matrices), cosine warmdown schedule
- `MATRIX_LR=0.10`, `WEIGHT_DECAY=0.2`, `WARMUP_RATIO=0.0`
- `TOTAL_BATCH_SIZE=2^16` (65,536 tokens), gradient accumulation: 2 microbatches
- `MAX_WALLCLOCK_SECONDS=300` (5 min budget)
- No `torch.compile` (Triton shared memory OOM on 4090)

## Method

Automated hyperparameter search ([karpathy/autoresearch](https://github.com/karpathy/autoresearch)) with systematic sweeps:
- **R1:** Device batch size, model width/depth, architecture variants (crystal vs standard)
- **R2:** Learning rate sweep, warmup ratio, weight decay
- **R3:** Batch size reduction (breakthrough), window patterns, vocab size, LR re-sweep

31 total experiments, 11 in R3. Each run wallclock-capped at 300s on a single 4090.

## Run Command

```bash
cd ~/autoresearch && uv run train.py
```

Key env/config overrides baked into `train.py`:
```python
TOTAL_BATCH_SIZE = 2**16
MAX_WALLCLOCK_SECONDS = 300
WARMUP_RATIO = 0.0
MATRIX_LR = 0.10
WEIGHT_DECAY = 0.2
```

## Compliance

- [x] val_bpb computed on standard fineweb validation set
- [x] No test-time training on validation data
- [x] No network calls during evaluation
- [x] No external compute
- [x] Train time: **non-record submission** (300s on 1× RTX 4090)

## Included Files

- `train_gpt.py` — Complete training script (code snapshot from autoresearch commit `88ada9a`)
- `train.log` — Full training log from verified reproducible run
- `submission.json` — Leaderboard metadata
- `README.md` — This file
