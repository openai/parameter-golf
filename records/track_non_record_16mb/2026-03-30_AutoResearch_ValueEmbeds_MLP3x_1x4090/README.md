# AutoResearch: Value Embeddings + MLP3x (1x RTX 4090)

**Non-record submission** — trained on a single RTX 4090 with a 5-minute wallclock budget. Submitted to the unlimited compute / non-record track.

## Score

| Metric | Value |
|--------|-------|
| val_bpb | **1.1801** |
| Training time | 300.6s |
| Total time (incl. eval) | 393.9s |
| Peak VRAM | 8,833 MB |
| MFU | 4.47% |
| Steps | 467 |
| Parameters | 91.1M |

## Architecture

- **12 layers**, 640d embedding, 5 attention heads, 5 KV heads (MHA)
- **MLP 3x** hidden dim (1920) — smaller expansion saves params, allowing more training steps within budget
- **Value Embeddings** (31.5M params): per-layer learned value projections with gating, alternating layers
- **SSSL sliding window pattern** (3 short + 1 long attention per 4 layers)
- **8192 BPE vocabulary** (SentencePiece)
- **RoPE** positional encoding
- **RMSNorm** pre-norm with residual lambdas and x0 skip connections
- **Sequence length**: 2048 (train and eval)

## Optimization

- **Muon optimizer** for matrix params (LR=0.10, momentum=0.95)
- **Adam** for embeddings (LR=0.6) and scalars (LR=0.5)
- **Batch size**: 65,536 tokens (device_batch=8, grad_accum=4)
- **No warmup**, 50% cosine warmdown
- **int8 quantization + zlib** compression for artifact

## Methodology

Built using an automated ablation framework ("autoresearch") that iteratively tests architecture and hyperparameter changes against a best-known baseline. Over 50 configurations were tested across 5 sweep rounds, each with a 300s training budget on a single RTX 4090.

### Key findings from ablation sweeps

1. **Value embeddings** provide the largest single improvement (~0.19 bpb over baseline)
2. **MLP 3x > 4x** at fixed param budget — fewer params per layer means more training steps
3. **Sequence length 4096** gives marginal improvement over 2048 but was used for best run
4. **device_batch=8** with more gradient accumulation beats device_batch=16
5. **EMA averaging** (0.995–0.998) produces negligible improvement at this step count
6. **Deeper networks** (10–11 layers) OOM or underperform 12L at 640d on 4090

## Reproduction

```bash
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp8192
NO_COMPILE=1 python3 records/track_non_record_16mb/2026-03-30_AutoResearch_ValueEmbeds_MLP3x_1x4090/train_gpt.py
```

Requires: PyTorch 2.x, CUDA, sentencepiece. Set `NO_COMPILE=1` to avoid Triton shared memory OOM on 24GB cards.

## Hardware

- 1x NVIDIA RTX 4090 (24GB GDDR6X)
- Ubuntu 22.04, PyTorch 2.5, CUDA 12.4
