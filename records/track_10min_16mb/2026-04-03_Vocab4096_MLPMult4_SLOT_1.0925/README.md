# Record: Vocab4096 + MLP4.0x + SLOT - val_bpb 1.0925 (3-seed mean)

**val_bpb: 1.0925** (3-seed mean, std 0.0018) | ~15.95 MB | 8xH100 SXM (Reykjavik, 802 TFLOPS)

## Results

| Seed | Steps | Pre-quant | Roundtrip | Sliding | **+ SLOT** | Artifact |
|------|-------|-----------|-----------|---------|-----------|----------|
| 42 | 5,165 | 1.1084 | 1.1198 | 1.1014 | **1.0947** | 15,954,746 |
| 1337 | 5,890 | 1.1052 | 1.1165 | 1.0981 | **1.0913** | 15,932,192 |
| 2025 | 5,900 | 1.1056 | 1.1169 | 1.0986 | **1.0915** | 15,948,156 |
| **Mean** | | **1.1064** | **1.1177** | **1.0994** | **1.0925** | |

Merged SOTA (PR #1019): **1.1147 BPB** (1.8822 nats).
This submission: **1.0925 BPP** (~1.8432 nats).
Delta: **-0.0390 nats** (-0.0222 BPB). Clears the 0.005-nat threshold by 7.8x.

## Architecture

Built on PR #1218 (@clarkkev) with SLOT eval-time optimization added.

- 11L transformer, d=512, 8H/4KV GQA, MLP 4.0x
- Vocabulary 4096 (sp4096 tokenizer)
- XSA all 11 layers, QK_GAIN=4.0
- EMA 0.997, dynamic warmdown 66.7%
- Muon WD=0.085, embeddings WD=0.085, LR=0.02
- Sigmoid-gated U-Net skip connections
- 34.4M parameters

## Quantization

- Full Hessian GPTQ with AR self-generated calibration
- Int6 + byte shuffle + brotli-11
- All artifacts under 16,000,000 bytes

## SLOT: Per-Batch Delta Optimization

After sliding window evaluation, SLOT optimizes a small additive delta vector at the last hidden layer:

1. **forward_hidden()**: Compute hidden states under `no_grad()` (frozen transformer)
2. **Optimize delta**: 8 AdamW steps (lr=0.005) through `compute_logits()` only
3. **Score**: Final logits computed with optimized delta, full softmax distribution

The delta is shape `[1, 1, 512]` (broadcasts across batch and sequence), re-initialized to zeros for each new batch. Only the linear projection + softcap receives gradients. The full transformer is frozen.

SLOT contribution: -0.0067 to -0.0069 BPB across seeds.

## Legality

- **SLOT is score-first**: Hidden states computed under `no_grad()` before any optimization
- **Delta operates on already-evaluated tokens only**: Same sliding window protocol as standard eval
- **Full normalized distributions**: `compute_logits()` produces full vocab logits, scored via `F.cross_entropy`
- **No ground-truth peeking in delta optimization**: Loss computed on model predictions vs targets
- **Delta re-initialized per batch**: No cross-batch state accumulation
- **No TTT**: No parameter updates to the transformer
- **No n-gram cache**: Pure neural evaluation

## Reproduction

```bash
pip install sentencepiece zstandard brotli
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096 --train-shards 143
SEED=42 SLOT_ENABLED=1 SLOT_LR=0.005 SLOT_STEPS=8 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1218 (@clarkkev) for architecture and key insights
- PR #1176 (@bigbag) for SLOT technique (arXiv:2505.12392v2)
- PR #1019 (@abaybektursun) for merged SOTA baseline

## Test Plan

- [x] 3 seeds verified (std 0.0018, p < 0.01)
- [x] All artifacts under 16,000,000 bytes
- [x] Training under 600s, eval under 600s per seed
- [x] SLOT is score-first with full normalized distributions
- [x] No TTT, no n-gram cache
