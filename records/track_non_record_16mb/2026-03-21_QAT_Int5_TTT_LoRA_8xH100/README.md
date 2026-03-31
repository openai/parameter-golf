# QAT Int5/Int6 + TTT LoRA (Non-Record)

**val_bpb: 1.14476** (seed 1337, post int5/int6+zstd quantization roundtrip, sliding window stride=64)

Non-record submission exploring two techniques on top of the #1 entry (thwu1, 1.14276 BPB).

## Run Command

```bash
# QAT run (tested, results below):
SEED=1337 TRIGRAM_VOCAB_SIZE=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

| Seed | val_bpb | Artifact Size | Valid |
|------|---------|--------------|-------|
| 42 | 1.14423 | 16,215,286 | no (over 16MB, had TrigramHash enabled) |
| 1337 | 1.14476 | 15,793,963 | yes |
| 2024 | — | — | pod terminated at step 4500 |

Does not beat #1 (1.14276).

## Approach 1: QAT with STE Fake-Quantization

Added Straight-Through Estimator fake-quantization to `CastedLinear.forward()`, matching the post-training quantization levels:

- **MLP layers**: int5 (clip_range=15) — same as #1's post-training int5
- **Attention layers**: int6 (clip_range=31) — same as #1's post-training int6
- **STE**: `w + (w_quantized - w).detach()` — forward uses quantized weights, backward uses original

Each `CastedLinear` module is tagged with its quantization category at model construction time using `_classify_param()`.

### Why QAT Failed

**Post-training quantization + SWA acts as beneficial regularization.** #1's pipeline applies SWA (averaging 24 checkpoints), 3% magnitude pruning, then int5/int6 quantization. This quantization noise acts as implicit regularization, actually *improving* BPB:

| Metric | #1 (no QAT) | Ours (QAT) |
|--------|-------------|------------|
| Pre-quant val_bpb | ~1.162 | 1.1628 |
| Post-quant val_bpb | 1.14276 | 1.14476 |
| Quantization "bonus" | -0.019 | -0.018 |

QAT makes weights pre-adapted to quantization levels, which removes the beneficial regularization effect of post-training quantization.

## Approach 2: TrigramHash (Tested in Seed 42)

Added a trigram hash embedding table (4096 buckets, dim=32) alongside the existing BigramHash(10240):

```python
hash = xor(48271 * t[i], 36313 * t[i-1], 27191 * t[i-2]) % 4095
```

**Result**: ~0.0005 BPB improvement (within noise). Pushed artifact over 16MB limit. Not included in the submitted `train_gpt.py` but the code is there (disabled via `TRIGRAM_VOCAB_SIZE=0`).

## Approach 3: TTT LoRA (Implemented, Not Tested)

Also implemented but could not test due to RunPod budget exhaustion. The idea: #1 uses only 170s of the 600s eval budget. TTT LoRA adapters (rank-8 on Q/V + LM head) trained per-document during eval could exploit the remaining 430s.

## Architecture

Identical to #1 (thwu1) with QAT added:
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init, U-Net skip connections, tied embeddings
- Muon WD=0.04, matrix_lr=0.02, momentum=0.99
- SWA start_frac=0.4, every=50
- Sliding window eval stride=64
- Mixed int5 MLP / int6 attn + zstd-22

## Hyperparameters

All defaults from #1, plus:

| Parameter | Value |
|-----------|-------|
| QAT_ENABLED | 1 |
| TRIGRAM_VOCAB_SIZE | 0 (disabled) |
