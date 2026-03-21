# Parameter Golf — H100 Execution Plan

## Target: sub-1.14 BPB

Current SOTA: 1.1428 (thwu1, 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04)

## Top 5 Breakdown

| Rank | BPB | Layers | MLP | Seq | Quant | Key Differentiators |
|---|---|---|---|---|---|---|
| 1 (1.1428) | thwu1 | 10 | 3x | 2048 | int5 MLP / int6 attn | BigramHash(10240), SWA(0.4), WD=0.04 |
| 2 (1.1458) | Raahil | 9 | 3x | 2048 | int6 | SmearGate, BigramHash(4096), SWA(0.5), WD=0.04 |
| 3 (1.1502) | aruniyer | 11 | 3x | 2048 | int6 QAT | 11 layers, WD=0.04, zstd-22 |
| 4 (1.1556) | aquarious | 9 | 3x | 1024 | int6 QAT | SmearGate, BigramHash, WD=0.01 |
| 5 (1.1586) | yahya010 | 10 | 2.6x | 2048 | int6 QAT | MLP=1344, zero quant gap |

## Consensus Techniques (all top 5 use)

- Sliding window eval stride=64
- FP16 tied embedding passthrough
- Muon momentum=0.99, warmup from 0.92 over 1500 steps
- Lower LRs: matrix=0.02, scalar=0.02, embed=0.03-0.04
- Warmdown=3000 iters
- Grad clip=0.3
- zstd-22 compression
- Int6 (minimum) quantization

## What Separates #1 from the Pack

1. **Int5 on MLP weights** — saves ~1.86MB vs int6, funding the 10th layer
2. **BigramHash(10240)** — 2.5x larger hash table than #2's 4096 buckets (+0.001 BPB)
3. **SWA start_frac=0.4** — only averages the most converged 40% of warmdown checkpoints
4. **WD=0.04** — 4x higher than #4's 0.01, keeps weights small for better quantization

## Execution Phases

### Phase 1: Reproduce #2's recipe (~3 runs, baseline)

```bash
NUM_LAYERS=9 MLP_MULT=3 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432
MUON_MOMENTUM=0.99 MUON_WEIGHT_DECAY=0.04 WARMDOWN_ITERS=3000
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03
GRAD_CLIP_NORM=0.3 FP16_EMBED=1 EVAL_STRIDE=64
USE_SMEARGATE=1 BIGRAM_HASH_BUCKETS=4096 BIGRAM_HASH_DIM=128
# + int6 + zstd-22 + SWA + OrthoInit
```

Target: ~1.146. This is the proven stack.

### Phase 2: Push toward #1 (~5 runs)

- Add int5 on MLP weights → fund 10th layer
- BigramHash(10240) instead of 4096
- SWA start_frac=0.4 (tighter averaging window)
- Try 11L (like #3) if int5 saves enough space

### Phase 3: Novel improvements (~10 runs)

- Try 11L + int5 MLP (not yet on leaderboard)
- BigramHash(16384) — even larger table
- QAT (zero quant gap like #5) + int5 (best compression)
- Muon-aware QAT (Gaussian noise mode from PR #130)
- Explore WD sweep (0.02-0.06)

### Phase 4: Submission (~5 runs)

- 3+ seeds on best config, p<0.01
- Target: sub-1.14 BPB

## Implementation Needed for CUDA Port

1. **SmearGate** — ~20 lines (gate + prev token blending)
2. **BigramHash** — ~30 lines (hash table + projection)
3. **SWA** — ~15 lines (checkpoint averaging in warmdown)
4. **OrthoInit** — ~10 lines (orthogonal_ on weight matrices)
5. **Int5 quantization** — ~10 lines (extend int6 to support step=8)
6. **zstd-22** — swap zlib for zstandard
7. **QAT with STE** — ~20 lines (fake quantize in forward pass)

## Key Architecture Details from Top Submissions

### SmearGate (~512 params)
```python
gate = sigmoid(self.gate)  # shape [dim], init via sigmoid(3.0) ≈ 0.95
output = gate * current_emb + (1 - gate) * prev_token_emb
```
Applied after embedding lookup + bigram hash, before RMS norm.

### BigramHash (4096-10240 buckets, dim=128)
```python
hash_idx = (prev_token * 31 + curr_token) % num_buckets  # or * 92821
bigram_emb = self.bigram_table[hash_idx]  # (B, T, 128)
bigram_proj = bigram_emb @ self.bigram_proj  # (B, T, model_dim)
x = tok_emb + bigram_proj  # added before SmearGate
```
~524K params at 4096 buckets, ~1.3M at 10240.

### SWA (Stochastic Weight Averaging)
```python
# During warmdown, every swa_every steps:
if step >= swa_start and step % swa_every == 0:
    swa_state = ema_update(swa_state, model.state, n_averaged)
# At end of training, load swa_state into model before export
```
start_frac=0.4-0.5, every=50 steps.

### OrthoInit
```python
for module in model.modules():
    if hasattr(module, 'weight') and module.weight.ndim == 2:
        nn.init.orthogonal_(module.weight)
# Output projections scaled by 1/sqrt(2*num_layers)
```

### Int5 Quantization (MLP weights only)
```python
# Same as int6 but clip to [-16, 15], scale = clip_abs / 16.0
# Stored in int8 container — top 3 bits are zero/sign, compresses extremely well
```

### QAT with STE
```python
def fake_quantize_int6(w):
    scale = w.abs().amax(dim=-1, keepdim=True) / 31.0
    w_q = (w / scale).round().clamp(-31, 31)
    return w + (w_q * scale - w).detach()  # STE: forward uses quantized, backward uses original
```

## Compute Budget

- Phase 1: 3 runs × 10 min = 30 min 8×H100
- Phase 2: 5 runs × 10 min = 50 min 8×H100
- Phase 3: 10 runs × 10 min = 100 min 8×H100
- Phase 4: 5 runs × 10 min = 50 min 8×H100
- Iteration on 1×H100: ~20 hours
- **Total: ~4 hours 8×H100 + ~20 hours 1×H100**

## MLX Validation (completed)

- 25+ experiments on Mac (Apple M2)
- Best Mac result: 1.9588 BPB (14L×416d, 750 steps, 10 shards, full val)
- Near-zero quant gap validated (0.001 BPB with FP16 embed + Muon WD)
- Dead ends eliminated: depth recurrence + int6, DWA, eval-time loops, NTK extrapolation
- PR #328 submitted as non-record
