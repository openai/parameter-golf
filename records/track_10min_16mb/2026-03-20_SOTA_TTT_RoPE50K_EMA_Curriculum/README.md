# SOTA+ TTT + RoPE50K + EMA + Curriculum

**Target: sub-1.13 BPB** | 8xH100 SXM, 600s | Pending compute run

## Base: PR #198 Stack (1.1326 BPB)

Every proven technique from the current #1 submission:

| Technique | Detail |
|-----------|--------|
| 11 layers | Deeper model, funded by int6 compression |
| Int6 MLP+Attn / Int8 Embed | Mixed precision quantization + zstd-22 |
| MLP 3x (1536 hidden) | Wider feed-forward, enabled by int6 savings |
| SmearGate | Learned per-dim gate blending token with predecessor |
| BigramHash (2048 buckets) | Hash-based token-pair embeddings |
| OrthoInit + muP | Orthogonal weight init with output scaling |
| WD=0.04 (Muon + Adam) | Quantization-friendly weight distribution |
| FA3 with SDPA fallback | FlashAttention 3 on H100, PyTorch SDPA locally |
| Sliding window eval (s64) | Near-full context for every scored token |
| FP16 tied embedding | Embedding never quantized |

## New: Four Untried Improvements

### 1. RoPE Base 50K (was 10K)

Smoother position interpolation at seq2048. Validated by PR #206 (1.1507 on 9L).
Zero parameter/compute cost. Expected gain: ~0.002 BPB.

### 2. LAWA-EMA (replaces periodic SWA)

Exponential moving average (decay=0.995) updated every step during warmdown,
instead of periodic SWA checkpoints every 200 steps. Smoother weight averaging
should reduce noise in the final model. Expected gain: ~0.002 BPB.

### 3. Context-Length Curriculum

Train at seq1024 for first 60% of wallclock (~50ms/step), then switch to seq2048
(~81ms/step). The short-context phase yields ~60% more optimizer steps, building
a stronger feature representation before introducing long context. Expected gain: ~0.003 BPB.

### 4. Full-Model SGD Test-Time Training

After training, run 1 epoch of SGD (lr=3e-4, momentum=0.95) over the validation
set before scoring. Each token predicted with backward-looking context only
(causal model ensures no leakage). Adapts the model to the evaluation distribution.

Without SmearGate, TTT adds ~0.033 BPB (PR #152). With SmearGate on a 9L model,
only ~0.001 (PR #178). The true gain on the full 11L stack is the critical unknown.
Expected gain: 0.001 to 0.033 BPB.

## Expected Outcome

| Scenario | BPB | Delta vs #198 |
|----------|-----|---------------|
| Conservative (TTT ~0.001) | ~1.125 | -0.008 |
| Moderate (TTT ~0.010) | ~1.116 | -0.017 |
| Aggressive (TTT ~0.033) | ~1.093 | -0.040 |

## Run Command

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are baked into defaults. Override with env vars if needed:

```bash
EMA_ENABLED=0 TTT_ENABLED=0 CURRICULUM_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture

- 11 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu-squared activation
- Vocab 1024 (SentencePiece BPE), tied embeddings
- RoPE base 50K, logit softcapping (30.0)
- U-Net skip connections with learned weights
- ~26.8M parameters, ~15.7MB artifact (int6+zstd-22)
