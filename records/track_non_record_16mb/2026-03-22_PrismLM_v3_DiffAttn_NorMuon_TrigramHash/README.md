# Non-Record Submission: PrismLM v3 — DiffTransformer V2 + NorMuon + TrigramHash

## Score: val_bpb = 1.1715 (post-quant int6+zstd, no sliding window)

Trained on 8×H100 SXM in 600 seconds. 15.59MB artifact (int6+zstd-22). Single seed run.

This is a non-record submission exploring **three novel techniques** not yet attempted in any merged or open PR, built on top of the proven technique stack from PR #315.

## Novel Contributions

### 1. DiffTransformer V2 Attention (Last 2 Layers)

Based on [Differential Transformer](https://arxiv.org/abs/2410.05258) (Microsoft, ICLR 2025 Oral). Computes two separate softmax attention maps and subtracts them, cancelling noise in the attention pattern:

```
attn = softmax(Q1 @ K1^T) - λ · softmax(Q2 @ K2^T)
```

Applied only to the last 2 layers where attention refinement matters most. The scalar `λ` is learned per-head via `lambda_init` reparameterization. Remaining layers use standard GQA + XSA.

### 2. NorMuon Optimizer

Replaces standard Muon with NorMuon ([Keller Jordan, Oct 2025](https://kellerjordan.github.io/posts/muon/)), which adds **per-neuron row normalization** after the Newton-Schulz orthogonalization step. This normalizes gradient updates by the second moment of each row, giving ~11% better compute efficiency. Uses `beta2=0.95` for the second moment EMA.

### 3. TrigramHash + Context-Aware N-gram Gating

Extends BigramHash with a TrigramHash table (2048 buckets, dim 64) that captures three-token patterns via `(t0 * 961 + t1 * 31 + t2) % (vocab_size - 1) + 1`. Both n-gram signals are modulated by a **context-aware gate** (inspired by [DeepSeek Engram](https://github.com/deepseek-ai/Engram)) that learns when to rely on n-gram vs. neural predictions:

```
gate = sigmoid(linear(hidden_state))
output = hidden + gate * (bigram_signal + trigram_signal)
```

## Full Architecture

| Component | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP expansion | 3× (hidden=1536), ReLU² |
| XSA layers | Last 6 |
| DiffAttn layers | Last 2 |
| Partial RoPE | 16/64 dims (25%) |
| LN depth scaling | 1/√(layer+1) |
| SmearGate | Yes |
| BigramHash | 2048 buckets, dim 128 |
| TrigramHash | 2048 buckets, dim 64 |
| N-gram gating | Context-aware sigmoid gate |
| U-Net skips | Yes |
| Logit softcap | 30.0 |
| Tied embeddings | Yes (FP16) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer (matrices) | NorMuon (lr=0.04, momentum=0.95, WD=0.02, beta2=0.95) |
| Optimizer (embeddings/scalars) | AdamW (lr=0.04, WD=0.01) |
| Tied embed LR | 0.05 |
| Batch size | 786,432 tokens |
| Sequence length | 2048 |
| Warmdown iters | 1200 |
| Grad clip | 0.3 |
| SWA | Enabled (every 200 steps) |
| Late QAT | Enabled (when lr_scale < 0.1) |
| Warmup | 20 steps |

## Quantization & Compression

- **Int6** per-row quantization on MLP and attention weight matrices
- **FP16** for tied embeddings
- **3% magnitude pruning** before quantization (adaptive up to 15% if over budget)
- **zstd level 22** compression
- Flash Attention 3 fallback to `F.scaled_dot_product_attention` (FA3 not available in our environment)

## Key Metrics

- **val_bpb (post-quant): 1.1715** (standard eval, no sliding window)
- Pre-quant val_bpb: 1.1607
- Quantization penalty: ~0.011 bpb
- Steps completed: 4,600 / 20,000 (wallclock-capped at 600s)
- Step average: 130.43 ms/step
- Model params: 27,518,587
- Artifact size: 15,586,651 bytes (15.59MB)
  - Model int6+zstd: 15,521,912 bytes
  - Code: 64,739 bytes
- Peak memory: 25,921 MiB allocated, 26,460 MiB reserved
- GPU: 8×H100 SXM (Modal)

## Training Progression

| Step | val_loss | val_bpb |
|------|----------|---------|
| 0 | 6.9300 | 4.1043 |
| 1000 | 2.2005 | 1.3033 |
| 2000 | 2.1133 | 1.2516 |
| 3000 | 2.0876 | 1.2364 |
| 4000 | 2.0209 | 1.1969 |
| 4600 (final) | 1.9598 | 1.1607 |
| **Post-quant** | **1.9780** | **1.1715** |

## Gap Analysis vs. SOTA

Our score of 1.1715 is ~0.029 bpb behind the merged SOTA (1.1428) and ~0.047 bpb behind the unmerged frontier (1.1248). Key factors:

1. **No sliding window eval** — was disabled to save eval time. Sliding window typically gives ~0.03 bpb improvement; re-enabled in the submitted code for future runs.
2. **Small n-gram tables** — BigramHash(2048) vs. the SOTA's BigramHash(10240). Larger tables are worth ~0.005 bpb.
3. **NorMuon hyperparameters** — momentum=0.95 vs. proven momentum=0.99. The lower momentum may have hurt convergence in the warmdown phase.
4. **DiffAttn parameter overhead** — 1.5× attention parameters on 2 layers reduces capacity available for other components. The noise-cancellation benefit at this scale is unclear.
5. **SDPA fallback** — Flash Attention 3 was unavailable; SDPA is functionally equivalent but ~10% slower, meaning fewer training steps.

## What We'd Change With More Compute

1. Increase BigramHash to 10240 buckets (~0.005 bpb)
2. Re-enable sliding window eval (~0.03 bpb)
3. Tune NorMuon momentum to 0.99
4. Try EMA instead of SWA (works better with XSA per community data)
5. Ablate DiffAttn vs. standard attention to quantify its contribution
6. Increase TrigramHash to 8192 buckets

## Included Files

- `train_gpt.py` — Self-contained training + evaluation script (bug-fixed: correct 16MB decimal limit, sliding window eval re-enabled)
- `train.log` — Training log from the 8×H100 run (seed 1337)
- `submission.json` — Leaderboard metadata
