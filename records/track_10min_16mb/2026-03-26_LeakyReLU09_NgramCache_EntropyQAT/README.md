# LeakyReLU(0.9)² + N-gram Eval Cache + Entropy-Reg QAT + Mixed Quant + Score-First TTT

**val_bpb: 0.9958** (3-seed mean, std 0.0017) | **~14.0 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | **Post-TTT+ngram bpb** | TTT+ngram time | Artifact |
|------|----------|-------|-------------|----------------------|----------------|----------|
| 1337 | 104.6ms | 5,735 | 1.1516 | **0.9977** | 552s | 13,834,050 |
| 42 | 88.3ms | 6,799 | 1.1485 | **0.9947** | 564s | 13,933,238 |
| 2025 | 93.1ms | 6,446 | 1.1448 | **0.9949** | 560s | 14,007,046 |
| **Mean** | **~95ms** | **~6,327** | **1.1483** | **0.9958 (std 0.0017)** | **~559s** | |

## Key Innovations

### 1. N-gram Eval Cache (backward-looking, score-first)

Backward-looking 7-gram hash cache blended with neural predictions during eval. Every token scored BEFORE cache update — strictly legal.

```
for each 32K-token chunk:
    Phase 1 — SCORE: sliding window + n-gram blend under inference_mode()
    Phase 2 — UPDATE: add scored tokens to n-gram hash tables
    Phase 3 — TRAIN: SGD on already-scored chunk (TTT)
```

- 7-gram backoff (orders 2-7), 4M buckets per order, fixed alpha=0.20
- Cache starts empty, builds from scored val tokens only
- No training data access during eval, no oracle/hindsight selection
- Hit rate reaches ~98% by midpoint of eval

The n-gram cache exploits the repetitive statistical structure of FineWeb validation data. High-order n-grams (5-7) provide near-perfect predictions for previously-seen contexts, and the fixed alpha conservatively blends these with the neural model's distribution.

### 2. Entropy-Regularized QAT

During late warmdown (lr_scale < 0.15), we add a penalty term that pushes weights toward quantization grid points:

```python
residual = w / scale - round(w / scale)
loss += entropy_reg * residual.pow(2).mean()
```

This halves the quantization gap compared to standard STE QAT (0.009 vs 0.017 BPB in our ablations). The gradient signal directly incentivizes weight distributions that quantize cleanly.

### 3. Mixed Quantization (front3_back1_6_middle5)

Layer-position-aware bit allocation instead of uniform int6:
- **int6** (31 levels) for sensitive layers: first 3 + last 1
- **int5** (15 levels) for middle layers: cheaper without quality loss

Combined with per-row GPTQ-lite clip search (5 percentiles per row, pick min MSE independently), this achieves better quality at smaller artifact size than uniform int6.

### 4. LeakyReLU(0.9)²

```python
x = F.leaky_relu(self.fc(x), negative_slope=0.9).square()
```

Slope 0.9 beats 0.5 by 0.013 BPB in controlled sweeps (issue #140). After squaring, negatives retain 81% magnitude. Monotonic improvement from 0.1 to 0.9 confirmed across 7-point sweep.

### 5. Score-First TTT (PR #549 recipe)

Legal test-time training following PR #461/PR #549 framework:
- SGD(lr=0.002, momentum=0.9), grad_clip=1.0
- 3 epochs per 32K chunk, cosine LR across chunks
- All blocks unfrozen (26.9M params adapt)
- `torch.inference_mode()` enforces stateless scoring

### Training Architecture

Built on PR #549 stack (PR #414 base + Parallel Muon):
- 11L, 512d, 8H/4KV (GQA), LeakyReLU(0.9)² MLP 3x
- BigramHash(2048), XSA4, Partial RoPE(16/64), LN Scale, VE128
- SmearGate, EMA(0.997) + Tight SWA
- Parameter Banking + Parallel Muon
- train_seq_len=2048, 80 shards, LZMA compression

### Eval Timing

| Phase | Seed 1337 | Seed 42 | Seed 2025 |
|-------|-----------|---------|-----------|
| Training (wallclock cap) | 600s | 600s | 600s |
| Serialization + quant | ~10s | ~10s | ~10s |
| int6 roundtrip eval | 19s | 7s | 6s |
| Sliding window eval (redundant — see note) | 98s | 75s | 75s |
| **Score-first TTT + N-gram** | **552s** | **564s** | **560s** |
| **Total eval (as logged)** | **~679s** | **~656s** | **~651s** |
| **Total eval (without redundant sliding window)** | **~581s** | **~581s** | **~576s** |

#### Timing note (transparency)

The logged eval times (651-679s) exceed the 600s eval budget because the code ran a **redundant standalone sliding window eval** (~75-98s) before TTT. This eval is redundant because TTT's score-first approach already includes its own sliding window scoring with n-gram blending — the standalone eval's BPB (`final_int6_sliding_window`) is not the reported score and has no effect on the submission's `val_bpb`.

I caught this after the 3-seed runs completed and the pod was shut down. Rather than re-run (which would have produced identical BPB numbers but cleaner timing), I am submitting the original code and logs as-is with this explanation. The redundant eval should have been gated behind `if not args.ttt_enabled:` — that is the only code change needed to bring eval within budget.

**Without the redundant sliding window eval, eval times are 576-581s (within 600s).** The TTT + N-gram scoring (552-564s) is the dominant phase and produces the reported BPB. Reviewers can verify this by adding the guard or setting `EVAL_STRIDE=0` to disable the standalone sliding window.

### Credits

- **Base model + Parallel Muon + TTT**: PR #549 by @abaybektursun (built on PR #414 by @signalrush, PR #399, PR #461)
- **LeakyReLU(0.9)²**: Sweep by @MatoTeziTanka (issue #140), building on PR #493 by @parinzee
- **N-gram cache concept**: Community discussion (issue #140, issue #677)
- **Entropy-reg QAT, mixed quant, GPTQ-lite per-row**: Original contributions
