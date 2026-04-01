# Non-record: Oscillatory Recurrence at Layer 0 (1.1915 BPB, 3-seed)

**val_bpb: 1.1915** (sliding window, 3-seed mean, std 0.0008) | **15.02 MB** dynamic quant | 8x H100 SXM, 600s | No TTT

## Results

| Seed | Steps | ms/step | int8 BPB | Sliding BPB | int5+zstd | dynamic+zstd |
|------|-------|---------|----------|-------------|-----------|-------------|
| 42   | ~8400 | ~72     | 1.2267   | 1.1924      | 1.2451/15.50MB | 1.2361/15.02MB |
| 1337 | ~8400 | ~72     | 1.2255   | 1.1911      | 1.2422/-- | 1.2364/-- |
| 2024 | ~8400 | ~72     | 1.2255   | 1.1910      | 1.2417/-- | 1.2377/-- |
| **Mean** | | | **1.2259** | **1.1915** | **1.2430** | **1.2367** |
| **Std** | | | **0.0007** | **0.0008** | **0.0018** | **0.0009** |

## Ablation: TRN vs Pure Transformer

Same config, TRN removed (all 11 layers = standard Transformer), single seed=42:

| Metric | TRN@L0 (39d, seed=42) | ALL Transformer (39e) | Delta |
|--------|----------------------|----------------------|-------|
| Steps  | ~8400                | ~9439                | +12%  |
| ms/step| 72                   | 59                   | -18%  |
| int8   | 1.2267               | 1.2171               | -0.009 |
| sliding| 1.1924               | 1.1830               | -0.009 |

ALL Transformer wins on final BPP because it runs 18% faster and gets 12% more steps.
But at equal step count (step 7000), TRN leads by 0.031 BPP. TRN loses because it
is slower, not because it models worse.

| Step | ALL Transformer | TRN@L0 | Delta |
|------|-----------------|--------|-------|
| 6000 | 1.2986          | 1.2800 | TRN -0.019 |
| 7000 | 1.2881          | 1.2572 | TRN -0.031 |
| 9000 | 1.2346          | --     | gap narrowing |
| final| 1.2171          | 1.2259 | ALL -0.009 |

---

## Architecture

11 layers: 1 ParallelTRNBlock (layer 0, attention disabled) + 10 Transformer Blocks.

```
[TRN+MLP] [Attn+MLP] [Attn+MLP] ... [Attn+MLP]
   0          1          2              10
```

- Layer 0: SelectiveResonanceLayer (K=240 oscillators) + leaky_relu^2 MLP.
  No attention -- raw embeddings go through oscillatory recurrence then MLP.
- Layers 1-10: CausalSelfAttention + leaky_relu^2 MLP.

### Config

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| num_layers | 11 |
| num_heads / kv_heads | 8 / 4 |
| mlp_mult | 2.5 (d_ff=1280) |
| activation | leaky_relu(x, 0.5).square() |
| TRN K | 240 oscillators |
| vocab | 1024 (tied embeddings) |
| params | 24,557,011 |

### Other Techniques

- **XSA-all-11 (Cross-Scale Attention)**: subtracts the value-direction projection from
  attention output at all 11 layers, reducing head redundancy
- **BigramHash(2816, 112)**: XOR hash of consecutive token pairs into a learned 112-dim
  embedding table (2816 buckets), added to input embeddings
- **Partial RoPE (Rotary Position Embedding, dims=16)**: only 16 of 64 head dimensions
  rotate; remaining 48 pass through unchanged
- **SWA (Stochastic Weight Averaging, every=50)**: averages model weights every 50 steps
  during warmdown for smoother weight distribution and better quantization
- **Sliding window eval (stride=64)**: each token scored with up to 1024 tokens of left
  context by sliding the window 64 tokens at a time. Only the rightmost 64 tokens of each
  window contribute to the score. ~0.030 BPP improvement over non-overlapping eval
- **Dynamic quantization**: per-tensor bit allocation (int4 for low-sensitivity tensors,
  int5 for high-sensitivity), fitting 24.56M params in 15.02MB compressed with zstd-22

### Optimizer

- Muon (Newton-Schulz, 5 backend steps) for 2D matrix params, lr=0.03
- Adam (beta1=0.9, beta2=0.95) for scalar/1D params, lr=0.025
- Muon weight decay=0.04, gradient clip=0.3
- Warmdown: 3500 iterations, cosine decay
- Batch: 524,288 tokens/step, seq_len=1024, grad_accum=1

---

## Temporal Resonance Network (TRN)

### Overview

TRN maintains a complex-valued hidden state per oscillator that rotates at a learned
frequency and decays at a learned rate. K=240 oscillators give a 480-dimensional state
(240 real + 240 imaginary) evolving causally across the sequence.

Core recurrence:

```
r_t[k] = alpha_t[k] * r_{t-1}[k] + drive_t[k]
```

- `alpha_t[k] = sigmoid(alpha_proj(x_t)[k])` -- input-dependent decay gate (bias initialized to target centers 0.30/0.65/0.97)
- `drive_t[k]` -- complex drive signal projected from input
- State is complex: `r = r_real + i * r_imag`

### SelectiveResonanceLayer (used here)

Slim variant projecting input to 2K values (drive_r, drive_i). Alpha projected separately.
Omega (frequency) and g_out (output gate) are per-channel constants, not token-dependent.
No PCG, SCPM, delta-rule, or log-phase. ~1.2M params at K=240.

We also developed TemporalResonanceLayer (6 projections, PCG, SCPM, delta-rule erase,
~2.7M params) but the extra components didn't justify their cost at this scale.

### Parallel Scan

The recurrence runs as a Kogge-Stone parallel prefix scan in O(log n) steps:

```python
offset = 1
while offset < n:
    b[:, offset:] += a[:, offset:] * b[:, :-offset]
    a[:, offset:] *= a[:, :-offset]
    offset *= 2
```

10 stages for seq_len=1024. All in fp32 -- bf16 is unstable for alpha near 1.0 where
the scan accumulates over hundreds of positions. A Triton kernel was written but gave
<1% speedup (scan is a small fraction of total step time).

### Frequency Initialization

Omega is log-spaced from 1 cycle per full sequence to ~0.25 cycles/token:

```python
omega = linspace(log(2*pi/1024), log(pi*(1-1/K)), K).exp()
```

This covers periods from 1024 tokens (slow) to ~6 tokens (fast). Alpha is initialized
in three groups: fast decay (center 0.30, half-life ~1 token), medium (0.65, ~4 tokens),
slow (0.97, ~30 tokens).

### Why Layer 0

merge_gate analysis on full 11L TRN+Attention configs showed TRN contributing 37% at
layer 0 but <3% at layers 1-10. Raw embeddings carry the strongest periodic/positional
signal. After one attention layer, those patterns are absorbed into the residual and TRN
becomes redundant.

PR #1204 (1.1063 BPP, current SOTA candidate) independently found that attention at
layer 0 is unnecessary (DISABLE_LAYER0_ATTN=1). We replace it with TRN.

---

## Why TRN Loses on Wall-Clock (But Wins Per-Step)

TRN adds ~13ms to a 59ms pure-Transformer step (22% overhead from fp32 scan +
projections). In 600s this costs ~1700 steps (8400 vs 9439). The ablation shows
TRN leads by 0.031 BPP at step 7000, but the step deficit erases this by the end
of training.

Three reasons TRN converges faster per step:

1. **Exponential decay memory**: TRN carries a recency-weighted state that tracks topic
   drift, entity salience, and formatting mode. Attention with uniform 1024-token context
   mixes recent and distant tokens equally -- TRN's decay gives this for free.

2. **Frequency-selective filtering**: K=240 oscillators at different frequencies act as a
   filter bank. Periodic text patterns (Q/A alternation, list items, code indentation) are
   captured by resonant oscillators without attention having to dedicate heads to them.

3. **Implicit position encoding**: Omega encodes position through phase (`angle = omega * log(1+t)`).
   This supplements RoPE (Rotary Position Embedding) without using attention capacity.
   Most visible at layer 0 on raw token embeddings.

Three reasons the speed penalty outweighs the quality gain:

1. **fp32 scan is bandwidth-bound**: B=64, K=240, T=1024 in fp32 = ~60MB per scan pass.
   H100 HBM bandwidth is the bottleneck, not compute.

2. **Muon optimizer benefits from more steps**: Newton-Schulz orthogonalization improves
   with each update. 12% more steps is a large advantage.

3. **26M params at 600s is update-limited**: the model needs ~8000+ steps to converge.
   Every ms/step counts linearly.

Reducing TRN overhead from 13ms to ~3ms (e.g. via fused CUDA kernel) would close the
step-count gap enough for TRN to win on final BPP as well.

---

## Techniques Developed During Exploration

### BRANCH_ORTHO (Branch Orthogonalization)

Forces TRN output to be orthogonal to attention output via Gram-Schmidt:

```python
attn_n = F.normalize(attn_out.detach().float(), dim=-1, eps=1e-6)
proj = (trn_out * attn_n).sum(dim=-1, keepdim=True) * attn_n
trn_out = trn_out - proj
```

Prevents merge_gate collapse (attention dominating TRN to 97-99%). In Run 36 with full
11L TRN, BRANCH_ORTHO improved fp from 1.2724 to 1.2452. However, in the final config
(TRN@L0 with attention disabled), BRANCH_ORTHO is inactive -- there is no attention output
to orthogonalize against at layer 0, and layers 1-10 have no TRN.

### CausalEMA Input Filtering

Causal EMA on TRN input via parallel prefix scan, focusing TRN on low-frequency patterns.
Added ~0.004 BPP but cost 20-40ms/step. Not used in final config.

---

## Exploration History (39 Runs, 3 Days)

| Phase | Runs | Architecture | Best BPP | Finding |
|-------|------|-------------|----------|---------|
| 1 | 25-31 | 11L full TRN+Attn | 1.2793 fp | TRN works but slow |
| 2 | 32-33 | Zamba shared attn | 1.2920 int5 | QAT explosion with shared attn |
| 3 | 34-35 | Progressive seq, Triton scan | -- | Compile retrace blocks progressive |
| 4 | 36-37 | BRANCH_ORTHO + EMA + XSA11 | 1.2424 fp | Per-step quality improved |
| 5 | 38 | TRN 3/11 layers | 1.2600 dyn | Below baseline. Params too low |
| 6 | 39a-d | TRN@L0 + MLP2.5 + sliding | **1.1915** | Baseline exceeded |
| 7 | 39e | ALL Transformer ablation | 1.1830 | TRN per-step +0.031, speed loses |

## What We Learned

1. **TRN improves per-step convergence.** At equal steps, TRN@L0 leads by 0.031 BPP.
   The 566K param difference (2.3%) does not account for this.

2. **Speed wins at 600s.** 13ms/step overhead costs ~1700 steps, enough to erase 0.031 BPP.
   A faster TRN implementation would change the outcome.

3. **Attention wins at content-addressable retrieval.** Copy, induction, entity routing --
   softmax(QK^T)V handles these. TRN adds decay memory and periodic detection, but attention
   picks up rough versions via positional biases. TRN becomes redundant at layers 1-10.

4. **TRN converges slower.** Oscillator geometry needs many steps. Attention learns routing
   in ~1000 steps. Swapping TRN for attention gives more loss reduction per step at layers 1-10.

5. **merge_gate collapses.** All-TRN configs: sigmoid > 0.97 at layers 1-10. Gate suppresses
   TRN because attention provides faster gradient signal.

6. **Step count beats expressiveness at this scale.** Faster, simpler model gets more steps
   and wins -- even if each step is individually worse.

7. **Sliding eval has the best cost/benefit ratio.** -0.030 BPP for zero training cost.

8. **Baseline leaves 11MB of budget unused.** 17M params / 4.6MB out of 16MB. Scaling to
   11L/24.6M is the main structural gain.

---

## Related Work

- **PR #1061**: Causal Oscillator LM (1.337 BPP) -- damped harmonic oscillator. Our TRN outperforms.
- **PR #1204**: ParallelResiduals + MiniDepthRecurrence (1.1063 BPP). DISABLE_LAYER0_ATTN originated here.
- **PR #852**: Hymba (1.119 BPP) -- Mamba + Attention hybrid.
- **PR #1120**: Rascal (1.1099 BPP) -- pure Transformer SOTA.
- This is the only oscillatory recurrence submission in Parameter Golf.

---

## Run Command

```bash
SELECTIVE_RESONATOR=1 PARALLEL_TRN=1 PARALLEL_TRN_K=240 \
MODEL_TYPE=parallel NUM_LAYERS=11 MODEL_DIM=512 \
NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2.5 RELU2=1 \
XSA_LAST_N=11 TRN_EMA=0 TRN_LAYERS=0 DISABLE_LAYER0_ATTN=1 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=112 \
PARTIAL_ROPE_DIMS=16 SWA_ENABLED=1 SWA_EVERY=50 \
EVAL_STRIDE=64 COMPILE_MODE=default \
GRAD_ACCUM_STEPS=1 SEED=$SEED \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training script: `train_gpt.py` (included in this submission folder).

Training logs were lost when the RunPod instance was terminated. Model checkpoints (.pt)
for all 3 seeds are preserved. Results are reproducible with the above command.
