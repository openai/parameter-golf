# Notable Non-Record: Switched Deep Supervision

**val_bpb: 1.08288** (TTT, single-seed) | **val_bpb: 1.08449** (sliding window, single-seed) | **15.997 MB artifact** | 8×H100 SXM, 588s

The first Deep Supervision (DS) submission in the Parameter Golf competition. Introduces **Switched Deep Supervision** — a training-time technique that adds intermediate cross-entropy supervision through the shared LM head at randomly-selected layers each step.

This submission does NOT beat SOTA (PR #1493 at 1.0810 BPB). It is presented as scientifically interesting research on auxiliary loss techniques in compute-constrained LM training, with detailed negative results and ongoing work.

## Results (Single Seed 42)

| Metric | Value |
|--------|-------|
| Pre-quantization post-EMA BPB | 1.08933 |
| Quantized BPB | 1.10110 |
| **Quantized sliding window BPB** | **1.08449** |
| **Quantized + TTT BPB** | **1.08288** |
| Total artifact size | 15,997,104 bytes |
| Training steps | 4316 |
| Training time | 588 seconds (8×H100) |

For comparison: merged SOTA (PR #1493) achieves 1.0810 TTT (3-seed mean). Our gap: +0.0019 BPB.

## Novel Contributions

### 1. Deep Supervision via Shared LM Head
At selected intermediate transformer layers (default 6, 7, 9), compute auxiliary cross-entropy loss using the shared LM head:

```
total_loss = main_CE + alpha * mean(layer_CE_for_each_DS_layer)
```

No new parameters — reuses the existing tied embedding / LM head. Zero artifact cost. The supervision provides direct gradient signals to intermediate layers, accelerating per-step convergence.

**Inspired by:** LayerSkip (Meta ACL 2024), Deeply Supervised Nets (Lee et al. 2015).

### 2. Switched DS — Random Single-Layer Supervision
Standard DS supervises ALL selected layers every step (3 auxiliary losses per step in our config). We introduce **Switched DS**: randomly pick ONE layer per step instead. Reduces compute overhead by ~3x while preserving most of the per-step benefit.

**Per-step supervision rotation:** Over thousands of steps, each DS layer receives ~1/N of the supervision events but with diverse training contexts. Our experiments show Switched DS produces better final BPB than non-switched DS at the same wallclock budget.

**Inspired by:** "Switched Auxiliary Loss" literature in multi-task learning.

### 3. Fraction-Based DS Decay
DS auxiliary loss alpha is ramped up over `DS_WARMUP_STEPS`, then linearly decayed to 0 between `DS_DECAY_START_FRAC=0.70` and `DS_DECAY_END_FRAC=0.85` of total training. This decouples DS-induced weight oscillation from the final EMA averaging window, allowing EMA to capture clean post-DS weights.

### 4. Per-Layer Adaptive GPTQ + int7 Embeddings
For artifact size compliance, we adopt per-layer adaptive GPTQ clipping (PR #1586 lineage):
- MLP weights: int6 with `MLP_CLIP_SIGMAS=12.0` (tighter)
- Attention weights: int6 with `ATTN_CLIP_SIGMAS=13.0`
- Embeddings: int7 with `EMBED_CLIP_SIGMAS=15.0` (saves ~530 KB vs int8)

This brings total artifact to 15.997 MB (within 16 MB limit).

## Architecture

Built on the April 2026 SOTA stack (PR #1493 by bigbag):

| Component | Setting |
|-----------|---------|
| Tokenizer | SP8192 |
| Layers | 11 physical, 512d, 8 heads, 4 KV heads |
| Depth Recurrence | Loop layers 3-5 three times, activate at 35% |
| Parallel Residuals | Layers 7-10 (GPT-J style) |
| MLP | 4x expansion (2048 hidden), LeakyReLU(0.5)^2 |
| Optimizer | MuonEq-R (row-normalized Muon), WD=0.095 |
| QK-Gain | 5.25 |
| Attention | XSA on all 11 layers, FlashAttention 3 |
| EMA decay | 0.9965 |
| Warmdown | Wallclock-fraction 0.72 |
| TTT | Score-first SGD, 3 epochs per chunk, cosine decay |
| **DS layers** | **6, 7, 9 (switched, alpha=0.01)** |
| **DS schedule** | **warmup 200 steps, decay 70%-85% of training** |

## Negative Results (What We Tried That Didn't Work)

These findings may be valuable to others exploring auxiliary loss approaches:

### Predictive Coding (PC) with Cosine Similarity
Tried cosine-similarity loss between intermediate layer outputs and (detached) next layer outputs. **Net negative across all alpha values tested (0.005, 0.01, 0.1).** Cosine similarity gradients shrink inversely with hidden state norms — pathological at scale.

### Multi-Token Prediction (MTP)
Combined DS with MTP heads predicting tokens t+2, t+3 via small transformer blocks sharing the LM head:

| MTP Variant | Sliding BPB | Verdict |
|-------------|-------------|---------|
| Block heads, horizons=2 | 1.09332 | -0.010 worse than pure DS |
| Block heads, horizons=1 | 1.08931 | -0.006 worse |
| Medusa heads (linear), horizons=2 | ~1.088 | -0.005 worse |
| Medusa heads (linear), horizons=1 | 1.08526 | -0.0016 worse |

**Verdict:** MTP provides genuine per-step convergence benefit (~0.005 BPB) but adds throughput overhead and EMA oscillation that consistently outweigh the gain. Even the lightest configuration (Medusa linear heads, 1 horizon) underperforms pure Switched DS at our compute budget.

This corroborates SPThole's broader finding (PR #1602): "Auxiliary losses are fatal in compute-starved regimes." However, our switched DS specifically is *not* fatal — it's slightly net-positive vs no-DS baseline at the per-step level, with throughput cost slightly exceeding the per-step gain.

## Ongoing / Future Work

### Top-K Sampled Softmax for DS Auxiliary Losses (in progress)
The dominant cost of DS is the LM head matmul (512 × 8192). We are exploring **sampled softmax with K random negatives** to reduce auxiliary loss compute by ~16x:

```
DS_aux_loss = CE([target, K random negatives], target_at_index_0)
```

This is mathematically a biased approximation of full CE but should preserve the gradient direction sufficiently for auxiliary supervision. Implementation is on a separate branch and pending H100 validation.

If successful, this would unlock **non-switched DS** (supervising all 3 layers every step at affordable compute), potentially providing strong enough per-step benefit to overcome the throughput penalty and beat SOTA.

This direction has zero precedent in the competition (verified across ~1600 PRs) — sparse/top-K LM head techniques are completely unexplored territory here.

## Reproduction

```bash
# Install dependencies
pip install brotli sentencepiece flash_attn_3

# Download SP8192 dataset (Kevin Clark's HF mirror)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# Run training (8×H100)
SEED=42 \
DS_ENABLED=1 DS_SWITCHED=1 DS_ALPHA=0.01 DS_WARMUP_STEPS=200 \
DS_LAYERS=6,7,9 DS_DECAY_START_FRAC=0.70 DS_DECAY_END_FRAC=0.85 \
QK_GAIN_INIT=5.25 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
WARMDOWN_FRAC=0.72 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
COMPRESSOR=brotli \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Compliance (Track B)

- **Condition 1 (Causality):** Sliding-window eval, prefix only ✓
- **Condition 2 (Normalized):** Standard softmax, no n-gram/logit bias ✓
- **Condition 3 (Score before update):** Each TTT chunk scored under `torch.no_grad()` BEFORE SGD ✓
- **Condition 4 (Single pass):** Each token scored once, no rescoring ✓

DS heads are training-only (not in artifact). All artifacts < 16,000,000 bytes. Training < 600s on 8×H100.

## Credits

Built on the SOTA stack from:
- PR #1493 (bigbag): SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT
- PR #1394 (Kevin Clark): SP8192 tokenizer + GPTQ Embeddings + Depth Recurrence + SDClip
- PR #1412 (Robby Sneiderman): Parallel Residuals
- PR #1586 (dexhunter): Per-Layer Adaptive GPTQ Clip + int7 Embeddings
- PR #1019 (abaybektursun): XSA-all + AR Self-Gen GPTQ + BigramHash
- PR #549 (abaybektursun): Score-First TTT framework

## Status

This is a **non-record submission** (does not beat current SOTA). Posted as documentation of:
1. The first Deep Supervision attempt in the competition
2. Switched DS as a novel auxiliary loss scheduling strategy
3. Negative results on PC and MTP variants
4. Roadmap for top-K sampled softmax (in progress)

3-seed validation pending. Single-seed (seed 42) result reported above.
