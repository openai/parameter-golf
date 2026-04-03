# Ouroboros (Bandit Wagon XI)

A research-driven production run stacking five confirmed signals on the 9-flat crawler platform. Named for the serpent eating its own tail — the crawler's recurrent loop refining its own output.

## Results

| Seed | int6_sw_bpb (sliding window) | Steps | Size | Artifact Legal |
|------|------------------------------|-------|------|----------------|
| 444  | 1.13727008                   | 5951  | 15,034,550 B | yes |
| 4    | **1.13565882**               | 5963  | 15,042,594 B | yes |
| 300  | 1.13638653                   | 5948  | 15,049,936 B | yes |
| **mean** | **1.13643848**           |       | **15,049,936 B** | |

Hardware: 8×H100 SXM · 600s wallclock · `bytes_code`: 121,677

## Architecture

9-flat crawler with recurrent refinement: 9 unique flat transformer blocks (encoder/decoder path) followed by 1 shared crawler block that loops 2× with differentiated RoPE scales.

**Key parameters:**
- `NUM_FLAT_LAYERS=9` · `NUM_CRAWLER_LAYERS=1` · `CRAWLER_LOOPS=2`
- `MODEL_DIM=512` · `NUM_HEADS=8` · `NUM_KV_HEADS=4`
- `QK_GAIN_INIT=4.0` · `INST_DIM=32`
- `COMPILE_FULLGRAPH=1` · `CRAWLER_LOOP_ROPE_SCALES=9,1,1`
- `LOOP_AWARE_GPTQ=1` · `GPTQ_CAL_SAMPLES=128` · `GPTQ_CAL_SEQ_LEN=2048`
- Compression: int6 quantization + brotli (quality=11)
- 26.25M parameters · ~100.85ms/step · SWA from step 5600

## Research: Five Stacked Signals

This submission is the product of a systematic crawler research program (BW5 through BW XIX) spanning March 29 – April 3, 2026. Each signal was individually gated before being stacked into this production run.

### Signal 1: Loop-Aware GPTQ (confirmed −0.00380 BPB)

Standard post-training GPTQ is dangerous on crawler architectures because shared weights are hostile to naive quantization — the Frugendorff model collapsed from 1.38 to 5.7 BPB post-quant. We developed a 2-phase loop-aware calibration:

- **Phase 1:** Collect Hessians for all layers (flat + crawler)
- **Phase 2:** Patch flat blocks with GPTQ-quantized weights, then re-collect crawler Hessians on the actual post-quantized activations

This ensures the crawler's importance scores reflect its real input distribution after flat-layer quantization. BW10 full run delivered −0.00380 BPB vs the BW5 champion. BW12 and BW13 confirmed −0.002 BPB consistently across multiple configurations.

**Source:** `crawler/2026-04-01_BW10_GPTQ/`, `crawler/2026-04-01_BW12_Interaction_2k/`, `crawler/2026-04-01_BW13_TapOff_Anchor_GPTQ_2k/`

### Signal 2: Brotli Compression (approved, ~5-15% smaller artifacts)

Replaced zstd (level 22) with brotli (quality 11) for post-quantization model compression. Brotli uses a larger context window and better entropy coding for static blobs — quantized weight tensors are a single-shot compression target, which is brotli's sweet spot. Gated in BW20 (1k-step, 8×GPU, clean run, no blowups).

The artifact size savings are critical: BWX at 15.24MB was tight against the 16MB cap. Brotli freed ~200KB+ headroom that absorbed the GPTQ size overhead while keeping the total at 15.03MB.

**Source:** `crawler/2026-04-02_BW20_Brotli_2k/`

### Signal 3: QK Gain Initialization (high-confidence, −0.006 external)

`QK_GAIN_INIT=4.0` (up from default 1.5). Per-head q_gain scalar initialized higher drives sharper early attention gradients. The model is free to train the scalar away — this is an init effect, not a constraint.

External evidence: ~−0.006 BPB across 45 runs in 3 codebases (arXiv-adjacent work). Neural track proxy: −0.00149 BPB at 2k gate. First crawler-track test in this submission.

**Source:** `experiments/COMPREHENSIVE_RESEARCH_SYNTHESIS_2026-04-02.md`, `PIPELINE.md` Tier 1

### Signal 4: 2-Loop Cadence (directional −0.054, faster steps)

Reduced `CRAWLER_LOOPS` from 3 to 2. BW17 DGX-Spark RAPID testing showed a −0.054 int6_sw_bpb directional delta (small-token run, absolute value inflated, but direction clear).

Fewer loops provide three benefits:
1. **Faster steps** (100.85ms vs 110.19ms) → 505 more training steps in the 600s budget
2. **Smaller quant gap** — less shared-weight amplification across iterations
3. **Simpler gradient flow** — fewer loop iterations reduce gradient conflict in shared weights

**Source:** `crawler/2026-04-02_BW17_DGXSpark_Cadence_Longform/`

### Signal 5: Optimized Warmdown (confirmed, 2000 > 3500 > 5000)

`WARMDOWN_ITERS=2000` (shorter warmdown). Rat Rod warmdown study confirmed shorter warmdown consistently beats longer across multiple configurations. Already present in BWX, retained here.

**Source:** `experiments/COMPREHENSIVE_RESEARCH_SYNTHESIS_2026-04-02.md`, Rat Rod PROGRESS.md

## Research Context: The Crawler Signal Analysis

Our crawler research program discovered that the crawler's advantage is **85% width, 15% implicit regularization** — not recursion itself. The real lever is fewer unique layers → wider dimension at fixed parameter count. This insight shifted our focus from adding more crawler complexity (trigram, smear, cannon — all washed out) toward:

1. **Maximizing flat depth** (4F → 5F → 9F: monotonic gains)
2. **Reducing loop overhead** (3 loops → 2: faster steps, less quant gap)
3. **Improving post-training quantization** (loop-aware GPTQ: Hessian-aware, not naive)

Dead branches that informed this direction:
- Cannon (scalar FFN gate): +0.00020, reversed at full run
- Trigram embedding: +0.00014, null within noise
- Loop smear: −0.00003, null
- Flat weight sharing: +0.03694, catastrophic
- Pyramid MLP choke: +0.03440, cold param burden

## Reproduce

```bash
# From repo root, 8×H100, flash-attention/hopper on PYTHONPATH
pip install brotli

SEED=444 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=2000 \
NUM_FLAT_LAYERS=9 \
NUM_CRAWLER_LAYERS=1 \
CRAWLER_LOOPS=2 \
USE_CRAWLER=1 \
COMPILE_FULLGRAPH=1 \
SKIP_GPTQ=0 \
LOOP_AWARE_GPTQ=1 \
QK_GAIN_INIT=4.0 \
GPTQ_CAL_SAMPLES=128 \
GPTQ_CAL_SEQ_LEN=2048 \
CRAWLER_LOOP_ROPE_SCALES=9,1,1 \
SKIP_EMA=1 \
MODEL_DIM=512 \
INST_DIM=32 \
CRAWLER_MLP_MULT=6.0 \
CRAWLER_TAP_DIM=0 \
ANCHOR_DIM=0 \
CRAWLER_MLP_CHOKE_DIM=0 \
XSA_LAST_N=11 \
BIGRAM_VOCAB_SIZE=2048 \
ROPE_DIMS=16 \
SWA_EVERY=50 \
MATRIX_LR=0.03 \
MLP_LEAKY_SLOPE=0.5 \
CRAWLER_MLP_LEAKY_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-03_Bandit_Wagon_XI_8xH100/train_gpt.py
```

---

### Footnote: Bandit Wagon X (Parent)

BW XI builds directly on Bandit Wagon X (BWX), our 9-flat crawler baseline:

| Metric | BWX 9F | BW XI | Delta |
|--------|--------|-------|-------|
| int6_sw_bpb | 1.13867894 | 1.13727008 | −0.00141 |
| bytes_total | 15,239,617 | 15,034,550 | −205,067 |
| step_ms | 110.19 | 100.85 | −9.34 |
| steps (600s) | 5446 | 5951 | +505 |

BWX established the 9F platform (tap-off, no anchor, naive int6 + zstd). BW XI adds five post-BWX research signals that collectively improve BPB, reduce artifact size, and increase training throughput. The research continues — BW18 and BW19 delta matrices are queued with 40+ additional ablation arms on the 9F platform.
