# Ouroboros — Crawler Architecture Research

Non-record research submission documenting signal-hunting through rapid ablation on a novel recurrent architecture.

## Results

| Seed | int6_sw_bpb | Steps | Size |
|------|-------------|-------|------|
| 444 | 1.13727008 | 5951 | 15,034,550 B |
| 4 | 1.13565882 | 5963 | 15,042,594 B |
| 300 | 1.13638653 | 5948 | 15,049,936 B |
| **mean** | **1.13643848** | | **15,049,936 B** |

Hardware: 8×H100 SXM · 600s · 26.25M params · ~100.85ms/step

## Research Context

This submission is one checkpoint in an ongoing crawler architecture research program. It is not a leaderboard attempt. The crawler is a recurrent refinement architecture distinct from the field's standard "depth recurrence" (simple layer replay) — it uses a separate shared block with its own MLP width, differentiated RoPE scales, and bidirectional cross-stream injection.

### The Arc: Frugendorff → Crawler → Ouroboros → Helix

**Frugendorff** ([PR #579](https://github.com/openai/parameter-golf/pull/579)) — Original recursive weight sharing research. Discovered cadence laws governing how shared transformer blocks interact with quantization. Key finding: recursion advantage is ~85% width reallocation, ~15% implicit regularization, ~0% from the recursion itself.

**ClownCar** ([PR #990](https://github.com/openai/parameter-golf/pull/990)) — Frugendorff compression baseline with canonical DeltaNet integration. Explored causal associative memory within crawler loops.

**Medusa** ([PR #1028](https://github.com/openai/parameter-golf/pull/1028), [PR #1047](https://github.com/openai/parameter-golf/pull/1047)) — DeltaNet crawler variants. Tested recurrent state carry across loops. Found cross-loop state carry violates causality — each loop must start from zero state.

**Crawler** ([PR #1140](https://github.com/openai/parameter-golf/pull/1140)) — 8.8MB, 1.1874 BPB. Established the U-Net encoder/decoder + bottleneck crawler architecture. Introduced FLOW instructions (per-loop learned state), RoPE battery (differentiated attention scales per loop), and the one-variable-at-a-time gate discipline.

**Nightcrawler** ([PR #1208](https://github.com/openai/parameter-golf/pull/1208)) — 10MB, 1.176 BPB. Scaled flat depth, validated tap-off > tap-on through interaction testing.

**Ouroboros** (this PR) — 15.0MB, 1.1364 BPB. Stacks five individually-gated signals on a 9-flat platform: loop-aware GPTQ, brotli compression, QK gain 4.0, 2-loop cadence, optimized warmdown.

**Helix** (in development) — Dual-stream co-firing architecture where crawler fires alongside every flat layer with bidirectional cross-injection through a position-agnostic content router. Micro-scale ablations (23+ arms on DGX Spark) show confirmed signal: +0.006 BPB from cross-injection alone on matched depth, with bridge width scaling monotonically through dim=96.

### Methodology: Signal Hunting Through Rapid Ablation

The crawler program runs 1-GPU 2000-step gates (~$0.50) before any 8×H100 production run (~$3-4). One variable per test, always. Over 50 ablation arms across BW5–BW19 mapped the interaction space systematically. Dead branches documented and closed: cannon (reversed at scale), trigram (null — recursion already approximates context), smear (null), flat weight sharing (catastrophic), pyramid choke (cold param burden).

Key technical contributions:
- **Loop-aware GPTQ**: 2-phase Hessian calibration addressing shared-weight quantization hostility. Standard GPTQ is dangerous on crawler (Frugendorff: 1.38 → 5.7 BPB post-quant). Loop-aware recalibrates crawler importance on post-flat-quantized activations.
- **Width-vs-recursion analysis**: Quantified that the crawler's advantage is width reallocation, not recursion signal. This redirected research from adding crawler complexity toward maximizing flat depth.
- **Position-agnostic cross-stream routing** (Helix): The bridge between flat and crawler streams deliberately strips positional encoding. Its job is content-based routing — connecting information clusters by semantics, not proximity. RoPE handles WHERE in each stream; the bridge handles WHAT flows between them.

## Reproduce

```bash
pip install brotli
SEED=444 MAX_WALLCLOCK_SECONDS=600 WARMDOWN_ITERS=2000 \
NUM_FLAT_LAYERS=9 NUM_CRAWLER_LAYERS=1 CRAWLER_LOOPS=2 \
USE_CRAWLER=1 COMPILE_FULLGRAPH=1 \
SKIP_GPTQ=0 LOOP_AWARE_GPTQ=1 QK_GAIN_INIT=4.0 \
GPTQ_CAL_SAMPLES=128 GPTQ_CAL_SEQ_LEN=2048 \
CRAWLER_LOOP_ROPE_SCALES=9,1,1 SKIP_EMA=1 \
MODEL_DIM=512 INST_DIM=32 CRAWLER_MLP_MULT=6.0 \
CRAWLER_TAP_DIM=0 ANCHOR_DIM=0 CRAWLER_MLP_CHOKE_DIM=0 \
XSA_LAST_N=11 BIGRAM_VOCAB_SIZE=2048 ROPE_DIMS=16 \
SWA_EVERY=50 MATRIX_LR=0.03 \
MLP_LEAKY_SLOPE=0.5 CRAWLER_MLP_LEAKY_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-03_Ouroboros_Crawler_Research_8xH100/train_gpt.py
```
