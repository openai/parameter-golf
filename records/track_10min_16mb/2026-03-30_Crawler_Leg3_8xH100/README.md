# Crawler — val_bpb 1.1874 (3-seed mean)

**Micro Crawler**: 4 flat XSA layers + 1 shared crawler block × 3 loops, mlp_mult=6.0. QAT via CRAWLER_QUANT_INT8=1. Naive int6 + zstd, ~9.4MB.

## Architecture Philosophy

The whole stack is a causal coordination engine operating at three temporal resolutions simultaneously through shared weights.

Each loop iteration is not doing different work — it is coordinating the same fuzzy input representation against the same learned shape space, but at a different causal horizon. Loop 0 attends to immediate causes (adjacent tokens). Loop 1 attends to medium-range causal structure. Loop 2 integrates distant causes at the sentence and paragraph level. The shared weights are the learned geometric attractor — the distributed representation of known truth that the input is being pulled toward through each pass. Weight sharing is not a parameter-budget compromise; it is the mechanism. The same causal law applied at three temporal resolutions, each loop leaving the representation less fuzzy than it found it.

## Results

| Seed | val_bpb (int6 SW exact) | Steps | Size |
|------|------------------------|-------|------|
| 1337 | 1.18720375             | 8087  | 8,842,981 bytes |
| 42   | 1.18761637             | 8119  | 9,362,069 bytes |
| 300  | 1.18745690             | 8103  | 9,332,848 bytes |
| **mean** | **1.18742567**     |       | **9,362,069 bytes (max)** |

Hardware: 8×H100 SXM, 600s wallclock cap.

## Config

- 4 flat XSA layers + 1 crawler block × 3 loops
- CRAWLER_MLP_MULT=6.0
- CRAWLER_QUANT_INT8=1 (QAT during training)
- GQA: 8 heads, 4 KV heads
- Bigram hash table: 2048
- RoPE: 16
- WARMDOWN_ITERS=2000
- SWA_EVERY=50
- SKIP_GPTQ=1 — naive int6 quantization, zstd compressed
- SKIP_EMA=1
- NGRAM_EVAL_ORDER=0 (no ngram)
- 14,462,508 parameters

## Reproduce

```bash
git clone https://github.com/newjordan/parameter-golf.git
cd parameter-golf
git checkout TEST_LAB
python3 data/cached_challenge_fineweb.py

# Seed 1337
SEED=1337 NPROC_PER_NODE=8 bash experiments/Crawler_Leg_3/run.sh

# Seeds 42 + 300
NPROC_PER_NODE=8 bash experiments/Crawler_Leg_3/run_multi_seed.sh
```

Training script: `experiments/Medusa/train_gpt.py`

## Active Ablation Work

The crawler architecture established above is the foundation. Current ablation series are investigating how to deepen the causal coordination mechanism:

**Choke (bandit_wagon_choke_shaped — BWCS):** Introduce per-loop bottleneck routing inside the crawler MLP. The fuzzy input must commit to a compressed shape before the loop can export its result. Per-loop routing means each causal horizon gets its own compression geometry. Testing flat, pyramid, grouped, and residual bottleneck shapes.

**Exporter / Cannon (planned — BWE):** Calibrate what each loop exports to the next. The choke compresses; the cannon fires the result at the right scale for the next loop's shared weights to receive cleanly. Per-channel soft clamp matched to the int6 dynamic range, plus per-loop bandwidth control so no single causal horizon dominates the residual stream.

**Battery (bandit_wagon_battery — BWB):** Per-loop RoPE frequency scaling (1, 3, 9) to specialize each loop's attention to a different causal distance. Pairs with skipgram features at matching skip distances as a future combination.

**Tap (bandit_wagon_tap — BWT):** Inject frozen encoder layer outputs per loop as stable, pre-drift ground truth anchors — giving each loop a direct read on what the encoder captured before any crawler-loop error accumulated.

The goal across all series: make the causal coordination at each temporal resolution explicit and controllable, rather than emergent and unbalanced.
