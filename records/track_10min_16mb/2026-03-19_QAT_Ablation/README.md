# 2026-03-19_QAT_Ablation

*Non-record: Does int8 quantization-aware training improve post-roundtrip val_bpb?*

**Answer: No — the overhead costs more than it recovers.**

---

## Question

The baseline loses ~0.007 BPB in the int8+zlib export step because bf16-trained weights are rounded cold onto the int8 grid. Every leaderboard entry so far attacks this gap indirectly — aggressive warmdown for tighter weight distributions, FP16 embedding bypass, or alternative quantization formats (int6). Nobody has trained directly against the int8 quantization grid.

This submission tests whether QAT (straight-through fake-quantize matching the export pipeline exactly) recovers some of that gap. The experiment isolates QAT as the only variable — baseline architecture, baseline hyperparameters, no other changes.

---

## Method

A `fake_quantize_int8_per_row` function is inserted into `CastedLinear.forward`. It matches the export pipeline's `quantize_float_tensor` exactly:
- Same `INT8_CLIP_Q = 0.9999984` percentile clipping via `torch.quantile`
- Same per-row scale: `clip_abs / 127.0`
- Same rounding: `round().clamp(-127, 127)`
- Straight-through estimator: gradients pass through as if no quantization happened

**Schedule:** QAT activates at 30% of training steps (~step 6,000). Training runs bf16-only before that to let the loss landscape stabilize.

**No other changes.** Architecture is 9L×512d, all hyperparameters are baseline defaults (WARMDOWN_ITERS=1200, MATRIX_LR=0.04, etc).

---

## Results

| Metric | SlidingWindowEval (no QAT) | This run (QAT) |
|--------|---------------------------|----------------|
| Steps completed | 13,450 | 8,011 |
| step_avg | 44.6ms | 75.2ms (64.5 pre-QAT, 77+ post-QAT) |
| Pre-quant val_bpb (standard eval) | 1.2196 | 1.2327 |
| **Post-quant val_bpb (sliding window)** | **1.1925** | **1.2052** |
| Artifact bytes | 15,874,829 | 15,868,103 |
| Eval time | 70s | 75s |

**val_bpb: 1.2052 vs 1.1925 — QAT is 0.013 worse.**

---

## Why it didn't work

The result is **not** evidence that QAT is a bad idea. It's evidence that **exact percentile-matching QAT is too expensive for int8 in this competition format.**

### The core problem: `torch.quantile` overhead

Matching the export pipeline exactly requires `torch.quantile(w.abs(), 0.9999984, dim=1)` on every weight matrix, every forward pass. This adds **~20% per-step overhead** (64ms → 77ms after QAT activates). Over a 600-second training budget, that costs ~2,000 training steps — roughly 1B fewer training tokens.

The lost training tokens hurt more than the quantization gap recovery helps. The int8 quantization gap (~0.007 BPB) is smaller than the convergence loss from 40% fewer training steps.

### Why this matters for the competition

| Approach | Per-step cost | Quant gap reduction | Net effect |
|----------|--------------|--------------------|----|
| Aggressive warmdown (WD=20000) | 0% overhead | ~0.009 BPB | **Positive** |
| FP16 tied embedding | 0% overhead, ~500KB artifact | ~0.004 BPB | **Positive** |
| Int8 QAT (this submission) | ~20% overhead → ~2000 fewer steps | ~0.003-0.006 BPB theoretical | **Negative** (overhead > recovery) |
| Int6 QAT (PRs #128, #137) | ~20% overhead | ~0.01+ BPB (larger gap) | **Likely positive** (larger gap to close) |

### When QAT would work

1. **With int6 quantization** — the quantization gap is larger (~0.01+ BPB), making the overhead worthwhile. PRs #128 and #137 confirm this with val_bpb 1.1594 and 1.1666 respectively.
2. **With `amax` instead of `torch.quantile`** — near-zero overhead, but doesn't match the export pipeline exactly. The 0.0001% percentile difference may not matter in practice.
3. **With a longer training budget** — if the wallclock cap were 30 minutes instead of 10, the overhead would be amortized over more steps.

---

## Graph priming finding

An earlier version pre-primed the QAT compiled graph during warmup (running one forward/backward pass with `_qat=True`, then resetting to `_qat=False`). This caused `torch.compile` to use a slower compilation path for the non-QAT forward pass — step_avg was 65ms from step 1, even before QAT activated. Removing the graph priming restored baseline speed for the non-QAT phase. This is a useful finding for anyone implementing conditional code paths under `torch.compile(dynamic=False, fullgraph=True)`.

---

## Reproduction

```bash
cd /workspace
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout qat-sliding-window
python3 data/cached_challenge_fineweb.py --variant sp1024

# Set env vars
export VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
export MLP_MULT=2 TIE_EMBEDDINGS=1 TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024
export ITERATIONS=20000 WARMDOWN_ITERS=1200 WARMUP_STEPS=20
export MAX_WALLCLOCK_SECONDS=600 TRAIN_LOG_EVERY=200 VAL_LOSS_EVERY=0
export QAT=1 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 DOC_ISOLATED_EVAL=0
export SEED=1337 RUN_ID=ablation_qat_slide64

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-19_QAT_Ablation/train_gpt.py
```

Hardware: 8×H100 SXM (RunPod), PyTorch 2.9.1+cu128

---

## Acknowledgments

- `train_gpt.py` is based on the SlidingWindowEval entry (#50) by @mattqlf, which provides the sliding window evaluation infrastructure
- Analysis informed by the WarmdownQuantization entry by @samuellarson (warmdown vs QAT tradeoffs) and the LoRA TTT ablation by @samacquaviva (doc-isolated eval gains)
- Int6 QAT comparison data from PRs #128 (@rsavitt) and #137 (@abhishekgahlot2)
- Built with [Claude Code](https://claude.com/claude-code)

## Author

GitHub: [@mrdavtan](https://github.com/mrdavtan)
Date: 2026-03-20
