# Gaussian Per-Token Loss Reweighting: A Negative Result

**Non-Record Submission (Research Contribution)**
**Author:** Julian Tang ([@JulianTang2027](https://github.com/JulianTang2027))
**Base:** PR #180 by thwu1 — 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04
**Final baseline:** 1.14416 val_bpb
**Final with Gaussian token weighting (sigma=2.0):** 1.15847 val_bpb
**Delta:** **+0.01431 bpb (regression)**

---

## TL;DR

I added a single hyperparameter to PR #180 that applies a **Gaussian per-token loss weight** centered on the batch mean of per-token losses, so the optimizer focuses on "learnable" tokens (near the mean) and downweights both trivial tokens (very low loss) and chaotic tokens (very high loss). It looked promising during training — the reported val_loss dropped dramatically — but **the final unweighted sliding-window BPB got worse by 0.0143**. The training-time improvement was an artifact of changing the objective being measured, not the objective being optimized.

This is a clean cautionary tale: if you reshape the per-token loss during training, you cannot read off "val_loss" at training time and expect it to correlate with the competition metric. I'm writing this up so nobody else wastes compute on the same idea without a skeptical eye on the evaluation.

---

## The Idea

**Motivation.** In the Parameter Golf regime, every parameter is expensive, so intuitively the model should spend its limited capacity on tokens where learning is actually possible. Per-token losses in a trained LM have a long-tailed distribution: most tokens have near-average loss, but a minority are either trivial (function words, punctuation after certain contexts) or essentially irreducible (the first byte of a proper noun, a rare subword). The trivial tokens consume gradient signal that is mostly noise, and the irreducible tokens dump gradient into dimensions the model can't hope to predict correctly at this scale.

**The mechanism.** I reshape the loss with a Gaussian weight in z-score space, applied per batch:

```python
if self.token_weight_sigma > 0:
    per_token_loss = F.cross_entropy(logits.float(), targets, reduction="none")
    ld = per_token_loss.detach()
    z = (ld - ld.mean()) / (ld.std() + 1e-8)
    w = torch.exp(-z.square() / (2.0 * self.token_weight_sigma ** 2))
    w = w / w.mean()  # normalize so weighted mean matches unweighted scale
    return (per_token_loss * w).mean()
return F.cross_entropy(logits.float(), targets, reduction="mean")
```

Tokens within ~1 sigma of the batch mean get weight near 1.0, tokens >2 sigma from the mean get aggressively downweighted. The `w / w.mean()` normalization keeps the total loss scale comparable to unweighted CE so Muon's LR schedule doesn't need retuning. I picked `sigma=2.0` as a single reasonable value to test; smaller sigmas are more aggressive, larger sigmas approach unweighted CE.

**Why I thought it might work.** The argument is essentially a curriculum / importance-sampling argument. In a tight parameter budget, you want every gradient step to count. If ~10% of tokens are contributing ~40% of the loss but are essentially unlearnable at 16MB, the unweighted objective spends a disproportionate share of gradient norm on things the model cannot improve. Downweighting them should free capacity for the bulk of the distribution.

**Why it doesn't work.** See the diagnosis section below.

---

## The Experiment

Two runs, identical seed (1337), identical hardware (8xH100), identical code except `TOKEN_WEIGHT_SIGMA`:

| Run | `TOKEN_WEIGHT_SIGMA` | Final train val_bpb (reported) | Final sliding-window val_bpb (metric) | Artifact size |
|---|---|---|---|---|
| baseline | 0.0 | 1.1542 | **1.1442** | 16,050,780 bytes |
| sigma2 | 2.0 | **1.0182** | 1.1585 | 15,976,011 bytes |

Both runs hit the 600 s wallclock cap at ~6510 steps out of 20000. Everything else (architecture, optimizer, schedule, quantization, compression, eval stride) is byte-identical to PR #180.

### What happens at the metric that matters

The sigma2 run is **0.0143 bpb worse** on the actual competition metric (final_int8_zstd sliding-window val_bpb, stride=64). That's a substantial regression — roughly the size of three stacked "known differentiator" techniques on the current leaderboard.

### What happens at the metric the training loop reports

During training, the sigma2 run shows val_loss dropping to 1.72 and val_bpb dropping to **1.0182** at step 6500. The baseline at the same step reports val_bpb 1.1542. If you just glance at the training log you would think sigma2 is crushing it by **over 0.13 bpb**. It isn't. It's winning on *its own reshaped objective*, which the eval loop is also computing using the reshaped loss during periodic in-training evals.

### Why the gap is real, not measurement noise

The `final_int8_zlib_roundtrip` eval at the end of training uses **unweighted** cross-entropy on the dequantized model over held-out FineWeb validation with stride-64 sliding-window attention. It is the same code path on both runs. The 1.1442 vs 1.1585 comparison is apples-to-apples on the metric the competition scores on.

---

## Why It Fails: The Diagnosis

When you apply a per-batch Gaussian weight in z-score space, **you are no longer training a language model under the competition's loss function.** You are training it under a reshaped loss where:

1. **The optimizer solves a different problem.** Per-token CE rewards every nat of information gained anywhere. Weighted CE rewards information gained only in a narrow band around the current mean. The model learns to concentrate its capacity on "average" tokens and literally stops trying on the tails.

2. **The tails are where the bits are.** In the bpb metric, a token that costs 10 nats costs 10 nats. You do not get credit for being very confident on easy tokens — you can't go below 0. You get punished for being very wrong on hard tokens. Downweighting the hard tokens during training means the model gets even more wrong on them, and those errors dominate the metric.

3. **Batch-relative reweighting is unstable as the model improves.** As training progresses, the per-token loss distribution tightens. `z = (ld - ld.mean()) / ld.std()` keeps the weight distribution roughly constant in shape regardless of how good the model is. This means late in training, tokens that are a few percent above the mean get downweighted just as hard as tokens that were genuinely unlearnable early in training. The reweighting doesn't decay.

4. **The weighted val_loss number is misleading by design.** `w / w.mean()` normalizes per batch, so the weighted loss is literally "what's the average loss on tokens the model happens to be predicting well right now." It looks monotonically better because the model is monotonically improving *on the tokens it was already best at*. It says nothing about the tokens driving the metric.

The short version: **any reweighting of per-token CE that downweights high-loss tokens will trade metric for apparent training-time progress.** The two metrics only coincide for `sigma=infinity`, which is just vanilla CE.

---

## Things I Did Not Try (and Why)

I kept the scope narrow on purpose — the point is a clean, single-variable ablation, not a hyperparameter sweep.

- **Multiple sigma values.** `sigma=2.0` is the middle of the reasonable range. Smaller sigma is more aggressive and would be worse by the same argument. Larger sigma (e.g., 5.0) approaches unweighted CE and would show diminishing gap; I expect the regression to monotonically shrink toward zero. I don't think there is any sigma at which this wins, and the argument in "Why It Fails" is sigma-independent.
- **Upweighting hard tokens (anti-Gaussian).** This is the symmetric idea. I suspect it also hurts because irreducible tokens dominate, but I haven't tested it and I'd be interested if someone does.
- **Scheduling sigma across training** (e.g., start at 2.0 and linearly increase to infinity). Plausibly recovers most of the baseline performance, but at that point you are just adding a warmup to a reweighting that ends at unweighted CE, which is strictly more complexity than the baseline for at best a marginal gain.
- **EMA'd loss statistics instead of per-batch.** Would stabilize the reweighting but does not address point (2) — the tails are still where the bits are.

---

## Artifact Details

- **Baseline (sigma=0.0, seed=1337):** val_bpb 1.14416008, bytes_total 16,050,780
- **Sigma2 (sigma=2.0, seed=1337):** val_bpb 1.15847054, bytes_total 15,976,011

Note that the baseline run is 50 KB over the 16 MB budget and the sigma2 run is 25 KB under. This variance is seed-dependent and is consistent with what PR #180's own submission.json reports (`bytes_total: 15,900,000` with different seeds). Different training objectives produce slightly different quantized weight entropies after zstd, which is how the sigma2 run ends up smaller — the extra few KB of "free" budget does not offset the 0.0143 bpb regression.

---

## Reproducing

Identical setup to PR #180. One env var added:

```bash
# Baseline
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# With Gaussian token weighting
SEED=1337 TOKEN_WEIGHT_SIGMA=2.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All other hyperparameters default to PR #180's values. `train_gpt.py` in this folder is the exact file used for both runs; setting `TOKEN_WEIGHT_SIGMA=0.0` (the default) is a no-op and recovers PR #180 exactly.

---

## Takeaway for Future Competitors

If you are considering per-token loss shaping in Parameter Golf, remember:

1. **The metric is unweighted CE with sliding-window attention.** Anything you do to the training loss that the final eval does not mirror is a change of objective, not an optimization trick.
2. **Training-time val_loss is only trustworthy if it uses the exact same reduction as the final metric.** If you reweight, log the *unweighted* val_loss in parallel so you can see the gap in real time.
3. **Downweighting high-loss tokens is particularly dangerous** because the high-loss tokens are exactly the ones the bpb metric is most sensitive to.

I'm submitting this because the competition rules explicitly invite negative results, and because I wasted a nontrivial chunk of RunPod credits finding this out the hard way. Hopefully somebody else skips the mistake.

---

## Included Files

- `train_gpt.py` — code snapshot used for both runs (identical, differs only via `TOKEN_WEIGHT_SIGMA` env var)
- `train_baseline_seed1337.log` — full baseline training log
- `train_sigma2_seed1337.log` — full sigma=2.0 training log
- `submission.json` — leaderboard metadata
