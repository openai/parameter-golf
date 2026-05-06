# Knowledge Distillation: A Negative Result (val_bpb: 1.1529, no improvement over baseline)

**val_bpb: 1.1529** (best distillation config, seed 42, 8xH100 SXM) | baseline: **1.1401** | **15.6 MB artifact**

## The Question

To clarify, this is a non-record submission on the Unlimited Compute track. The goal was to see whether distillation is even feasible under the constraints of this competition. I'm conveniently ignoring the fact that you'd also have to figure out how to train the teacher model within the time budget (spoiler: 6 hours of H200 time doesn't fit in 10 minutes).

Can a bigger teacher model help train a smaller student when every training step counts? I trained a 105M parameter teacher, cached its logits, and ran a "systematic ablation" across two distillation approaches, two teachers, and multiple mixing ratios.

Distillation doesn't improve anything at this budget. Any benefit from distillation and the step penalty roughly cancel each other out. Which makes me think the next experiment isn't a better neural teacher, but rather, a cheaper one. But that's a different submission.

## Teacher Model

| Metric        | Value                          |
|---------------|--------------------------------|
| Architecture  | 16L, 768-dim, 12 heads, MLP 4x |
| Parameters    | 105.5M                         |
| Training      | 50K steps, 5.6 hours on 4xH200 |
| Float BPB     | 1.099                          |
| Training cost | ~6 hours of H200 time          |

0.05 BPB better than the student on its own. Legit teacher.

## What I Tried

Two distillation approaches, two teachers, multiple mixing ratios. All compared against the same student trained on hard labels.

### Hard Distillation (teacher's top-1 prediction as label)

Swap some fraction of the ground truth labels with whatever the teacher is most confident about. Like asking a friend who took the test last year for the answers instead of studying the textbook.

| Teacher              | Alpha | Sliding BPB | vs Baseline |
|----------------------|-------|-------------|-------------|
| none (baseline)      | 0.0   | 1.152       | —           |
| self (27M, 1.15 BPB) | 0.1   | 1.242       | +0.090      |
| self (27M, 1.15 BPB) | 0.5   | 1.559       | +0.407      |
| big (105M, 1.10 BPB) | 0.1   | 1.242       | +0.090      |

Catastrophic at every setting. Turns out your friend wasn't as solid on the material as they thought. The teacher's top-1 prediction is just a noisy version of the ground truth, wrong often enough to actively hurt training. Teacher quality doesn't even matter here. The 105M teacher and the 27M self-teacher produce nearly identical results at alpha=0.1 (1.2417 vs 1.2423).

### Soft Distillation (KL divergence against teacher's distribution)

Instead of the teacher's best guess, use the probability distribution as a soft target. The student learns "the teacher thinks it's probably X, maybe Y, definitely not Z." Standard Hinton et al. approach.

I cached the teacher’s top-32 logits per position to avoid running the teacher during training. The output vocabulary is 1024 tokens (BigramHash is input-side only, doesn’t affect the prediction head), so that’s 3.1% of the distribution. Necessary compromise to keep I/O overhead from killing the run. The lookup added ~11ms overhead per step (159ms vs 148ms baseline).

| Teacher    | Alpha | Temp | Sliding BPB | vs Baseline |
|------------|-------|------|-------------|-------------|
| none       | 0.0   | —    | 1.152       | —           |
| big (105M) | 0.1   | 2.0  | 1.155       | +0.003      |
| big (105M) | 0.3   | 2.0  | 1.156       | +0.003      |
| big (105M) | 0.5   | 2.0  | 1.156       | +0.004      |

Way better than hard distillation. But still can't beat the baseline. The gap is roughly 0.003 BPB regardless of alpha. Flat across the sweep.

## Why It Doesn't Work (given our time constraints)

The ~11ms step overhead from cached logit lookup costs ~280 training steps (3773 vs 4051 in 600s). The distillation benefit has to overcome that step penalty. It doesn't. Knowledge transfer and step cost roughly cancel out.

It's like reading CliffsNotes instead of the actual book. Sure, CliffsNotes are a shortcut. But if you only have 10 minutes to study, it doesn't really matter which version you study, you still only studied for 10 minutes. The shortcut has to be dramatically more efficient per-second to make up for this.

### The Online Distillation Problem

I also tried running the teacher live during training (no caching). Teacher forward pass pushed step time to 761ms. This resulted in only 789 steps instead of 4051. The student barely trained. 1.564 BPB. Cached logits are mandatory.

## Extended Training: Does Distillation Ever Cross Baseline?

The 10-minute results left an open question: maybe distillation just needs more time. So I ran both configs for 2 hours and tracked val_bpb along the way.

| Step  | Baseline | Distillation | Delta  |
|-------|----------|--------------|--------|
| 1K    | 1.306    | 1.308        | +0.002 |
| 5K    | 1.232    | 1.233        | +0.001 |
| 10K   | 1.207    | 1.210        | +0.003 |
| 20K   | 1.195    | 1.195        | 0.000  |
| 30K   | 1.191    | 1.190        | -0.001 |
| 40K   | 1.188    | 1.189        | +0.001 |
| 45K   | 1.121    | 1.134        | +0.013 |
| final | 1.106    | 1.107        | +0.001 |

The curves track almost exactly. No crossover. At step 30K the distillation model briefly pulls ahead by 0.001, then falls back.

The step 45K jump is warmdown kicking in. Both models drop sharply, but baseline benefits more (1.188 → 1.121, a −0.067 drop) than distillation (1.189 → 1.134, only −0.055). That +0.013 delta is the largest gap in the entire extended run. They converge again by the final checkpoint. I don't have a clean explanation for why distillation gets less out of warmdown, but the gap doesn't stick around so I'm not reading too much into it.

One thing that changed at the longer timescale: per-step overhead dropped from ~11ms to ~7.7ms (back-calculated from 46356 vs 48777 total steps in 7200s). Probably the OS page cache warming up for the logit lookups. So the overhead argument actually gets weaker over time... but the distillation benefit doesn't grow to fill that gap. The curves just stay locked together.

## The Top-32 Caveat

I only cached the teacher’s top-32 logits per position out of a 1,024-token output vocabulary. That’s 96.9% of the distribution I’m throwing away. Hinton’s whole "dark knowledge" argument is that the tail carries useful signal ("the teacher thinks this is definitely not Z"), and I’m cutting that tail off entirely.

This was a deliberate tradeoff, not an oversight. The top-32 tokens capture the vast majority of the teacher’s probability mass, and caching the full 1,024 distribution would have increased cache size and lookup time. At a 10-minute budget, I/O is the enemy.

So this experiment really tests whether the plausible alternatives (the top of the teacher’s distribution) can provide enough signal to pay for their own overhead. They couldn’t.

## What Actually Happened

Hard distillation is catastrophic. The teacher's most confident prediction is a noisy label that hurts every time, and teacher quality doesn't change this. A 105M teacher does the same damage as a 27M self-teacher.

Soft KL distillation nearly matches baseline but can't beat it. The full distribution (well, top-32 of it) is useful signal, but the step overhead offsets it. Net result: ~0.003 BPB worse at all alpha values.

More training time doesn't fix it. Extended 2-hour runs show the curves tracking identically across ~49K baseline steps (~46K for distillation, since it loses steps to overhead). The step overhead drops at longer timescales, but distillation benefit doesn't grow to fill the gap.

The teacher's soft predictions aren't worth the cost. At this model scale, on this data distribution, whatever the teacher knows about inter-token correlations doesn't translate into faster student learning. That doesn't mean the dark knowledge isn't real (it probably is, and caching only top-32 logits limits what I can say here). It means whatever information is there isn't enough to overcome even modest overhead per step.

Running the teacher live during training is a non-starter. The forward pass overhead guts the training budget. Cached logits are the only viable path, and even those aren't enough.

## 3-Seed Validation (8xH100 SXM, 600s)

Self-distillation (student as its own teacher, same architecture):

| Seed | Baseline BPB | Distillation BPB | Delta | Notes |
|------|-------------|-----------------|-------|-------|
| 42   | 1.140       | 1.153           | +0.013 | |
| 1337 | 1.139       | 1.140           | +0.001 | baseline artifact slightly over 16MB* |
| 2024 | 1.140       | 1.141           | +0.001 | distill artifact slightly over 16MB* |

Distillation never beats baseline across any seed. The average gap (+0.005) is consistent with the H200 findings. Seed 42 shows a larger gap, likely due to random variation in warmdown timing.

*Seeds 1337 baseline (16.15 MB) and 2024 distillation (16.17 MB) are slightly over the 16,000,000-byte artifact limit due to seed-dependent quantization variance. Included as auxiliary evidence for the negative result, not as compliant runs. The remaining 4 of 6 runs are within limits and independently confirm the finding.

## Architecture

Student model (same as baseline):

| Component     | Detail           |
|---------------|------------------|
| Layers        | 11               |
| Dim           | 512              |
| Heads         | 8 (4 KV, GQA)    |
| MLP           | 3x, relu-squared |
| XSA           | Last 4 layers    |
| EMA           | 0.997            |
| Stored params | ~25M             |

## Run Commands

```bash
# Cache teacher logits (one-time, ~5 min)
CACHE_TEACHER_LOGITS=1 TEACHER_PATH=workspace/teacher_model.pt \
  torchrun --standalone --nproc_per_node=4 train_gpt.py

# Train student with cached soft KL distillation
DISTILL=1 DISTILL_CACHED=1 DISTILL_SOFT=1 DISTILL_ALPHA=0.1 \
  DISTILL_TEMP=2.0 VE_ENABLED=1 WARMDOWN_ITERS=1600 \
  NUM_LAYERS=11 XSA_LAST_N=4 EMA_ENABLED=1 LATE_QAT=1 \
  BIGRAM_VOCAB_SIZE=6144 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py

# Baseline (no distillation)
VE_ENABLED=1 WARMDOWN_ITERS=1600 NUM_LAYERS=11 XSA_LAST_N=4 \
  EMA_ENABLED=1 LATE_QAT=1 BIGRAM_VOCAB_SIZE=6144 \
  torchrun --standalone --nproc_per_node=4 train_gpt.py
```

## Credits

First distillation experiment in Parameter Golf. Inspired by Hinton et al. (2015) "Distilling the Knowledge in a Neural Network.
