# ACN Output Accumulator on the Naive Baseline

**Non-Record Submission (Idea Probe)**
**Author:** Chris Coffee ([@chrico-bu-uab](https://github.com/chrico-bu-uab))
**Base:** Official `train_gpt.py` baseline (1.2244 bpb)
**Result:** 1.2265 bpb (ACN on) vs 1.2262 bpb (ACN off) — wallclock-tied
**Idea:** Auto-Compressing Networks (Bose et al., 2025 — arxiv [2506.09714](https://arxiv.org/abs/2506.09714)) output accumulator, gated behind `ACN_OUTPUT=1`.

## TL;DR

I added one knob to the official baseline: each transformer block's hidden state is added (with a learnable per-layer scalar, init to zero) to the final pre-`final_norm` representation. With the cap on, the result is essentially tied with the unmodified baseline. With the cap conceptually removed and runs compared at equal step counts, ACN is ~0.006 bpb better. The naive accumulator costs ~2 % wallclock per step, which exactly eats the per-step gain in this 10-minute regime.

That's the whole story: ACN gives a small but real signal on the baseline architecture, but the naive implementation is too expensive to convert into a record. This is a one-seed idea probe, not a tuned submission.

## What ACN does (and what's actually in the diff)

The Auto-Compressing Networks paper proposes letting every block contribute directly to the final representation, rather than having information flow only through the residual stream. With per-layer scales initialized to zero, the model starts identical to the baseline and only opts into auxiliary contributions where they pay for themselves.

The full diff against the official `train_gpt.py` is 16 lines. In the `GPT.forward` loop:

```python
hidden_states.append(x)              # collected after each block
...
if self.output_scales is not None:
    scales = self.output_scales.to(dtype=x.dtype)
    for i, h in enumerate(hidden_states):
        x = x + scales[i] * h
x = self.final_norm(x)
```

`output_scales` is a single `nn.Parameter` of shape `[num_layers]` (9 floats — adds 9 params total, +36 bytes uncompressed). It's added to the scalar-LR optimizer group, picked up automatically by `CONTROL_TENSOR_NAME_PATTERNS` for fp32 retention. Everything else is unchanged from the baseline.

Toggle: `ACN_OUTPUT=1` enables it. Default is off, so this is a strict superset of the official baseline.

## Results (1 seed, 8xH100 SXM, 600s wallclock cap)

Both runs use the official baseline config: 9 layers × 512d × vocab 1024, GQA 8/4, 2x MLP, tied embeddings, seed 1337, full 80 training shards.

### At equal step counts (the per-step picture)

| step | val_bpb (ACN off) | val_bpb (ACN on) | Δ |
|----:|---:|---:|---:|
| 1000  | 1.3834 | 1.3825 | −0.0009 |
| 5000  | 1.2746 | 1.2747 | +0.0001 |
| 9000  | 1.2536 | 1.2536 |  0.0000 |
| 12000 | 1.2439 | 1.2437 | −0.0002 |
| **13000** | **1.2326** | **1.2264** | **−0.0062** |

The gap opens up late, in the warmdown phase. The output_scales drift away from zero gradually; once they're nonzero the ACN forward starts to actually do something different.

### At equal wallclock (the headline picture)

|                                  | ACN off    | ACN on     |
|----------------------------------|-----------:|-----------:|
| model_params                     | 17,059,912 | 17,059,921 |
| step_avg                         | 43.92 ms   | 44.88 ms   |
| steps reached in 600 s           | 13,662     | 13,370     |
| pre-quant val_bpb (final step)   | 1.2191     | 1.2195     |
| **post-quant val_bpb (final)**   | **1.2262** | **1.2265** |
| compressed artifact (bytes)      | 15,864,344 | 15,875,628 |

ACN is +2.2 % wallclock per step (the per-block accumulator is on the hot path), so it gets 292 fewer training steps in the 600 s budget. The per-step win and the per-step cost roughly cancel.

## What this is and isn't

**What it is.** A single-seed, single-config probe to see whether the ACN paper's idea has any signal on the official baseline. The question was whether routing each block's hidden state directly into the final representation gives the model useful auxiliary supervision at this scale. The answer appears to be "a little, on the per-step axis."

**What it isn't.** A record submission. One seed, no hyperparameter tuning beyond the defaults, naive Python-loop accumulator with no kernel work. The 0.006-nat per-step gap is plausibly real (it is well above the ~0.001 jitter typically reported across seeds at this size in other PRs), but a single seed cannot statistically establish that on its own.

## Why I'm submitting this anyway

Three reasons, in order of honesty:

1. The README explicitly invites "interesting negative results" and "in-progress or unoptimized solutions" in the non-record track. This fits.
2. The diff is so small (16 lines, one env var) that anyone curious can run their own A/B in 20 minutes. Lowering the activation energy for someone with more compute to multi-seed this seems worth a few hundred lines of writeup.
3. The wallclock tax is 100 % implementation cost: nine `x = x + s * h` calls in Python in the hot loop. A fused kernel, or even just collapsing the accumulator into a single `einsum`, plausibly recovers the per-step gain at zero wallclock cost. That's the natural follow-up I'd run if I had the budget.

## Reproducing

From the official repo, on the `acn-baseline` branch of [chrico-bu-uab/parameter-golf](https://github.com/chrico-bu-uab/parameter-golf/tree/acn-baseline) (or just take the `train_gpt.py` in this folder):

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024

# ACN off — should reproduce the baseline ~1.2244 bpb
RUN_ID=baseline_off VOCAB_SIZE=1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee baseline_off.log

# ACN on
RUN_ID=baseline_acn VOCAB_SIZE=1024 ACN_OUTPUT=1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee baseline_acn.log
```

Seed 1337 (default), 80 shards, 600 s wallclock cap. Logs from this submission are in `baseline_off.log` and `baseline_acn.log`.

## What would change my mind / suggested follow-ups

- **More seeds.** 3-seed pairs at minimum. The 0.006-nat per-step gap is plausibly real but unproven at one seed.
- **No-cap runs.** Run both to the full 20 k iterations to see whether the gap persists, grows, or collapses past where the 600 s cap currently cuts off.
- **Kill the wallclock tax.** Replace the per-layer accumulator loop with a single fused operation (or skip layers whose `output_scale` is below some threshold). If the per-step win survives a near-zero-cost implementation, ACN becomes a free lunch on the baseline.
- **Stack on top of a competitive run.** This was tested on the naive baseline only. Whether the auxiliary-supervision benefit holds on top of SP8192 + parallel residuals + TTT (where the model is already heavily over-supervised) is open.
