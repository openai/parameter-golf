# Winning PRs #657 And #693 Breakdown

This note complements [winning_pr_639_breakdown.md]( nanoevolve/pgolf/parameter-golf/stage2_1/winning_pr_639_breakdown.md).

The goal is not to repeat leaderboard summaries. The goal is to isolate:

- what each PR actually changed
- what it inherited from an earlier strong stack
- what mechanism most plausibly lowered `val_bpb`

## PR #657

Upstream PR:
- [#657](https://github.com/openai/parameter-golf/pull/657)

Headline:
- `1.1229` 3-seed mean

### What It Actually Did

`#657` is not a fresh-from-scratch design. It is a cleaned-up, stronger derivative of the `#414` family.

The stack described in the PR is:

- `11L`, `512d`, `8/4` GQA, U-Net skips
- `MLP 3x`
- `LeakyReLU(0.5)^2`
- `XSA4`
- `Partial RoPE 16/64`
- `LN Scale`
- `VRL`
- `BigramHash 2048`
- `VE128`
- `SmearGate`
- `EMA(0.997)` + tight SWA
- late QAT
- Muon WD `0.04`
- warmdown `3500`
- GPTQ-lite int6
- `lzma`
- FA3 on Hopper

### What It Inherited

The core point is that `#657` did not discover a new first-order deployment path.

It mostly inherited:

- the `#414` architecture family
- the strong late-QAT / EMA / warmdown recipe
- the known architecture-side refinements already circulating in frontier stacks

So the meaningful new deltas are narrower than the full PR summary makes them sound.

### What Probably Mattered Most

#### 1. LeakyReLU(0.5)^2

This looks like a real, but modest, modeling gain.

Why it helps:
- avoids hard ReLU dead zones
- preserves negative gradient flow through the squared MLP activation
- especially useful in small models where dead channels are expensive

My read:
- real gain, but not the whole story
- probably worth a couple of thousandths, not a tenth

#### 2. VRL

This is the most interesting new modeling mechanism in `#657`.

Why it helps:
- value paths often degrade into over-local, over-concentrated attention
- re-injecting layer-0 value information into later layers gives the network a persistent low-level value reference
- that is a stronger causal change than another scalar schedule tweak

My read:
- VRL is the part of `#657` most worth taking seriously as a modeling hypothesis
- it is structurally different from the stage2_1 helper set

#### 3. lzma

This is a budget reallocation mechanism, not a direct modeling win.

Why it helps:
- tighter compression buys artifact headroom
- that headroom can be spent on restoring `MLP 3x`, `BigramHash 2048`, or other capacity helpers

My read:
- `lzma` matters because it lets them keep capacity that would otherwise be cut
- the causal win is “keep more model under the cap,” not “lzma improves loss”

### Why #657 Gets To ~1.123

From first principles, `#657` gets low because it is a dense frontier stack with:

- an already-strong architecture family
- a small but real activation improvement
- a more interesting value-path mechanism (`VRL`)
- enough compression headroom to avoid cutting useful capacity

It is less about a revolutionary export path than `#639`.
It is more about a better balanced train-time architecture under the artifact budget.

### Main Lesson From #657

If you compress it to one sentence:

- `#657` wins by keeping a strong architecture-rich stack under the 16MB cap and adding one real structural improvement (`VRL`) instead of just another optimizer-side helper.

## PR #693

Upstream PR:
- [#693](https://github.com/openai/parameter-golf/pull/693)

Headline:
- `1.1186` 3-seed mean

### What It Actually Did

The stack described in the PR is:

- `11L`, `512d`, `8/4` GQA
- `MLP 3x LeakyReLU(0.5)^2`
- `XSA` on the last `4` layers
- `VRL`
- `BigramHash 3072`
- `SWA/EMA 50/50`
- pure inference eval, no TTT
- sliding window eval `stride=64`
- full Cholesky GPTQ
- new `CROWN-Q` warmdown penalty

### What It Inherited

Like `#639`, this PR is already operating in the “strong base + strong export” regime.

It does not start from a simple baseline. It starts from a stack where:

- architecture is already frontier-grade
- sliding eval is already part of the scoring path
- export quality is already first-order

### What Probably Mattered Most

#### 1. Full GPTQ

Still the biggest practical mechanism.

Why it helps:
- preserves the trained model through int6 export
- directly attacks the deployed-score bottleneck

My read:
- full GPTQ is still the main reason this stack can live in the `1.11x` band
- without it, the rest of the stack would lose too much during export

#### 2. CROWN-Q

This is the most interesting new idea in `#693`.

Mechanism described in the PR:
- during warmdown, apply a rowwise penalty proportional to quantization step size
- intentionally over-penalize relative to the actual int6 step
- push weights into flatter, more quantization-tolerant basins

Why it helps:
- it internalizes deployment damage into late training
- unlike ordinary QAT, it is explicitly trying to reduce quantization sensitivity before export

My read:
- this is exactly the type of mechanism stage2_1 was missing
- it is not a local helper
- it is a train-to-deploy bridge

#### 3. SWA/EMA Blend

Again, this is a deployment helper.

Why it helps:
- smoother late weights quantize better
- the blend stabilizes the final export target

My read:
- important in combination with GPTQ and CROWN-Q
- not likely the lead mechanism by itself

#### 4. No TTT

The PR explicitly says TTT was removed because it hurt quantized models.

Why this matters:
- it confirms that once the deployed artifact is strong enough, TTT can be neutral or harmful
- that is strong evidence against over-crediting TTT in frontier stacks

My read:
- the negative result is as informative as the positive mechanisms
- it reinforces that the game is about the deployed artifact first

### Why #693 Gets To ~1.1186

From first principles, the score is low because the stack aligns the whole late pipeline to the deployed objective:

- strong architecture
- sliding eval
- full GPTQ
- flatter late basin via CROWN-Q
- smoothed late weights via SWA/EMA

That is a very coherent story.

It is not:
- "one more helper"
- "one more optimizer tweak"
- "more throughput"

It is:
- late training explicitly shaped for quantization robustness
- then exported with a high-quality quantizer

### Main Lesson From #693

If you compress it to one sentence:

- `#693` wins because it turns quantization robustness into a late training objective instead of treating export loss as an afterthought.

## Cross-PR Takeaways

Across `#639`, `#657`, and `#693`, the consistent themes are:

- strong base architecture first
- deployment loss treated as first-order
- late smoothing matters
- artifact budget is actively managed
- tiny helper patches are not enough on their own

The real differences are:

- `#639`: strongest emphasis on full GPTQ + XSA-all + compliant deployment path
- `#657`: strongest emphasis on architecture richness under the cap, especially `VRL`
- `#693`: strongest emphasis on internalizing quantization robustness during warmdown

## What Matters For Us

The useful lesson is not “copy all three.”

The useful lesson is:

- if we want a real next-stage hypothesis, it should look more like `VRL` or `CROWN-Q` than like another small training helper
- if we want a real next-stage win, it should probably attack late deployability or structural value/context flow, not just early training dynamics
