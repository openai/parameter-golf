# Parameter Golf Council Brief — March 25, 2026 (Evening)

## Situation Update

We have a full run in progress on 8xH100 with the complete new stack:
- LeakyReLU(0.5)² + VRL + Gated Attention + BigramHash 3072 + CROWN-Q + lzma
- AdamW TTT (PR #688 recipe: lr=1e-4, 9 frozen blocks, Polyak averaging, cosine LR)
- FA3 Hopper (installed via pre-built wheel in 30 seconds!)

Benchmark shows ~106ms/step at step 30, expected to settle to ~87-95ms after torch.compile warmup. Results in ~20 min.

## Key Findings Since Last Brief

### 1. Pod Lottery is MASSIVE
Same GPU SKU (H100 SXM), same template, wildly different speeds:
- US-NE-1 pods: ~87ms/step (our best runs, 1.1229 bpb)
- India pods (some): ~87-106ms/step (usable)
- Japan pods: 260-320ms/step (3-4x slower, unusable)

This means the competition leaderboard is partly a hardware lottery. Whoever gets a fast pod gets ~2,000 more training steps in 10 min.

### 2. FA3 Pre-Built Wheel Works
`pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291`
Installs in 30 seconds. We spent ~$100 and 10+ hours building from source before discovering this. We've published the from-source build as a GitHub release for the community: https://github.com/anthony-maio/openai-parameter-golf-fa3-wheel/releases/tag/v1.0

### 3. Full GPTQ is ILLEGAL (issue #677)
Multiple PRs disqualified for using Hessian-based GPTQ with calibration data during eval. GPTQ-lite (clip search, no calibration data) remains legal. This invalidated the council's previous top recommendation.

### 4. Our VRL is Spreading
- PR #745 (1.0222 bpb) credits us directly for VRL
- ChideraIbe123 adopted our VRL implementation verbatim
- Validates that VRL is a real, composable gain

### 5. New Techniques Researched

**Gated Attention (GA)**: Per-head sigmoid gate on attention output. ~0.002-0.003 bpb gain. 6 lines of code. Stacks additively with VRL. Implemented. (PR #638)

**CROWN-Q**: Curvature-weighted quantization penalty during warmdown. 10 lines. Training-time only (legal). Pushes weights toward flat minima where int6 rounding hurts less. Implemented. (PR #693)

**Hedge Mixer TTT (PR #688)**: 5-expert online ensemble (neural + unigram/bigram/trigram + entropy) using Hedge algorithm. Gets -0.05 bpb combined with AdamW TTT. Key finding: PR #688 uses **AdamW(lr=1e-4)** not SGD(lr=0.002), and only unfreezes **last 2 blocks** (9 frozen). This is likely why our SGD TTT failed (20x higher LR, all blocks unfrozen).

## Questions for the Council

### Q1: PR Strategy — Update or New PR?

Our current PR #657 shows val_bpb=1.1229 (3-seed mean). If the current run succeeds with the new stack (GA + BH3072 + CROWN-Q + AdamW TTT), we'll have a significantly better number. Options:

A) **Update PR #657** with new results (same branch, just push new code + logs). Keeps our timestamp advantage but changes the submission significantly.

B) **Close PR #657 and open a new PR**. Clean slate, clear description of the new stack. But we lose timestamp priority.

C) **Keep PR #657 as-is (non-record) and open a new record PR**. Shows progression. But rules say only one open record PR at a time.

Which is strategically optimal? Does the timestamp matter for the leaderboard?

### Q2: If AdamW TTT Works — Expected Ceiling?

Our previous SGD TTT failed (bpb went UP). The AdamW recipe from PR #688 uses:
- AdamW lr=1e-4 (vs our SGD lr=0.002 — 20x lower)
- 9 frozen blocks (vs 0 — protects VRL gates)
- Polyak averaging (decay=0.998) for scoring stability
- Cosine LR decay across chunks

PR #688 gets -0.05 bpb from their full TTT+mixer stack. Realistically, without the Hedge Mixer, what should we expect from AdamW TTT alone on our base? -0.01? -0.02? -0.05?

### Q3: Hedge Mixer — Should We Implement It?

The Hedge Mixer is ~170 lines, self-contained, operates purely on logits (doesn't touch model weights). It runs 5 "experts" and blends their predictions online:
- Expert 0: Neural model log-softmax
- Expert 1: Unigram frequency table (from scored tokens)
- Expert 2: Bigram P(next|prev) table
- Expert 3: Trigram hash table (64K buckets)
- Expert 4: Entropy regularizer

The n-gram tables are built incrementally from already-scored tokens only. The Hedge algorithm updates expert weights via multiplicative weights (no gradients).

Is this legal under issue #677? The n-gram tables are built from validation tokens that have already been scored — similar to the contested n-gram caching techniques. If it's legal, this could be a massive gain on top of AdamW TTT.

### Q4: What's the Realistic Frontier We Should Target?

Given:
- Merged SOTA: 1.1194 (PR #549)
- Frontier with legal TTT: ~1.10-1.12
- Frontier with Hedge Mixer: ~1.02-1.07 (legality debated)
- N-gram caching frontier: sub-1.0 (legality heavily debated)
- Our current: 1.1229 (no TTT)

Where should we aim? Is 1.10 achievable with legal techniques, or should we target 1.115-1.118 as our realistic ceiling?

### Q5: Competition Meta — Is It Worth Chasing SOTA?

The competition runs until April 30 (5 more weeks). New techniques are appearing daily. PRs are getting disqualified regularly. Is the optimal strategy:
A) Submit our best number now and iterate weekly
B) Go heads-down on implementation and submit one strong PR near the deadline
C) Focus on non-record submissions with novel techniques (custom kernels, depth recurrence) since those get accepted more easily

Our budget is ~$60 remaining. That's ~4 full 3-seed runs.

## Current Run Status
Training on India H100 SXM x8, ~106ms/step benchmark. Full stack with AdamW TTT enabled. Results expected in ~20 min.
