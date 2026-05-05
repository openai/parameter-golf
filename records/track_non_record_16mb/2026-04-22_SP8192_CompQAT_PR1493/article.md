# OpenAI Parameter Golf — what I built, what worked, what didn't

OpenAI Parameter Golf is a challenge that started in March 2026. The goal is simple: train a language model that fits in 16 MB and runs training in 10 minutes on 8×H100 GPUs. The model is evaluated by how well it compresses FineWeb validation text. Lower bits-per-byte (BPB) wins.

The current record is [PR #1493](https://github.com/openai/parameter-golf/pull/1493) by "bigbag" at 1.0810 BPB. It uses a stack of techniques the community built over a few weeks: depth recurrence, parallel residuals, score-first test-time training, GPTQ with SD-clip, and code obfuscation to save artifact bytes.

I worked on this challenge in two phases, about 4 weeks total.

**Phase 1 — 3DCF research.** About 3 weeks, ~$40 spent, 21 cloud experiments. No leaderboard result. But it produced a clean negative finding and a principle that shaped everything I did after.

**Phase 2 — 5-day sprint on top of [PR #1493](https://github.com/openai/parameter-golf/pull/1493).** Three ideas, $43 of fresh credits, actual spend $51.50. Two of the ideas were direct responses to lessons from Phase 1.

**Total across both phases: ~$91.50.**

In this article, I will describe what I built, what the numbers showed, and what I'd try next. I'll start with 3DCF, because the rest of my work only makes sense after you see why it failed.

---

## The three pieces of work

1. **3DCF research** (phase 1, ~3 weeks, ~$40) — Rust quantization toolkit, 5 quantization methods tested, 21 cloud experiments. Clean negative result. Produced the principle that led to comp-QAT.
2. **Compression-aware QAT** (phase 2, 5 days, ~$51) — train the model to produce weights that compress well, instead of treating quantization and compression as separate steps.
3. **LoRA-TTT** (phase 2) — low-rank adapters for test-time training. 90K trainable parameters per chunk instead of 36M.

Two of three got tested at 8×H100 scale. One produced a measurable compressibility signal in pilot ablations. None moved the leaderboard.

## What I built on top of

[PR #1493](https://github.com/openai/parameter-golf/pull/1493) is not one idea. It is a stack of contributions. I kept each one intact and added my work as extensions.

The integration is additive — there's an identity test in my repo that strips my extensions from the integrated script and asserts the MD5 matches the unmodified [PR #1493](https://github.com/openai/parameter-golf/pull/1493) source byte-for-byte.

Here is what each contribution gave me:

**[PR #1493 (bigbag)](https://github.com/openai/parameter-golf/pull/1493) — the 1.0810 record and my base.**
I kept the LZMA obfuscation wrapper. Without it, my extensions would not fit under 16 MB. I added: comp-QAT loss in the training step, native LoRA-TTT eval alongside the existing full-param TTT, and a re-obfuscation pipeline that rebuilds the shipped `train_gpt.py` before every cloud run.

**[PR #1394 (@clarkkev)](https://github.com/openai/parameter-golf/pull/1394) — SP8192 tokenizer + GPTQ with SD-clip.**
SP8192 is dominant across top submissions. SD-clip is a per-row clipper: `clip = k × std(row)`. I tested my own Rust GPTQ variant against it — mine was worse on int6 grids, so I kept [PR #1394](https://github.com/openai/parameter-golf/pull/1394) exactly. Useful side-effect: comp-QAT concentrates weights toward grid centers during training, which gives SD-clip fewer outliers to shave off at quantization time. They pair well, basically by accident.

**[PR #1331 (@dexhunter)](https://github.com/openai/parameter-golf/pull/1331) — depth recurrence.**
Free depth. Loop layers 3–5 three times and you get 17 virtual layers from 11 physical ones. I considered pairing it with per-pass LoRA adapters (different adapters per virtual visit) — the code is in the `universal_ttt/` module but I ran out of budget for the 8×H100 test.

**[PR #549 (@abaybektursun)](https://github.com/openai/parameter-golf/pull/549) — score-first test-time training.**
The only TTT protocol the competition accepts as legal. Score under `no_grad`, then train, then move to the next chunk, never re-score. My LoRA-TTT variant is the same protocol, with AdamW on 90K adapter parameters instead of SGD on 36M base parameters.

**[PR #1412 (@Robby955)](https://github.com/openai/parameter-golf/pull/1412) — parallel residuals.**
GPT-J-style parallel attention/MLP from layer 7. I didn't modify it. One observation worth writing down: the MLP projections are the largest weight matrices (~2M params each), so they dominate the comp-QAT surrogate loss. If someone applies the surrogate selectively, that's where to start.

---

## Innovation 3 (out of order): the 3DCF research

I am starting with this because everything else in the article depends on what it taught me.

### Where the code came from

Before Parameter Golf, I was working on [doc2dataset](https://github.com/3DCF-Labs/doc2dataset), a Rust toolkit in my [3DCF-Labs](https://github.com/3DCF-Labs) org. Different problem — doc2dataset is about getting documents into models, not about compressing the models themselves.

When I started Parameter Golf, I needed Rust-native weight quantization fast enough to run inside a QAT training loop. Starting from scratch would take weeks. So I took doc2dataset's Rust + pyo3 foundation (the build setup, the binding patterns, a few low-level utilities) and started a new sibling crate on top of it for neural-network weight compression.

That crate is public at [github.com/3DCF-Labs/model-compress](https://github.com/3DCF-Labs/model-compress), MIT-licensed.

The infrastructure came from doc2dataset. Everything quantization-specific is new and was written as the Parameter Golf experiments demanded it.

### The original hypothesis

[PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s artifact is 16 MB. Most of that is weights: 36M parameters quantized to int6, stored in int8 bytes (one byte per weight, 2 bits wasted per byte), then brotli-compressed.

Obvious optimization: pack the int6 values at 6 bits per value instead of 8. Save 25% raw bytes before compression. With that 25% saved, you could fit a larger model — 14 or 15 layers instead of 11 — and the extra parameters might overcome the small precision loss.

This was the hypothesis. It seemed obviously right.

### What I built

I picked Rust instead of Python because the int6 pack/unpack operations need to be fast. They run at every QAT training step.

The crate ended up implementing every quantization method I thought might be relevant:

| Module | Lines | What it does |
|---|---|---|
| `quantize.rs` | ~100 | Symmetric int6/int8 scalar quantization |
| `lloyd_max.rs` | 230 | Lloyd-Max optimal codebook (iterative MSE-minimal) |
| `residual_quant.rs` | 85 | Multi-stage residual codebooks |
| `product_quant.rs` | 155 | Product quantization (independent sub-vector codebooks) |
| `gptq.rs` | 297 | GPTQ Hessian-based rounding |
| `compress.rs` | ~80 | zstd wrapper with bit-packing |
| `lib.rs` | ~150 | Python bindings via pyo3 |

Plus 4 iterations of the export pipeline, a full training script integration (`train_gpt_3dcf.py`, 639 lines), cloud orchestration scripts, and tests.

About 3 weeks of work. 21 cloud experiments. Before the [PR #1493](https://github.com/openai/parameter-golf/pull/1493) campaign even started.

### First failure: low MSE does not mean low BPB

The first thing I tried was not packing. It was the codebook methods — Lloyd-Max, Residual Quantization, Product Quantization. The theory says codebooks should beat scalar int8 because they are more flexible.

**Run 5** was the MSE diagnostic. Same trained model, different quantization methods:

| Method | MSE | vs int8 |
|---|---|---|
| Residual Quant 6+2 | 0.0000010 | **2× better** |
| int8 | 0.0000020 | baseline |
| Lloyd-Max | 0.0000028 | 1.4× worse |
| int6 | 0.0000054 | 2.7× worse |
| Product Quant 2×256 | 0.0000134 | 6.8× worse |

Residual Quantization had half the MSE of int8. This looked like a silver bullet.

**Run 6**: same RQ-quantized model, full evaluation. BPB came back at 3.35. Baseline was 1.59. Much worse.

I tried different RQ configurations. Different codebook sizes. Different residual stages. Same result every time. Lloyd-Max gave the same kind of failure. Product Quant was bad from the start.

The mechanism, once I figured it out: codebook methods produce **correlated quantization errors**. Two weights with originally different values both get mapped to the same codebook entry. From the network's perspective, those two weights were always identical. The network's expressiveness depends on being able to make different weights do different things, so this is catastrophic.

Per-weight scalar quantization (int8, int6) produces **uncorrelated** noise. Small random jitter per weight. The network can absorb it.

Codebook quantization produces **structured** noise. The network cannot recover from it, regardless of how small the MSE looks.

**Lesson 1: MSE is not the right metric for quantization in language models. Two methods with identical MSE can have completely different BPB if one produces correlated errors.**

This cost about $5 of cloud to discover. Killed the codebook direction entirely. The Rust code is still in the repo — over 700 lines of Lloyd-Max + RQ + PQ implementations — as a record of what does not work and why.

### Pivot to bit-packed int6

Codebooks are out. Plain int6 with denser storage is the remaining handle. 6 bits per value instead of 8. 25% raw savings could fund a 14-layer model in the 11-layer budget.

I built the bit-packing infrastructure (the "3DCF sym int6" 348-line path), wrote the QAT integration, and ran it at scale.

**Run 18** (4×H100, 600s, 11L):

| Metric | Value |
|---|---|
| Steps | 4,837 |
| Pre-quant BPB | 1.2101 |
| int8+zstd roundtrip | 1.2119 |
| **3DCF int6 roundtrip** | **1.2318 (+0.020 worse)** |

+0.020 BPB gap. Looked broken.

A day of debugging. The cause was a **QAT STE mismatch**. My training-time quantization noise simulator used a sqrt-companding asymmetric mapping. The actual export used uniform symmetric int6. The model was being trained against one quantization and shipped with another.

**Run 19** fixed the STE (uniform symmetric [−31, +31] in both places). Gap collapsed to +0.002. 3DCF was now nearly identical to int8 on BPB.

Looked promising.

### Run 20 — the measurement that killed 3DCF

The whole point of 3DCF was to fit a larger model. So Run 20 tested 11L, 14L, 15L side by side:

| Config | Params | int8 BPB | int8 size | 3DCF BPB | 3DCF size |
|---|---|---|---|---|---|
| 11L | 26.5M | 1.2892 | 17.5 MB | 1.2913 | **19.4 MB (+11%)** |
| 14L | 33.6M | 1.3071 | 20.5 MB | 1.3089 | **24.1 MB (+18%)** |
| 15L | 35.9M | 1.3170 | 21.3 MB | 1.3189 | **25.5 MB (+20%)** |

3DCF with bit-packed int6 made the artifact **11–20% larger** than int8-stored int6, after compression. And it got worse as the model grew — which was supposed to be the whole point of packing.

The mechanism is information-theoretic. zstd (and brotli, and every other practical compressor) exploits byte-level redundancy. When you store int6 values in int8 bytes, only 63 of 256 byte values are ever used, clustered near zero. Wildly redundant. Typical compression ratio: 0.65.

When you pack densely, you spend that redundancy for compactness. The packed bytes are close to uniform over all 256 values. Compressor has nothing to consume. Typical ratio: 0.97.

So the 25% raw savings from packing got entirely eaten by 32% worse compression. Net: bigger artifact, plus the larger models trained 28% fewer steps in the same wallclock, so BPB was also slightly worse.

**Lesson 2: byte-level redundancy that an entropy coder can exploit beats packing density. Quantization compactness and post-compression size are in tension, not aligned. The "wasteful" int8 storage of int6 values is doing actual work — its sparseness is what the compressor consumes.**

This was the death of 3DCF. Not a bug in my code. A fundamental tension between two layers of the storage stack.

About $4.70 on Run 20. $40 total across the 3 weeks and 21 runs. Substantial Rust code now sitting in the repo as a record of "this doesn't work, here is why."

### The pivot to compression-aware QAT

Two days after Run 20 I sat with the data and asked the obvious question. If I can't make the bytes denser via packing, can I make them more compressible via training?

Bit-packing was fighting the compressor. Give it less to work with. The opposite direction is feed the compressor more of what it wants.

What does brotli want? Byte-level redundancy. What weight distributions produce byte-level redundancy after int6 quantization? Distributions where weights cluster near a small number of int6 grid centers. Most weights at 0. Some at ±1. Fewer at ±2. The more concentrated, the more brotli has to chew on.

So instead of packing harder, **train so the post-quantization byte distribution is intrinsically more compressible**.

That's the seed of comp-QAT. It came directly from understanding why 3DCF failed.

3DCF was a clean negative result with a generalizable principle. It was also the lesson that produced the next idea. Both outcomes are real, even though neither moved the leaderboard.

---

## Innovation 1: Compression-aware QAT

### The mechanism

Compressors exploit byte-level redundancy. If quantized weights cluster near a small number of int6 grid centers, the byte stream is predictable and compresses well. If they spread uniformly, the bytes look random and compression fails.

Problem: actual zstd is not differentiable. No useful gradient.

My solution: a soft-assignment surrogate.

```python
def compression_surrogate(W, n_levels=63, beta=10.0):
    scale = W.abs().quantile(0.999).detach()
    normalized = W / (scale + 1e-8) * 31.0
    levels = torch.arange(-31, 32, device=W.device, dtype=torch.float32)
    soft_assign = torch.softmax(-beta * (normalized.unsqueeze(-1) - levels).pow(2), dim=-1)
    hist = soft_assign.reshape(-1, n_levels).mean(dim=0)
    return -(hist * (hist + 1e-10).log()).sum()  # Shannon entropy
```

Low entropy → weights cluster near a few int6 levels → better compression. High entropy → weights spread uniformly → worse compression. The whole thing is differentiable.

Added to training as `loss_total = loss_CE + λ · surrogate_entropy`.

### Does the surrogate actually predict compression?

Before any cloud spending, I tested it locally on 6 synthetic weight distributions: tight Gaussian, wide Gaussian, uniform, bimodal, concentrated near zero, sparse.

For each distribution I measured my surrogate and the actual zstd compression ratio.

**Pearson correlation: +0.994.**

The surrogate basically IS the compressibility signal. If the correlation had come back at 0.5, I would have killed the idea right there. (Lesson learned from 3DCF: do not burn cloud dollars on something that has not passed the cheap local sanity check.) It came back 0.994, and I went forward.

### What went wrong at 1×H100

I ran a λ sweep at 1×H100 to pick a value for the main 8×H100 run.

| λ | Train steps | Weights blob (B) | Δ vs baseline |
|---|---|---|---|
| 0 (baseline) | 589 | 15,989,490 | — |
| 0.0003 | 426 | 15,992,937 | **+3,447** |
| 0.001 | 427 | 15,993,109 | **+3,619** |
| 0.003 | 427 | 15,989,969 | **+479** |

Artifact got larger, not smaller. Opposite of what the mechanism is supposed to do.

I stared at this for a day. First hypothesis: surrogate is wrong. Re-read the math. Nothing wrong.

Then I looked at the training step counts. Baseline did 589 steps in 600 seconds. My comp-QAT runs did 426–427. The extra forward/backward pass through the surrogate adds 28% per-step overhead.

At 1×H100 I was already compute-starved. The model was 28% less converged than baseline. Its weight distribution was still chaotic from undertraining. The surrogate gradient was pulling weights toward grid centers, but the grid centers were moving because training was not done yet. The brotli ratio looked worse on the noisy undertrained weights, even though the surrogate was decreasing.

**Lesson: at 1×H100, the signal I was measuring was undertraining, not comp-QAT. The 1×H100 proxy is not valid for any mechanism with training-time compute overhead.**

### What happened at 8×H100

At 8×H100 with the same 600-second budget:

- [PR #1493](https://github.com/openai/parameter-golf/pull/1493) baseline: 4,550 training steps
- My comp-QAT (λ=0.001): 2,652 training steps

42% fewer steps (torch.compile warmup adds extra overhead on top of the 28%). But 2,652 is well into convergence.

Results:

| Metric | [PR #1493](https://github.com/openai/parameter-golf/pull/1493) baseline | Mine (λ=0.001, single-seed pilot) | Δ |
|---|---|---|---|
| val_bpb (TTT) | 1.08079 | 1.10326 | +0.0225 |

This pilot run suggested the mechanism can move compressibility in the intended direction, but the final submitted 3-seed package was still larger on mean artifact size and worse on BPB than public [PR #1493](https://github.com/openai/parameter-golf/pull/1493). The useful result is the measured tradeoff: the mechanism may help compressibility, but the per-step overhead still dominates the final submission outcome.

The surrogate entropy decreased monotonically during training:
- Step 250 (post-warmup): 3.566
- Step 2652 (end): ~3.41
- Max possible: ln(63) ≈ 4.14

Starting at 86% of max entropy, ending at 82%. Weights concentrated toward int6 grid centers as designed.

The same mechanism that 3DCF could not reach with packing tricks, comp-QAT reached by directly training for the property the compressor wants.

### The per-seed variance finding

I ran 4 seeds total. Artifact sizes:

| Seed | Total artifact (B) | Margin under 16 MB |
|---|---|---|
| 1337 | 15,991,258 | 8,742 |
| 1338 | 15,996,195 | 3,805 |
| 1339 | 15,999,417 | **583 ⚠️** |
| 1340 | 15,994,396 | 5,604 |

Range across seeds: 8,159 bytes.

[PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s 3-seed artifact range: 1,302 bytes.

**Comp-QAT amplifies per-seed artifact variance by 6×.**

Seed 1339 had only 583 bytes of margin under the 16 MB hard cap. A different seed could have pushed the artifact over the limit and failed verification entirely.

Why does this happen? My best guess: the surrogate pushes weights toward low-entropy configurations, but different seeds land in different low-entropy basins. The basins have different post-compression sizes. Without comp-QAT, the compression ratio is dominated by training-data statistics (similar across seeds). With comp-QAT, it is also affected by which particular basin each seed finds.

**Lesson: if you use compression-aware objectives near the 16 MB cap, leave extra margin, run more seeds than usual, or apply the objective selectively.**

### BPB variance got tighter

At the same time, BPB variance across seeds got tighter:

- [PR #1493](https://github.com/openai/parameter-golf/pull/1493) 3-seed σ: 0.00020
- Mine 4-seed σ: 0.00009

**Half the BPB variance of the current record.**

I do not have a clean explanation for this. Best guess: the surrogate acts as a mild regularizer, making different seeds converge to functionally similar solutions (same BPB) via structurally different weights (different artifacts).

Opposite of the artifact variance direction. Same mechanism, opposite effect on the two metrics. Needs proper analysis. I did not have budget for it.

### The +0.022 BPB gap

Final 3-seed mean: 1.10314. [PR #1493](https://github.com/openai/parameter-golf/pull/1493): 1.0810. Gap: +0.022 BPB.

Most of this gap is not from the regularization. It is from the 42% training-step reduction. Comp-QAT training overhead is the real cost, not the regularization quality.

What would reduce the gap:
- Lower λ (smaller overhead at the same effect)
- Compute surrogate every Nth step instead of every step
- Apply surrogate only to largest weight matrices (MLP fc/proj contain >80% of weight bytes)
- More compute budget to train longer

None of these shift my submission. They are ideas for someone extending the work.

### The lambda that shouldn't have happened

Small story worth including. I chose λ=0.003 after Phase 3 because that was where the artifact-size signal overcame undertraining noise at 1×H100. My Phase 4 bash script had this line:

```bash
export COMP_LAMBDA="${PHASE4_COMP_LAMBDA:-0.001}"
```

Default is 0.001. Override is `PHASE4_COMP_LAMBDA`, which I never set. So Phase 4 ran with λ=0.001, not 0.003. I noticed when the artifact saved as `phase4_pilot_seed1337_lambda_0.001.int6.ptz`.

The accidental λ=0.001 turned out to be the better choice on both axes. Locked it in for Phase 5.

I would love to say I was smart. What actually happened is a bash default was more correct than my deliberate decision.

---

## Innovation 2: LoRA-TTT

### The question

[PR #1493](https://github.com/openai/parameter-golf/pull/1493) uses score-first test-time training. At eval time, the model fine-tunes itself on each chunk: score, then train the full model on those tokens with SGD, then move to the next chunk.

It works — about −0.002 BPB improvement at 8×H100. But it is expensive. 370 seconds for TTT eval vs 120 seconds for sliding-window baseline.

Question: do I need to update all 36M parameters to get this lift? Or can low-rank adapters do the same with a fraction of the compute?

### The bug that cost me $1.50

My first Phase 2 integration crashed on the cloud. The wrapper called an existing `lora_ttt_eval` function with:

- Wrong positional argument order (4 arguments out of place)
- Wrong keyword names (`rank` instead of `lora_rank`, `epochs` instead of `ttt_steps`)
- A made-up argument (`freeze_first`) that does not exist
- Wrong return type expectation (`(val_loss, val_bpb)` instead of `(total_loss, num_tokens)`)

Unit tests passed because they only tested the `LoRAAdaptor` class, not the wrapper function. Wrapper crashed only at runtime. $1.50 of GPU time already spent.

**Lesson: unit tests on components are necessary but not sufficient. Integration glue needs exec-level tests too.**

I wrote `tests/test_lora_ttt_integration.py` after the fact. Runs the wrapper end-to-end on a CPU mini-model. Takes 5 seconds. Catches exactly this class of bug. Should have existed on day one.

### The inference_mode bug — a real PyTorch footgun

After fixing the wrapper, I hit a second bug that took another hour to find.

The score-first protocol needs:
1. Forward pass without gradients (score phase)
2. Normal forward + backward (train phase)

The natural PyTorch idiom for "no gradients" is `torch.inference_mode()`. I used it. It silently broke everything.

Error message: `Cannot set version_counter for inference tensor.`

What is actually happening: `torch.inference_mode()` marks tensors created inside it with a flag that makes them incompatible with autograd forever. The rotary positional embedding cache in [PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s model is a buffer that gets populated on first forward pass. If that first forward pass runs inside `inference_mode()`, the cache gets poisoned. Every subsequent forward pass that uses the cache inherits the poison. Backward pass fails.

The fix is one character: `torch.no_grad()` instead of `torch.inference_mode()`. Both disable gradient tracking. Both preserve score-first legality (weights cannot mutate during scoring). Only `no_grad()` does not poison the rotary cache.

This bug is not specific to Parameter Golf. Anyone implementing score-first TTT in modern PyTorch with a rotary cache will hit this.

### What I didn't test

Here is the honest part. I integrated LoRA-TTT into the training script. I caught the bug. I wrote the integration test. I did NOT compare LoRA-TTT vs full-param TTT at 8×H100.

My Phase 4 and Phase 5 runs all used `TTT_ENABLED=1, LORA_TTT_ENABLED=0` — [PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s full-param TTT. I wanted cross-recipe consistency across the 4 seeds.

The question I set out to answer — does LoRA-TTT beat full-param TTT at frontier scale? — is unanswered. I have the code. I have tests proving the code works. I do not have a measured 8×H100 head-to-head. That comparison is ~$10 of future work.

Earlier evidence: on a weaker SP1024 base, LoRA-TTT gave −0.034 BPB lift. Stronger signal than full-param at that scale. But SP1024 is not SP8192, and 1×H100 is not 8×H100.

---

## Cost ledger

### 3DCF research (~3 weeks before this campaign)

| Stage | Spend |
|---|---|
| Runs 1–9 (smoke tests, codebook diagnostics) | ~$10 |
| Runs 10–17 (sym int6 validation, scaling tests) | ~$10 |
| Run 18 (4×H100 full run, found STE mismatch) | $6.83 |
| Run 19 (STE fix sweep) | $3.50 |
| Run 20 (the killer measurement) | $4.70 |
| Run 21 partial (universal transformer attempt) | ~$3 |
| **3DCF subtotal** | **~$40** |

### [PR #1493](https://github.com/openai/parameter-golf/pull/1493) + comp-QAT campaign (5 days)

| Phase | Budgeted | Actual |
|---|---|---|
| Phase 2 (integration validation, 1×H100) | $1.78 | $4.98 |
| Phase 3 (λ sweep, 1×H100) | $5.34 | $6.45 |
| Phase 4 (8×H100 pilot) | $11 | $13.16 |
| Phase 5 (3 seeds, 8×H100) | $28 | $26.71 |
| Volumes | $0.20 | $0.20 |
| **Campaign subtotal** | **$46.32** | **$51.50** |

**Total across both phases: ~$91.50.**

The 5-day campaign was $8.50 over target. Driven by:
- 8×H100 SXM stock concentrated in a data center without persistent volumes (dataset re-downloads on expensive GPU time)
- Eval timing on 1×H100 was 6× slower than 8×H100 because evals do not parallelize on 1 GPU
- Two small integration bugs caught only at cloud runtime

---

## Final result

3-seed mean val_bpb: **1.10314**, σ=0.00009.

Not a leaderboard record. [PR #1493](https://github.com/openai/parameter-golf/pull/1493) is 1.0810. My submission is +0.022 above the SOTA threshold.

What I actually produced:

1. **Compression-aware QAT** — working mechanism at frontier scale. In a same-script pilot it reduced artifact size relative to the PR #1493 reference, but the final 3-seed submission was still larger on mean artifact size and worse on BPB. Artifact variance per seed increased, BPB variance tightened, and the net score cost remained dominated by training overhead.

2. **LoRA-TTT** — integrated, tested, inference_mode bug fixed. Head-to-head comparison at 8×H100 not run (budget scope).

3. **3DCF research** — 3 weeks of Rust quantization infrastructure. 5 quantization methods tested. Two clean negative results: (a) MSE is not BPB — codebook methods produce correlated errors that destroy in-context computation, and (b) bit-packing is in tension with entropy coding — packed bytes lose more compressibility than they save in raw size. The second finding is what produced the comp-QAT idea.

Three non-record contributions. The challenge has an explicit non-record track for this kind of thing.

---

## What I'd try next with more compute

1. **Comp-QAT at λ=0.0001 on 8×H100.** 1×H100 made small λ look broken, but that was undertraining noise. Full compute might give artifact shrinking with less BPB cost. Most promising unexplored direction.

2. **Surrogate computed every 10th step instead of every step.** Would cut the 28% overhead down to ~3%. Weights drift slowly. The surrogate does not need to be re-measured every gradient step.

3. **Surrogate applied only to MLP fc/proj layers.** They contain >80% of total weight bytes. Attention layers are ~10% each. Skipping attention saves compute with minimal effect on artifact size.

4. **LoRA-TTT vs full-param TTT A/B at 8×H100.** Code is ready. ~$10 of pilot time tells you which wins at frontier scale.

5. **Formal characterization of per-seed artifact variance.** 6 seeds with comp-QAT, 6 without, compute variance of post-brotli size. If the 6× amplification is real and consistent, that is a paper on its own.

6. **Try the idea 3DCF was supposed to produce, but with comp-QAT instead.** Larger model (14L or 15L) with comp-QAT pushing weights toward extra-compressible distributions. The 25% bytes that bit-packing tried to free for extra parameters might be reachable through training-time compression instead. Risk: high. Reward: this is the path to actually beating [PR #1493](https://github.com/openai/parameter-golf/pull/1493).

---

## Closing

3DCF did not produce a leaderboard contribution. It produced a clean negative result, a generalizable principle (compression and packing are in tension), and the seed of comp-QAT. Without those 3 weeks of failed experiments, comp-QAT would not exist.

Comp-QAT did not produce a leaderboard record. It produced a working mechanism at frontier scale, four-seed reproducibility tighter than the current record, and a measurable artifact-size effect. And a clean answer to "what does this mechanism cost?" — the answer is "compute overhead, not regularization quality." That is the kind of answer that tells the next person what to fix.

LoRA-TTT did not produce the head-to-head comparison I wanted. It produced a fully integrated implementation, an inference_mode bug fix that is general-purpose, and tests that prove the integration is correct.

What I did worked. Just not well enough to win the leaderboard. The mechanisms are real. The measurements are clean. The reproduction instructions are complete. Someone else can pick it up from here instead of starting from zero.

That is what I set out to do. Not what I hoped to do.

---

## Reproduction

- **Parameter Golf research repo**: [github.com/yevh/parameter-golf](https://github.com/yevh/parameter-golf/tree/compqat-sp8192-submission)
- **Quantization crate**: [github.com/3DCF-Labs/model-compress](https://github.com/3DCF-Labs/model-compress) — MIT-licensed
- **Sibling project** (doc2dataset, source of the Rust infrastructure): [github.com/3DCF-Labs/doc2dataset](https://github.com/3DCF-Labs/doc2dataset)
- **Submission PR**: [github.com/openai/parameter-golf/pull/1805](https://github.com/openai/parameter-golf/pull/1805)
- **Reproduction recipe**: `REPRODUCE.md` in the research repo
- **Per-phase results**: `phase{2,3,4,5}_results.md`

Follow me for the next one.
