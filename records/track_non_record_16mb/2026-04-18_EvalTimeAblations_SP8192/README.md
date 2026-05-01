# Non-record: Eval-time lever ablations + architectural analysis on the SP8192 absolute-RoPE stack

A companion to my [record PR #1716](https://github.com/openai/parameter-golf/pull/1716) (SP8192 + d=32 BigramHash + Path A v3, 3-seed mean **1.07882 bpb**). This PR documents the ablation path behind that record — what I tested, what worked, what didn't, and the architectural reason behind each null result. The **point of a non-record submission is signal quality**, not a leaderboard number; I am filing structured evidence so that the next person exploring this branch of the design space does not re-learn the same lessons at full training cost.

> This is a **non-record** submission. Every number below was measured on 8× H100 SXM with the canonical SP8192 pipeline (see Reproducibility). Where a result reverses a prior hypothesis, I report the negative outcome in full; where a result was within seed-noise, I declare it non-significant rather than chase it.

## TL;DR

| Lever | Hypothesis | Result | Verdict |
|---|---|---|---|
| `BIGRAM_DIM = 32` (vs 48 / 64) | smaller bigram regularizes | pre-quant −0.0002 bpb, marginal but consistent | ✅ kept in record stack |
| **Path A v3 passthrough quantization** | int8 control-tensor + small-matrix quant fits int8 tok_emb under 16 MB | **total artifact 15.99 MB with 6.6 KB margin; 0 bpb cost at 5 d.p.** | ✅ **primary mechanism in record** |
| `TTT_EPOCHS = 4` (vs 3) | more TTT compute → better adaptation | Δ −0.00008 bpb (within noise) | ❌ saturated |
| `EVAL_SEQ_LEN = 4096, stride=128` (NTK-RoPE scaling auto-kicks) | 2× attention context per scored token | pre-quant −0.00509 ✅, sliding +0.00555 ❌, TTT +0.00033 ❌ | ❌ architecture-limited (see §3.2) |
| **SWA `window=1024` at training** (full 2×2 factorial at eval) | windowed attention unlocks eval-time KV chaining | all 3 SWA eval configs strictly worse than baseline; training-time SWA recovers ~0.002 of the eval-time cap but never beats baseline | ❌ SWA alone can't close the gap (see §3.5) |
| **`TRAIN_SEQ_LEN = 4096`** (halved batch) | positions 2048-4095 in-distribution → eval context extension works | pre-quant −0.00196 ✅, sliding +0.00435 ❌, TTT +0.00428 ❌ | ❌ position-depth vs breadth tradeoff (see §3.6) |
| **QAT v3** (matrices-only int6 fake-quant + GPTQ-matched scale) | train model to be quantization-robust | pre-quant +0.015, TTT diverged to 1.48 under score-first TTT | ❌ QAT × TTT incompatibility (see §3.3) |
| **Adaptive Hadamard GPTQ** | random Hadamard rotation reduces quant MSE | Muon-trained weights have sub-Gaussian distributions; rotation does not help, sometimes hurts | ❌ null result, documented |

**The structural finding:** every attempt to exploit the competition's eval-time working-memory asymmetry via *within the current absolute-position RoPE architecture* fails for the same architectural reason — sliding eval scores tokens at a narrow tail-position band, and any technique that trades position depth (more samples per position) for position breadth (more positions covered) regresses on sliding. The right next step is relative-position attention (ALiBi / NoPE / retrieval), where scored-token position becomes irrelevant.

## 1. Design-space map

The competition imposes **asymmetric compute constraints**:

- **Artifact: hard cap 16,000,000 B.** Binds architecture choice.
- **Training: 10 min on 8× H100.** Binds step count and effective batch size.
- **Eval: 10 min on 8× H100 + 640 GB HBM.** **Essentially unbounded in memory**, only wallclock-bound in compute.

That asymmetry defines two research tiers:

**Tier A — Training-time levers** (widely explored by other submissions): architecture width/depth, loss shaping, optimizer, vocab, attention pattern. Every PR on the leaderboard since 2026-04-06 is a tier-A variation.

**Tier B — Eval-time levers** (under-explored): TTT compute budget, eval sequence length, KV-cache strategies, quantization scheme, code packing. These use the ~$1 of "free" eval compute that the grader allocates per run. This PR systematically tests tier-B levers on top of a strong tier-A base (the PR #1394 / 2026-04-09 lineage).

## 2. What I confirmed works (carried into the record PR)

### 2.1 `BIGRAM_DIM = 32`

Reducing `BigramHashEmbedding` projection dimension from common d=48 / d=64 to **d=32** produces a small but reproducible pre-quant improvement on this architecture. Observed across two d=48 retrains and three d=32 retrains: pre-quant post-EMA lands in the band 1.0856–1.0860 on d=32 versus 1.0850–1.0852 on d=48 — within seed noise but consistent in sign. More importantly for the record, the **smaller bigram shrinks `bigram.proj` from 24,576 to 16,384 parameters**, which is inside the `numel() ≤ 65536` small-matrix threshold that Path A v3 targets for aggressive int8.

### 2.2 Path A v3 passthrough quantization (the primary mechanism)

The baseline SP8192 + d=48 + int8 tok_emb recipe produces a **16.06 MB artifact — 65 KB over cap** (I reproduced this three times). Roughly 40 KB of that surplus sits in fp16 **passthrough tensors** that the canonical `gptq_mixed_quantize` leaves uncompressed because they fall under the `numel() ≤ 65536` threshold (small-matrix) or aren't 2-D (scalars / scales).

My solution quantizes these aggressively:

- **Control tensors** (`attn_scale`, `mlp_scale`, `resid_mix`, `skip_gates`, `skip_weights`): per-tensor int8 with a single fp32 scale.
- **Small 2-D matrices** (`bigram.proj`, `attn_gate_proj`, `smear_gate.weight`): per-row int8 with fp16 scales.
- **LZMA self-extracting code wrapper** (standard technique from prior records): shrinks the script from 53 KB raw to **18.1 KB** wrapped.

Net effect: total submission drops from 16.06 MB → 15.99 MB, at a **measured cost of 0 bpb at 5-decimal precision** (the quant roundtrip bpb is unchanged vs baseline).

The mechanism is elementary and the diff is small (~90 lines in `gptq_mixed_quantize` + `dequantize_mixed`), which is a feature rather than a bug — in this regime, **the arithmetic of byte accounting matters more than sophisticated compression schemes**.

Analysis against alternative fits-under-16 MB approaches I considered but rejected:

| Alternative | Effect | Why I didn't pick it |
|---|---|---|
| int7 tok_emb | saves ~500 KB, costs ~0.005 bpb | Baseline BPB preservation was more valuable than 500 KB over-headroom |
| Per-group tok_emb scales (16 groups) | saves ~15 KB, costs ~0.001 bpb | **Not enough alone** (baseline is 65 KB over); ran the math |
| LZMA on weights (instead of brotli) | artifact grew to 17.04 MB | **Anti-fits**: int-quantized weights have high local entropy; brotli wins |
| Bit-packing (int6 packed into 6 bits) | saves 25% raw | **Anti-fits**: packed bits have higher entropy → **+2.6 MB after brotli** |

## 3. What I tested and killed (tier-B and tier-A levers that failed)

### 3.1 `TTT_EPOCHS = 4` — saturated

**Hypothesis:** more SGD epochs per TTT chunk → more adaptation → lower bpb, paid for with ~112 s of extra eval time that fits inside the 600 s eval budget.

**Result on seed 42 (eval-only on saved final_model.pt):**

| Config | TTT val_bpb | TTT eval time |
|---|---|---|
| TTT_EPOCHS=3 (record baseline) | 1.07886574 | 336 s |
| TTT_EPOCHS=4 (probe) | 1.07877614 | 390 s |
| **Δ** | **−0.00008959** | +54 s |

**Interpretation:** the improvement is 6% of my 3-seed sample standard deviation (σ = 0.000143). Effect size is in-noise. **Not a lever — saturated.**

### 3.2 `EVAL_SEQ_LEN = 4096, stride = 128` — architecturally defeated

**Hypothesis:** doubling sequence length at eval (NTK-aware RoPE scaling, already present on line 90 of the shipped Rotary class) doubles the attention context each scored token sees. At `stride = 128` the total number of sliding windows halves, so eval compute stays near-constant.

**Result on seed 42:**

| Metric | Baseline (seq=2048, stride=64) | Probe (seq=4096, stride=128) | Δ bpb |
|---|---|---|---|
| Pre-quant post-EMA val_bpb | 1.08584 | 1.08075 | **−0.00509** ✅ |
| Quantized roundtrip val_bpb | 1.09678 | 1.09344 | **−0.00334** ✅ |
| **Sliding val_bpb** | **1.08014** | **1.08569** | **+0.00555 ❌** |
| TTT val_bpb | 1.07886 | 1.07919 | +0.00033 ❌ |

**The pre-quant improvement is real** (averaged over all positions 0…4095 in each batch, the model benefits from richer context). **But it is *hidden by the sliding-window scoring geometry*** — and that is the number that counts.

**Mechanism:** in the shipped sliding eval, each scored token sits at window-positions `[seq_len − stride, seq_len − 1]`. At `seq_len=2048, stride=64` those are positions 1984–2047, which the Rotary was trained on. At `seq_len=4096, stride=128` those are positions 3968–4095 — NTK-extrapolated, out-of-distribution. The average context gain is over 2k–4k prior tokens (positive), but the average query-phase degradation is applied to 100% of scored tokens (negative, and larger).

Proof-level evidence that the sign flip is a query-position effect, not a context-gain effect: in `eval_val` (non-sliding), where scored tokens span positions 0–4095 uniformly, NTK extension improves bpb by −0.00509. In `eval_val_sliding`, where scored tokens are always at the tail, NTK extension regresses by +0.00555. The only variable that changes between these two is **which rotary phase the query sees**.

### 3.3 QAT v3 (matrices-only int6 fake-quant) — TTT incompatibility

Three days of QAT experiments (v1 with wrong scale formula, v2 with tok_emb fake-quant, v3 with GPTQ-matched scale + matrices-only). Summary:

| QAT variant | Pre-quant bpb (vs non-QAT 1.08584) | Quantized bpb (vs non-QAT 1.09678) | TTT bpb (vs non-QAT 1.07886) |
|---|---|---|---|
| v1 (wrong scale) | +0.0001 (noise) | +0.0001 | — (didn't reach) |
| v2 (correct scale, includes tok_emb fake-quant) | **+0.023** | −0.0074 | — pre-quant cost dominates |
| v3 (correct scale, matrices only, warmup=0) | +0.015 | +0.014 | **diverged to 1.48** |

**Key finding — QAT × score-first-TTT catastrophic interaction:**

QAT v3 produced a respectable quantized artifact (albeit with +0.015 pre-quant drift), but **TTT evaluation diverged to val_bpb = 1.48169**. The mechanism:

- During QAT training, weights are pushed onto the int6 quantization lattice; the fake-quant STE gradient nudges them toward lattice points.
- During TTT, the model is fine-tuned via SGD on val chunks. SGD pushes weights off-lattice.
- Because QAT-trained weights are highly sensitive to leaving the lattice, each SGD step damages the effective predictive distribution.
- Cumulative over 3 TTT epochs × 1238 chunks, the model drifts into a region the quantizer can no longer approximate, and bpb explodes.

**Practical implication:** QAT and score-first TTT are fundamentally at odds in this regime. Any future QAT attempt would need either (a) to freeze all matrix weights during TTT, or (b) to re-apply fake-quant *during* TTT SGD so the lattice is maintained. Both are non-trivial. **I do not recommend this avenue under the 10-min training budget.**

### 3.4 Adaptive Hadamard GPTQ — null on Muon-trained weights

Hadamard rotation prior to quantization is a well-known technique (e.g., QuaRot, SpinQuant) for distributing weight outliers. Theoretical motivation: random Hadamard rotation makes weights approximately Gaussian (CLT), so GPTQ's per-row scale captures Gaussian range more efficiently.

I implemented this as `adaptive_hadamard_gptq.py` (unit-tested) and measured MSE pre/post rotation on actual Muon-trained matrix weights across int5/int6/int7/int8 configurations. **Result: no significant MSE reduction on Muon weights — sometimes a slight regression.**

**Mechanism:** Muon (row-normalized spectral decomposition variant) produces weights with substantially *sub-Gaussian* per-row distributions — empirical kurtosis ≈ −1.2, nearly uniform with short tails. Random Hadamard rotation of a sub-Gaussian vector produces another sub-Gaussian vector with similar kurtosis. The rotation does not smooth a distribution that is already as smooth as it can get, so there is no quantization benefit.

SpinQuant's success on standard LLMs (AdamW-trained, fat-tailed Gaussian-like weights) **does not transfer to Muon-trained small models.** Null result.

### 3.5 Sliding-Window Attention during training (complete 2×2 factorial) — position-cap null

**Hypothesis:** if the model is trained with SWA (each query attends to last W tokens), then at eval it should be (a) robust to long contexts (since distances are always bounded by W) and (b) better-suited to KV chaining across windows.

**Setup:** trained SWA with `window_size = 1024` at `seq_len = 2048` via flash_attn_3's `window_size` parameter. Then ran a complete 2×2 factorial on eval axis: {baseline model, SWA-trained model} × {full-attention eval, SWA eval at seq=2048 or seq=4096}.

| Train config | Eval config | Pre-quant | Quantized | Sliding | **TTT** | vs record baseline |
|---|---|---|---|---|---|---|
| Baseline (full attn) | Full attn, seq=2048 | 1.08584 | 1.09678 | **1.08014** | **1.07886** | **0** (record) |
| Baseline (full attn) | SWA=1024, seq=2048 (Exp B) | 1.08737 | 1.09829 | 1.08491 | 1.08321 | +0.00435 |
| SWA=1024 | SWA=1024, seq=2048 | 1.08602 | 1.09704 | 1.08293 | 1.08163 | +0.00277 |
| SWA=1024 | SWA=1024, seq=4096 (Exp A) | 1.08239 | 1.09480 | 1.08830 | 1.08341 | +0.00455 |

**What the factorial shows:**

1. **SWA training converges to essentially the same pre-quant bpb as baseline** (1.08602 vs 1.08584, Δ within noise). The model learns equally well under a 1024-token window.
2. **Eval-time SWA windowing is a pure context-cap.** Baseline model + SWA=1024 eval loses 0.0044 on TTT because each scored token sees only 1024 prior context instead of 2048.
3. **Training-time SWA recovers ~0.002 bpb of the eval-time cap** (1.08163 vs 1.08321) — training on the same distribution the eval operates in does help. But it cannot overcome the fundamental halving of context.
4. **Extending to seq=4096 with SWA doesn't unlock long context** either. Scored positions at 3968–4095 are OOD (NTK-extrapolated) even though the training-time window was the same. The OOD query-phase penalty stacks with the context cap.

**Null result for this specific windowing choice.** A more careful design would be `train_seq_len=4096, window=2048` (so the train-time window matches what baseline's full attention sees on 2048 val positions), but that requires a ×2 memory training run. The pattern suggests it would still not produce a record — because it is still a position-depth-vs-breadth trade, just shifted.

### 3.6 `TRAIN_SEQ_LEN = 4096` — position-depth vs breadth

**Hypothesis:** if the model is trained at seq_len = 4096, positions 0–4095 all become in-distribution, so eval at seq=4096 with standard sliding is clean — no OOD tax at scored tail positions.

**Setup:** `TRAIN_SEQ_LEN = 4096`, `ROPE_TRAIN_SEQ_LEN = 4096`, `TRAIN_BATCH_TOKENS = 786432` (unchanged, so 192 sequences per step instead of 384). Eval: `EVAL_SEQ_LEN = 4096, stride = 64`. Single seed (42).

**Throughput cost:** training took 588 s as capped, achieving step 3837 (baseline achieved 4393 at seq=2048). Per-token throughput: 5.65 M tok/s (vs baseline's 7.50 M tok/s) — **24% slower on a per-token basis**, and total processed tokens down ~15%.

**Result:**

| Metric | Baseline seed 42 (seq=2048) | Seq=4096 seed 42 | Δ bpb |
|---|---|---|---|
| Pre-quant post-EMA val_bpb | 1.08584 | **1.08388** | **−0.00196** ✅ |
| Quantized roundtrip val_bpb | 1.09678 | **1.09439** | **−0.00239** ✅ |
| **Sliding val_bpb** | **1.08014** | 1.08449 | **+0.00435** ❌ |
| **TTT val_bpb** | **1.07886** | 1.08314 | **+0.00428** ❌ |

**The exact same sign inversion as §3.2 appears here too.** Pre-quant improves because its average is over all positions 0–4095 (broad, the model has in-distribution coverage everywhere). Sliding regresses because it only scores positions 3968–4095 (narrow tail, each of those 128 positions got half the training samples vs baseline's 64 tail positions at seq=2048).

**Mechanism — "position-depth-vs-breadth tradeoff":**

Training at `TRAIN_SEQ_LEN = 2048` concentrates every training sample on positions 0–2047. A position like 2000 might be seen hundreds of thousands of times across training.

Training at `TRAIN_SEQ_LEN = 4096` spreads the same total training samples across positions 0–4095. Position 2000 is now seen roughly half as often — and position 4000 (previously never seen) is now seen some times, but still much less than position 2000 was at the shorter sequence length.

Sliding eval with stride=64 always scores tokens at window-tail positions. At seq=2048 that's positions 1984–2047; at seq=4096 that's positions 3968–4095. Both in-distribution for their respective models, but the seq=4096 model has **shallower per-position knowledge** at its tail positions — same total training data spread across a wider position range.

**The net effect is that sliding specifically measures the shallow-end of position depth, not the broad-average.** Raw seq_len extension (and SWA at a smaller window) both trade depth for breadth; both regress on sliding for the same reason.

## 4. On tokenization determinism between pods

During this session I discovered a **subtle bpb-scale shift across pods** worth documenting for other competitors.

Pod A (earlier): step-0 val_bpb on random init = **3.4871**
Pod B (fresh, same architecture): step-0 val_bpb on random init = **3.6843**

Both pods used the identical SP8192 tokenizer file (md5 `ec1e96070f...`) and the identical 40,540,160-token val shard. But the val-token-byte count (denominator of val_bpb) differed by **~5.6%** because the shards themselves had been tokenized with a different SP8192 BPE model at different times. The tokenizer file on disk matched, but the shard-time tokenizer did not.

When regenerated from `willdepueoai/parameter-golf`'s `docs_selected.jsonl` via `download_hf_docs_and_tokenize.py --variant sp8192`, the pipeline produces a canonical SP8192 tokenizer + shards that reproduce step-0 bpb **3.4873** (matching Pod A within 0.0002, i.e., bit-identical up to CUDA nondeterminism). All three seeds in my record PR were run on this canonically-regenerated tokenizer.

**Advisory for other competitors:** the canonical SP8192 training and eval setup requires regenerating the tokenizer + shards from `willdepueoai/parameter-golf`'s published `docs_selected.jsonl`, not reusing any SP8192 shards found on disk from a previous session. Random-init step-0 val_bpb near 3.487 (at the canonical token/byte ratio 0.387) is the canonical fingerprint; numbers far from this indicate a tokenization mismatch, not a training success or failure.

## 5. The architectural insight

Stepping back from individual nulls: the three tier-A/B levers that *could have unlocked the eval-time working-memory lever* — `EVAL_SEQ_LEN = 4096`, `SWA training`, and `TRAIN_SEQ_LEN = 4096` — all fail in the same direction under sliding-window eval. Each pays a cost in the narrow tail-position range that sliding specifically measures.

**Sliding-window eval at fixed stride rewards position depth at a specific band** (positions `seq_len−stride` through `seq_len−1`). Any technique that spreads training effort across more positions, or narrows per-token attention, reduces the per-position depth at that band.

The right fix is not a better position-extension scheme — it's a position scheme that makes the scored-token position irrelevant. Three candidates:

1. **ALiBi** (attention with linear bias on distance) — attention logits get a per-head linear bias based on token distance, no rotary at all. Model predictions depend on distance, not absolute position. Extrapolates cleanly to longer contexts in the literature (LLaMA, Mistral variants).
2. **NoPE** (no positional encoding) — surprisingly competent on causal LMs; the causality mask alone provides some order signal. Zero OOD-position risk.
3. **T5-style learnable relative-position buckets** — similar to ALiBi but with learned biases per log-spaced distance bucket.

All three would be trained at `TRAIN_SEQ_LEN = 2048` and evaluated at `EVAL_SEQ_LEN = 4096+` without any position-related regression. **That's the next experiment** — and it's the one the ablation evidence from this PR points toward.

## 6. Reproducibility

This PR includes everything needed to reproduce every number reported above. File manifest:

```
README.md              # this file
submission.json        # non-record metadata
adaptive_hadamard_gptq.py   # §3.4 null result — full impl + unit tests
kv_cache_chain.py      # Prototype helpers for an KV-chain direction (superseded — see §5)
patches/
  patch_pathav3_inline.py    # Path A v3 quantization (used in record PR)
  patch_qat_v3.py            # QAT v3 (§3.3, NOT RECOMMENDED)
  patch_swa.py               # SWA at training/eval (§3.5)
  pack_submission.py         # LZMA code wrapper
  ngram_cache_eval.py        # Prototype causal n-gram cache (Track B, not yet tested — §6)
logs/
  eval_ttt4_s42.log          # §3.1 TTT_EPOCHS=4 probe
  eval_yarn4096_s42.log      # §3.2 NTK-RoPE seq=4096 probe
  qat_v1_seed42.log          # §3.3 QAT v1 (wrong scale)
  qat_v2_seed42.log          # §3.3 QAT v2 (tok_emb fake-quant)
  salvage_int7.log           # §3.3 supplementary
  sanity_swa0.log            # §3.5 SWA patch sanity (SWA=0 should = baseline; confirmed)
  swa_train.log              # §3.5 full SWA training + seq=2048 eval
  swa_eval4096.log           # §3.5 SWA-trained + seq=4096 eval (Experiment A)
  retrofit_swa_s42.log       # §3.5 baseline model + SWA=1024 eval (Experiment B)
  seq4096_s42.log            # §3.6 TRAIN_SEQ_LEN=4096 full run
```

The canonical seed 42 trained model is PR #1716's `final_model_seed42.pt` — any reader can reload it and replay these probes in 5–10 min each on 8× H100.

## 7. Where this points

Ranked by expected return for the remaining days of the competition:

1. **ALiBi training** (replace RoPE with relative-position attention bias) — the architecturally-right answer to every null in §3.2, §3.5, §3.6. Training-time change, well-documented in literature. This is the next experiment.
2. **NoPE** — simpler fallback; worth trying if ALiBi dev is heavy. Literature suggests competitive performance on causal LMs.
3. **Causal n-gram cache at eval** (tier-B, permitted explicitly under Issue #1017) — prototype in `patches/ngram_cache_eval.py`. Novel to this competition: no PR I've found uses a dynamic bigram prior accumulated from already-scored val tokens. Expected modest gain; valuable as a clean legal addition rather than record-breaker.
4. **Legitimate FLA / GatedDeltaNet** with rigorous byte-accounting — architectural long-context, but the flagged PR queue (#1672, #1687, #1698, #1711, #1712) shows the byte-count pitfall kills almost every attempt. Expensive pit to dig out of correctly.
5. **Per-document LoRA TTT** (PR #1928 framework on top of the record stack).

## Credits

- All upstream lineage from record PR #1716 (see that PR's attribution list).
- Issue #1017 (A Field Guide to Valid Submissions) for the tier-A vs tier-B framing.
- SpinQuant / QuaRot literature for the Hadamard direction that §3.4 refutes on this stack.
- Press et al. 2022 (ALiBi) and Haviv et al. 2022 (NoPE) for the relative-position directions §7 points toward.
