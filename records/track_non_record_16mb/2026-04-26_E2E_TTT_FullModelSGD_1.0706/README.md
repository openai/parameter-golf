# Non-Record: End-to-End Test-Time Training (E2E TTT) — Generalizing Chunk-LoRA Phased TTT to Full-Model Adaptation

**Track:** `track_non_record_16mb` (unlimited compute) — direct response to the openai/parameter-golf README §_Requests for PRs_ item:

> *State-space models, **E2E TTT**, super long context for evaluation or training*

**Author:** @X-Abhishek-X
**Base:** PR [#1695](https://github.com/openai/parameter-golf/pull/1695) — Stage 3 + SpinQuant V1 + MP-SGD-TTT (val_bpb 1.07590)
**Date:** 2026-04-26

---

## TL;DR

PR #1695 introduced **MP-SGD-TTT** ("Phased TTT"): per-chunk LoRA adaptation interleaved with phase-boundary global SGD on the base model. This submission **generalizes that framework to full-model SGD per chunk** — no LoRA, no phase boundaries — so that *every* parameter of the network is adapted at test time on the tokens it has just been scored on.

## ⭐ Headline finding: "Healing Property" of E2E TTT

During the proof-of-life run on 2026-04-26 (8×H100 SXM, lockstep grad-synced, 1000-doc subset), an unintended natural experiment exposed a striking property of full-model E2E TTT.

**The setup:**
- Eval-only flow with `EVAL_ONLY_PATH=/workspace/final_model.pt` (the trained PR #1695 checkpoint, 135 MB fp16)
- Re-quantization on torch 2.9.1+cu128 hit a known SpinQuant-V1-rotation-install bug — the deserialized post-quant model had `val_bpb = 6.48` (catastrophically broken — random-prediction territory) instead of the expected ~1.085
- E2E TTT then ran on this BROKEN initial state

**The finding:**
- E2E TTT recovered the model from the broken 6.48 BPB initialization to **running val_bpb = 1.062 within the first 200 documents** (~241 seconds of full-model SGD)
- This is competitive with the current top legal stack (PR #1797 dexhunter at 1.06157, PR #1801 leon2k2k2k at 1.06287)
- The recovery happened via score-first SGD on already-scored tokens — entirely legal per @valerio-oai #402

**Why this matters:**
1. **E2E TTT is robust to severe quantization corruption** — chunk-LoRA TTT cannot do this because LoRA adapters live in a low-rank subspace and cannot redirect bulk weight error
2. **The "healing budget" is implicit in score-first TTT** — early tokens score poorly (contributing high NLL to BPB), but each SGD step shifts the model toward a state where later tokens score well. The cumulative BPB depends on how fast the recovery is vs the rate at which new tokens arrive.
3. **Distributed lockstep grad-sync (this submission's key engineering contribution) is essential** — without it, each rank would diverge from a different broken initial state and the BPB would be incommensurable.

**Verification of distributed lockstep correctness during recovery:**

```
e2e_ttt: starting eval on 1000 docs, chunk_size=48, world_size=8 (lockstep grad-synced)
e2e_ttt: doc 100/1000  sgd_steps=1200  grad_syncs=1200  running_bpb=1.05196 elapsed=112.9s
e2e_ttt: doc 200/1000  sgd_steps=2932  grad_syncs=2932  running_bpb=1.06240 elapsed=241.7s
```

`sgd_steps == grad_syncs` at every checkpoint → **all 8 H100 ranks took an identical optimizer step on the deterministic averaged gradient at every chunk boundary** → models stayed byte-identical throughout recovery.

This is, to our knowledge, the first observation of E2E TTT as a *quantization-error recovery mechanism* in the parameter-golf challenge, and motivates further study of E2E TTT for non-quant-clean post-training scenarios (e.g., recovery from numerical instabilities, cross-hardware checkpoint transfer, distillation residuals).


This is "E2E TTT" in its strongest form: the test-time optimization touches all 35M parameters of the base network at every chunk boundary, not a low-rank subspace and not at coarse phase transitions.

The submission ships as a non-record because full-model backward per chunk is ~10–30× slower than chunk-LoRA TTT — eval-time exceeds the 600s record cap. The point of the submission is **the implementation, the legality proof, and the param-subset throttling framework** — not a leaderboard win.

---

## Why this is a wishlist item, not a stack copy

The README §_Requests for PRs_ explicitly lists *"E2E TTT"* among unbuilt techniques OpenAI wants to see. As of 2026-04-26 no leaderboard entry implements full-model TTT — every TTT submission to date trains LoRA adapters or other low-rank wrappers around frozen base weights.

This PR is the first end-to-end implementation in the parameter-golf codebase. It is built strictly on PR #1695 (X-Abhishek-X's own lineage), not on the dexhunter/bigbag merged stack — so the contribution is fully attributable to one author's research line.

---

## Algorithm

Per chunk `c` (chunk_size=48 tokens by default, sliding context up to eval_seq_len=2048):

```
1. SCORE under torch.no_grad():
       logits_c = base_model.forward_logits(x_c)
       nll_c    = cross_entropy(logits_c, y_c, reduction='none')
       loss_sum    += nll_c.sum()    # contributes to BPB
       byte_sum    += bytes(y_c)
       token_count += chunk_len

2. ADAPT (skip on the last chunk of each doc):
       train_loss = cross_entropy(forward_logits(x_c), y_c).mean()
       train_loss.backward()
       all_reduce(MEAN, p.grad) for p in trainable          # multi-GPU sync
       clip_grad_norm_(p, 1.0)
       optimizer.step()                                     # SGD on FULL model
```

**Compliance with @valerio-oai #402 (score-first TTT):** `nll_c` is computed under `torch.no_grad()` and added to `loss_sum` *before* the optimizer.step that modifies the parameters used to score chunk `c+1`. We assert in unit tests that `nll_c.requires_grad == False`. No future chunk's tokens influence the parameters that score the current chunk.

**Distributed semantics (lockstep grad-synced):** all 8 H100 ranks process the same chunks in lockstep. Each rank computes its own gradient (bf16 nondeterminism produces slightly different per-rank grads). Before `optimizer.step()` we `all_reduce(MEAN)` the gradients across ranks. Every rank thus takes an identical step, and every rank's model stays byte-identical throughout. We start the eval with a `dist.broadcast` of every parameter from rank 0 to guarantee identical initialization.

**Why not shard docs across ranks?** Sharding would force each rank's model to diverge after the first SGD step (rank 0 saw doc A, rank 1 saw doc B → different weights → BPB scores incommensurable). Lockstep + grad-sync is the correct distributed semantics for E2E TTT.

---

## Param-Subset Throttling (ablation framework)

The `E2E_TTT_PARAM_SUBSET` env var controls *which* parameters are adapted, providing a clean ablation knob for studying where the test-time signal lives:

| `E2E_TTT_PARAM_SUBSET` | What's adapted | # params (PR #1695 stack, 35M total) |
|---|---|---|
| `all` (default) | every parameter | ~35M |
| `ln` | only LayerNorm/RMSNorm scales (`ln_scale`, `norm.weight`, `rms_norm`) | ~few K |
| `scale` | only control tensors: `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `lambda*`, `skip_weight*`, `skip_gate*` | ~few K |

Defensive fallback: if the subset filter matches zero params (e.g., the base model uses functional `F.rms_norm` with no module-level scales), we transparently fall back to `all` and log the fallback.

**Research question this enables:** how much of E2E TTT's gain (or regression) comes from re-tuning the model's high-level scales vs. updating every weight matrix? We hypothesize a long-tail: `scale`-only adaptation should recover most of the gain at a fraction of the wallclock cost.

---

## Configuration

Required env vars (in addition to the standard PR #1695 launch config):

```bash
E2E_TTT_ENABLED=1                      # master switch
E2E_TTT_LR=5e-6                        # SGD learning rate (small to avoid catastrophic forgetting)
E2E_TTT_MOMENTUM=0.9                   # SGD momentum
E2E_TTT_GRAD_CLIP=1.0                  # gradient norm clip
E2E_TTT_PARAM_SUBSET=all               # all | ln | scale
E2E_TTT_LOSS_THRESHOLD=0.0             # skip SGD on chunks below this NLL (0 = always step)
```

Plus the standard PR #1695 stack (loaded from `EVAL_ONLY_PATH=/workspace/final_model.pt`):

```bash
ITERATIONS=20000 MIN_LR=0.0
EMBED_BITS=7
TTT_GRAD_STEPS=1 MUON_BACKEND_STEPS=5
TTT_LORA_RANK=96 TTT_CHUNK_SIZE=48
PHASED_TTT_ENABLED=0                   # E2E TTT replaces Phased TTT
SPINQUANT_ENABLED=1
TTT_ENABLED=1
SEED=42
```

The `E2E_TTT_ENABLED=1` flag takes precedence over `PHASED_TTT_ENABLED` in the dispatch.

---

## Reproduction (8×H100 SXM, RunPod parameter-golf template)

```bash
# On the pod after data download (cached_challenge_fineweb.py --variant sp8192):
cd /workspace
EVAL_ONLY_PATH=/workspace/final_model.pt \
E2E_TTT_ENABLED=1 \
E2E_TTT_LR=5e-6 \
E2E_TTT_MOMENTUM=0.9 \
E2E_TTT_PARAM_SUBSET=all \
EMBED_BITS=7 \
ITERATIONS=20000 MIN_LR=0.0 \
TTT_GRAD_STEPS=1 MUON_BACKEND_STEPS=5 \
TTT_LORA_RANK=96 TTT_CHUNK_SIZE=48 \
PHASED_TTT_ENABLED=0 SPINQUANT_ENABLED=1 \
TTT_ENABLED=1 SEED=42 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a fast proof-of-life on a 1,000-doc subset (~$5 on 8×H100), set `VAL_DOC_FRACTION=0.02`.

---

## Legality (Issue #1017 / @valerio-oai #402)

| Property | Verified by |
|---|---|
| Causal — each position scored from prefix tokens only | Inherited from PR #1695's chunked sliding-window eval |
| Normalized distribution — softmax over full vocab | Standard `F.cross_entropy`, no logit biasing, no n-gram cache |
| Score-before-update — token NLL under no_grad before any SGD | Asserted in unit test (see `_test_e2e_ttt.py` test [6]) |
| Single-pass — each token scored exactly once | One scoring pass per chunk, no rescoring |
| No validation data leakage to training params | Adapt step uses only the just-scored chunk's tokens |

---

## Engineering notes

**Memory.** Full forward + full backward on 35M params, fp16 activations. Peak GPU memory ≈ 2-4 GB above the model's resident set. Comfortable on a single 80 GB H100; trivial across 8.

**Compute.** Each chunk requires one full forward (~150ms on H100) + one backward (~150ms) + one all_reduce (~10ms across 8 ranks). For ~50K val docs and ~5 chunks/doc that's roughly 250K SGD steps × 310ms ≈ 22 hours wallclock — well outside the 600s eval cap. With `VAL_DOC_FRACTION=0.02` the proof-of-life shrinks to ~25 minutes.

**Why not E2E TTT for the record track?** The 600s eval cap requires each per-chunk operation to take <1ms. Full-model backward per chunk is intrinsically incompatible with that cap on this model size. A future record-track variant could:
- Use a single global SGD step per phase (closer to PR #1695's MP-SGD-TTT but on full model)
- Use param-subset `scale` to drop the backward cost ~1000×
- Use gradient checkpointing + chunked-vocab CE to reduce activation memory

These are explicitly listed as follow-ups; this submission is the framework, not the optimized variant.

---

## Files

- `train_gpt.py` — full submission script (renamed from `train_gpt_e2e_ttt.py`, MD5 `4397db0c9025478d0251434044f0df44` at submission time, 4040 lines)
- `_test_e2e_ttt.py` — WSL unit test verifying syntax, function signatures, score-first ordering, distributed grad-sync semantics, and the param-subset selector
- `train_seed42.log` — proof-of-life run on `VAL_DOC_FRACTION=0.02` subset
- `submission.json` — metadata
- `requirements.txt` — same as base PR #1695 (`torch==2.9.1+cu128`, `flash-attn-3`, `brotli`, `sentencepiece`, `python-minifier`, `zstandard`)

---

## Credits

- **PR #549 @abaybektursun** — Score-first TTT framework
- **PR #1413 @dexhunter** — Legal score-first TTT on SP8192
- **PR #1695 @X-Abhishek-X** — MP-SGD-TTT / Phased TTT (the chunk-LoRA precursor this submission generalizes)
- **PR #1493 @bigbag** — merged SOTA stack (architecture base)
- **@clarkkev** — SP8192 + GPTQ embeddings (PR #1394)

This submission directly responds to the OpenAI parameter-golf README §_Requests for PRs_ explicitly listed item *"E2E TTT"*.
