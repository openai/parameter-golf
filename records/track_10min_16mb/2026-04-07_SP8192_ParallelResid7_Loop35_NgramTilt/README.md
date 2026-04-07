# Record: SP8192 + Parallel Residuals + 3-Layer Recurrence + Token-Only N-gram Tilt — val_bpb 1.08091 (5-seed mean, causal-corrected)

**val_bpb: 1.08091** (5-seed mean, std 0.00043) | **2.79210 nats per token** | **~16.00 MB** | 8×H100 SXM, 600 s | Legal Score-First TTT + Causal Token-Only N-gram Tilt

Beats [PR #1394](https://github.com/openai/parameter-golf/pull/1394) (1.08563) by **+0.01219 nats per token** — comfortably clearing the 0.005-nat record threshold (2.4× the bar). Also beats merged SOTA [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (1.11473) by **+0.08736 nats per token**.

> **2026-04-07 PM correction note** — see [Legality Fix](#legality-fix-2026-04-07-pm) section. The originally posted 5-seed mean (1.07807) was produced with a non-causal n-gram kernel inherited from [PR #1420](https://github.com/openai/parameter-golf/pull/1420). @abaybektursun [has acknowledged the bug and proposed the same fix I applied here](https://github.com/openai/parameter-golf/pull/1420#issuecomment-4199452189). The current 5-seed mean (1.08091) is ~+0.00284 BPB above the originally posted (illegal) 1.07807, but it still passes the 0.005-nat record bar against PR #1394 by 2.4×, so this remains a valid record submission. Pre-fix per-seed values are preserved in `submission.json` under `seed_results_pre_fix` for the public record.

## Bar comparisons (5-seed mean 1.08091, val_loss 2.79210 nats/token)

| Comparator | val_bpb | Δ (nats per token) | 0.005-nat bar |
|---|---:|---:|---|
| Merged SOTA [PR #1019](https://github.com/openai/parameter-golf/pull/1019) (abaybektursun) | 1.11473 | **+0.08736** | ✅ comfortably |
| [PR #1394](https://github.com/openai/parameter-golf/pull/1394) (clarkkev) | 1.08563 | **+0.01219** | ✅ clears (2.4× the bar) |
| Our [PR #1413](https://github.com/openai/parameter-golf/pull/1413) | 1.08279 | +0.00486 | ❌ misses by 0.00014 (essentially tied) |
| [PR #1420](https://github.com/openai/parameter-golf/pull/1420) (same kernel family; direct pre-fix comparison is not apples-to-apples) | 1.08014 | -0.00199 | ⚠️ see note below |

The unit is nats per token (per the README's record threshold). The bpb-to-nats conversion factor is the mean bytes-per-token in the sp8192 val set: 1 bpb ≈ 2.5831 nats per token (verified against this submission's own `val_bpb / val_loss` ratio).

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, causal token-only n-gram tilt)

### Core (TTT) table — 5-seed verification, all seeds re-run via shipped mini wrapper with the patched kernel

| Seed | Steps | Pre-quant BPB | Sliding BPB | **Post-TTT (causal token-only) BPB** | val_loss (nats) | Artifact (bytes) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4911 | 1.08730 | 1.08219 | **1.08035** | 2.79067 | **15,994,644** ✅ |
| 42 | 4906 | 1.08792 | 1.08272 | **1.08097** | 2.79225 | **15,995,572** ✅ |
| 1234 | 4915 | 1.08823 | 1.08336 | **1.08127** | 2.79303 | **15,993,531** ✅ |
| 1337 | 4905 | 1.08759 | 1.08235 | **1.08060** | 2.79131 | **15,988,802** ✅ |
| 2025 | 4911 | 1.08833 | 1.08302 | **1.08135** | 2.79324 | **15,993,360** ✅ |
| **5-seed mean** | | **1.08787** | **1.08273** | **1.08091** | **2.79210** | all < 16,000,000 |

**Verification status:**
- All 5 seeds independently re-run via the shipped `train_gpt.py` (~18.9 KB code) with the **patched** `fused_expert_kernel.cpp` and `NGRAM_WITHIN_BETA=0 NGRAM_WORD_BETA=0`. Each artifact is the actual `Total submission size quantized+brotli` from the mini-wrapper run.
- All 5 artifacts fit under 16,000,000 bytes (corrected runs use the same model weights as the original submission; only the eval-time kernel changed).
- 5-seed standard deviation: **0.00043 BPB**.
- Pre-fix (illegal) per-seed values are preserved in `submission.json` under `seed_results_pre_fix`.

## Legality Fix (2026-04-07 PM)

The original kernel from [PR #1420](https://github.com/openai/parameter-golf/pull/1420) (which this submission ported with `nanobind` removed) had a causality bug in `get_hints_batch`:

- Lines 384-386 read `tok = tokens_[p]` (the **target** token at the position being scored) and derived `is_bnd = is_bnd_[tok]` and `is_ws = has_ls_[tok]`.
- Lines 399-400 then passed those flags to `within_hint(is_bnd, is_ws, ...)` and `word_hint(is_ws, ...)`, gating hint emission on whether the **current target** is mid-word vs word-start vs boundary.

This means the predictive distribution at position `p` depended on metadata derived from `x_p` itself, leaking 1-2 bits per scored position about the answer. Under the [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) framing, this is a violation of the prefix-only causality requirement. The original 1.07807 5-seed mean reported in PR #1437's first version is therefore tainted.

**The fix** (matches @abaybektursun's [proposed patch](https://github.com/openai/parameter-golf/pull/1420#issuecomment-4199452189)):

1. **Kernel patch**: derive `prev_is_bnd`/`prev_is_ws` from `tokens_[p-1]` (last prefix token) for hint gating only. The current-token reads at lines 384-386 are kept only for the *update* calls at lines 437-439 (causal because they run after hint emission for that position).
2. **Disable within/word experts**: set `NGRAM_WITHIN_BETA=0 NGRAM_WORD_BETA=0`. Empirically, the within/word experts under prefix-only gating fire for the wrong positions (within fires for word-starts, word fires for mid-word) and contribute *negative* BPB. Only `token_hint` (which has always been causal — `compute_hashes` only reads `tokens[pos - k - 1]` for `k ≥ 0`) is left active.

**Measured leak magnitude (this submission, 5-seed mean):** TTT `1.07807 BPB` → `1.08091 BPB`, delta **+0.00284 BPB ≈ +0.00734 nats per token** (using 1 bpb ≈ 2.5831 nats per token, the mean bytes-per-token in the sp8192 val set). Sliding (no tilt) and pre-quant numbers are unchanged because the kernel only affects the TTT eval pass.

**PR #1420 cross-reference**: PR #1420 originally shipped the same kernel-family bug. @abaybektursun has [acknowledged it in their thread](https://github.com/openai/parameter-golf/pull/1420#issuecomment-4199452189) and proposed the same fix. Because the original `1.08014` number was reported before that correction, direct pre-fix comparison is not apples-to-apples.

## Key Innovations

A 3-lever stack on top of [@clarkkev's PR #1394](https://github.com/openai/parameter-golf/pull/1394) sp8192 baseline:

### 1. Parallel Residuals on layers 7–10 (from [PR #1412](https://github.com/openai/parameter-golf/pull/1412) by @Robby955)

GPT-J-style parallel attention + MLP for the last 4 layers. Both attention and MLP read the same pre-residual input and their outputs are summed in parallel. Reduces interference between attention and MLP during GPTQ calibration → tighter quantization gap.

```python
# Parallel (layers 7-10):
x_out = x + attn_scale * Attn(norm(x)) + mlp_scale * MLP(norm(x))

# Sequential (layers 0-6, unchanged):
h = x + attn_scale * Attn(norm(x))
x_out = h + mlp_scale * MLP(norm(h))
```

Verified standalone contribution: **−0.00048 BPB** on 3-seed mean (par7 alone vs control).

### 2. 3-Layer Depth Recurrence (extending PR #1394's 2-layer recurrence)

Loop layers **3–5 twice** instead of 4–5 twice. Encoder pattern `[0,1,2,3,4,5,3,4]` and decoder `[5,3,4,5,6,7,8,9,10]`. Costs ~200 training steps but the additional virtual depth (17 vs 15 layers) more than compensates.

Verified standalone contribution on top of par7: **−0.00128 BPB** on s42.

### 3. Eval-Time Causal N-gram Tilt (from [PR #1420](https://github.com/openai/parameter-golf/pull/1420) by @abaybektursun, lineage [PR #1145](https://github.com/openai/parameter-golf/pull/1145) @AnirudhRahul)

A causal open-addressing n-gram cache (token orders 8/10/12/14/16, within-word orders 1–3, word-start order 4) proposes a single hint token from strict prefix state. The model's full softmax distribution is then **rescaled with a one-token exponential tilt**:

```
p_tilt(t) = p_model(t) · exp(β · 𝟙[t==hint]) / Z
Z = 1 + p_model(hint) · (exp(β) − 1)
```

This is a **renormalized full-vocab distribution**, not a `p(correct_token)`-only blend. The hint at position `p` is computed from `tokens[0..p−1]` only; the cache is updated with `tokens[p]` AFTER position `p`'s score is locked.

Per-position NLL becomes:
```python
mixed_nll = scored_nll + has_hint * (Z.log() - β * is_hit)
```

C++ kernel ported from PR #1420 with the nanobind dependency removed (replaced with a `extern "C"` shim and ctypes loader). Build is a single `g++ -O3 -march=native -std=c++17 -fPIC -shared` invocation against `fused_expert_kernel.cpp`. The kernel processes ~3M tokens/sec; the precompute over the full ~40.5M val tokens runs in ~32 s on rank 0 then broadcasts hints/betas to other ranks.

Verified standalone contribution on top of par7: **−0.00297 BPB** on s42 (PR #1420 reports −0.0029 — port is byte-correct).

## Stacking decomposition (s42)

| Stack | TTT BPB | Δ vs control |
|---|---|---|
| Control (PR #1413) | 1.08315 | — |
| + Parallel residuals layers 7+ | 1.08239 | −0.00076 |
| + 3-layer recurrence | 1.08111 | −0.00204 |
| + N-gram tilt | **1.07808** | **−0.00507** |

The three levers stack approximately linearly with slight positive synergy (predicted −0.00473, actual −0.00507).

## Changes from baseline (PR #1394 → this PR)

| Component | PR #1394 | This PR |
|---|---|---|
| Tokenizer | SentencePiece BPE 8192 | (same) |
| Architecture core | 11L / 512d / 8H / 4KV, MLP 4× | (same) |
| Depth recurrence | Loop layers 4–5 twice | **Loop layers 3–5 twice** |
| Block forward pattern | Sequential attn → MLP all 11 layers | **Parallel attn+MLP for layers 7–10**, sequential layers 0–6 |
| Optimizer | MuonEq-R, WD=0.085 | (same) |
| Quantization | GPTQ int6 + int8 embed + SDClip | (same) |
| Eval | sliding window | sliding window **+ score-first TTT + causal n-gram tilt** |
| QK_GAIN_INIT | 4.0 | **5.0** |
| TTT | none | **score-first, LR=0.005, epochs=3, freeze=0** |
| val_bpb (3-seed mean) | 1.08563 | **1.07800** |
| Δ vs PR #1394 (per-token nats) | — | **−0.01971** |

## Architecture

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)² activation, Partial RoPE (16/64 dims), tied token embeddings. Depth recurrence: encoder `[0,1,2,3,4,5,3,4]`, decoder `[5,3,4,5,6,7,8,9,10]` (loops layers 3–5 twice, activated at frac=0.5 of training, ~step 2924). Layers 7–10 use the GPT-J parallel attention+MLP pattern; layers 0–6 stay sequential.

Quantization: full-Hessian GPTQ on all attention/MLP matrices at int6 with SD-based clip (`row_std × 12.85 / 31`); token embedding at int8 with clip `20 × row_std`; small control tensors and scalars kept float16/float32 via passthrough. Compression: byte-shuffle + Brotli-11. Self-extracting LZMA mini runner (~18,905 bytes code).

N-gram tilt subsystem: 5 token-order open-addressing hash tables (orders 8, 10, 12, 14, 16) at `open_table_bits=26` ≈ 67M slots × 16 B/entry = 1 GB each (5 GB token-cache) + 3 within-word tables and 1 word-start table at `bits=20` (≈ 16 MB total) + 1 `WordStartState` Python dict. **Host RAM only** — not counted toward the 16 MB artifact. Built fresh from val tokens on rank 0 in ~32 s, hints/betas broadcast to other ranks before TTT eval starts.

## Rule Compliance

Per [repo README](https://github.com/openai/parameter-golf) and [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) four conditions:

- **Condition 1 (Causality)**: The n-gram cache state at position `p` is built solely from `tokens[0..p−1]`; the C++ kernel's `compute_hashes` reads only `tokens[pos − k − 1]` for `k ≥ 0`. The hint at position `p` is written to the output buffer BEFORE the kernel mutates any table with `tokens[p]`. The model forward pass is the standard causal transformer; sliding-window eval never references future tokens. See `fused_expert_kernel.cpp` `get_hints_batch` lines around the explicit `hints[i] = best_hint; betas[i] = best_beta; ... token_update(...);` ordering.
- **Condition 2 (Normalized full distribution)**: Standard softmax over the full sp8192 vocab. The n-gram tilt rescales each per-position distribution as `p_tilt(t) = p_model(t) · exp(β · 𝟙[t==hint]) / Z` with `Z = 1 + p_model(hint) · (exp(β) − 1)`. This is a proper probability distribution over the entire alphabet — not a `p_t(correct_token)`-only blend. The hint token is chosen from prefix-only state BEFORE the realized target is consulted; the only target dependence is the indicator `𝟙[tgt==hint]`, which is the legitimate "did the realized token land on the boosted token" term.
- **Condition 3 (Score before update)**: Every TTT chunk is scored under `torch.no_grad()` before any parameter update. Every n-gram tilt position is scored before its target token is mixed into the cache state. No same-symbol adaptation, no self-exclusion.
- **Condition 4 (Single pass)**: Each token is scored exactly once. Sliding-window eval is forward-only (`stride < seq_len`). The C++ kernel's `get_hints_batch` walks positions in monotonic order. No rescoring, no oracle selection.

Additional:
- **No SLOT** (standard or causal). No eval-time delta optimization in hidden space.
- **No pre-quant TTT on val data**. The model is quantized once after training, then the quantized model is evaluated under score-first TTT + n-gram tilt.
- **No ETLB**.
- **No tokenizer change** — uses PR #1394's SentencePiece BPE 8192 unchanged.
- **GPTQ calibration uses `fineweb_train_*` exclusively**, inside the 588 s training cap (12 s GPTQ reserve).
- **N-gram cache state lives in host RAM only**, not in the 16 MB artifact.
- **C++ kernel and Python wrapper live alongside `train_gpt.py`** in the records folder; only `train_gpt.py` (the LZMA self-extracting mini wrapper, ~18.9 KB) counts toward the 16 MB artifact, matching the precedent set by [PR #1145](https://github.com/openai/parameter-golf/pull/1145).
- **3 distinct seeds** (0, 42, 1234) — independent runs on the same hardware.

## Requirements

```
torch==2.9.1+cu128
flash-attn==2.8.3
flash-attn-3 (interface wheel; Hopper build)
sentencepiece
numpy
brotli
gcc (any version supporting C99/C++17)
```

GCP 8×H100 80GB SXM pod with `NCCL_NET=Socket` (GCP-specific; NCCL 2.27.5 + gIB device issue).

## Run Command

```bash
export NCCL_NET=Socket
export QK_GAIN_INIT=5.0
export PARALLEL_RESIDUAL_START=7
export LOOP_START=3
export LOOP_END=5
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export NGRAM_TILT_ENABLED=1
export NGRAM_BASE_BETA=2.0
export NGRAM_AGREE_BONUS=0.1
export NGRAM_WITHIN_THRESHOLD=0.25
# CAUSAL CORRECTION: disable within/word experts
export NGRAM_WITHIN_BETA=0.0
export NGRAM_WORD_BETA=0.0

for SEED in 0 42 1234 1337 2025; do
    SEED=$SEED uv run torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

The first run will compile `fused_expert_kernel.cpp` to `libfused_ngram.so` via gcc; subsequent runs reuse the cached `.so`.

## Lineage

- **[PR #1394](https://github.com/openai/parameter-golf/pull/1394)** (@clarkkev) — sp8192 + GPTQ embeddings + SDClip + MuonEq-R + 2-layer depth recurrence — base stack
- **[PR #1413](https://github.com/openai/parameter-golf/pull/1413)** (@dexhunter, ours) — sp8192 + QK-Gain 5 + legal score-first TTT — direct predecessor
- **[PR #1412](https://github.com/openai/parameter-golf/pull/1412)** (@Robby955) — Parallel Residuals + Hessian-Aware SDClip — parallel residuals lever
- **[PR #1420](https://github.com/openai/parameter-golf/pull/1420)** (@abaybektursun) — Triple Loop + Fused Kernels + N-gram Tilt — n-gram tilt kernel and tilt math
- **[PR #1145](https://github.com/openai/parameter-golf/pull/1145)** (@AnirudhRahul) — Online Best-Agree N-gram — first legal normalized n-gram cache, organizer-discussed precedent in [issue #677](https://github.com/openai/parameter-golf/issues/677)
- **[PR #1019](https://github.com/openai/parameter-golf/pull/1019)** (@abaybektursun, merged) — AR Self-Gen GPTQ + XSA + BigramHash 3072 — current merged SOTA, GPTQ pipeline ancestor
- **[PR #549](https://github.com/openai/parameter-golf/pull/549)** (@abaybektursun, merged) — LeakyReLU² + score-first TTT — legal-TTT precedent

## Credits

- **@clarkkev** for the sp8192 base stack (PR #1394) this submission builds on
- **@Robby955** for parallel residuals on layers 7–10 (PR #1412)
- **@abaybektursun** for the n-gram tilt mechanism, the C++ kernel, and the merged-precedent legal-TTT (PRs #1420, #1019, #549)
- **@AnirudhRahul** for the original normalized causal n-gram cache pattern (PR #1145)
- **@msisovic** for depth recurrence (PR #1204)
- **@bigbag** for MuonEq-R (PR #1217)
- **@unnir** for XSA (PR #265)
- **@simon-marcus** for the corrected Scylla byte-accounting reference (PR #1314) — used for legality discussions, not in this submission
- **@NoesisGenesis** for the four-conditions framework (issue #1017)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` — self-extracting LZMA mini wrapper, ~18.9 KB. The only file counted toward the 16 MB artifact.
- `ngram_tilt.py` — Python ctypes wrapper for the C++ n-gram kernel. Imported at runtime by `train_gpt.py`. Not counted toward artifact (parallel pattern to PR #1145's separate `online_best_agree_eval.py`).
- `fused_expert_kernel.cpp` — C++ source for the n-gram cache. Built to `libfused_ngram.so` at runtime via `gcc -O3 -march=native -std=c++17 -fPIC -shared`. Not counted toward artifact.
- `train_seed0.log`
- `train_seed42.log`
- `train_seed1234.log`
