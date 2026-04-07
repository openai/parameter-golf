# Record: SP8192 + Parallel Residuals + 3-Layer Recurrence + Legal N-gram Tilt — val_bpb 1.07807 (5-seed mean)

**val_bpb: 1.07807** (5-seed mean, std 0.00040) | **2.78478 nats per token** | **~15.99 MB** | 8×H100 SXM, 600 s | Legal Score-First TTT + Causal N-gram Tilt

Beats [PR #1394](https://github.com/openai/parameter-golf/pull/1394) (1.08563) by **0.00756 bpb / 0.01952 nats per token** on a 5-seed mean, comfortably clearing the 0.005-nats record threshold. Beats [PR #1420](https://github.com/openai/parameter-golf/pull/1420) (1.08014) by **0.00207 bpb / 0.00534 nats per token**, clearing the 0.005-nats threshold against the next-best legal open PR. Beats our own [PR #1413](https://github.com/openai/parameter-golf/pull/1413) (1.08279) by **0.00472 bpb / 0.01218 nats per token**.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, legal score-first TTT with causal n-gram tilt)

### Core (TTT) table — 5-seed verification, all seeds re-run via shipped mini wrapper

| Seed | Steps | Pre-quant BPB | Sliding BPB | **Post-TTT (n-gram tilted) BPB** | val_loss (nats) | Artifact (bytes) |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 4918 | 1.08728 | 1.08209 | **1.07751** | 2.78333 | **15,992,304** ✅ |
| 42   | 4911 | 1.08785 | 1.08268 | **1.07809** | 2.78481 | **15,993,733** ✅ |
| 1234 | 4908 | 1.08794 | 1.08280 | **1.07813** | 2.78492 | **15,990,539** ✅ |
| 1337 | 4909 | 1.08772 | 1.08246 | **1.07801** | 2.78461 | **15,988,039** ✅ |
| 2025 | 4908 | 1.08842 | 1.08306 | **1.07862** | 2.78620 | **15,992,215** ✅ |
| **5-seed mean** | | **1.08784** | **1.08262** | **1.07807** | **2.78478** | all < 16,000,000 |

**Verification status:**
- **All 5 seeds independently re-run via the shipped `train_gpt.py` self-extracting LZMA mini wrapper** (~18.9 KB code, ~57 KB decoded payload). Each artifact is the actual `Total submission size quantized+brotli` from the mini-wrapper run, NOT a projection.
- **All 5 artifacts fit under 16,000,000 bytes** with 6,267–11,961 byte headroom.
- 5-seed standard deviation: **0.00040 BPB** (5-seed standard error of the mean: ~0.00018).
- BPB values are reported from the legal score-first TTT eval pass with causal n-gram tilt applied; sliding (no-TTT) and pre-quant numbers are also shown for diagnostic transparency.

### Diagnostics (mini-wrapper runs)

| Seed | Pre-quant BPB | Quantized roundtrip BPB | Sliding BPB | TTT BPB | TTT eval (s) | N-gram precompute (s) | N-gram hint coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0    | 1.08728 | 1.09923 | 1.08209 | 1.07751 | 335.5 | 31.9 | 22.38% |
| 42   | 1.08785 | 1.09937 | 1.08268 | 1.07809 | 316.6 | 32.2 | 22.38% |
| 1234 | 1.08794 | 1.09941 | 1.08280 | 1.07813 | 332.2 | 32.0 | 22.38% |
| 1337 | 1.08772 | 1.09918 | 1.08246 | 1.07801 | 338.4 | 31.9 | 22.38% |
| 2025 | 1.08842 | 1.09957 | 1.08306 | 1.07862 | 333.4 | 32.0 | 22.38% |

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
export NGRAM_WITHIN_BETA=0.92

for SEED in 0 42 1234; do
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
