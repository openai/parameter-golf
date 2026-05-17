# Record: SP8192 + Muon momentum 0.97 + Legal Score-First TTT + Causal N-gram Token Tilt — val_bpb 1.07983 (3-seed mean)

**val_bpb: 1.07983** (3-seed mean, std 0.00050) | **2.78932 nats** (per token, mean) | **~15.99 MB** | 8xH100 SXM, 600s | Legal Score-First TTT + causal token n-gram tilt

Beats [PR #1493](https://github.com/openai/parameter-golf/pull/1493) (1.0810) by **0.00117 bpb / 0.00302 nats per token** on the current merged legal-track SOTA, and beats our own [PR #1413](https://github.com/openai/parameter-golf/pull/1413) (1.08279) by **0.00296 bpb / 0.00764 nats per token** on a 3-seed mean.

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, legal score-first TTT + causal n-gram token tilt)

### Core (TTT) table

| Seed | Steps | Pre-TTT sliding bpb | Post-TTT bpb | TTT gain | TTT time | Artifact |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 4911 | 1.08102 | **1.07928** | -0.00174 | 341.6 s | 15,993,346 |
| 42   | 4915 | 1.08167 | **1.07997** | -0.00170 | 330.2 s | 15,992,995 |
| 1234 | 4909 | 1.08194 | **1.08025** | -0.00169 | 335.0 s | 15,994,604 |
| **mean** | 4912 | **1.08154** | **1.07983** | -0.00171 | 335.6 s | 15,993,648 |

### Diagnostics

| Seed | Post-EMA bpb | Quant roundtrip bpb | Sliding bpb | val_loss (nats) | Code bytes | Total submission | Train ms | Eval ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0    | 1.08687 | 1.09772 | 1.08102 | 2.78790 | 19,387 | 15,993,346 | 588,052 | 442,336 |
| 42   | 1.08754 | 1.09822 | 1.08167 | 2.78967 | 19,387 | 15,992,995 | 588,045 | 436,478 |
| 1234 | 1.08783 | 1.09857 | 1.08194 | 2.79039 | 19,387 | 15,994,604 | 588,131 | 435,641 |

(Eval ms = roundtrip + sliding + ngram_tilt precompute + TTT; all values under the 600s eval budget.)

## Key Innovations

Three small but compounding changes on top of [@clarkkev's PR #1394](https://github.com/openai/parameter-golf/pull/1394) sp8192 baseline and our own PR #1413 legal-TTT pass:

1. **Muon momentum = 0.97** (vs PR #1394's default 0.99) — a single-knob hyperparameter change that consistently reduces BPB by ~0.0005 across seeds at our step budget. Warmup schedule unchanged: 0.92 -> 0.97 over the first 1500 steps.
2. **Legal score-first TTT** — already present in our PR #1413. Each sliding-window chunk is scored under `inference_mode()` BEFORE any gradient update; each chunk is only trained on AFTER it has been fully scored. No chunk is trained on before scoring.
3. **Causal n-gram token tilt** — a prefix-only exponential tilt on the model's softmax using the token expert from [@abaybektursun's PR #1420](https://github.com/openai/parameter-golf/pull/1420) kernel. The within-word and word-start experts are **explicitly disabled** (`within_beta=0`, `word_beta=0`) because they cannot be made fully causal without losing most of their benefit (see Legality section below). Only the token expert (`base_beta=2.0`, `agree_bonus=0.1`) contributes.

```python
# Score-first TTT (same pattern as PR #1413 / PR #549 precedent)
for chunk_idx, chunk_windows in enumerate(chunks):
    # Phase 1: SCORE (no grad, no model update) with causal n-gram tilt on the NLL
    with torch.inference_mode():
        logits = model.forward_logits(batch)
        nll = F.cross_entropy(logits, targets, reduction='none')
        mixed_nll = ngram_state.tilt_nll(nll, logits, targets, global_positions)
    loss_sum += mixed_nll.sum()

    # Phase 2: TRAIN (only on the chunk just scored, never on anything still-to-score)
    if not is_last_chunk:
        for _ in range(ttt_epochs):
            for x, y in chunk_seqs:
                loss = model(x, y)
                loss.backward()
                optimizer.step()
```

## N-gram tilt legality (summary)

The n-gram tilt kernel comes from PR #1420. That PR was found (and the author acknowledged) to have a subtle causality bug in the within-word and word-start expert gating: both experts read `is_bnd[tokens[p]]` / `is_ws[tokens[p]]`, which depend on the TARGET token at position p, leaking 1-2 bits about the answer per scored position.

The fix applied in this submission:

1. **Hint gating uses prefix-only metadata.** The within and word hints are gated on `is_bnd[tokens[p-1]]` and `is_ws[tokens[p-1]]` — i.e. the last PREFIX token, not the target. This is the fix @abaybektursun proposed in the PR #1420 thread.
2. **Within/word experts are disabled at eval time.** Empirically, the within and word experts contribute negligibly once the leak is removed, so we set `within_beta=0` and `word_beta=0` in the config. Only the `token_hint` expert is active, and it is causal by construction — it reads the prefix hash table BEFORE the update for position p is written.
3. **Update functions are causal.** `token_update` / `within_update` / `word_update` for position p happen AFTER the hint for p has been read out; they use the target token at p (which the model has already scored) to build state for positions p+1, p+2, ... This is strictly score-before-update at the kernel level.

Both modifications are in `ngram/fused_expert_kernel.cpp` (see the `// CAUSAL FIX` comment block). The compiled shared library is built on-the-fly by g++ the first time the eval path runs.

## Changes from baseline (PR #1394)

| Component | PR #1394 | This submission |
|---|---|---|
| Tokenizer | SentencePiece BPE 8192 | SentencePiece BPE 8192 (same) |
| Architecture | 11L / 512d / 8H / 4KV, MLP 4x, Partial RoPE 16d | (same) |
| Depth recurrence | Loop layers 4-5 twice from 50% training | Loop layers 3-5 twice from 50% training (PR #1413) |
| Parallel residual | none | From layer 7 onwards (PR #1413) |
| Optimizer | MuonEq-R (row-normalized Muon), WD=0.085 | Same, but **muon_momentum=0.97** (vs 0.99) |
| Quantization | GPTQ int6 matrices + int8 embeddings + SD-clip | (same) |
| QK_GAIN_INIT | 4.0 | 5.0 (from PR #1413) |
| **TTT** | **none** | **Legal score-first, LR=0.005, epochs=3, freeze=0** |
| **N-gram tilt** | **none** | **Causal token expert only (base_beta=2.0, within/word disabled)** |
| val_bpb (3-seed mean) | 1.08563 | **1.07983** |
| Delta vs PR #1394 baseline (nats/token) | - | **-0.01497** |

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2 activation, Partial RoPE (16 / 64 dims), layerwise LN scale, tied token embeddings. Depth recurrence: loops layers 3-5 twice, activated at step ~2920 (50% training). Parallel residual from layer 7 onwards.

Quantization: full-Hessian GPTQ on all attention/MLP matrices at int6 with SD-based clip (row_std x 12.85 / 31 step); token embedding at int8 with clip 20 x row_std; small control tensors and scalars kept float16/float32 via passthrough. Compression: byte-shuffle + Brotli-11. Self-extracting LZMA mini runner (~19.4 KB code).

## Rule Compliance

Per [repo README](https://github.com/openai/parameter-golf) and [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) four conditions:

- **Condition 1 (Causality)**: Strict causal forward pass. Sliding-window eval never references future tokens for current-position scoring. The n-gram tilt kernel reads prefix-only hash tables; `// CAUSAL FIX` block in `ngram/fused_expert_kernel.cpp` documents the prefix-only gating.
- **Condition 2 (Normalized distribution)**: The tilted distribution remains a proper probability distribution over the full vocabulary. For each position with a hint, we compute `p_tilt(t) = p_model(t) * exp(beta * 1[t==hint]) / Z` where `Z = 1 + p_hint * (exp(beta) - 1)`, which is a positive re-weighting followed by renormalization across the full vocab. No logit biasing outside the vocab, no BigramHash, no two-pass rescoring.
- **Condition 3 (Score before update)**: Every TTT chunk is scored under `inference_mode()` BEFORE any parameter update. Gradient updates only use already-scored tokens. Score-first pattern matches merged precedent PR #549. Additionally, the n-gram kernel's `token_update` for position p happens strictly AFTER the hint for p has been read from the open-addressing hash table.
- **Condition 4 (Single pass)**: Each token is scored exactly once. No rescoring, no two-pass eval, no selection between alternatives.

Additional:
- **No SLOT** (standard or causal). No eval-time delta optimization in hidden space.
- **No pre-quant TTT on val data**. The model is quantized once after training, then evaluated.
- **No ETLB** (eval-time logit bias on future tokens).
- **No hashed n-gram cache with uncontrolled tables**. The n-gram state table is deterministic, fixed, and prefix-only.
- **No tokenizer change**. Uses PR #1394's SentencePiece BPE 8192 unchanged.
- **Artifact under 16 MB** on all 3 seeds (margins 5,396 - 7,005 bytes).
- **Training under 600s** on all 3 seeds (~588 s actual, wallclock cap).
- **Eval under 600s** on all 3 seeds (~436-442 s actual: ~8 s roundtrip + ~92 s sliding + ~33 s n-gram precompute + ~330-342 s TTT).
- **3 distinct seeds** (0, 42, 1234) - independent runs on the same hardware.

## Requirements

```
torch==2.9.1+cu128
flash-attn==2.8.3
flash-attn-3 (interface wheel; Hopper build)
sentencepiece
numpy
torch.distributed (NCCL)
g++ with C++17 (for on-the-fly ngram kernel compilation)
```

GCP 8xH100 80GB SXM pod with `NCCL_NET=Socket` (GCP-specific; NCCL 2.27.5 + gIB device issue).

## Run Command

```bash
export NCCL_NET=Socket
export MUON_MOMENTUM=0.97
export QK_GAIN_INIT=5.0
export PARALLEL_RESIDUAL_START=7
export LOOP_START=3
export LOOP_END=5
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768
export TTT_FREEZE_BLOCKS=0
export NGRAM_TILT_ENABLED=1
export NGRAM_WITHIN_BETA=0.0
export NGRAM_WORD_BETA=0.0

for SEED in 0 42 1234; do
    SEED=$SEED uv run torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Lineage

- **[PR #1394](https://github.com/openai/parameter-golf/pull/1394)** (@clarkkev) - SP8192 + GPTQ embeddings + SD-clip + MuonEq-R + depth recurrence - base stack
- **[PR #1413](https://github.com/openai/parameter-golf/pull/1413)** (dexhunter, ours) - QK_GAIN_INIT=5.0 + legal score-first TTT on top of PR #1394 - direct predecessor
- **[PR #1420](https://github.com/openai/parameter-golf/pull/1420)** (@abaybektursun) - N-gram tilt kernel (ContextMixer) - we use the token expert only, with the within/word causal fix from the PR thread
- **[PR #1019](https://github.com/openai/parameter-golf/pull/1019)** (@abaybektursun) - Full-Hessian GPTQ + XSA + BigramHash - GPTQ calibration pipeline
- **[PR #549](https://github.com/openai/parameter-golf/pull/549)** (@abaybektursun) - LeakyReLU^2 + score-first TTT precedent
- **[PR #461](https://github.com/openai/parameter-golf/pull/461)** (@Christopher-Lee-McClendon) - LoRA TTT framework - earlier legal-TTT reference

## Credits

- **@clarkkev** for the sp8192 base stack (PR #1394)
- **@abaybektursun** for the GPTQ-XSA lineage, the legal-TTT precedent (PR #549), and the n-gram tilt kernel (PR #1420) including the causal fix acknowledged in the PR #1420 thread
- **@Christopher-Lee-McClendon** for the LoRA TTT reference (PR #461)
- **@unnir** for XSA (PR #265)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` (self-extracting LZMA runner, ~19.4 KB)
- `ngram_tilt.py` (eval-time helper that wraps the C++ kernel via ctypes; imported only when `NGRAM_TILT_ENABLED=1`)
- `ngram/fused_expert_kernel.cpp` (C++17 source, compiled on-the-fly by g++ -O3 at the start of eval; emits the prefix-only hint tables)
- `train_seed0.log`
- `train_seed42.log`
- `train_seed1234.log`
