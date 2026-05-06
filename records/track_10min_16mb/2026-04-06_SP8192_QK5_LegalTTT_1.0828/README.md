# Record: SP8192 + QK-Gain 5 + Legal Score-First TTT — val_bpb 1.08279 (3-seed mean)

**val_bpb: 1.08279** (3-seed mean, std ~0.00049) | **2.79697 nats** (per token, mean) | **~15.99 MB** | 8×H100 SXM, 600s | Legal Score-First TTT

Beats [PR #1394](https://github.com/openai/parameter-golf/pull/1394) (1.08563) by **0.00283 bpb / 0.00731 nats per token** on a 3-seed mean, clearing the 0.005 nats record threshold.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, legal score-first TTT)

### Core (TTT) table

| Seed | Steps | Pre-TTT sliding bpb | Post-TTT bpb | TTT gain | TTT time | Artifact |
|---:|---:|---:|---:|---:|---:|---:|
| 0    | 5088 | 1.08397 | **1.08210** | −0.00187 | 293.4 s | 15,991,018 |
| 42   | 5088 | 1.08470 | **1.08315** | −0.00155 | 289.9 s | 15,992,546 |
| 1234 | 5088 | 1.08590 | **1.08314** | −0.00276 | 295.3 s | 15,989,058 |
| **mean** | | **1.08486** | **1.08279** | −0.00206 | 292.9 s | 15,990,874 |

### Diagnostics

| Seed | Post-EMA bpb | Quant roundtrip bpb | Sliding bpb | val_loss (nats) | Code bytes | Total submission | Train ms | Eval ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0    | 1.08924 | 1.10019 | 1.08397 | 2.79517 | 16,719 | 15,991,018 | 588,004 | 385,050 |
| 42   | 1.08950 | 1.10068 | 1.08470 | 2.79788 | 16,719 | 15,992,546 | 588,009 | 381,500 |
| 1234 | 1.08967 | 1.10146 | 1.08590 | 2.79785 | 16,719 | 15,989,058 | 588,000 | 386,880 |

## Key Innovation

A single-knob improvement on top of [@clarkkev's PR #1394](https://github.com/openai/parameter-golf/pull/1394) sp8192 baseline + a legal score-first TTT eval pass:

1. **QK_GAIN_INIT = 5.0** (vs PR #1394's 4.0) — raised attention query/key scaling on the same arch.
2. **Legal score-first TTT** — score each sliding window chunk under `inference_mode()` BEFORE any gradient update; each chunk is only trained on after it has been fully scored. No chunk is trained on before scoring.

```python
for chunk_idx, chunk_windows in enumerate(chunks):
    # Phase 1: SCORE (no grad, no model update)
    with torch.inference_mode():
        nll = model.forward_logits(batch).cross_entropy(targets)
    loss_sum += nll.sum()

    # Phase 2: TRAIN (only on the chunk just scored, never on anything still-to-score)
    if not is_last_chunk:
        for _ in range(ttt_epochs):
            for x, y in chunk_seqs:
                loss = model(x, y)
                loss.backward()
                optimizer.step()
```

Strict score-before-update ordering matches the PR #549 precedent and satisfies [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) conditions 1–4. No eval-time delta optimization (no SLOT), no pre-quant TTT on val data, no two-pass rescoring, no n-gram cache.

## Changes from baseline (PR #1394)

| Component | PR #1394 | This PR |
|---|---|---|
| Tokenizer | SentencePiece BPE 8192 | SentencePiece BPE 8192 (same) |
| Architecture | 11L / 512d / 8H / 4KV, MLP 4x, Partial RoPE 16d | (same) |
| Depth recurrence | Loop layers 4–5 twice from 50% training | (same) |
| Optimizer | MuonEq-R (row-normalized Muon), WD=0.085 | (same) |
| Quantization | GPTQ int6 matrices + int8 embeddings + SD-clip (matrix_clip_sigmas=12.85, embed_clip_sigmas=20.0) | (same) |
| **QK_GAIN_INIT** | **4.0** | **5.0** |
| **TTT** | **none** | **Legal score-first, LR=0.005, epochs=3, freeze=0** |
| val_bpb (3-seed mean, sliding no-TTT) | 1.08563 | **1.08486** |
| val_bpb (3-seed mean, post-TTT) | — | **1.08279** |
| Δ vs PR #1394 baseline (nats/token) | — | **−0.00731** |

## Architecture

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)² activation, Partial RoPE (16 / 64 dims), layerwise LN scale, tied token embeddings. Depth recurrence: encoder [0,1,2,3,4,5,4], decoder [5,4,5,6,7,8,9,10] (loops layers 4–5 twice, activated at step 2885 ≈ 50% training).

Quantization: full-Hessian GPTQ on all attention/MLP matrices at int6 with SD-based clip (row_std × 12.85 / 31 step); token embedding at int8 with clip 20 × row_std; small control tensors and scalars kept float16/float32 via passthrough. Compression: byte-shuffle + Brotli-11. Self-extracting LZMA mini runner (~16.7 KB code).

## Rule Compliance

Per [repo README](https://github.com/openai/parameter-golf) and [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) four conditions:

- **Condition 1 (Causality)**: Strict causal forward pass. Sliding-window eval never references future tokens for current-position scoring.
- **Condition 2 (Normalized distribution)**: Standard softmax over full vocab, no normalization trickery. No BigramHash, no two-pass rescoring, no logit biasing.
- **Condition 3 (Score before update)**: Every TTT chunk is scored under `inference_mode()` BEFORE any parameter update. Gradient updates only use already-scored tokens. Score-first pattern matches merged precedent PR #549.
- **Condition 4 (Single pass)**: Each token is scored exactly once. No rescoring, no cache lookups.

Additional:
- **No SLOT** (standard or causal). No eval-time delta optimization in hidden space.
- **No pre-quant TTT on val data**. The model is quantized once after training, then evaluated.
- **No ETLB** (eval-time logit bias).
- **No n-gram cache** at eval.
- **No tokenizer change** — uses PR #1394's SentencePiece BPE 8192 unchanged.
- **Artifact under 16 MB** on all 3 seeds (margins 7,454–10,942 bytes).
- **Training under 600s** on all 3 seeds (~588 s actual).
- **Eval under 600s** on all 3 seeds (~382 s actual: 8 s roundtrip + 83 s sliding + 290 s TTT).
- **3 distinct seeds** (0, 42, 1234) — independent runs on the same hardware.

## Requirements

```
torch==2.9.1+cu128
flash-attn==2.8.3
flash-attn-3 (interface wheel; Hopper build)
sentencepiece
numpy
torch.distributed (NCCL)
```

GCP 8×H100 80GB SXM pod with `NCCL_NET=Socket` (GCP-specific; NCCL 2.27.5 + gIB device issue).

## Run Command

```bash
export NCCL_NET=Socket
export QK_GAIN_INIT=5.0
export TTT_ENABLED=1
export TTT_LR=0.005
export TTT_EPOCHS=3

for SEED in 0 42 1234; do
    SEED=$SEED uv run torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Lineage

- **[PR #1394](https://github.com/openai/parameter-golf/pull/1394)** (@clarkkev) — SP8192 + GPTQ embeddings + SD-clip + MuonEq-R + depth recurrence — base stack used unchanged
- **[PR #1019](https://github.com/openai/parameter-golf/pull/1019)** (@abaybektursun) — Full-Hessian GPTQ + XSA + BigramHash — GPTQ calibration pipeline
- **[PR #549](https://github.com/openai/parameter-golf/pull/549)** (@abaybektursun) — LeakyReLU² + score-first TTT precedent — our TTT implementation follows this pattern
- **[PR #461](https://github.com/openai/parameter-golf/pull/461)** (@Christopher-Lee-McClendon) — LoRA TTT framework — earlier legal-TTT reference

## Credits

- **@clarkkev** for the sp8192 base stack (PR #1394) this submission builds on unchanged
- **@abaybektursun** for the GPTQ-XSA lineage and the legal-TTT precedent (PR #549)
- **@Christopher-Lee-McClendon** for the LoRA TTT reference (PR #461)
- **@unnir** for XSA (PR #265)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed0.log`
- `train_seed42.log`
- `train_seed1234.log`
