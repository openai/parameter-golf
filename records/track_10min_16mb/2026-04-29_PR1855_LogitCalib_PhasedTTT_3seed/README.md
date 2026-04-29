# Record: PR #1855 base + Smear + LQER Asym + ATTN_CLIP_SIGMAS=14 + Logit Calibration + Phased TTT

**val_bpb: 1.06080** (3-seed mean, std 0.00095) | **2.32143 nats** | **~15.80 MB** | 8×H100 SXM, 600s train / 600s eval | Phased TTT

> Extends the PR #1855 family (PR #1787 native base + NUM_LOOPS=2 triple recurrence) with our full stack: Smear gate (BOS-masked), LQER asymmetric rank-4 correction, and phased TTT — plus one new mechanism: **logit calibration**, an affine per-token-category correction (scale + per-category bias vector) fitted on the first 100K train tokens post-GPTQ. The correction takes ~5s and costs ≈5,200 compressed bytes from the 16MB budget.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

### Core table (phased TTT)

| Seed | Steps | Pre-TTT BPB | Post-TTT BPB | TTT gain | Eval time | Artifact (bytes) |
|------|------:|------------:|-------------:|---------:|----------:|-----------------:|
| 314  | 4969  | 1.07281     | **1.06011**  | -0.01270 | 479.7s    | 15,789,408       |
| 42   | 4974  | 1.07304     | **1.06040**  | -0.01264 | 437.9s    | 15,787,251       |
| 1234 | 4938  | 1.07460     | **1.06189**  | -0.01271 | 433.4s    | 15,795,987       |
| **Mean** | 4960 | **1.07348** | **1.06080** | **-0.01268** | **450.3s** | **15,790,882** |
| **Std**  | —    | 0.00097     | **0.00095** |              | 26.1s      | 4,632          |

### Supplemental diagnostics

| Seed | Post-EMA BPB (pre-quant) | Quantized BPB (no TTT) | Post-TTT BPB | val_loss (nats) | Train time | Eval time |
|------|-------------------------:|-----------------------:|-------------:|----------------:|-----------:|----------:|
| 314  | 1.06388                  | 1.07281                | 1.06011      | 2.31992         | 599.56s    | 479.7s    |
| 42   | 1.06403                  | 1.07304                | 1.06040      | 2.32056         | 599.59s    | 437.9s    |
| 1234 | 1.06556                  | 1.07460                | 1.06189      | 2.32380         | 599.63s    | 433.4s    |

All seeds must clear both 600s budgets (train + eval) and the 16,000,000-byte decimal artifact cap.

## Key innovation — logit calibration

After GPTQ quantization, the model's output logit distribution shifts slightly. We fit a lightweight affine correction:

```python
# post-GPTQ, on first 100K train tokens (train data only — no val):
# 14 token categories: upper, lower, alpha, digit, space, len1-len5, starts_space, contains_digit
# For each position t: logit[v] += bias[category[v]]
# Plus a global log-scale: logits *= scale  (scale ≈ 0.997)
```

Fitting is O(100K tokens × 14 groups × vocab), takes ~5s on 8 GPUs, and costs ~5,200 compressed bytes in the artifact. Condition 3 compliance: calibration uses TRAIN tokens only (no val tokens, no score-before-update concern).

Example calibration output (s314): `scale:0.99798 upper:-0.0325 len1:-0.0261 len3:+0.0201 contains_digit:+0.0149 starts_space:-0.0129`

### Mechanism stack

| Component | Origin | Role |
|-----------|--------|------|
| CaseOps bijective case transform | PR #1729 (romeerp) / PR #1736 (ours) | ~1.5% token savings, full byte-level bijection |
| SparseAttnGate | PR #1787 (nprime06) | sparse per-head gate inside attention |
| NUM_LOOPS=2 (triple recurrence) | PR #1855 (codemath3000) | encoder layers 3-5 looped 2× before decoder |
| Smear gate (BOS-masked) | PR #1797 (ours) | causal 12-dim gate + 1-token lookback; doc-boundary safe |
| LQER asymmetric rank-4 correction | PR #1797 (ours) | post-GPTQ int6 residual recovery |
| Phased TTT (score-first, 3 phases, 2500-doc prefix) | PR #1394 / PR #1797 (ours) | per-document LoRA adapter, score-before-update |
| ATTN_CLIP_SIGMAS=14.0 | this family | retuned attention-output GPTQ clip |
| **Logit calibration** | **this submission** | affine scale + per-category bias on output logits, train-only fit |

## Changes from prior submission (PR #1797, banked at 1.06412)

| Component | PR #1797 | This submission |
|-----------|----------|-----------------|
| Base | PR #1787 native + Smear + LQER + Phased TTT | + NUM_LOOPS=2 (PR #1855 base) |
| `ATTN_CLIP_SIGMAS` | 13.0 | **14.0** |
| `MLP_CLIP_SIGMAS` | 12.0 | **11.5** |
| `PHASED_TTT_PREFIX_DOCS` | 2000 | **2500** |
| `TTT_LORA_RANK` | 128 | **80** |
| `WARMDOWN_FRAC` | 0.75 | **0.85** |
| `BETA2` / `TTT_BETA2` | 0.95 / default | **0.99 / 0.99** |
| Logit calibration | disabled | **enabled (100K train tokens)** |
| Compressor | brotli | **per-group lrzip + brotli** |

## Architecture

| Item | Value |
|------|------:|
| num_layers | 11 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 |
| mlp_mult | 4.0 |
| num_loops | 2 (encoder layers 3-5 looped) |
| parallel_start_layer | 8 |
| eval_seq_len / eval_stride | 2048 / 64 |
| matrix_bits / embed_bits | 6 / 7 |
| attn / mlp / embed clip σ | 14.0 / 11.5 / 14.0 |
| LQER rank / top-K / A-bits / B-bits / asym group | 4 / 3 / 2 / 4 / 64 |
| smear gate window | 12 |
| compressor | per-group lrzip + brotli |
| logit_calib tokens / stride / lr | 100,000 / 64 / 0.003 |

## Rule compliance

- **Artifact ≤ 16,000,000 bytes**: s314 = 15,789,408 bytes, s42 = 15,787,251 bytes (both under cap; ~211-213 KB headroom).
- **train_time ≤ 600s**: s314 = 599.56s, s42 = 599.59s (both `stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: s314 = 479.7s, s42 = 437.9s. Phased TTT amortizes across 3 phases; logit calibration adds ~5s before TTT.

### Issue #1017 four-condition compliance

- **Condition 1 — strict causal dependence**: transformer attention reads only positions ≤ t. Smear gate BOS-mask zeros the previous-token term at doc boundaries (`train_gpt.py:1336`, `:1435`). Logit calibration is a static affine correction applied to logits — no per-token context dependence, no future tokens.
- **Condition 2 — full normalized distribution**: standard softcapped CE over full 8192-token softmax at every position. Logit calibration does not restrict the distribution (it's an affine shift before softmax, which still normalizes over all 8192 tokens).
- **Condition 3 — score-before-update**: phased TTT: `per_tok_loss` accumulated BEFORE `cur_opt.step()`. Logit calibration: fitted on TRAIN tokens only — no val token data, no score-before-update concern.
- **Condition 4 — single left-to-right pass**: each token scored exactly once via `_accumulate_bpb`. Stride-64 sliding eval. Logit calibration is a single inference-time transform (no rescoring).

### Length-sort defense (validation batching)

The TTT eval path length-sorts validation docs for batching efficiency (`_build_ttt_global_batches`, `train_gpt.py:3007`: `sorted(doc_entries, key=lambda x: x[1][1])`). **Each val token is scored exactly once via `_accumulate_bpb` BEFORE any TTT update touches it** — the length-sort only affects compute layout, not which-token-sees-which-model-state. Cond 4 ("single L→R pass") holds at the token level: the per-doc `chunk_offset/chunk_len` window in `_accumulate_bpb` (`train_gpt.py:3046`) selects each val position exactly once.

**Merged precedent**: this exact pattern (length-sorted batching for TTT) is in merged PR #77 (LoRA_TTT, 2026-03-17) — `rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)` at `2026-03-17_LoRA_TTT/train_gpt.py:871`. Also used by the PR #1394 / PR #1736 phased-TTT lineage (merged). Score-before-update at `:3372` accumulate, `:3387 if needs_train` train.

### Logit calibration defense (post-GPTQ train-data fit)

Logit calibration is a STATIC post-quant correction fit ONCE on TRAIN tokens (`fit_logit_calibration`, `train_gpt.py:658-741`; train shards loaded at `:669` via `h.train_files`, `_bytes_` sidecars filtered out). The learned `(scale, bias)` is frozen for the entire eval phase and applied uniformly per token-id (`scale * logits + bias` at `:1693`). It does NOT touch val data, does NOT depend on x_t at runtime, and does NOT restrict Σ (the affine transform commutes with the softmax denominator over the full 8192 vocab). This is the same class of mechanism as ValCalib in the PR #1019 lineage — a stationary statistic learned on train and frozen for eval.

### Section V — byte-level BPB

Canonical PR #1019 formula: `base_bytes_np[token_id] = len(piece.encode("utf-8"))` after stripping `▁`, with +1 space credit applied once via `is_boundary_token_lut`.

## Requirements

```bash
apt-get install -y lrzip
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece brotli numpy huggingface_hub python-minifier
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

## Data setup (run ONCE)

```bash
python3 prepare_caseops_data.py \
    --docs ./fineweb10B_raw/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

## Run command (3-seed reproduction)

```bash
for SEED in 314 42 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  PHASED_TTT_NUM_PHASES=3 PHASED_TTT_PREFIX_DOCS=2500 \
  EMBED_BITS=7 \
  MATRIX_LR=0.026 MIN_LR=0.1 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=14.0 EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 \
  TTT_CHUNK_SIZE=48 \
  WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  WARMDOWN_FRAC=0.85 \
  BETA2=0.99 TTT_BETA2=0.99 \
  TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  VAL_LOSS_EVERY=0 \
  GATED_ATTN_QUANT_GATE=1 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  GATE_WINDOW=12 SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 \
  LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  FUSED_CE_ENABLED=1 \
  COMPRESSOR=pergroup \
  LOGIT_CALIB_ENABLED=1 LOGIT_CALIB_TOKENS=100000 LOGIT_CALIB_STRIDE=64 \
  LOGIT_CALIB_BATCH_SEQS=8 LOGIT_CALIB_LR=0.003 LOGIT_CALIB_L2=0.01 \
  LOGIT_CALIB_EPOCHS=1 LOGIT_CALIB_APPLY_TTT_UPDATE=1 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Lineage

- **PR #549** — original modded-nanogpt (Keller Jordan).
- **PR #1019** (merged) — byte-level BPB SentencePiece accounting.
- **PR #1394** (merged) — SP8192 + multi-phase score-first TTT baseline.
- **PR #1530** (samacqua) — Loop4-5 + parallel residual.
- **PR #1586** — per-group lrzip compression.
- **PR #1626** (ours) — GPTQ trimming + multi-phase SGD + adaptive clip.
- **PR #1729** (romeerp) — CaseOps bijective transform + byte sidecar.
- **PR #1736** (ours) — CaseOps + gated attention + phased TTT.
- **PR #1767** — TTT warm-start-A initialization.
- **PR #1787** (nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE.
- **PR #1797** (ours, submitted, 1.06412 BOS-fix) — PR #1787 base + Smear + LQER + Phased TTT.
- **PR #1855** (codemath3000) — NUM_LOOPS=2 triple recurrence on PR #1787 base.
- **This submission** — PR #1855 family + ATTN_CLIP_SIGMAS=14 + Smear + LQER + Phased TTT + logit calibration.

## Credits

- @codemath3000 — PR #1855 NUM_LOOPS=2 triple recurrence.
- @nprime06 — PR #1787 base stack.
- @msisovic — SmearGate BOS-mask fix on PR #1797.
- @samacqua — PR #1530 loop + parallel residuals.
- @romeerp — PR #1729 CaseOps.
- @bigbag — PR #1493 merged SOTA (1.0810).

## Included files

- `train_gpt.py` — training script (168,434 bytes).
- `submission.json` — metadata (3-seed results).
- `README.md` — this file.
- `train_seed314.log`, `train_seed42.log`, `train_seed1234.log` — run logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model.
- `lossless_caps.py` — bijective CaseOps transform.
- `prepare_caseops_data.py` — data prep script.
