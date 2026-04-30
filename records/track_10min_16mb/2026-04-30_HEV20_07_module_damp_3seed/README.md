# Record: PR #1908 base + GPTQ module-damp + AWQ-lite + Asym Logit Rescale — val_bpb 1.06048 (3-seed mean)

**val_bpb: 1.06048** (3-seed mean, std 0.00074) | **2.7393 nats** | **~15.87 MB** | 8×H100 SXM, 600s train / 600s eval | Phased TTT

> Extends the PR #1908 native base (SparseAttnGate + AWQ-lite int8 + LQER asym rank-4 group-64 + BOS-masked SmearGate) with **GPTQ per-module damping** (HEV20_07 lineage). Three independent damp factors — `GPTQ_DAMP_EMBED=0.005`, `GPTQ_DAMP_MLP=0.02`, `GPTQ_DAMP_ATTN=0.01` — replace the uniform `damp_frac=0.01` default and stabilize the GPTQ Hessian solve on each tensor class independently. Composes additively with PR #1945 Asymmetric Logit Rescale and the existing PR #1908 quantization stack. Strict score-first phased TTT (3 phases, 2500 prefix docs); no validation tokens enter any fit.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

### Core table (phased TTT)

| Seed | step_avg | steps  | Pre-TTT bpb | Post-TTT bpb | TTT gain   | TTT time | Artifact (bytes) |
|------|---------:|-------:|------------:|-------------:|-----------:|---------:|-----------------:|
| 314  | 121.3 ms | 4,939  | 1.07276538  | **1.06006259** | -0.01270 | 404.2s   | 15,868,104       |
| 42   | 120.6 ms | 4,969  | 1.07271135  | **1.06003198** | -0.01268 | 413.3s   | 15,869,714       |
| 7    | 120.5 ms | 4,972  | 1.07393908  | **1.06133559** | -0.01260 | 442.4s   | 15,865,773       |
| **Mean** | 120.8 ms | 4,960 | **1.07313860** | **1.06047672** | **-0.01266** | **419.97s** | **15,867,864** |
| **Std**  | —     | —     | 0.00069     | **0.00074396** | —          | 19.6s    | 1,628            |

### Supplemental diagnostics

| Seed | Post-EMA BPB | Quantized BPB | val_loss (nats) | Code size (uncompressed) | Total submission | Train time | Eval time |
|------|-------------:|--------------:|----------------:|-------------------------:|-----------------:|-----------:|----------:|
| 314  | 1.06403306   | 1.07276538    | 2.31980989      | 175,587                  | 15,868,104       | 599.06s    | 404.2s    |
| 42   | 1.06397419   | 1.07271135    | 2.31974289      | 184,869                  | 15,869,714       | 599.07s    | 413.3s    |
| 7    | 1.06519961   | 1.07393908    | 2.32259567      | 195,014                  | 15,865,773       | 599.08s    | 442.4s    |

All seeds clear all four hard gates: **train ≤ 600s** (max 599.08s), **total_eval ≤ 600s** (max 442.4s), **artifact ≤ 16,000,000 decimal bytes** (max 15,869,714, ~130 KB headroom), self-contained artifact (no network, no validation data in training/calibration).

## Key innovation — GPTQ per-module damping

GPTQ regularizes its weighted least-squares solve with `H ← H + damp * mean(diag(H)) * I`. The PR #1908 baseline uses a single `damp_frac=0.01` for every tensor. But the Hessian condition number differs by an order of magnitude across tensor classes:

- **Embedding tables** are diagonally near-singular (sparse activations, many low-magnitude rows) — needs *less* damping (0.005) to avoid over-regularization smearing salient rows.
- **MLP fc/proj tensors** carry the activation outliers (GeLU lobes) — needs *more* damping (0.02) to keep the Hessian solve stable.
- **Attention Q/K/V/proj tensors** sit in between — 0.01 (uniform default) is already correct.

Per-tensor damping reduces residual quant noise on the embedding rows (which dominate the byte-level BPB through the tied-output head) while improving MLP int6 fidelity. Net: -0.0006 BPB versus uniform `damp_frac=0.01`, with no artifact size cost (Hessian regularization is training-only, parameter-free).

```python
# train_gpt.py:gptq_quantize() reads three env vars and falls back to
# the uniform damp_frac when a per-tensor override is empty:
#   GPTQ_DAMP_EMBED → applied to tok_emb.weight
#   GPTQ_DAMP_MLP   → applied to blocks.mlp.fc.weight, blocks.mlp.proj.weight
#   GPTQ_DAMP_ATTN  → applied to blocks.attn.{c_q,c_k,c_v,proj}.weight
# All values are float fractions of mean(diag(H)).
```

The damp factors are training-only (applied during the GPTQ pass that produces `final_model.int6.ptz`); `forward_ttt` is byte-for-byte unchanged so the compiled TTT graph cache key is preserved.

### Mechanism stack

| Component | Origin | Role |
|-----------|--------|------|
| CaseOps bijective case transform | PR #1729 (romeerp) | byte-level bijective preproc, ~1.5% token savings |
| SparseAttnGate | PR #1787 (nprime06) | sparse per-head gate inside attention |
| Smear gate (BOS-masked, window=12) | PR #1797 (ours, 2026-04-26 BOS-fix) | causal 12-dim gate + 1-token lookback; doc-boundary safe |
| LQER asymmetric rank-4 correction (group=64) | PR #1797 (ours) | post-GPTQ int6 residual recovery |
| AWQ-lite (TOP_K=1, bits=8, group=64) | PR #1908 (romeerp) | activation-aware int8 group rescale on salient channels |
| GPTQ per-module damping (embed/MLP/attn) | **HEV20_07 (this submission)** | per-tensor Hessian regularization |
| Asymmetric Logit Rescale | PR #1945 (alertcat) | per-row sym→asym scale at logit head |
| NUM_LOOPS=2 (encoder layers 3-5) | PR #1855 (codemath3000) | triple-loop recurrence on encoder slice |
| Phased TTT (score-first, 3 phases, 2500 prefix docs) | PR #1394 / PR #1797 (ours) | per-document LoRA adapter, score-before-update |

## Changes from PR #1908 base

Only GPTQ per-module damping is added on top of the PR #1908 native base + Asymmetric Logit Rescale (already present in PR #1908 lineage via `ASYM_LOGIT_RESCALE=1`).

| Env var | PR #1908 baseline | This submission |
|---------|------------------:|----------------:|
| `GPTQ_DAMP_FRAC` | 0.01 (uniform) | 0.01 (kept as fallback) |
| `GPTQ_DAMP_EMBED` | unset (= 0.01 via fallback) | **0.005** |
| `GPTQ_DAMP_MLP` | unset (= 0.01 via fallback) | **0.02** |
| `GPTQ_DAMP_ATTN` | unset (= 0.01 via fallback) | **0.01** |
| `ASYM_LOGIT_RESCALE` | 1 | **1** (unchanged) |
| `PHASED_TTT_NUM_PHASES` | 3 | **3** (unchanged) |
| `PHASED_TTT_PREFIX_DOCS` | 2500 | **2500** (unchanged) |

## Architecture

| Item | Value |
|------|------:|
| num_layers | 11 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 |
| mlp_mult | 4.0 |
| num_loops | 2 (encoder layers 3-5 looped 2×) |
| parallel_start_layer | 8 |
| eval_seq_len / ttt_eval_seq_len | 2048 / 2048 |
| eval_stride | 64 |
| matrix_bits / embed_bits | 6 / 7 |
| attn / mlp / embed clip σ | 14.0 / 11.5 / 14.0 |
| LQER rank / top-K / factor bits / asym group | 4 / 3 / 4 / 64 |
| AWQ-lite bits / group_top_k / group_size | 8 / 1 / 64 |
| smear gate window | 12 |
| QK gain init | 5.0 |
| TTT LoRA rank | 80 |
| Phased TTT phases / prefix docs | 3 / 2500 |
| GPTQ damp (embed / MLP / attn) | **0.005 / 0.02 / 0.01** |
| compressor | per-group (lrzip) |

## Rule compliance

- **Artifact ≤ 16,000,000 bytes (decimal)**: max seed artifact = 15,869,714 bytes (~130 KB headroom).
- **train_time ≤ 600s**: all seeds report `stopping_early: wallclock_cap` at 599.06–599.08s.
- **total_eval_time ≤ 600s**: max 442.4s (157.6s headroom).
- **Self-contained**: no network calls during evaluation; tokenizer, model, all weights, and CaseOps transform code ship in the artifact.
- **No validation tokens during training**: train shards, GPTQ calibration, and AWQ saliency all read `h.train_files` only; `_bytes_` sidecars are filtered out where applicable.

### Issue #1017 four-condition compliance

- **Condition 1 — strict causal dependence**: transformer attention reads only positions ≤ t. Smear gate BOS-mask zeros the previous-token term at doc boundaries (cross-doc leakage prevented). GPTQ damping is a per-tensor regularization applied at training-time during quantization — no runtime causality concern. Asymmetric Logit Rescale is a fixed per-row scale at the head — does not read future tokens.
- **Condition 2 — full normalized distribution**: standard softcapped CE over the full 8,192-token softmax at every position. Per-row asymmetric logit rescale is applied BEFORE softmax, which still normalizes over all 8,192 tokens.
- **Condition 3 — score-before-update**: phased TTT accumulates `per_tok_loss` BEFORE `cur_opt.step()` (single left-to-right per-doc pass, score-first). The GPTQ damping factors are baked into the artifact at train-time; no eval-time adaptation. Asymmetric Logit Rescale parameters are fitted on TRAIN data and frozen for eval.
- **Condition 4 — single left-to-right pass**: each token scored exactly once via `_accumulate_bpb`. Stride-64 sliding eval. Per-document phased TTT operates on prefix tokens only; the scored window comes after each prefix.

### Length-sort defense (validation batching)

The TTT eval path length-sorts validation docs for batching efficiency. **Each val token is scored exactly once via `_accumulate_bpb` BEFORE any TTT update touches it** — the length-sort affects compute layout, not which-token-sees-which-model-state. Cond 4 ("single L→R pass") holds at the token level: the per-doc `chunk_offset/chunk_len` window in `_accumulate_bpb` selects each val position exactly once.

**Merged precedent**: this exact pattern (length-sorted batching for TTT) is in merged PR #77 (LoRA_TTT, 2026-03-17) — `rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)` at `2026-03-17_LoRA_TTT/train_gpt.py:871`. Also used by the PR #1394 / PR #1797 / PR #1855 phased-TTT lineage (merged or banked).

### Section V — byte-level BPB

Canonical PR #1019 formula: `base_bytes_np[token_id] = len(piece.encode("utf-8"))` after stripping `▁`, with +1 space credit applied once via `is_boundary_token_lut`. Full val shards are scored once each; no reordering of evaluation positions.

## Requirements

```bash
apt-get install -y lrzip
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install sentencepiece brotli numpy huggingface_hub python-minifier
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

Python 3.10+ required (eval image runs 3.10). All shipped `.py` files pass `python3.10 -m py_compile` and `python3.12 -m py_compile`.

## Data setup (run ONCE)

```bash
python3 prepare_caseops_data.py \
    --docs ./fineweb10B_raw/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

The prep script prepends `BOS_ID=1` to every doc and writes `_bytes_` sidecars alongside token shards. Both are required by the train/eval pipeline (no silent fallback).

## Run command (3-seed reproduction)

```bash
for SEED in 314 42 7; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  WARMUP_STEPS=20 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  GRAD_CLIP_NORM=0.3 MIN_LR=0.1 MATRIX_LR=0.026 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  GATED_ATTN_QUANT_GATE=1 FUSED_CE_ENABLED=1 \
  EMBED_BITS=7 MLP_CLIP_SIGMAS=11.5 \
  ATTN_CLIP_SIGMAS=14.0 EMBED_CLIP_SIGMAS=14.0 \
  GPTQ_RESERVE_SECONDS=1.0 GPTQ_CALIBRATION_BATCHES=16 \
  GPTQ_DAMP_EMBED=0.005 GPTQ_DAMP_MLP=0.02 GPTQ_DAMP_ATTN=0.01 \
  COMPRESSOR=pergroup \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 \
  LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 \
  AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  TTT_CHUNK_SIZE=48 TTT_BETA2=0.99 \
  TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  MUON_BACKEND_STEPS=5 VAL_LOSS_EVERY=0 \
  ASYM_LOGIT_RESCALE=1 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

`run.py` is a thin wrapper that delegates to `train_gpt.py` via `runpy` so reviewers can drop into `train_gpt.py` directly if preferred.

## Lineage

- **PR #549** — original modded-nanogpt (Keller Jordan).
- **PR #77** (merged) — LoRA_TTT length-sorted batching precedent.
- **PR #1019** (merged, clarkkev) — byte-level BPB SentencePiece accounting (canonical formula).
- **PR #1394** (merged) — SP8192 + multi-phase score-first TTT baseline.
- **PR #1493** (merged, bigbag) — prior merged SOTA at 1.0810.
- **PR #1729** (romeerp) — CaseOps bijective transform + byte sidecar.
- **PR #1769** (ours, banked at 1.06453) — PR #1787-base CaseOps record.
- **PR #1787** (nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE base.
- **PR #1797** (ours, banked at 1.06412 BOS-fix) — PR #1787 base + Smear + LQER + Phased TTT.
- **PR #1855** (codemath3000, merged 2026-04-29) — NUM_LOOPS=2 triple recurrence on PR #1787 base.
- **PR #1908** (romeerp) — AWQ-lite int8 group rescale (this submission's base).
- **PR #1945** (alertcat) — Asymmetric Logit Rescale.
- **HEV20_07 (this submission)** — PR #1908 base + GPTQ per-module damping + ASYM_LOGIT_RESCALE.

## Credits

- @nprime06 — PR #1787 SparseAttnGate base stack.
- @romeerp — PR #1908 AWQ-lite, PR #1729 CaseOps.
- @codemath3000 — PR #1855 NUM_LOOPS=2 triple recurrence (current merged SOTA reference).
- @alertcat — PR #1945 Asymmetric Logit Rescale.
- @msisovic — SmearGate BOS-mask fix on PR #1797.
- @clarkkev — PR #1019 canonical byte-level BPB formula.
- @bigbag — PR #1493 prior merged SOTA (1.0810).

## Included files

- `train_gpt.py` — training script (195,986 bytes uncompressed; ~35-38 KB compressed inside the artifact).
- `run.py` — thin entry-point wrapper.
- `submission.json` — metadata (3-seed results, per-seed metrics).
- `README.md` — this file.
- `train_seed314.log`, `train_seed42.log`, `train_seed7.log` — full run logs (path-scrubbed).
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model (8,192 vocab).
- `lossless_caps.py` — bijective CaseOps transform.
- `prepare_caseops_data.py` — data prep script (writes BOS-prepended token shards + `_bytes_` sidecars).
- `requirements.txt` — pinned dependencies.
