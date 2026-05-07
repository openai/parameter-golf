# Record: PR #1797 base + Smear Gate + LQER Asymmetric + Phased TTT + per-group lrzip — val_bpb 1.05993

**val_bpb: 1.05993** (3-seed mean, std 0.00049) | **val_loss: 2.31951 nats/token** (std 0.00106) | **~15.98 MB artifact** | 8×H100 SXM, 600s train / ≤600s eval | Phased TTT

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

### Core table (phased TTT)

| Seed | Steps | Pre-TTT (quantized) BPB | Post-TTT BPB | TTT gain  | TTT / total eval time | Artifact (bytes) |
|------|------:|------------------------:|-------------:|----------:|----------------------:|-----------------:|
| 314  | 4974  | 1.07273999              | **1.05993748** | -0.01280 | 513.3s                | 15,981,858       |
| 42   | 4984  | 1.07209178              | **1.05932556** | -0.01277 | 525.7s                | 15,979,215       |
| 999  | 4981  | 1.07324627              | **1.06051274** | -0.01273 | 514.1s                | 15,982,243       |
| **Mean** | **4980** | **1.07269**         | **1.05993**   | **-0.01277** | **517.7s**       | **15,981,105**   |
| **Std**  |           | 0.00058             | **0.00059**   |              | 6.9s              | 1,636            |

### Supplemental diagnostics

| Seed | Post-EMA BPB (pre-quant) | Quantized BPB (no TTT) | Post-TTT BPB | val_loss (nats) | Train time | Total eval time |
|------|-------------------------:|-----------------------:|-------------:|----------------:|-----------:|----------------:|
| 314  | 1.06436921               | 1.07273999             | 1.05993748   | 2.31953610      | 598.91s    | 525.7s          |
| 42   | 1.06369027               | 1.07209178             | 1.05932556   | 2.31819699      | 598.91s    | 513.3s          |
| 999  | 1.06507008               | 1.07324627             | 1.06051274   | 2.32079499      | 599.04s    | 514.1s          |

All 3 seeds clear both 600s budgets (train + eval) and the 16,000,000-byte decimal artifact cap. 3-seed post-TTT std is 0.00059 BPB.

## Key innovation — PR #1797 base + orthogonal Smear gate + inline LQER asymmetric factorization + per-group lrzip

This submission combines three components on top of the PR #1797 (dexhunter) upstream base:

1. **PR #1797 base stack** (CaseOps + SparseAttnGate + PolarNS + MIN_LR + FusedCE + TTT_WARM_A). `SPARSE_ATTN_GATE_ENABLED=1` is PR #1787's sparse per-head multiplicative gate applied inside attention, inherited through PR #1797.
2. **PR #1855** per-group `lrzip` + `brotli` compression pipeline (`COMPRESSOR=pergroup`).
3. Tightened quant clips: `ATTN_CLIP_SIGMAS=12.0`, `EMBED_CLIP_SIGMAS=12.0` to make GPTQ quantization results better about 0.001 BPB. `EMBED_WD=0.06` is also made total result better.

### Mechanism stack

| Component | Origin | Role |
|-----------|--------|------|
| CaseOps bijective case transform | PR #1729 (romeerp) / PR #1736 (ours) | ~1.5% token savings, full byte-level bijection |
| SparseAttnGate | PR #1787 (nprime06) | sparse per-head gate inside attention |
| Smear gate | PR #1797 (dexhunter) | causal content-conditioned gate on first 12 residual dims, adding 1-token lookback |
| LQER asymmetric rank-4 correction | PR #1797 (dexhunter) | post-GPTQ int6 residual recovery, INT2/INT4 asym factors on top-3 tensors |
| Phased TTT (score-first, 3 phases, 2000-doc prefix) | PR #1394 / PR #1736 | per-document LoRA adapter, score-before-update |
| Int6 GPTQ + Brotli + per-group lrzip compressor | PR #1019 / PR #1530 / PR #1855 | fits int6 model + factors + code under 16,000,000 bytes |

### Empirical result (3 seeds)

| Seed | val_bpb   | val_loss (nats) |
|------|----------:|----------------:|
| 42   | 1.05932556 | 2.31819699     |
| 314  | 1.05993748 | 2.31953610     |
| 999  | 1.06051274 | 2.32079499     |
| **Mean** | **1.05993** | **2.31951** |
| **Std**  | 0.00049   | 0.00106         |

3-seed mean clears the merged SOTA (PR #1493 at 1.0810) by **~0.0211 BPB ≈ 0.0547 nats/token**, well above the 0.005-nat record bar (sp8192: 0.005 nats ≈ 0.00194 BPB).

## Changes from PR #1797

| Component        | PR #1797 | This submission |
|------------------|----------|-----------------|
| Compressor       | brotli only | **PR #1855 per-group lrzip + brotli** (`COMPRESSOR=pergroup`) |
| ATTN_CLIP_SIGMAS | 13.0     | **12.0** |
| EMBED_CLIP_SIGMAS| 15.0     | **12.0** |
| EMBED_WD         | 0.085    | **0.06** |

Net on 3-seed mean: **−0.00564 BPB / −0.01217 val_loss (nats/token)** vs PR #1736 (1.06557 / 2.33168).

## Architecture (inherits PR #1797 shape)

| Item | Value |
|------|------:|
| num_layers | 11 |
| model_dim | 512 |
| num_heads / num_kv_heads | 8 / 4 |
| mlp_mult | 4.0 |
| rope_base / rope_dims | 10000 / 16 |
| logit_softcap | 30.0 |
| loop_start / loop_end | 3 / 5 (NUM_LOOPS=2) |
| parallel_start_layer | 8 |
| eval_seq_len / eval_stride | 2048 / 64 |
| matrix_bits / embed_bits | 6 / 7 |
| LQER rank / top-K / A-bits / B-bits / asym group | 4 / 3 / 2 / 4 / 64 |
| smear gate window | 12 |
| compressor | per-group lrzip + brotli |

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: all 3 seeds 15,979,215–15,982,243 bytes (~18–21 KB headroom).
- **train_time ≤ 600s**: all 3 seeds 598.91–599.04s (`stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: all 3 seeds 513.3–525.7s.
- **Issue #1017 Condition 1 (causal dependence)**: (a) SparseAttnGate and Smear gate are pure functions of previous-token context (the Smear gate reads only the current token's prefix `x_t[:GATE_WINDOW]` and the immediately previous token `x_{t-1}`). (b) Phased TTT updates the per-document LoRA adapter AFTER scoring every chunk; no position-t prediction is ever conditioned on y_t or on positions > t.
- **Issue #1017 Condition 2 (full normalized distribution)**: CE over the full 8192-token softmax at each position; no x_t-dependent restriction of Σ.
- **Issue #1017 Condition 3 (score-before-update)**: the TTT path snapshots the pre-update per-chunk logits and scores them BEFORE the adapter SGD step. Per-document LoRA reset (`reusable_lora.reset()`) prevents cross-document leakage.
- **Issue #1017 Condition 4 (single left-to-right pass)**: eval is one left-to-right pass with sliding stride 64; no rescore/selection.
- **Section V — byte-level BPB**: BPB is scored on original pre-transform UTF-8 bytes via the per-token byte sidecar (`fineweb_val_bytes_XXXXXX.bin`), parallel to the val token shards. No hardcoded bytes/token.
- **No val data during training**: training uses only `fineweb_train_*.bin` shards. The TTT prefix (first 2000 val docs) follows the score-first protocol.
- **CaseOps bijectivity**: `decode_lossless_caps_v2(encode_lossless_caps_v2(x)) == x` for all test strings (transform is verifiable in `lossless_caps.py`).
- **LQER bijectivity is not required**: the rank-4 factors are additive correction on top of int6 GPTQ and do not alter the distribution support; they are fully reproducible from the stored factor tensors.
- **No external network during eval**: self-contained; tokenizer + transform + CaseOps SentencePiece model ship with this folder.
- **Reproducibility**: `train_gpt.py` is a single self-contained file; all mechanism flags are set via the Run Command environment.

## Requirements

```bash
# Python >= 3.12 required.
apt install -y lrzip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy brotli python_minifier
```

## Data setup (run ONCE)

The submission ships with the trained CaseOps SentencePiece model (`tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`) and the bijective transform module (`lossless_caps.py`). Train/val shards and the byte sidecar are rebuilt from the canonical FineWeb-10B doc stream:

```bash
# 1. Ensure docs_selected.jsonl exists (standard repo setup step).
python3 ../../data/download_hf_docs_and_tokenize.py  # or point to existing file

# 2. Build CaseOps-transformed shards + val byte sidecar.
python3 prepare_caseops_data_mp.py \
    --docs ./fineweb10B_raw/docs_selected.jsonl \
    --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
    --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Output layout (what `train_gpt.py` expects with `CASEOPS_ENABLED=1`):

```
data/datasets/fineweb10B_sp8192_caseops/datasets/
  tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
  datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
    fineweb_train_000000.bin
    ...
    fineweb_val_000000.bin
    fineweb_val_bytes_000000.bin
```

## Run command (3-seed reproduction)

```bash
for SEED in 314 42 999; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=12.0 \
  MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=12.0 \
  MATRIX_LR=0.026 \
  MIN_LR=0.1 \
  FUSED_CE_ENABLED=1 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  TTT_WARM_START_A=1 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  EMBED_WD=0.06 COMPRESSOR=pergroup \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Lineage

- **PR #549** — original modded-nanogpt stack (Keller Jordan).
- **PR #1019** (merged) — byte-level BPB SentencePiece accounting (`piece.encode`).
- **PR #1394** (merged) — SP8192 + multi-phase score-first TTT baseline.
- **PR #1530** (samacqua) — Loop4-5 depth recurrence + parallel residual start layer 8.
- **PR #1626** (ours, submitted) — GPTQ trimming + multi-phase SGD + adaptive clip.
- **PR #1729** (romeerp) — CaseOps bijective case transform + byte sidecar accounting.
- **PR #1736** (ours, submitted) — CaseOps + gated attention + quant-gate + phased TTT.
- **PR #1767** — TTT warm-start-A initialization.
- **PR #1769** (ours, submitted) — MLP GPTQ outlier-clip retune (10.0 → 12.0).
- **PR #1787** (nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE stack, 4-mechanism combo over the CaseOps base.
- **PR #1797** (dexhunter) — Smear gate + inline LQER asymmetric rank-4 correction stacked on top of PR #1787; this submission's direct base.
- **PR #1855** (codemath3000) — per-group lrzip compressor.
- **This submission** — PR #1797 base + PR #1855 per-group lrzip + tightened clips (`ATTN_CLIP_SIGMAS=12.0`, `EMBED_CLIP_SIGMAS=12.0`, `EMBED_WD=0.06`).

## Credits

- @codemath3000 — PR #1855 per-group lrzip.
- @dexhunter — PR #1797 Smear gate + LQER asymmetric rank-4 correction (direct base of this submission).
- @nprime06 — PR #1787 base stack (SparseAttnGate + PolarNS + MIN_LR + FusedCE + TTT warm-A).
- @samacqua — PR #1530 base stack (Loop4-5 + parallel residuals).
- @romeerp — PR #1729 CaseOps concept + byte sidecar accounting.
- @bigbag — PR #1493 merged SOTA (1.0810 val_bpb).
- @MarioPaerle — PR #1667 AttnOutGate pattern.
- PR #549 / PR #1019 / PR #1394 authors — merged baselines this stack descends from.

## Included files

- `train_gpt.py` — training script (157,445 bytes).
- `submission.json` — metadata (3-seed results).
- `README.md` — this file.
- `train_seed314.log`, `train_seed42.log`, `train_seed999.log` — 3-seed run logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model.
- `lossless_caps.py` — bijective CaseOps transform (used by `prepare_caseops_data_mp.py`).
- `prepare_caseops_data_mp.py` — multi-process CaseOps data preparation script.
