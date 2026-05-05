# Record: PR #1787 base + Smear Gate (BOS-masked) + LQER Asymmetric + Phased TTT — val_bpb 1.06412

**val_bpb: 1.06412** (3-seed mean, std 0.00172) | **val_loss: 2.32869 nats/token** (std 0.00373) | **~15.95 MB** | 8×H100 SXM, 600s train / 600s eval | Phased TTT

> **Updated 2026-04-27**: SmearGate forward path now masks the previous-token term at document boundaries (`input_ids == BOS_ID`), per @msisovic's catch in [#1797 (comment)](https://github.com/openai/parameter-golf/pull/1797#issuecomment-2783310834). The metric below is the rebanked 3-seed result with the BOS mask applied at both `_forward_hidden` and `forward_ttt`. The original 1.06157 headline was favorably biased by the cross-doc smear leak (+0.00255 BPB).

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

### Core table (phased TTT)

| Seed | Steps  | Pre-TTT BPB | Post-TTT BPB | TTT gain | TTT time | Artifact (bytes) |
|------|-------:|------------:|-------------:|---------:|---------:|-----------------:|
| 314  | 4883   | 1.07599     | **1.06307**  | -0.01292 | 422.8s   | 15,951,189       |
| 42   | 4878   | 1.07606     | **1.06319**  | -0.01287 | 429.4s   | 15,953,178       |
| 1234 | 4655   | 1.07898     | **1.06610**  | -0.01288 | 473.1s   | 15,953,718       |
| **Mean** | **4805** | **1.07701** | **1.06412** | **-0.01289** | **441.8s** | **15,952,695** |
| **Std**  |          | 0.00172     | **0.00172** |          | 27.27s   | 1,332            |

### Supplemental diagnostics

| Seed | Post-EMA BPB (pre-quant) | Quantized BPB (no TTT) | Post-TTT BPB | val_loss (nats) | Train time | Eval time |
|------|-------------------------:|-----------------------:|-------------:|----------------:|-----------:|----------:|
| 314  | 1.06684                  | 1.07599                | 1.06307      | 2.32639         | 596.13s    | 422.8s    |
| 42   | 1.06705                  | 1.07606                | 1.06319      | 2.32665         | 596.13s    | 429.4s    |
| 1234 | 1.06988                  | 1.07898                | 1.06610      | 2.33302         | 596.10s    | 473.1s    |

All 3 seeds clear both 600s budgets (train + eval) and the 16,000,000-byte decimal artifact cap. 3-seed std is 0.00172 BPB.

## Key innovation — PR #1787 native base + orthogonal Smear gate + inline LQER asymmetric factorization

This submission combines three components on top of the PR #1787 (nprime06) upstream base:

1. **Native PR #1787 base stack** (CaseOps + SparseAttnGate + PolarNS + MIN_LR + FusedCE + PR #1767-style TTT with `TTT_WARM_START_A=1`). The SparseAttnGate (`SPARSE_ATTN_GATE_ENABLED=1`) is PR #1787's replacement for the earlier QuantGate — it's a sparse per-head multiplicative gate applied inside attention.
2. **Smear gate** (`SMEAR_GATE_ENABLED=1`, `GATE_WINDOW=12`): a lightweight content-conditioned gate over the **first `GATE_WINDOW=12` feature dimensions** of the current-token residual, modulating a **1-token causal lookback** `x_t ← x_t + λ · sigmoid(W · x_t[:12]) · x_{t-1}`. Orthogonal to SparseAttnGate because it operates on the residual (not on attention outputs) and uses only the previous token, not the full attention window.
3. **LQER asymmetric rank-k correction** (`LQER_ENABLED=1`, `LQER_RANK=4`, `LQER_TOP_K=3`, `LQER_ASYM_ENABLED=1`, `LQER_ASYM_GROUP=64`): inline post-GPTQ asymmetric low-rank error compensation. The **top-K entire weight tensors (K=3)** are selected globally by Frobenius norm of the quantization residual `E = W - W_q`; each selected tensor is factored as `E ≈ A · B` via rank-4 SVD. In asymmetric mode, `A` is stored as **INT2 per-matrix (single fp16 scalar scale)** and `B` as **INT4 per-group-64**; both are Brotli-compressed with the model. Recovers ≈0.009 BPB of the int6 quantization tax at a ≈30 KB artifact cost. (`LQER_FACTOR_BITS=4` is consumed only by the symmetric fallback path and is unused here.)

### Mechanism stack

| Component | Origin | Role |
|-----------|--------|------|
| CaseOps bijective case transform | PR #1729 (romeerp) / PR #1736 (ours) | ~1.5% token savings, full byte-level bijection |
| SparseAttnGate | PR #1787 (nprime06) | sparse per-head gate inside attention |
| Smear gate | this submission | causal content-conditioned gate on first 12 residual dims, adding 1-token lookback |
| LQER asymmetric rank-4 correction | this submission | post-GPTQ int6 residual recovery, INT2/INT4 asym factors on top-3 tensors |
| Phased TTT (score-first, 3 phases, 2000-doc prefix) | PR #1394 / PR #1736 | per-document LoRA adapter, score-before-update |
| Int6 GPTQ + Brotli compressor | PR #1019 / PR #1530 | fits int6 model + factors + code under 16,000,000 bytes |

### Empirical result (3 seeds)

| Seed | val_bpb | val_loss (nats) |
|------|--------:|----------------:|
| 314  | 1.06307 | 2.32639         |
| 42   | 1.06319 | 2.32665         |
| 1234 | 1.06610 | 2.33302         |
| **Mean** | **1.06412** | **2.32869** |
| **Std**  | 0.00172    | 0.00373        |

3-seed mean clears the merged SOTA (PR #1493 at 1.0810) by **0.0169 BPB ≈ 0.0436 nats/token ≈ 8.7× the 0.005-nat record bar inflection** (sp8192: 0.005 nats ≈ 0.00194 BPB).

## Changes from PR #1736 (our prior banked submission)

| Component | PR #1736 (ours, banked) | This submission |
|-----------|-------------------------|-----------------|
| Base stack | PR #1530 + CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT | PR #1787 native (CaseOps + SparseAttnGate + PolarNS + MIN_LR + FusedCE + TTT_WARM_A) |
| Gated attention | `GATED_ATTN_ENABLED=1` (per-head scalar) | `SPARSE_ATTN_GATE_ENABLED=1` (sparse gate, PR #1787 native) |
| Smear gate | not used | `SMEAR_GATE_ENABLED=1`, `GATE_WINDOW=12` |
| LQER | not used | `LQER_ENABLED=1`, rank=4, top_k=3, factor_bits=4, asym group=64 |
| MIN_LR | 0.0 | 0.1 |
| FUSED_CE | disabled | `FUSED_CE_ENABLED=1` |
| TTT warm-start A | off | `TTT_WARM_START_A=1` |
| Other hparams | — | identical (SP8192, 11L, dim=512, 8/4 heads, MLP 4×, Loop3-5, 2 iters, parallel_start=8, int6 MLP/matrix, int7 embed, eval stride 64) |

Net on 3-seed mean: **−0.00137 BPB / −0.00299 val_loss (nats/token)** vs PR #1736 (1.06549 / 2.33168).

## Architecture (inherits PR #1787 shape)

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
| compressor | brotli |

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: all 3 seeds 15,951,189–15,953,718 bytes (~46–49 KB headroom).
- **train_time ≤ 600s**: all 3 seeds 599.47–599.64s (`stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: all 3 seeds 423.3–494.8s.
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
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy brotli
```

## Data setup (run ONCE)

The submission ships with the trained CaseOps SentencePiece model (`tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`) and the bijective transform module (`lossless_caps.py`). Train/val shards and the byte sidecar are rebuilt from the canonical FineWeb-10B doc stream:

```bash
# 1. Ensure docs_selected.jsonl exists (standard repo setup step).
python3 ../../data/download_hf_docs_and_tokenize.py  # or point to existing file

# 2. Build CaseOps-transformed shards + val byte sidecar.
python3 prepare_caseops_data.py \
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

### Reproduction sanity check (run after step 2)

Each shard must contain `BOS_ID=1` at the start of every document — `train_gpt.py`'s phased TTT eval path (`_find_docs`) requires it. Quick check on the first val shard:

```python
python3 -c "
import numpy as np
d = np.fromfile('data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/fineweb_val_000000.bin', dtype=np.uint16)
tokens = d[512:]
bos_count = int((tokens == 1).sum())
print(f'BOS markers in val shard: {bos_count}  (must be > 0)')
assert bos_count > 0, 'prep script broken: re-run prepare_caseops_data.py (must prepend BOS_ID=1 to each doc)'
"
```

## Run command (3-seed reproduction)

```bash
for SEED in 314 42 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=12.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 \
  MIN_LR=0.1 \
  FUSED_CE_ENABLED=1 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  TTT_WARM_START_A=1 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
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
- **PR #1787** (nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE stack, 4-mechanism combo over the CaseOps base. Base for this submission.
- **This submission** — PR #1787 native base with our Smear gate and inline LQER asymmetric rank-4 correction stacked on top.

## Credits

- @nprime06 — PR #1787 base stack (SparseAttnGate + PolarNS + MIN_LR + FusedCE + TTT warm-A).
- @samacqua — PR #1530 base stack (Loop4-5 + parallel residuals).
- @romeerp — PR #1729 CaseOps concept + byte sidecar accounting.
- @bigbag — PR #1493 merged SOTA (1.0810 val_bpb).
- @MarioPaerle — PR #1667 AttnOutGate pattern.
- PR #549 / PR #1019 / PR #1394 authors — merged baselines this stack descends from.

## Included files

- `train_gpt.py` — training script (151,554 bytes).
- `submission.json` — metadata (3-seed results).
- `README.md` — this file.
- `train_seed314.log`, `train_seed42.log`, `train_seed1234.log` — 3-seed run logs.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model.
- `lossless_caps.py` — bijective CaseOps transform (used by `prepare_caseops_data.py`).
- `prepare_caseops_data.py` — one-time data prep: tokenizes FineWeb via CaseOps + emits per-token byte sidecar.
