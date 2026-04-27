# Record: PR #1787 base + PPM-D OMP byte mixture — val_bpb 1.0322

**val_bpb: 1.0322** (3-seed mean, std 0.00064) | **val_loss: 0.7155 nats/byte** (std 0.00045) | **~16.00 MB** | 8×H100 SXM, 600s train | PPM-D byte mixture (no neural TTT)

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, PPM-D byte mixture, no neural TTT)

### Core table

| Seed | Steps | ms/step | Post-EMA BPB (pre-quant) | Post-PPM BPB (sliding) | val_loss (nats) | Artifact (bytes) |
|------|------:|--------:|-------------------------:|-----------------------:|----------------:|-----------------:|
| 314  | 4658  | 128.0   | 1.07320                  | **1.03191**            | 0.71526         | 15,996,077       |
| 42   | 4679  | 127.4   | 1.07231                  | **1.03176**            | 0.71516         | 15,995,309       |
| 1234 | 4675  | 127.5   | 1.07354                  | **1.03294**            | 0.71598         | 15,998,552       |
| **Mean** | 4671 | 127.6 | 1.07301                  | **1.03220**            | 0.71547         | 15,996,646       |
| **Std**  |      |       | 0.00065                  | **0.00064**            | 0.00045         | 1,624            |

### Supplemental diagnostics

| Seed | Post-EMA BPB | Post-PPM BPB | val_loss (nats) | Code size | Total submission | Train time | Eval time |
|------|-------------:|-------------:|----------------:|----------:|-----------------:|-----------:|----------:|
| 314  | 1.07320      | 1.03191      | 0.71526         | 183,428   | 15,996,077       | 596.09s    | 297.9s    |
| 42   | 1.07231      | 1.03176      | 0.71516         | 183,428   | 15,995,309       | 596.08s    | 124.3s    |
| 1234 | 1.07354      | 1.03294      | 0.71598         | 183,428   | 15,998,552       | 596.08s    | 131.1s    |

All 3 seeds clear both 600s budgets (train + eval) and the 16,000,000-byte decimal artifact cap.
The seed-314 PPM eval pass is longer (~298s) because the PPM-D context-table collection ran with a cold L3 cache; subsequent seeds populate the cache and complete in ~130s. All three are well under the 600s eval cap.

## Key innovation — PPM-D byte-level mixture with OpenMP-parallelized native scoring

This submission combines four components on top of the PR #1787 (nprime06) upstream base:

1. **PR #1787 native base stack** (SparseAttnGate + PolarNS + MIN_LR + FusedCE) — same as our prior submission.
2. **Smear gate** (`SMEAR_GATE_ENABLED=1`, `GATE_WINDOW=12`) — content-conditioned gate over the first 12 residual dims, modulating a 1-token causal lookback `x_t ← x_t + λ · sigmoid(W · x_t[:12]) · x_{t-1}`. Includes the BOS-mask fix (smear is reset to zero at every document boundary), addressing the cross-document leakage flagged on the prior submission.
3. **LQER asymmetric rank-4 correction** (`LQER_ENABLED=1`, `LQER_RANK=4`, `LQER_TOP_K=3`, `LQER_ASYM_ENABLED=1`, `LQER_ASYM_GROUP=64`) — inline post-GPTQ asymmetric low-rank residual correction on the top-3 weight tensors by Frobenius norm of the quantization residual.
4. **PPM-D byte-level mixture** (`PPM_NATIVE_ENABLED=1`, `PPM_ORDER=4`) — port of the PPM-D class from PR #1850, rewritten in C and parallelized with OpenMP (`PPM_OMP_THREADS=8`, `PPM_OMP_CHUNK_TOKENS=4194304`). The PPM-D contexts are byte-level Markov tables of orders 0..4 with escape-D smoothing; mixed with the NN per-byte logits as `p_mix = (1−λ) · p_NN + λ · p_PPM`, where λ adapts between `PPM_LAMBDA_HI=0.9` and `PPM_LAMBDA_LO=0.05` based on PPM context confidence (`PPM_CONF_THRESHOLD=0.9`). The PPM table is updated **after** scoring each byte (strictly score-before-update), and is local to each chunk so the chunked OMP parallelism does not change the strictly causal scoring order within a chunk.

The OpenMP parallelization across `PPM_OMP_THREADS=8` chunks reduces PPM scoring wall-time from a baseline of ~957s to ~95-190s on the 40.5M-token validation set, fitting the 600s eval budget with substantial headroom.

### Mechanism stack

| Component | Origin | Role |
|-----------|--------|------|
| SparseAttnGate | PR #1787 (nprime06) | sparse per-head gate inside attention |
| PolarNS / MIN_LR / FusedCE | PR #1787 (nprime06) | base optimizer + scheduler refinements |
| Smear gate (BOS-masked) | prior submission (ours) | causal content-conditioned gate on first 12 residual dims, with doc-boundary reset |
| LQER asymmetric rank-4 | prior submission (ours) | post-GPTQ int6 residual recovery |
| PPM-D order-4 byte mixture | PR #1850 (port) + native gcc/OMP (this submission) | byte-level Markov mixture, score-before-update, OpenMP-parallelized |
| Int6 GPTQ + Brotli compressor | PR #1019 / PR #1530 | fits int6 model + LQER factors + code under 16,000,000 bytes |

## Changes from our prior banked submission

| Component | Prior banked submission (val_bpb 1.06157) | This submission |
|-----------|-------------------------------------------|-----------------|
| Base stack | PR #1787 native + CaseOps | PR #1787 native (CaseOps OFF — canonical SP8192 tokenizer) |
| Smear gate | enabled | enabled (with BOS-mask fix) |
| LQER asymmetric | enabled | enabled |
| Phased TTT | enabled (per-doc LoRA) | DISABLED — replaced by PPM byte mixture |
| PPM-D byte mixture | not used | `PPM_NATIVE_ENABLED=1`, order=4, λ_hi=0.9, λ_lo=0.05, conf_thr=0.9, OMP threads=8, chunk=4M tokens |
| CaseOps tokenizer | yes (lossless_caps_v1) | no (canonical fineweb_8192_bpe) |

PPM-D byte mixture replaces phased TTT and adds substantial additional bits via byte-level Markov modelling on the residual stream.

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
| eval_seq_len / eval_stride | 2048 / 64 |
| matrix_bits / embed_bits | 6 / 7 |
| LQER rank / top-K / asym group | 4 / 3 / 64 |
| smear gate window | 12 |
| PPM order / λ_hi / λ_lo / conf_threshold | 4 / 0.9 / 0.05 / 0.9 |
| PPM OMP threads / chunk_tokens | 8 / 4,194,304 |
| compressor | brotli (q=11) |

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: max across 3 seeds = 15,998,552 bytes (~1.4 KB headroom). Mean 15,996,646.
- **train_time ≤ 600s**: max = 596.09s (`stopping_early: wallclock_cap`).
- **total_eval_time ≤ 600s**: max = 297.9s (s314, cold-cache PPM collection); other seeds ~130s.
- **Issue #1017 Condition 1 (strict causal dependence)**: (a) The model forward pass uses only causal attention. (b) The PPM-D byte mixture is updated **byte-by-byte AFTER** that byte has been scored — the context table state at byte position `i` depends only on bytes 0..i-1. No future-byte leakage. (c) The OpenMP chunking parallelizes across independent chunks; within each chunk, the score-before-update order is preserved sequentially.
- **Issue #1017 Condition 2 (full normalized distribution over Σ)**: (a) NN logits are softmaxed over the full 8192-token vocabulary. (b) PPM-D produces a full 256-byte distribution at each byte position (normalized, with escape-D smoothing covering all bytes including unseen ones). (c) The mixture `p_mix = (1−λ) · p_NN_byte + λ · p_PPM` is a convex combination of two normalized distributions over the same byte alphabet, hence is itself normalized over Σ=256 bytes per byte position.
- **Issue #1017 Condition 3 (score-before-update)**: This is the core legality property of PPM-D as implemented. For each byte `b_i` in the validation stream:
  1. Compute `p_mix(b_i | context)` from the current NN logits (already committed by the eval loop) AND the current PPM-D context-table state (which has NOT yet been updated by `b_i`).
  2. Add `−log p_mix(b_i)` to the running NLL.
  3. **Then** update the PPM-D tables to incorporate `b_i` into the context.
  See `score_byte()` in the embedded native C source inside `train_gpt.py` — the table update happens after the log-prob accumulation.
- **Issue #1017 Condition 4 (single L→R pass)**: The eval is a single left-to-right pass over the validation stream. No rescore/selection/reordering. (Note: the TTT length-sort batching helper `_build_ttt_global_batches` is present in the source for code-path completeness but is **not called** at eval time — the active path is gated by `ppm_only_path = h.ppm_native_enabled and not h.ttt_enabled`, and goes directly to `run_ppm_native_pass` which scores in shard order without doc-level reordering. With `TTT_ENABLED=0` in the Run command, the sorted helper is dead code at eval time.)
- **Section V — byte-level BPB**: BPB is scored on the original UTF-8 byte stream via the SentencePiece piece table (`build_sentencepiece_luts`), with the standard PR #1019 +1 space-credit rule applied exactly once per token at boundary tokens. PPM scoring runs over the same byte stream the BPB is computed on. Full 40,540,160 validation tokens scored (151,078,222 bytes); no subset.
- **No val data during training**: training uses only `fineweb_train_*.bin` shards. PPM-D context tables are built from scratch at eval start (no pre-trained PPM state shipped in the artifact).
- **No external network during eval**: self-contained. The PPM-D native module is compiled at eval start via `gcc -O3 -march=native -fopenmp` from a C source string embedded inside `train_gpt.py`; no download.
- **Reproducibility**: `train_gpt.py` is a single self-contained file. All mechanism flags are set via environment variables in the Run Command. Requires gcc with OpenMP support on the eval host (standard on all Linux distros with `gcc` package).

## Requirements

```bash
# Python >= 3.10 (eval image runs 3.10 per Issue #17)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy brotli
# System packages: gcc with OpenMP support (libgomp). Standard on all Linux distros.
# Verify: `gcc -fopenmp -dM -E - < /dev/null | grep _OPENMP` should print a non-empty line.
```

## Data setup

Uses the canonical FineWeb-10B SentencePiece-8192 token shards (no transform, no per-token byte sidecar). The standard repo data download / tokenization pipeline produces them at `data/datasets/fineweb10B_sp8192/`. No special prep script is required for this submission (CASEOPS_ENABLED=0).

Expected layout under `$DATA_DIR`:

```
data/
  datasets/fineweb10B_sp8192/
    fineweb_train_000000.bin
    ...
    fineweb_val_000000.bin
    ...
  tokenizers/fineweb_8192_bpe.model
```

## Run command (3-seed reproduction)

```bash
for SEED in 314 42 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=0 \
  GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=12.0 \
  MATRIX_CLIP_SIGMAS=12.85 \
  MATRIX_LR=0.026 \
  GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=12 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 \
  LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
  TTT_ENABLED=0 \
  PPM_NATIVE_ENABLED=1 \
  PPM_ORDER=4 \
  PPM_LAMBDA_HI=0.9 PPM_LAMBDA_LO=0.05 \
  PPM_CONF_THRESHOLD=0.9 \
  PPM_LOG_CACHE_SIZE=1048576 \
  PPM_OMP_THREADS=8 \
  PPM_OMP_CHUNK_TOKENS=4194304 \
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
- **PR #1787** (nprime06) — SparseAttnGate + PolarNS + MIN_LR + FusedCE base stack.
- **PR #1850** (someone114514) — PPM-D byte-level mixture mechanism class.
- **This submission** — PR #1787 base + Smear gate (BOS-fixed) + LQER asymmetric + PPM-D byte mixture (port of PR #1850 with native gcc + OpenMP scoring, replaces neural TTT).

## Credits

- @nprime06 — PR #1787 base stack (SparseAttnGate + PolarNS + MIN_LR + FusedCE).
- @someone114514 — PR #1850 PPM-D byte-mixture mechanism class.
- @aamodbhatt — Phased TTT precedent (the score-first byte-level mixture pattern parallels PR #1394's score-first per-document LoRA).
- @samacqua — PR #1530 base stack (Loop4-5 + parallel residuals).
- @bigbag — PR #1493 merged SOTA (1.0810 val_bpb).
- @msisovic — caught the SmearGate cross-document leakage bug on our prior submission (BOS-mask fix is included in this submission).
- PR #549 / PR #1019 / PR #1394 authors — merged baselines this stack descends from.

## Included files

- `train_gpt.py` — training + PPM-D scoring script (183,428 bytes).
- `submission.json` — metadata (3-seed results).
- `README.md` — this file.
- `train_seed314.log`, `train_seed42.log`, `train_seed1234.log` — 3-seed run logs.
