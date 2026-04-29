# Record: SP8192 CaseOps + Per-Document Score-First TTT

**val_bpb = 1.06504520** (3-seed mean, std 0.00073036) | **15.922 MB max artifact** | 8xH100 SXM

This submission starts from the open PR #1855 frontier stack and makes the eval-time adaptation path legality-clean by using **per-document score-first LoRA only**: no global SGD, no cross-document adaptive state, and `TTT_WARM_START_A=0`.

AWQ-Lite was also tested as a paired quantization-only probe. It selected `tok_emb.weight[0:64]`, but the resulting artifact was 16,001,784 bytes, above the 16,000,000 byte hard cap, so AWQ-Lite is not used.

## 3-Seed Results

| Seed | Quantized BPB | Per-doc TTT BPB | Final loss | Submitted artifact bytes | Train steps | Total eval wallclock |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1.07565002 | **1.06438547** | 2.32926995 | 15,919,874 | 4994 | 405.7s |
| 0 | 1.07670295 | **1.06492009** | 2.33043990 | 15,918,942 | 4961 | 404.4s |
| 1234 | 1.07735123 | **1.06583003** | 2.33243118 | 15,922,155 | 4966 | 404.9s |
| **Mean** | | **1.06504520** | **2.33071368** | **15,920,323.67** | **4973.67** | |
| **Std** | | **0.00073036** | **0.00159830** | | | |

One-sided one-sample t-test vs 1.0810: `t=-37.83681065`, `df=2`, `p=0.000348888147222`.

## Key Techniques

1. **SP8192 CaseOps tokenizer + byte sidecar** -- official full-vocab token scoring with exact byte denominator accounting.
2. **#1855 frontier architecture** -- 11L x 512d transformer, XSA, SparseAttnGate, BOS-fixed SmearGate, recurrence, QK gain, and tuned Muon/AdamW schedule.
3. **Stock top-k LQER** -- rank-4 int4 asymmetric factors on the stock top-3 tensors after GPTQ.
4. **Per-group lrzip/Brotli compressed GPTQ** -- int6 matrices, int7 embeddings, and per-group compression.
5. **Legal per-document score-first LoRA TTT** -- LoRA state resets per document, tokens are scored before update, no global SGD, no warm-started cross-document LoRA A.
6. **Self-extracting code wrapper** -- submitted `train_gpt.py` is a 47,640-byte LZMA/base85 wrapper that expands to the full training/evaluation script at runtime.

## Size Accounting

The submitted `train_gpt.py` is the counted code artifact.

| Field | Bytes |
|---|---:|
| Submitted `train_gpt.py` wrapper | 47,640 |
| Max compressed model (`final_model.int6.ptz`) | 15,874,515 |
| `bytes_total` / max code + weights | 15,922,155 |
| Decimal cap | 16,000,000 |
| Margin | 77,845 |

The run logs were produced before final PR packaging and report a smaller internal Brotli wrapper estimate of 46,482 bytes. `submission.json` uses the actual submitted wrapper byte count, so it is the conservative number for the public code+weights cap.

## Evaluation-Time Ordering

Validation uses physical length bucketing only as a batching optimization across independent documents. The base quantized model state is fixed across documents, LoRA state is reset for each document, `TTT_WARM_START_A=0`, and there is no global SGD or cross-document adaptive state. Within each document, tokens are scored before the per-document LoRA update.

The audit logs include:

- `ttt_order_audit warm_start_a_zero verified`
- `ttt_order_audit perdoc_mode no_global_sgd_expected`
- `ttt_order_audit perdoc_no_global_sgd verified`
- `caseops_sanity enabled:1 ... scored_tokens:47851520 scored_bytes:151074499`

## Compliance

- **C1 causality:** scoring is causal within each document; no document can alter the base model state seen by another document.
- **C2 normalized distribution:** standard softmax over the full SP8192 CaseOps token alphabet; no byte-PPM, n-gram cache, or custom validation alphabet.
- **C3 score before update:** each chunk is scored before any per-document LoRA update on those tokens.
- **C4 single pass:** every validation token is scored once; no rescoring or validation-label selection.
- **All validation shards:** 47,851,520 scored tokens and 151,074,499 scored bytes per seed.
- **Artifact cap:** all three submitted code+weights totals are below 16,000,000 decimal bytes.
- **Eval time:** all three total eval wallclocks are under 600s.

## Reproduction

Runtime dependencies include PyTorch 2.9.1+cu128, CUDA 12.8, FA3 / `flash_attn_interface`, Triton, SentencePiece, Brotli, and `lrzip`.

The final seeds used the same stack with only `SEED` changed:

```bash
DATA_DIR=./data \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
VOCAB_SIZE=8192 CASEOPS_ENABLED=1 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 GLOBAL_TTT_MOMENTUM=0.9 \
WARMDOWN_FRAC=0.85 BETA2=0.99 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 GATED_ATTN_QUANT_GATE=1 \
SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 SMEAR_GATE_ENABLED=1 LQER_ENABLED=1 LQER_ASYM_ENABLED=1 \
LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 LQER_SELECT=topk AWQ_LITE_ENABLED=0 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup SUBMISSION_SIZE_CAP=16000000 NCCL_NET=Socket \
TTT_ENABLED=1 TTT_WARM_START_A=0 PHASED_TTT_PREFIX_DOCS=0 PHASED_TTT_NUM_PHASES=1 \
TTT_BATCH_BUCKETING=global_length TTT_ORDER_WINDOW_DOCS=1000000 TTT_ORDER_AUDIT=1 \
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=0` and `SEED=1234` for the other two logs.

## Credits

This submission is primarily a legality-clean and packaging-safe continuation of the PR #1855 / open-frontier lineage. Credit to the contributors behind the inherited stack: codemath3000, dexhunter, clarkkev, aquariouseworkman, Robby955, abaybektursun, and others in the Parameter Golf PR queue. OpenAI's RunPod compute grant made the final 3-seed validation possible.

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed0.log`
- `train_seed1234.log`
