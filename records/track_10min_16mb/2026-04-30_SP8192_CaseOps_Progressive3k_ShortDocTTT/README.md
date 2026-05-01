# Record candidate: SP8192 CaseOps + Progressive 3k Context Growth + Short-Doc Score-First TTT

**val_bpb: 1.05759** (3-seed mean, std 0.00034) | **val_loss: 2.31441 nats** (std 0.00075) | **15.98 MB max** | 8xH100 SXM | 600s train / 600s eval

**Improvement over merged PR #1855 leaderboard record (1.06107587 BPB):**
**-0.00348 BPB / -0.00762 nats**

This stacks a progressive training-context schedule and a short-document TTT schedule on top of the late-April CaseOps/SP8192/LQER/SparseAttnGate/BOS-fixed SmearGate lineage. The direct leaderboard comparison is PR #1855, which is the current merged leader used here as the baseline.

## Results

| Seed | Steps | ms/step | Train ms | Pre-quant BPB | Quant BPB | **Post-TTT BPB** | TTT eval s | Artifact bytes |
|-----:|------:|--------:|---------:|--------------:|----------:|-----------------:|-----------:|---------------:|
| 42   | 4,888 | 121.9 | 596,025 | 1.05993108 | 1.06833072 | **1.05740567** | 572.4 | 15,981,945 |
| 314  | 4,882 | 122.1 | 595,976 | 1.05975470 | 1.06832443 | **1.05730104** | 489.9 | 15,984,387 |
| 0    | 4,884 | 122.0 | 596,022 | 1.06072266 | 1.06902034 | **1.05807084** | 493.5 | 15,981,122 |
| **Mean** | **4,884.7** | **122.0** | **596,008** | **1.06013615** | **1.06855850** | **1.05759252** | **518.6** | **15,982,485** |

3-seed population std: **0.00034091 BPB / 0.00074604 nats**.

All included seeds are under the 16,000,000-byte artifact cap and the 600s train/eval budgets as logged. The maximum artifact is **15,984,387 bytes** and the maximum validation-data TTT pass is **572.4s**.

## Full validation coverage

All three logs evaluate the full CaseOps validation shard target set:

| Seed | `val_tokens` | `target_tokens` |
|-----:|-------------:|----------------:|
| 42   | 47,853,343 | 47,853,343 |
| 314  | 47,853,343 | 47,853,343 |
| 0    | 47,853,343 | 47,853,343 |

The training script explicitly keeps the validation tail via `EVAL_INCLUDE_TAIL=1`. This avoids the older multiple-of-context truncation and makes the standard diagnostic eval and quantized TTT eval agree on the same target count.

The tokenizer, CaseOps transform, training shards, validation shard, and byte sidecar format are the same canonical HF-hosted CaseOps export used by the merged PR #1855 setup. If a reviewer already has the clean #1855/HF CaseOps data staged, those same staged shards can be reused here. The included tokenizer/prep files are present only to make this submission self-contained; the preferred reproduction path is to download the canonical HF CaseOps export directly.

## What changed vs PR #1855

This submission keeps the same overall 11-layer SP8192 CaseOps recurrent-transformer family as PR #1855, then adds the following levers:

| Lever | Setting | Purpose |
|-------|---------|---------|
| Progressive train context | `TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000` | Train cheaply at 1k early, move to 2k for most of training, then finish at 3k context. |
| Final/eval context | `TRAIN_SEQ_LEN=3072`, `EVAL_SEQ_LEN=3072`, `TTT_EVAL_SEQ_LEN=3072`, `EVAL_STRIDE=1536` | Extend the final model and TTT scoring context beyond 2k without the 4k eval-time cost. |
| Long-context TTT mask | `TTT_MASK=no_qv`, `TTT_Q_LORA=0`, `TTT_V_LORA=0` | Keep K/O/MLP LoRA adaptation while removing Q/V adapters that were less helpful at longer context. |
| TTT local LR | `TTT_LOCAL_LR_MULT=0.75` | Slightly softer per-document LoRA adaptation. |
| Short-doc score-first chunks | `TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24`, default chunk 48 | Use smaller score-before-update chunks for short documents, preserving causality while improving adaptation. |
| TTT phases | `PHASED_TTT_NUM_PHASES=1`, `PHASED_TTT_PREFIX_DOCS=2500` | Single score-first phased pass with a 2500-doc prefix budget. |
| QK gain | `QK_GAIN_INIT=5.25` | Public long-context sweep result from the PR #1953 lineage. |
| Compression/quant stack | `COMPRESSOR=pergroup`, AWQ-lite, asymmetric logit rescale | Inherited from public late-April quantization/compression work stacked on the PR #1855 base. |

The short-doc TTT schedule does **not** train on future validation tokens. It only changes the chunk granularity used inside the existing score-before-update loop: each chunk is scored first, then the LoRA update is applied for future chunks.

## Architecture and training stack

| Component | Setting |
|-----------|---------|
| Model | 11 layers, 512d, 8 query heads, 4 KV heads, MLP 4x |
| Tokenizer/data | SP8192 CaseOps lossless caps with byte sidecar accounting |
| RoPE | Partial RoPE, 16 dims |
| Recurrence | Layers 3-5 looped, enabled at `frac=0.35` |
| Parallel decoder | Parallel lane from layer 8, mean final lane |
| XSA | All 11 layers |
| Gates | BOS-fixed SmearGate, SparseAttnGate with `gate_window=12`, scale 0.5 |
| Optimizer | Muon on matrix params, Adam on embedding/scalars, `BETA2=0.99` |
| EMA | `ema_decay=0.9965` |
| Quantization | GPTQ int6 matrices, int7 embeddings, LQER asymmetric rank-4 correction |
| GPTQ reserve | `GPTQ_RESERVE_SECONDS=4.0`; logs show `gptq:reserving 4s, effective=596000ms` |
| Compression | Per-group compression |
| TTT | Quantized phased LoRA TTT, score-first, no_qv mask, short-doc chunk schedule |

## Compliance notes

- **Artifact cap:** all seeds <= 15,984,387 bytes.
- **Training wallclock:** all training loops stop around 596.0s with `GPTQ_RESERVE_SECONDS=4.0`; GPTQ hessian collection is logged immediately after (`67 Hessians in 4.1s`) for transparency.
- **Eval wallclock:** all validation-data TTT passes are <= 572.4s. The `ttt_lora:compile warmup` uses random tokens and no validation data; it is logged separately from `total_eval_time`.
- **Score-before-update:** `quantized_ttt_phased` scores each chunk before applying that chunk's LoRA update. The short-doc schedule only changes chunk size.
- **Full validation targets:** `val_tokens == target_tokens == 47853343` in all included logs.
- **No validation data in training:** training uses only training shards. TTT accesses validation documents left-to-right under the score-first rule.
- **No external cache or direct memorization:** no SLOT, n-gram cache, PPM mixture, logit bias table, or validation-derived precomputation.
- **Original-byte BPB:** CaseOps byte sidecar accounting is preserved.

## Reproduction

Install the dependencies in `requirements.txt`. FlashAttention 3 and the `lrzip` system binary are noted there because they require separate install paths.

This submission uses the clean canonical CaseOps SP8192 export hosted on Hugging Face. The logs were produced from a 50,000-document validation split with 80 training shards (`train_shards: 80`, `ttt_phased: total_docs:50000`, and `val_tokens == target_tokens == 47853343` in every included log).

Preferred data setup:

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="romeerp/parameter-golf-caseops-v1",
    repo_type="dataset",
    local_dir="./data/datasets/fineweb10B_sp8192_caseops",
    allow_patterns=[
        "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/*",
        "datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    ],
    max_workers=8,
)
PY
```

Then set:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOKENIZER_PATH=./data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Fallback local rebuild: if the HF export is unavailable, rebuild from the canonical `docs_selected.jsonl` with the included `prepare_caseops_data.py`, `lossless_caps.py`, and tokenizer. Use `--val-docs 50000` and write into a fresh output directory. The prep script now defaults to 50,000 validation docs and refuses to write over existing `fineweb_*.bin` shards unless `--overwrite` is passed, to avoid accidentally mixing stale validation shards with a new train split.

Run one seed at a time, replacing `DATA_PATH` and `TOKENIZER_PATH` with the staged CaseOps paths:

```bash
for SEED in 42 314 0; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 \
  VOCAB_SIZE=8192 \
  ITERATIONS=20000 \
  MAX_WALLCLOCK_SECONDS=600 \
  EVAL_INCLUDE_TAIL=1 \
  TRAIN_SEQ_LEN=3072 \
  ROPE_TRAIN_SEQ_LEN=3072 \
  TRAIN_SEQ_SCHEDULE=1024@0.100,2048@0.700,3072@1.000 \
  TRAIN_SEQ_SCHEDULE_MODE=wallclock \
  SEQ_CHANGE_WARMUP_STEPS=32 \
  EVAL_SEQ_LEN=3072 \
  EVAL_STRIDE=1536 \
  TTT_ENABLED=1 \
  TTT_EVAL_SEQ_LEN=3072 \
  TTT_BATCH_SIZE=24 \
  TTT_CHUNK_SIZE=48 \
  TTT_SHORT_SCORE_FIRST_ENABLED=1 \
  TTT_SHORT_DOC_LEN=2000 \
  TTT_SHORT_CHUNK_SIZE=24 \
  TTT_SHORT_SCORE_FIRST_STEPS=256:8,2000:24 \
  TTT_LORA_RANK=80 \
  TTT_LORA_LR=0.0001 \
  TTT_LOCAL_LR_MULT=0.75 \
  TTT_MASK=no_qv \
  TTT_Q_LORA=0 \
  TTT_V_LORA=0 \
  TTT_WEIGHT_DECAY=0.5 \
  TTT_BETA2=0.99 \
  PHASED_TTT_PREFIX_DOCS=2500 \
  PHASED_TTT_NUM_PHASES=1 \
  WARMDOWN_FRAC=0.85 \
  BETA2=0.99 \
  QK_GAIN_INIT=5.25 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GATED_ATTN_QUANT_GATE=1 \
  SMEAR_GATE_ENABLED=1 \
  GATE_WINDOW=12 \
  FUSED_CE_ENABLED=1 \
  MATRIX_LR=0.026 \
  MIN_LR=0.1 \
  GRAD_CLIP_NORM=0.3 \
  EMBED_BITS=7 \
  EMBED_CLIP_SIGMAS=14.0 \
  MATRIX_CLIP_SIGMAS=12.85 \
  ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=11.5 \
  LQER_ENABLED=1 \
  LQER_RANK=4 \
  LQER_TOP_K=3 \
  LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 \
  LQER_ASYM_GROUP=64 \
  AWQ_LITE_ENABLED=1 \
  AWQ_LITE_BITS=8 \
  AWQ_LITE_GROUP_TOP_K=1 \
  AWQ_LITE_GROUP_SIZE=64 \
  ASYM_LOGIT_RESCALE=1 \
  GPTQ_RESERVE_SECONDS=4.0 \
  GPTQ_CALIBRATION_BATCHES=16 \
  COMPRESSOR=pergroup \
  VAL_LOSS_EVERY=0 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Included files

- `train_gpt.py` - full training/eval script used for the logs.
- `train_seed42.log`, `train_seed314.log`, `train_seed0.log` - full per-seed logs.
- `submission.json` - structured metadata and per-seed results.
- `README.md` - this file.
- `requirements.txt` - Python dependencies plus notes for FA3 and `lrzip`.
- `prepare_caseops_data.py` - fallback CaseOps dataset/token/byte-sidecar preparation; defaults to the canonical 50,000-doc validation split and refuses mixed/stale output directories by default.
- `lossless_caps.py` - reversible CaseOps transform, same as the PR #1855 CaseOps setup.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - SentencePiece tokenizer used by the logs; identical CaseOps tokenizer lineage as PR #1855.

## Lineage and credits

This submission is a stack on top of the public CaseOps/SP8192 record lineage:

- PR #1855 by @codemath3000 - merged leaderboard record and direct comparison baseline.
- PR #1945 / PR #1908 / PR #1923 public late-April quantization stack - AWQ-lite and asymmetric logit rescale lineage.
- PR #1953 - long-context/no_qv/QK-gain sweep ideas.
- PR #1797 by @dexhunter - SmearGate and LQER asymmetric rank-4 lineage.
- PR #1787 by @nprime06 - Polar Express Muon, MIN_LR, SparseAttnGate, fused CE.
- PR #1736 and PR #1729 by @dexhunter / @romeerp - CaseOps integration and byte sidecar accounting.
- PR #1667 by @MarioPaerle - SmearGate lineage.
- PR #1626 / PR #1610 - phased score-first TTT lineage.
- Issue #1017 by @cocohearts - score-first validation criteria.

The new contribution here is the combination of progressive 3k train/eval context growth with the short-document score-first TTT chunk schedule, while preserving the full validation target count and staying under the artifact/eval budgets.
