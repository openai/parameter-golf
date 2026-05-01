# Record candidate: SP8192 CaseOps + Progressive 3k Context + Short-Doc Score-First TTT — GPTQ_CALIBRATION_BATCHES=32

**Comparison baseline: PR #2014 (1.05759 BPB)** — direct stack baseline.
**Merged-leaderboard comparison: PR #1855 (1.06108 BPB).**

This submission keeps the PR #2014 architecture/training stack identical and only changes: **GPTQ_CALIBRATION_BATCHES=32**. Every other hyperparameter, env var, and code path is byte-for-byte the PR #2014 reproduction command.

## Full validation coverage

The 3 per-seed logs evaluate the full CaseOps validation shard target set:

| Seed | `val_tokens` | `target_tokens` |
|-----:|-------------:|----------------:|
| 42 | 47,853,343 | 47,853,343 |
| 314 | 47,853,343 | 47,853,343 |
| 0 | 47,853,343 | 47,853,343 |

The training script keeps the validation tail via `EVAL_INCLUDE_TAIL=1`, matching PR #2014 exactly.

The tokenizer, CaseOps transform, training shards, validation shard, and byte sidecar format are the canonical HF-hosted CaseOps export (`romeerp/parameter-golf-caseops-v1`) used by the merged PR #1855 setup.

## What changed vs PR #2014

| Lever | PR #2014 | This submission | Mechanism |
|-------|----------|-----------------|-----------|
| `GPTQ_CALIBRATION_BATCHES` | 16 | **32** | Doubles the number of GPTQ Hessian calibration batches from 16 to 32, reducing post-quantization error at fixed compute by giving GPTQ a denser activation estimate. |

Everything else — model architecture, optimizer, schedule, TTT, quantization, compression — is byte-for-byte identical to the PR #2014 stack.

## Architecture and training stack

| Component | Setting |
|-----------|---------|
| Model | 11 layers, 512d, 8 query heads, 4 KV heads, MLP 4x |
| Tokenizer/data | SP8192 CaseOps lossless caps with byte sidecar accounting |
| RoPE | Partial RoPE, 16 dims |
| Recurrence | Layers 3-5 looped, enabled at `frac=0.35` |
| Parallel decoder | Parallel lane from layer 8, mean final lane |
| XSA | All 11 layers |
| Gates | BOS-fixed SmearGate, SparseAttnGate (`scale=0.5`) |
| Optimizer | Muon on matrix params, Adam on embedding/scalars, `BETA2=0.99` |
| EMA | `ema_decay=0.9965` |
| Quantization | GPTQ int6 matrices, int7 embeddings, LQER asymmetric rank-4 correction, AWQ-lite int8 group quant, asymmetric logit rescale |
| Compression | Per-group compression (lrzip + brotli) |
| Context schedule | Progressive train context: 1024 @ 10%, 2048 @ 70%, 3072 final (wallclock-driven) |
| Final/eval context | `TRAIN_SEQ_LEN=3072`, `EVAL_SEQ_LEN=3072`, `TTT_EVAL_SEQ_LEN=3072`, `EVAL_STRIDE=1536` |
| TTT | Quantized phased LoRA TTT, score-first, `no_qv` mask, `TTT_LORA_RANK=80`, `TTT_LORA_LR=0.0001`, `TTT_LOCAL_LR_MULT=0.75`, short-doc chunk schedule (`256:8, 2000:24`), `PHASED_TTT_PREFIX_DOCS=2500`, `PHASED_TTT_NUM_PHASES=1` |

## Compliance notes

- **Artifact cap:** all seeds <= 16,000,000 bytes.
- **Training wallclock:** training stops at the 600s wallclock budget (matching PR #2014).
- **Eval wallclock:** all validation passes are <= 600s.
- **Score-before-update:** `quantized_ttt_phased` scores each chunk before applying that chunk's LoRA update. Inherited unchanged from PR #2014.
- **Full validation targets:** `val_tokens == target_tokens == 47,853,343` in all included logs.
- **No validation data in training:** training uses only training shards. TTT accesses validation documents left-to-right under the score-first rule.
- **No external cache or direct memorization:** no SLOT, persistent n-gram cache, PPM mixture, logit bias table, or validation-derived precomputation.
- **Original-byte BPB:** CaseOps byte sidecar accounting is preserved.

## Reproduction

Install the dependencies in `requirements.txt`. FlashAttention 3 and the `lrzip` system binary are noted there because they require separate install paths.

This submission uses the canonical CaseOps SP8192 export hosted on Hugging Face. Logs are produced from a 50,000-document validation split with 80 training shards.

Preferred data setup (matches PR #2014):

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

Training command (only the lever override differs from PR #2014):

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
  GPTQ_CALIBRATION_BATCHES=32 \
  COMPRESSOR=pergroup \
  VAL_LOSS_EVERY=0 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```


## Included files

- `train_gpt.py` - full training/eval script (identical to PR #2014).
- `train_seed*.log` - full per-seed logs.
- `submission.json` - structured metadata.
- `README.md` - this file.
- `requirements.txt` - Python dependencies plus notes for FA3 and `lrzip`.
- `prepare_caseops_data.py` - fallback CaseOps dataset/token/byte-sidecar preparation.
- `lossless_caps.py` - reversible CaseOps transform.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - SentencePiece tokenizer.

## Lineage and credits

This submission is a single-knob (resp. two-knob) refinement on top of PR #2014, which itself stacks on the public CaseOps/SP8192 record lineage:

- PR #2014 by @simonbissonnette - direct comparison baseline; provides the progressive-context + short-doc TTT chunk scheduling on top of the merged PR #1855 stack.
- PR #1855 by @codemath3000 - merged leaderboard record.
- PR #1797 by @dexhunter - SmearGate and LQER asymmetric rank-4 lineage.
- PR #1787 by @nprime06 - Polar Express Muon, MIN_LR, SparseAttnGate, fused CE.
- PR #1736 / PR #1729 by @dexhunter / @romeerp - CaseOps integration and byte sidecar accounting.

The new contribution here is the targeted hyperparameter change (GPTQ_CALIBRATION_BATCHES=32) on the PR #2014 stack, isolated as a clean ablation so reviewers can compare the lever effect against PR #2014 directly.
