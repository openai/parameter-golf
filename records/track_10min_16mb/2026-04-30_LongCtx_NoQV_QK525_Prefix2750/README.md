# SP8192 + LongCtx NoQV QK5.25 Prefix2750

This is a deliberately small follow-up candidate on PR #1953:

- Base: PR #1953, `PR #1945 base + 2560 long-context + no_qv TTT mask + TTT LR 0.75 + QK_GAIN 5.25`.
- Only intended change: `PHASED_TTT_PREFIX_DOCS=2750` instead of `2500`.
- No tokenizer change, no PPM, no n-gram, no SLOT, no logit bias, no pre-quant validation adaptation.
- Artifact size is effectively unchanged; the risk is eval time, not bytes.

## Why this change

PR #1953 reports max eval time `513.1s`, leaving roughly `87s` under the 600s eval cap. Its lineage already shows that increasing phased-TTT prefix docs from earlier values to `2500` was useful. This candidate spends part of the remaining eval budget on a slightly larger TTT prefix (`2750`) while keeping every other mechanism unchanged.

Single-seed testing shows this change is essentially neutral versus the #1953 seed 42 reference. It is included as a narrow phased-TTT prefix schedule experiment with the full seed 42 log.

## Result

| Run | Seed | Prefix docs | Final BPB | Eval time | Total bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| #1953 reference | 42 | 2500 | 1.05824720 | 430.0s | 15,988,861 |
| This experiment | 42 | 2750 | 1.05826976 | 495.0s | 15,978,173 |

The longer prefix increases eval time by about 65s and lands within `0.00003 BPB` of the #1953 seed 42 reference. The full log is included as `train_seed42.log`.

## Data

This script uses the CaseOps SP8192 dataset. Do not use the ordinary `sp8192` FineWeb download.

Expected layout after running `download_caseops_data.py`:

```text
/workspace/caseops_data/datasets/
  tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
  datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
    fineweb_train_*.bin
    fineweb_val_*.bin
    fineweb_val_bytes_*.bin
```

Download and validate:

```bash
python3 records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_Prefix2750/download_caseops_data.py \
  --local-dir /workspace/caseops_data
```

## Dependencies

Python packages are listed in `requirements.txt`. FlashAttention 3 and `lrzip` are required:

```bash
apt-get update
apt-get install -y lrzip
pip3 install -r records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_Prefix2750/requirements.txt
pip3 install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

## Single-seed test

```bash
RUN_ID=1953_prefix2750_seed42 \
SEED=42 \
DATA_DIR=/workspace/caseops_data/datasets \
DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
VOCAB_SIZE=8192 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
WARMDOWN_FRAC=0.85 \
BETA2=0.99 \
MUON_MOMENTUM=0.97 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
EMBED_BITS=7 \
MATRIX_CLIP_SIGMAS=12.85 \
ATTN_CLIP_SIGMAS=13.0 \
MLP_CLIP_SIGMAS=11.5 \
EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 \
FUSED_CE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_SCALE=0.5 \
SPARSE_ATTN_GATE_INIT_STD=0.0 \
GATED_ATTN_QUANT_GATE=1 \
LQER_ENABLED=1 \
LQER_RANK=4 \
LQER_TOP_K=3 \
LQER_GROUP_SIZE=64 \
LQER_FACTOR_BITS=4 \
LQER_ASYM_ENABLED=1 \
LQER_ASYM_GROUP=64 \
AWQ_LITE_ENABLED=1 \
ASYM_LOGIT_RESCALE=1 \
GPTQ_RESERVE_SECONDS=4.0 \
GPTQ_CALIBRATION_BATCHES=16 \
COMPRESSOR=pergroup \
TTT_ENABLED=1 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_NUM_PHASES=3 \
PHASED_TTT_PREFIX_DOCS=2750 \
TTT_LORA_RANK=80 \
TTT_MASK=no_qv \
TTT_Q_LORA=0 \
TTT_V_LORA=0 \
TTT_LOCAL_LR_MULT=0.75 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
EVAL_SEQ_LEN=2560 \
TTT_EVAL_SEQ_LEN=2560 \
QK_GAIN_INIT=5.25 \
NCCL_NET=Socket \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-30_LongCtx_NoQV_QK525_Prefix2750/train_gpt.py
```

## Decision rule

Compare seed 42 against PR #1953 seed 42:

- PR #1953 seed 42 post-TTT: `1.05824720`.
- This experiment seed 42 post-TTT: `1.05826976`.
- This is a single-seed schedule experiment; the result is effectively tied with the #1953 seed 42 reference while using a larger phased-TTT prefix.
