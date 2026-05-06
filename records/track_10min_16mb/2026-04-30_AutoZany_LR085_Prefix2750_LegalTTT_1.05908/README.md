# AutoZany LR 0.85 + Prefix2750 Legal Phased TTT (val_bpb 1.05908)

**val_bpb = 1.05907559** (3-seed mean, population std 0.00041335) on 8x H100 SXM with strict `<600s` training, `<600s` evaluation, and `<16,000,000` byte artifacts.

This is a final-day conservative submission built on the public PR #1953 / PR #1945 lineage. It keeps the legal score-first phased TTT path and changes only the final TTT evaluation neighborhood: `TTT_LOCAL_LR_MULT=0.85` with `PHASED_TTT_PREFIX_DOCS=2750`.

## Results

| Seed | Stop step | Train ms | Pre-quant BPB | Quant no-TTT BPB | Prefix2750 TTT BPB | Eval s | Artifact bytes |
|-----:|----------:|---------:|--------------:|-----------------:|-------------------:|-------:|---------------:|
| 42   | 4889 | 596017 | 1.06193713 | 1.07022525 | **1.05849788** | 473.8 | 15,976,870 |
| 0    | 4888 | 596020 | 1.06282344 | 1.07113858 | **1.05928718** | 464.8 | 15,980,787 |
| 1234 | 4906 | 596115 | 1.06264173 | 1.07122159 | **1.05944171** | 468.3 | 15,984,508 |
| **Mean** | **4894** | **596051** | **1.06246743** | **1.07086181** | **1.05907559** | **468.9** | **15,980,722** |

The train logs (`train_seed*.log`) are the full training + quantization runs. The `ttt_prefix2750_seed*.log` files are `TTT_EVAL_ONLY=1` re-evaluations of those exact saved artifacts with `PHASED_TTT_PREFIX_DOCS=2750`; those are the scores reported above.

## Changes

This uses the same `train_gpt.py` source as the PR #1953 stack and runs with:

```bash
EVAL_SEQ_LEN=2560
TTT_EVAL_SEQ_LEN=2560
TTT_MASK=no_qv
TTT_Q_LORA=0
TTT_V_LORA=0
TTT_LOCAL_LR_MULT=0.85
PHASED_TTT_PREFIX_DOCS=2750
PHASED_TTT_NUM_PHASES=3
QK_GAIN_INIT=5.25
ASYM_LOGIT_RESCALE=1
AWQ_LITE_ENABLED=1
COMPRESSOR=pergroup
```

## Compliance

- Score-first phased TTT only: each validation chunk is scored before any adaptation update.
- No n-gram cache, PPM, validation pretraining, pre-quant TTT, validation lookahead, external data, or network access.
- Full fixed validation set with CaseOps byte sidecar.
- Decimal artifact cap respected for all seeds. The largest observed artifact is `15,984,508` bytes.
- Training cap respected for all seeds. The largest observed train timer is `596115ms`.
- Evaluation cap respected for all seeds. The largest observed TTT eval timer is `473829ms`.

## Reproduction

Run the full training command once per seed, then optionally rerun `TTT_EVAL_ONLY=1` from the saved `ARTIFACT_DIR` with `PHASED_TTT_PREFIX_DOCS=2750`.

```bash
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
VOCAB_SIZE=8192 \
ITERATIONS=20000 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TTT_ENABLED=1 \
PHASED_TTT_NUM_PHASES=3 \
PHASED_TTT_PREFIX_DOCS=2750 \
TTT_LORA_RANK=80 \
TTT_MASK=no_qv \
TTT_Q_LORA=0 \
TTT_V_LORA=0 \
TTT_LOCAL_LR_MULT=0.85 \
TTT_BETA2=0.99 \
TTT_WEIGHT_DECAY=0.5 \
EVAL_SEQ_LEN=2560 \
TTT_EVAL_SEQ_LEN=2560 \
TTT_CHUNK_SIZE=48 \
QK_GAIN_INIT=5.25 \
WARMDOWN_FRAC=0.85 \
MATRIX_LR=0.026 \
MIN_LR=0.1 \
EMBED_BITS=7 \
EMBED_CLIP_SIGMAS=14.0 \
MATRIX_CLIP_SIGMAS=12.85 \
ATTN_CLIP_SIGMAS=13.0 \
MLP_CLIP_SIGMAS=11.5 \
GRAD_CLIP_NORM=0.3 \
FUSED_CE_ENABLED=1 \
SMEAR_GATE_ENABLED=1 \
GATE_WINDOW=12 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_SCALE=0.5 \
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
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat for `SEED=0` and `SEED=1234`.

## Lineage

This submission builds on public Parameter Golf work, especially PR #1953 and its PR #1945 / #1855 lineage: AWQ-lite, Asymmetric Logit Rescale, CaseOps tokenizer, SparseAttnGate, SmearGate, LQER, QK gain, and legal phased TTT. The contribution here is the final-day legal TTT neighborhood selection and 3-seed verification under the hard time/artifact limits.
