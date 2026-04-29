# Non-Record Unlimited-Compute Submission: Sophonics LoRA Int5 DeepMLP

This is both a **non-record** submission and an **unlimited-compute** submission. It is not intended for the 10-minute leaderboard; it documents a multi-stage post-training pipeline that stays under the `16,000,000` byte artifact cap.

The workflow is:

1. Train a compact tied-embedding GPT base model.
2. Quantize the base to an `int5` repair substrate.
3. Train rank-16 LoRA repair modules on the deepest two MLP blocks.
4. Merge the repair weights back into ordinary checkpoint weights.
5. Quantize the merged checkpoint to uniform `int8` and zlib-compress it.

Longer-form writeup:

- White paper available here: https://tomatocultivator.com/sophonics_white_paper.pdf

## Result

Winning repair configuration:

- Base checkpoint: fresh `1xRTX5090` 600s reproduction run in `base_train.log`
- Repair target: `^blocks\.[7-8]\.mlp\.(fc|proj)$`
- Base precision during repair: `int5`
- Reference precision during repair: `int8`
- LoRA rank: `16`
- Trainable repair parameters: `98,304`
- Repair steps: `600`

Validation results:

- `1024`-sequence slice: repaired merged checkpoint `1.3610 BPB`, recovering `90.2%` of the `int5 -> int8` gap
- Full validation, pre-quant: `val_loss=2.2831`, `val_bpb=1.352204`
- Full validation, `int8+zlib` roundtrip: `val_loss=2.2854`, `val_bpb=1.353521`

Artifact size:

- compressed model (`int8+zlib`): `10,769,165 bytes`
- counted code: `train_gpt.py` only
- code bytes: `73,555`
- total bytes: `10,842,720`
- headroom under cap: `5,157,280 bytes`

The counted implementation is intentionally consolidated into `train_gpt.py`, per the challenge rule that all counted code should live in the self-contained training script. `sophonic_submission_check.py` is an uncounted verification helper.

## Reproduction

These commands are the copy-pasteable reproduction path for the recorded run. They assume the challenge dataset and tokenizer are available at the same `/workspace/parameter-golf` paths used in the logs.

### 1. Base Training

```bash
cd /workspace/parameter-golf/records/track_non_record_16mb/2026-03-27_Sophonics_LoRA_Int5_DeepMLP

RUN_ID=sophonic_5090_base_20260419_145335 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
GRAD_ACCUM_STEPS=2 \
TIE_EMBEDDINGS=1 \
EMBED_LR=0.05 \
HEAD_LR=0.0 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TRAIN_BATCH_TOKENS=131072 \
TRAIN_SEQ_LEN=1024 \
ITERATIONS=20000 \
WARMUP_STEPS=20 \
MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py

mkdir -p artifacts/sophonic_5090_base
cp final_model.pt artifacts/sophonic_5090_base/final_model.pt
```

Recorded base config:

- tokenizer: SentencePiece, `/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
- dataset: `fineweb10B_sp1024`, `80` training shards
- world size: `1`
- final base roundtrip in `base_train.log`: `val_loss=2.25913988`, `val_bpb=1.33798989`

### 2. LoRA Repair

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
TRAIN_SEQ_LEN=1024 \
VAL_BATCH_SIZE=4096 \
python train_gpt.py repair \
  --model artifacts/sophonic_5090_base/final_model.pt \
  --base-bits 5 \
  --high-bits 8 \
  --rank 16 \
  --alpha 16.0 \
  --target-regex '^blocks\.[7-8]\.mlp\.(fc|proj)$' \
  --max-steps 600 \
  --max-wallclock-seconds 600 \
  --train-batch-tokens 8192 \
  --lr 0.002 \
  --weight-decay 0.0 \
  --grad-clip 1.0 \
  --eval-every 100 \
  --eval-max-seqs 256 \
  --final-eval-max-seqs 1024 \
  --max-train-files 2 \
  --seed 1337 \
  --save-best-path artifacts/lora_int5_blocks78_mlp_r16_600_best.pt
```

The canonical merged checkpoint is:

- `artifacts/lora_int5_blocks78_mlp_r16_600_best.pt`

### 3. Verification

```bash
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VAL_BATCH_SIZE=131072 \
python sophonic_submission_check.py \
  --model artifacts/lora_int5_blocks78_mlp_r16_600_best.pt \
  --device cuda
```

Expected verification output:

```text
bytes_model_int8_zlib: 10769165
bytes_code: 73555
bytes_total: 10842720
fp32_val_bpb: 1.352204
int8_val_bpb: 1.353521
```

The existing compressed artifact is:

- `artifacts/lora_int5_blocks78_mlp_r16_600_best.int8.ptz`

## What Sophonics Means Here

The original static Sophonics idea was a low-bit base plus compressed higher-precision residual patches. That did not work well: low-rank residuals recovered only about `1%` to `3%` of the lost `int5/int6 -> int8` performance.

The working formulation here is a low-bit base plus tiny learned repair modules attached only where quantization damage matters most. In this submission, the “sophons” are those small learned repair modules.

Important scope:

- The final submitted artifact is not a live runtime package of `int5 base + explicit Sophonic modules`.
- The repair modules are trained on a frozen quantized base and then merged back into ordinary weights.
- The merged checkpoint is uniformly quantized to `int8` for the final artifact.

This result shows that a strongly compressed base can be behaviorally restored by small localized learned expansions while still fitting comfortably under the `16MB` cap. It does not yet prove a runtime that keeps an `int5` substrate and separate conditional Sophonic modules alive during inference.

## Localization Evidence

This model has transformer blocks `blocks.0` through `blocks.8`. The winning target set was:

- `blocks.7.mlp.fc`
- `blocks.7.mlp.proj`
- `blocks.8.mlp.fc`
- `blocks.8.mlp.proj`

A shallow-block control with the same rank, precision, parameter count, and `600` repair steps targeted `blocks.0-1.mlp.(fc|proj)`. It reached `1.4092 BPB` on the same `1024`-sequence slice and recovered only `56.5%` of the `int5 -> int8` gap, versus `90.2%` for the deepest MLP target.

## Included Files

- `train_gpt.py` - complete self-contained counted script: base training, eval helpers, quantization helpers, and LoRA repair
- `sophonic_submission_check.py` - uncounted full-validation and artifact-size verification helper
- `base_train.log` - exact 1xRTX5090 base-model training log
- `repair_train.log` - exact LoRA repair training log for the winning `int5` run
- `submission_check.log` - original full-validation and artifact-size verification log
- `environment_5090.log` - environment details for the reproduced run
- `requirements.txt` - minimal Python package requirements for these scripts
- `submission.json` - submission metadata
