# Submission: 4xH100 QK5.25 TTT-disabled promotion

Candidate: `best4x_ttt_disabled_qk525`
Mean val_bpb: `1.26066159`
Artifact bytes: `15080366`
Hardware: `4xNVIDIA H100 80GB HBM3`

> Note: this generated package contains fewer than 3 seeds. For a new SOTA record PR, rerun the selected candidate with additional seeds or submit as a non-record result.

## Results

| Seed | val_bpb | val_loss | artifact bytes | train seconds |
|---|---:|---:|---:|---:|
| 1337 | 1.26066159 | 2.12857428 | 15080366 | 696.463 |

## Technique

4xH100 promotion of the best 1xH100 TTT-disabled QK5.25 control

Candidate environment is captured in `candidate_env.json`; source batch rows are copied into `run_result*.json`.

## Reproduction

From this record folder:

```bash
export ARTIFACT_BUDGET_STRICT="0"
export CANDIDATE_IMPL="autoregressive_gpt"
export MAX_WALLCLOCK_SECONDS="600"
export QK_GAIN_INIT="5.25"
export SKIP_PRE_QUANT_FINAL_EVAL="1"
export TRAIN_BATCH_TOKENS="2097152"
export TRAIN_LOG_EVERY="100"
export TTT_ENABLED="0"
export VAL_LOSS_EVERY="0"
export VOCAB_SIZE="1024"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export DATA_PATH="${DATA_PATH:-../../data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-../../data/tokenizers/fineweb_1024_bpe.model}"
export RUN_ID="${RUN_ID:-submission_rerun}"
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train_gpt.py
```

For a 3-seed verification run:

```bash
for SEED in 42 314 1234; do
  SEED=$SEED RUN_ID=submission_seed${SEED} \
    torchrun --standalone --nproc_per_node=8 train_gpt.py > train_seed${SEED}_rerun.log 2>&1
done
```

## Compliance Notes

- Training under 600s in packaged run(s): `False`
- Artifact under 16,000,000 bytes in packaged run(s): `True`
- Three-seed evidence included: `False`

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train.log`
- `candidate_env.json`
- `train*.log`
- `run_result*.json`
