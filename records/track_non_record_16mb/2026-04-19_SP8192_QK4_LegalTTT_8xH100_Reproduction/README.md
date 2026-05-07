# Non-Record Submission: SP8192 QK4 + Legal TTT 8xH100 Reproduction

This is a non-record 8xH100 reproduction/promotion run for the April 2026
SP8192 record family. It combines the April 5 SP8192 GPTQ embeddings + SDClip +
Loop45x2 training stack with a legal score-first TTT evaluation pass adapted
from the April 6 QK5 Legal TTT record.

This is not submitted as a SOTA record. The final legal TTT score is
`1.08448769` val_bpb, which is better than the April 5 base record-family score
reported in the leaderboard (`1.0856`) but behind the later QK5, parallel
residual, and 3-layer recurrence records.

## Result

| Metric | Value |
| --- | ---: |
| Seed | `1337` |
| Hardware | `8x NVIDIA H100 80GB HBM3` |
| Runtime | `torch 2.11.0+cu128`, CUDA `12.8` |
| Train shards | `128` SP8192 shards |
| Validation tokens | `40,540,160` |
| Stop step | `4955/20000` |
| Training wallclock | `588.054s` effective training time |
| Post-EMA pre-quant val_bpb | `1.09111489` |
| Quantized full val_bpb | `1.10253513` |
| Quantized sliding-window val_bpb | `1.08587554` |
| Legal TTT exact val_bpb | `1.08448769` |
| Quantized artifact bytes | `15,969,046` |
| Training submission total bytes | `15,984,562` |
| TTT submission total bytes | `15,985,765` |

## Configuration

Training used the April 5 SP8192 stack with these notable settings:

- `QK_GAIN_INIT=4.0`
- `MATRIX_CLIP_SIGMAS=12.86`
- `GPTQ_CALIBRATION_BATCHES=64`
- `EMA_DECAY=0.997`
- `SLIDING_WINDOW_ENABLED=1`
- `TRAIN_BATCH_TOKENS=786432`
- `VAL_BATCH_TOKENS=524288`
- `MAX_WALLCLOCK_SECONDS=600`

Legal TTT evaluation used:

- `TTT_LR=0.005`
- `TTT_EPOCHS=3`
- `TTT_CHUNK_TOKENS=32768`
- `TTT_BATCH_SEQS=32`
- `TTT_FREEZE_BLOCKS=0`
- `EVAL_STRIDE=64`

The TTT evaluator uses score-first ordering: validation tokens in a chunk are
scored before the model update for that chunk, matching the challenge rule that
test-time training may only update on already-scored validation tokens.

## Reproduction Notes

The RunPod instance had CUDA driver `12.8`, so the run used PyTorch
`2.11.0+cu128` and the CUDA 12.8 Flash Attention 3 wheel. CUDA 13 wheels failed
on this pod class because the driver was too old.

The pod also hit a workspace write quota when Hugging Face cached blobs were
copied into `data/datasets`. The working setup used symlinks from the dataset
directory to the Hugging Face cache:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
MATCHED_FINEWEB_MATERIALIZE=symlink \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

To reproduce the training pass from this folder after data is present:

```bash
RUN_ID=sp8192_qk4_ttt_8xh100_seed1337_train \
SEED=1337 \
DATA_DIR=/workspace/parameter-golf/data \
MATRIX_CLIP_SIGMAS=12.86 \
GPTQ_CALIBRATION_BATCHES=64 \
SLIDING_WINDOW_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To run the legal TTT evaluation against the exported artifact:

```bash
SOURCE_ARTIFACT=/workspace/parameter-golf/phase2_runs/sp8192_qk4_ttt_8xh100_seed1337/final_model.int6.ptz \
DATA_DIR=/workspace/parameter-golf/data \
QK_GAIN_INIT=4.0 \
TTT_ENABLED=1 \
TTT_LR=0.005 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_BATCH_SEQS=32 \
TTT_FREEZE_BLOCKS=0 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 eval_legal_ttt_from_artifact.py
```

## Interpretation

This run establishes that our SP8192 8xH100 path is operational end-to-end:
full data download, training, GPTQ/SDClip export, sliding-window validation,
and legal score-first TTT all completed under the artifact cap.

It is useful as a funded-compute milestone and reproduction baseline. The next
record-oriented attempt should move beyond QK4 to the later official deltas:
QK5/legal TTT first, then parallel residuals or 3-layer recurrence.

## Included Files

- `train_gpt.py` — April 5 SP8192 QK4 training/export script snapshot.
- `eval_legal_ttt_from_artifact.py` — location-independent wrapper for legal
  score-first TTT evaluation using the April 6 implementation.
- `train_seed1337.log` — training, export, and sliding eval log.
- `legal_ttt_seed1337.log` — legal TTT eval log.
- `submission.json` — metadata for non-record submission review.
