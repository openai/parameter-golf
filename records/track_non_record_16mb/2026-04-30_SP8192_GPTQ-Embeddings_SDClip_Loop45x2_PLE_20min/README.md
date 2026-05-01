# Non-Record Submission: SP8192 GPTQ Embeddings + SDClip + Loop45x2 + PLE

This is a non-record exploratory submission adding per-layer embeddings (PLE) to Kevin Clark's SP8192 GPTQ embeddings + SDClip + Loop45x2 stack from [PR #1394](https://github.com/openai/parameter-golf/pull/1394).

It is not a leaderboard record attempt. The run used a 20-minute wallclock cap (`MAX_WALLCLOCK_SECONDS=1200`) and the exported quantized+brotli artifact was `20,886,863` bytes, which is `4,886,863` bytes over the `16,000,000` byte artifact cap. The result is included because it is a useful PLE-on-PR1394 datapoint.

## Provenance

- Base submission: Kevin Clark's [PR #1394](https://github.com/openai/parameter-golf/pull/1394), "SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R + SDClip"
- Local implementation commit: `54bb087` (`54bb087ea167d7a23d95d4638e91783c574b2388`)
- PLE commit author: `BumaldaOverTheWater94`
- Run ID: `baseline_sp8192_GPTQ_embeddings_SDClip_loop_PLE_r1`

## What Changed

The run keeps the PR #1394 SP8192, GPTQ embeddings, standard-deviation clipping, MuonEq-R, and Loop45x2 baseline shape, then adds PLE:

- `PER_LAYER_EMBED_DIM=64`
- `PER_LAYER_EMBED_INIT_STD=0.02`
- Learned token-side per-layer embeddings in `embed_tokens_per_layer`
- A learned model-side `per_layer_model_projection`
- Per-block gated PLE injection after attention and MLP updates
- Rowwise int8 export for `embed_tokens_per_layer.weight`

The provided run used `MTP=1`, so this is a next-token objective run despite the PLE architecture change.

## Results

| Metric | Value |
|--------|------:|
| Quantized exact val_bpb | `1.21951793` |
| Quantized exact val_loss | `3.15010472` |
| Pre-quant post-EMA val_bpb | `1.21469745` |
| Pre-quant post-EMA val_loss | `3.13765307` |
| Stopped step | `1101 / 20000` |
| Train time | `1,188,555ms` |
| Wallclock cap | `1200s` |
| Model params | `42,792,024` |
| Quantized+brotli model bytes | `20,795,676` |
| Code bytes | `91,187` |
| Total submission bytes | `20,886,863` |

For comparison, PR #1394 reported a 5-seed mean sliding BPB of `1.08563` under the 16MB cap. This PLE run is therefore a negative result in this exact configuration: it increases artifact size substantially and does not improve quality within the logged 20-minute single-run setup.

## Run Command

The log was produced with the defaults from commit `54bb087` plus the explicit run identity and validation cadence shown below:

```bash
RUN_ID=baseline_sp8192_GPTQ_embeddings_SDClip_loop_PLE_r1 \
WANDB=1 \
WANDB_PROJECT=parameter-golf \
WANDB_RUN_NAME=baseline_sp8192_GPTQ_embeddings_SDClip_loop_PLE_r1 \
SEED=1337 \
MAX_WALLCLOCK_SECONDS=1200 \
VAL_LOSS_EVERY=250 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Track-relevant defaults from the logged hyperparameters:

```text
DATA_PATH=./data/datasets/fineweb10B_sp8192/
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
VOCAB_SIZE=8192
NUM_LAYERS=11
MODEL_DIM=512
EMBEDDING_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=4.0
TIE_EMBEDDINGS=1
TRAIN_BATCH_TOKENS=786432
TRAIN_SEQ_LEN=2048
EVAL_SEQ_LEN=2048
EVAL_STRIDE=64
MTP=1
NUM_LOOPS=2
LOOP_START=4
LOOP_END=5
ENABLE_LOOPING_AT=0.5
PER_LAYER_EMBED_DIM=64
PER_LAYER_EMBED_INIT_STD=0.02
MATRIX_BITS=6
EMBED_BITS=8
MATRIX_CLIP_SIGMAS=12.85
EMBED_CLIP_SIGMAS=20.0
GPTQ_CALIBRATION_BATCHES=64
GPTQ_RESERVE_SECONDS=12.0
COMPRESSOR=brotli
EMA_DECAY=0.997
MUON_ROW_NORMALIZE=1
MUON_WD=0.085
EMBED_WD=0.085
```

## Included Files

- `train_gpt.py` - exact code snapshot from commit `54bb087`
- `train_seed1337.log` - provided training and export log
- `submission.json` - non-record metadata, including explicit over-cap status
