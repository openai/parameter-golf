# Non-Record: State-Space Hybrid with Attention Anchors

This folder records a local sign-of-life run for the README wishlist `state-space models` lane.

This is **not** a leaderboard record attempt.
This is **not** an 8xH100 run.
This is **not** a full-train-shards claim.

What it is:

- A scorer-clean hybrid architecture study on the standard `train_gpt.py` path.
- Standard primary metric: `final_int8_zlib_roundtrip_exact val_bpb`.
- Full official validation split (`fineweb_val_*.bin`, `62,021,632` scored tokens).
- Local workspace training on the single available `fineweb_train_000000.bin` shard, fixed-step matched against the all-attention control.
- An architecture family designed to remain compile-friendly, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AASSSASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Export policy: keep `ssm.*` tensors in `float16` during export; quantize the rest with the standard int8+zlib path
- Seed: `1337`
- Fixed-step budget: `80` steps
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`
- Primary score: `val_bpb = 3.15875948`
- Primary loss: `val_loss = 5.33343307`
- Model params: `17,087,512`

## Controlled Comparison

All rows below use the same local setup, the same scorer path, the same full validation split, and the same `80` training steps.

| Run | Layout | Export policy | val_bpb | val_loss | step_avg |
|---|---|---|---:|---:|---:|
| Baseline control | `AAAAAAAAA` | default int8+zlib | `3.18401827` | `5.37608148` | `449.33 ms` |
| Hybrid default | `AASSSASSS` | default int8+zlib | `3.17178172` | `5.35542057` | `351.52 ms` |
| **Kept hybrid** | `AASSSASSS` | **`ssm.* -> float16`** | **`3.15875948`** | **`5.33343307`** | **`315.26 ms`** |

Delta vs fixed-step baseline control: `-0.02525879` BPB.

## Architecture

The block layout keeps exact attention in the first two layers, inserts a mid attention anchor, and replaces the remaining blocks with an S4D-style SSM mixer from a compile-friendly architecture family:

```text
layers 0-1: attention
layers 2-4: S4D-style SSM
layer 5:    attention anchor
layers 6-8: S4D-style SSM
```

Each SSM block uses:

- pre-norm residual structure from the baseline
- learned exponential depthwise conv kernels (`kernel_size=64`, `rank=4`)
- gated input projection
- learned per-channel direct term (`D` skip)
- the same MLP and residual stack as the attention baseline

## Quantization / Export Finding

The hybrid path is measurably more export-sensitive than the all-attention baseline.

Retained quantization study on the same proxy training state:

- Default export: `3.19792320` BPB on `tttmicro`
- Keep `ssm.*` in `float16`: `3.18893676` BPB, artifact `12,425,827` bytes
- Keep `ssm.*` in `float32`: `3.18862502` BPB, artifact `21,344,486` bytes, **illegal**

The kept policy is therefore:

- legal: yes (`12,403,935` bytes total on the full retained run)
- better than default quantization
- specific to recurrent / state-space tensors rather than a global export change

## Validity Notes

Passed for this non-record folder:

- Same scorer path for control and hybrid (`train_gpt.py`, `final_int8_zlib_roundtrip_exact`)
- Full official validation split, standard `val_bpb`
- Artifact byte audit under the decimal `16,000,000` byte cap
- No validation-data training
- No evaluation-time downloads or external services
- Quantization/export policy explicitly accounted for in bytes
- Kept run configuration explicitly recorded with `ENABLE_TORCH_COMPILE=0`

Not claimed here:

- full training set usage
- 10-minute / 8xH100 legality
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,756`
- Model bytes (`final_model.int8.ptz`): `12,346,179`
- Total bytes: `12,403,935`

## Wallclock Breakdown

From the kept full run:

- Training time: `25,221 ms`
- Evaluation time: `114,443 ms`
- Export / serialization / roundtrip overhead: about `5,736 ms`
- End-to-end run duration: `145.40 s`

## Exact Command

PowerShell command used for the kept run from this workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='80'
$env:TRAIN_LOG_EVERY='20'
$env:VAL_LOSS_EVERY='0'
$env:MAX_WALLCLOCK_SECONDS='0'
$env:WARMUP_STEPS='0'
$env:ENABLE_TORCH_COMPILE='0'
$env:SDP_BACKEND='math'
$env:SAVE_RAW_MODEL='0'
$env:FINAL_PREQUANT_EVAL='0'
$env:BLOCK_LAYOUT='AASSSASSS'
$env:SSM_CORE='s4d'
$env:SSM_KERNEL_SIZE='64'
$env:SSM_RANK='4'
$env:INT8_FORCE_FLOAT_NAME_PATTERNS='ssm.'
$env:SEED='1337'
python train_gpt.py
```

## Included Files

- `train_gpt.py`: exact script snapshot used for the kept run
- `train.log`: kept run stdout log
- `best_run_summary.json`: machine-readable summary for the kept full run
- `ablation_scoreboard.tsv`: retained cycle table for this campaign
- `submission.json`: metadata for this non-record folder
