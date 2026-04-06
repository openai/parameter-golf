# Non-Record: State-Space Hybrid with Attention Anchors

This folder records a local scaled run for the README wishlist `state-space models` lane.

This is **not** a leaderboard record attempt.
This is **not** an official 8xH100 / 10-minute lane run.
This is **not** a full-train-shards claim.
This is **not** a statistical-significance claim for a record.

What it is:

- A scorer-clean hybrid architecture study on the standard `train_gpt.py` path.
- Standard primary metric: `final_int8_zlib_roundtrip_exact val_bpb`.
- Full official validation split (`fineweb_val_*.bin`, `62,021,632` scored tokens).
- Local one-shard training on the single available `fineweb_train_000000.bin` shard.
- A Blackwell workstation scaling study with a same-lane training-wallclock-matched baseline control.
- An architecture family designed to remain compile-friendly, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAASSASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Attention anchors: layers `0-2` and `5`
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Export policy: keep `ssm.*` tensors in `float16` during export; quantize the rest with the standard int8+zlib path
- Seed: `1337`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `480` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.84399667`
- Primary loss: `val_loss = 3.11351113`
- Model params: `17,082,912`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The promoted comparison is matched by measured training wallclock on the same Blackwell lane rather than by fixed step count.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss |
|---|---|---:|---:|---:|---:|---:|
| Baseline control | `AAAAAAAAA` | `116,230 ms` | `218,850 ms` | `8,969,332` | `1.91983524` | `3.24156138` |
| Mid-anchor hybrid | `AASSSASSS` | `113,016 ms` | `101,313 ms` | `15,805,433` | `1.84814088` | `3.12050846` |
| **Kept hybrid** | **`AAASSASSS`** | **`119,096 ms`** | **`121,491 ms`** | **`14,781,218`** | **`1.84399667`** | **`3.11351113`** |

Delta vs the matched Blackwell baseline control: `-0.07583857` BPB.

End-to-end runtime was also lower for the kept hybrid than for the matched baseline:

- Kept hybrid duration: `245.95 s`
- Matched baseline duration: `344.39 s`

## Scaling Notes

This lane first produced a fixed-step one-shard sign-of-life at `80` steps:

- `AASSSASSS` + `ssm.* -> float16`: `3.15875948` BPB

The later Blackwell scaling study retained the same scorer path and export policy while increasing training time:

- `AASSSASSS`, `480` steps: `1.84814088` BPB, `15,805,433` bytes
- `AAASSASSS`, `480` steps: `1.84399667` BPB, `14,781,218` bytes

The kept result therefore reflects both stronger scaling and a small but retained anchor-placement improvement toward more lower-layer attention.

## Quantization / Export Finding

The hybrid path remains measurably more export-sensitive than the all-attention baseline.

Retained quantization study on the same proxy training state:

- Default export: `3.19792320` BPB on `tttmicro`
- Keep `ssm.*` in `float16`: `3.18893676` BPB, artifact `12,425,827` bytes
- Keep `ssm.*` in `float32`: `3.18862502` BPB, artifact `21,344,486` bytes, **illegal**

The kept policy is therefore:

- legal: yes (`14,781,218` bytes total on the promoted scaled run)
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
- Same-lane matched baseline retained for the promoted comparison

Not claimed here:

- full training set usage
- official record-lane legality
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,756`
- Model bytes (`final_model.int8.ptz`): `14,723,462`
- Total bytes: `14,781,218`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `119,096 ms`
- Evaluation time: `121,491 ms`
- Export / serialization / roundtrip overhead: about `5,359 ms`
- End-to-end run duration: `245.95 s`

Matched baseline control:

- Training time: `116,230 ms`
- Evaluation time: `218,850 ms`
- End-to-end run duration: `344.39 s`

## Exact Command

PowerShell command used for the kept run from the research workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='480'
$env:TRAIN_LOG_EVERY='20'
$env:VAL_LOSS_EVERY='0'
$env:MAX_WALLCLOCK_SECONDS='0'
$env:WARMUP_STEPS='0'
$env:ENABLE_TORCH_COMPILE='0'
$env:SDP_BACKEND='math'
$env:SAVE_RAW_MODEL='0'
$env:FINAL_PREQUANT_EVAL='0'
$env:BLOCK_LAYOUT='AAASSASSS'
$env:SSM_CORE='s4d'
$env:SSM_KERNEL_SIZE='64'
$env:SSM_RANK='4'
$env:INT8_FORCE_FLOAT_NAME_PATTERNS='ssm.'
$env:SEED='1337'
python train_gpt.py
```

## Included Files

- `train_gpt.py`: exact script snapshot used for the promoted run
- `train.log`: promoted kept run stdout log
- `best_run_summary.json`: machine-readable summary for the promoted kept run
- `ablation_scoreboard.tsv`: retained cycle table for this campaign
- `submission.json`: metadata for this non-record folder
