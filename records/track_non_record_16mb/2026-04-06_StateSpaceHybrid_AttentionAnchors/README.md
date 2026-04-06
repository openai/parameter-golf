# Non-Record: State-Space Hybrid with Attention Anchors

This folder records a local scaled run for the README wishlist `state-space models` lane.

This is **not** a leaderboard record attempt.
This is **not** an official 8xH100 / 10-minute lane run.
This is **not** a full-train-shards claim.
This is **not** a statistical-significance claim for a record.

Track label for this folder:

- fixed-predictor state-space hybrid
- not an adaptive-compression result
- no eval-time adaptation or TTT

What it is:

- A scorer-clean hybrid architecture study on the standard `train_gpt.py` path.
- Standard primary metric: `final_int8_zlib_roundtrip_exact val_bpb`.
- Full official validation split (`fineweb_val_*.bin`, `62,021,632` scored tokens).
- Local one-shard training on the single available `fineweb_train_000000.bin` shard.
- A Blackwell workstation scaling study with same-lane training-wallclock-matched all-attention controls.
- An architecture family designed to remain compile-friendly, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAAAAAASS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: front-loaded exact attention with a short SSM tail
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Kept export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `2027`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `480` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.79279482`
- Primary loss: `val_loss = 3.02705896`
- Model params: `17,077,304`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The promoted comparison is matched by measured training wallclock on the same Blackwell lane rather than by fixed step count.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss |
|---|---|---:|---:|---:|---:|---:|
| Baseline control | `AAAAAAAAA` | `134,901 ms` | `221,886 ms` | `9,444,341` | `1.82125026` | `3.07510478` |
| Previous promoted hybrid | `AAASSASSS` | `119,096 ms` | `121,491 ms` | `14,781,218` | `1.84399667` | `3.11351113` |
| **Kept hybrid** | **`AAAAAAASS`** | **`138,815 ms`** | **`185,347 ms`** | **`9,709,339`** | **`1.79279482`** | **`3.02705896`** |

Delta vs the matched Blackwell baseline control: `-0.02845544` BPB.

Delta vs the previous promoted kept result: `-0.05120185` BPB.

## Stability Package

Before searching wider, the previous promoted winner `AAASSASSS` was rerun on the same Blackwell lane to test whether the signal survived seed changes.

Retained runs for `AAASSASSS` + `ssm.* -> float16`:

- seed `1337`: `1.84399667` BPB
- seed `2027`: `1.80196893` BPB
- seed `4242`: `1.82589781` BPB
- mean: `1.82395447`
- stddev: `0.01721269`

Matched baseline reruns at `390` steps:

- seed `1337`: `1.91983524` BPB
- seed `2027`: `1.90784061` BPB
- mean: `1.91383793`
- stddev: `0.00599732`

This continuation therefore treats the earlier hybrid gain as stable enough to search around, not a single-seed accident.

## Anchor / Layout Search Finding

The continuation kept pointing toward more lower and mid-layer exact attention:

- `AAASSASSS` + `ssm.* -> float16`: `1.80196893` BPB at seed `2027`
- `AAAASASSS` + `ssm.* -> float16`: `1.79647499` BPB
- `AAAAAASSS` + `core recurrent fp16`: `1.79508744` BPB
- `AAAAAAASS` + `core recurrent fp16`: `1.79291250` BPB
- `AAAAAAASS` + `core recurrent fp16` + `SSM_RANK=8`: `1.79279482` BPB

In this local one-shard setting, the best retained hybrid so far is therefore a strongly front-loaded attention stack with only a short SSM tail.

## Quantization / Export Finding

The recurrent / state-space tensors are still export-sensitive, but the continuation found a cleaner policy than the earlier blanket `ssm.* -> float16` rule.

On `AAAASASSS` at seed `2027`:

- `ssm.* -> float16`: `1.79647499` BPB, `13,758,841` total bytes
- `ssm_coeff,ssm_log_decay,ssm_d -> float16`: `1.79647573` BPB, `9,715,005` total bytes

The score delta is only `+0.00000074` BPB while saving `4,043,836` bytes.

The kept policy is therefore:

- legal: yes (`9,709,339` bytes total on the promoted run)
- effectively score-preserving on the tested layout
- specific to recurrent / state-space tensors rather than a global export change

## Headroom Study Finding

Once the tighter recurrent export policy was in place, the lane had over `6 MB` of artifact headroom left.

Retained recurrent-capacity study on `AAAAAAASS`:

- `SSM_RANK=4`: `1.79291250` BPB, `9,678,230` total bytes
- `SSM_RANK=8`: `1.79279482` BPB, `9,709,339` total bytes

The gain is small but positive and remains comfortably legal.

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
- Model bytes (`final_model.int8.ptz`): `9,651,583`
- Total bytes: `9,709,339`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `138,815 ms`
- Evaluation time: `185,347 ms`
- Export / serialization / roundtrip overhead: about `5,675 ms`
- End-to-end run duration: `329.84 s`

Matched baseline control:

- Training time: `134,901 ms`
- Evaluation time: `221,886 ms`
- Export / serialization / roundtrip overhead: about `8,531 ms`
- End-to-end run duration: `365.32 s`

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
$env:BLOCK_LAYOUT='AAAAAAASS'
$env:SSM_CORE='s4d'
$env:SSM_KERNEL_SIZE='64'
$env:SSM_RANK='8'
$env:INT8_FORCE_FLOAT_NAME_PATTERNS='ssm_coeff,ssm_log_decay,ssm_d'
$env:SEED='2027'
python train_gpt.py
```

## Included Files

- `train_gpt.py`: exact script snapshot used for the promoted run
- `train.log`: promoted kept run stdout log
- `best_run_summary.json`: machine-readable summary for the promoted kept run
- `ablation_scoreboard.tsv`: retained cycle table for this campaign
- `variance_summary.json`: machine-readable rerun / variance package
- `submission.json`: metadata for this non-record folder
