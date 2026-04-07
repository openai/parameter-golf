# Non-Record: State-Space Hybrid with Attention Anchors

This folder records a local continuation of the README wishlist `state-space models` lane.

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
- A Blackwell workstation continuation that tightened the legal all-attention control frontier to the byte ceiling before promoting a stronger hybrid point.
- A compile-friendly architecture family, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAAASASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: four early exact-attention blocks, one mid attention anchor, and a four-block SSM tail
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Kept export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `2027`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `700` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.68195191`
- Primary loss: `val_loss = 2.83990535`
- Model params: `17,111,080`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The strongest retained legal all-attention control in this continuation is the `517`-step artifact-budget-aware control that preserves `tok_emb`, blocks `7-8`, and block `6` attention weights in `float16`.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss | Legality |
|---|---|---:|---:|---:|---:|---:|---|
| Previous promoted hybrid | `AAAASASSS` | `176,889 ms` | `130,148 ms` | `10,476,226` | `1.72653150` | `2.91517612` | legal |
| Strongest legal all-attention control | `AAAAAAAAA` | `147,844 ms` | `207,443 ms` | `15,997,493` | `1.76063540` | `2.97275912` | legal |
| Illegal boundary control | `AAAAAAAAA` | `148,872 ms` | `198,573 ms` | `16,002,381` | `1.76150820` | `2.97423281` | illegal |
| **Kept hybrid** | **`AAAASASSS`** | **`177,883 ms`** | **`125,716 ms`** | **`11,248,890`** | **`1.68195191`** | **`2.83990535`** | **legal** |

Delta vs the strongest legal all-attention control: `-0.07868349` BPB.

Delta vs the previous promoted kept result: `-0.04457959` BPB.

Important legality note:

- the `518`-step all-attention control crossed the artifact cap at `16,002,381` bytes
- it is retained as a legality boundary only and is not admissible as a counted control
- the strongest retained legal control is therefore the `517`-step point at `1.76063540` BPB

## Variance / Stability Package

Before promoting a new winner, the previous public winner `AAAASASSS` was rerun on the same Blackwell lane to test whether its gain survived additional seeds.

Retained reruns for `AAAASASSS` + `SSM_RANK=8` + `ssm_coeff,ssm_log_decay,ssm_d -> float16` at `580` steps:

- seed `2027`: `1.72653150` BPB
- seed `1337`: `1.74584230` BPB
- seed `4242`: `1.72653573` BPB
- seed `9001`: `1.73082916` BPB
- mean: `1.73243467`
- stddev: `0.00793705`

Matched `480`-step artifact-aware all-attention reruns:

- seed `2027`: `1.79187065` BPB
- seed `4242`: `1.79729407` BPB
- seed `9001`: `1.79558744` BPB
- mean: `1.79491739`
- stddev: `0.00226423`

Mean edge for the previous public winner over that control package: `-0.06248271` BPB.

The promoted `AAAASASSS` + `SSM_RANK=12` continuation also has a retained rerun package:

- seed `2027`: `1.68195191` BPB
- seed `1337`: `1.69065755` BPB
- mean: `1.68630473`
- stddev: `0.00435282`

## Data / Scale Reality

The biggest realism bottleneck in this local campaign remains unchanged:

- detected local train shards: `1`
- available shard: `fineweb_train_000000.bin`

No additional local `fineweb_train_*.bin` shards were available during this continuation, so the kept result is still a one-shard non-record Blackwell result.

## Stronger Control Frontier

This continuation tightened the legal all-attention control package to the byte ceiling before promoting a stronger hybrid point.

Retained all-attention controls on the same Blackwell lane:

- `480` steps: `1.79187065` BPB, `15,803,271` total bytes, legal
- `500` steps: `1.77639623` BPB, `15,911,144` total bytes, legal
- `515` steps: `1.76264468` BPB, `15,988,097` total bytes, legal
- `517` steps: `1.76063540` BPB, `15,997,493` total bytes, legal
- `518` steps: `1.76150820` BPB, `16,002,381` total bytes, illegal

This matters for interpretation:

- the public lane is no longer being compared against the earlier weaker `480`-step control alone
- the kept hybrid still clears the tightened legal control ceiling by `0.07868349` BPB

## Export Granularity Study

The recurrent / state-space tensors remain export-sensitive, but this continuation rechecked whether the broader `ssm.* -> float16` policy was actually worth its bytes on the current layout.

On `AAAASASSS`, `SSM_RANK=8`, `580` steps:

- narrow recurrent-core fp16 (`ssm_coeff,ssm_log_decay,ssm_d`): `1.72653150` BPB, `10,476,226` total bytes
- broad `ssm.* -> float16`: `1.72652974` BPB, `14,406,178` total bytes

The broader policy improved score by only `0.00000176` BPB while costing `3,929,952` extra bytes.

The kept export policy therefore remains the narrow recurrent-core allowlist.

## SSM Headroom Study

This continuation spent additional legal bytes on the SSM side of the incumbent layout before promoting the final branch.

On `AAAASASSS`, `600` steps:

- `SSM_RANK=8`: `1.71464907` BPB, `10,603,338` total bytes
- `SSM_RANK=12`: `1.71415861` BPB, `10,668,076` total bytes

The rank increase was score-positive and remained comfortably legal.

Scaling that stronger SSM-side point to `700` steps produced the kept result:

- `AAAASASSS`, `SSM_RANK=12`, `700` steps: `1.68195191` BPB, `11,248,890` total bytes

## Fixed-Predictor Transfer Study

This run also tested a static frontier trick instead of only doing more layout churn.

Retained BigramHash transfer on the scaled incumbent:

- `AAAASASSS`, `SSM_RANK=8`, `600` steps, no BigramHash: `1.71464907` BPB
- same branch + `BIGRAM_VOCAB_SIZE=1024`, `BIGRAM_DIM=64`: `1.71975209` BPB

The small fixed-predictor BigramHash side path was therefore a negative transfer on this hybrid branch in the current one-shard setting.

## Validity Notes

Passed for this non-record folder:

- Same scorer path for control and hybrid (`train_gpt.py`, `final_int8_zlib_roundtrip_exact`)
- Full official validation split, standard `val_bpb`
- Artifact byte audit under the decimal `16,000,000` byte cap for the kept promoted run
- No validation-data training
- No evaluation-time downloads or external services
- Quantization/export policy explicitly accounted for in bytes
- Kept run configuration explicitly recorded with `ENABLE_TORCH_COMPILE=0`
- Fixed-predictor labeling explicit; no eval-time adaptation or TTT
- Expanded rerun package for the previous public winner completed
- Stronger legal all-attention control frontier completed
- Fixed-predictor transfer study completed
- Illegal higher-score all-attention point retained explicitly and not counted

Not claimed here:

- full training set usage
- official record-lane legality
- H100 or 8xH100 confirmation
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,756`
- Model bytes (`final_model.int8.ptz`): `11,191,134`
- Total bytes: `11,248,890`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `177,883 ms`
- Evaluation time: `125,716 ms`
- Export / serialization / roundtrip overhead: about `5,075 ms`
- End-to-end run duration: `308.67 s`

Strongest legal all-attention control:

- Training time: `147,844 ms`
- Evaluation time: `207,443 ms`
- Export / serialization / roundtrip overhead: about `10,314 ms`
- End-to-end run duration: `365.60 s`

## Exact Command

PowerShell command used for the kept run from the research workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='700'
$env:TRAIN_LOG_EVERY='20'
$env:VAL_LOSS_EVERY='0'
$env:MAX_WALLCLOCK_SECONDS='0'
$env:WARMUP_STEPS='0'
$env:ENABLE_TORCH_COMPILE='0'
$env:SDP_BACKEND='math'
$env:SAVE_RAW_MODEL='0'
$env:FINAL_PREQUANT_EVAL='0'
$env:BLOCK_LAYOUT='AAAASASSS'
$env:SSM_CORE='s4d'
$env:SSM_KERNEL_SIZE='64'
$env:SSM_RANK='12'
$env:INT8_FORCE_FLOAT_NAME_PATTERNS='ssm_coeff,ssm_log_decay,ssm_d'
$env:SEED='2027'
python train_gpt.py
```

## Included Files

- `train_gpt.py`: exact script snapshot used for the promoted run
- `train.log`: promoted kept run stdout log
- `best_run_summary.json`: machine-readable summary for the promoted kept run
- `ablation_scoreboard.tsv`: retained cycle table for this campaign
- `variance_summary.json`: machine-readable rerun / control / legality / transfer dossier
- `submission.json`: metadata for this non-record folder
