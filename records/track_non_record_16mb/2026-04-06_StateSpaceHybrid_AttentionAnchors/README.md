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
- A Blackwell workstation continuation with same-lane all-attention controls, including a stronger artifact-budget-aware control package.
- An architecture family designed to remain compile-friendly, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAAAAASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: six front-loaded exact-attention blocks followed by a three-block SSM tail
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Kept export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `2027`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `520` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.76279061`
- Primary loss: `val_loss = 2.97639811`
- Model params: `17,086,000`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The strongest retained all-attention control in this continuation is not the default baseline; it is a legal artifact-budget-aware control that preserves `tok_emb`, blocks `7-8`, and block `6` attention weights in `float16`.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss |
|---|---|---:|---:|---:|---:|---:|
| Default all-attention control | `AAAAAAAAA` | `134,901 ms` | `221,886 ms` | `9,444,341` | `1.82125026` | `3.07510478` |
| Strongest legal all-attention control | `AAAAAAAAA` | `133,911 ms` | `217,560 ms` | `15,642,808` | `1.82051743` | `3.07386743` |
| Previous promoted hybrid | `AAAAAAASS` | `138,815 ms` | `185,347 ms` | `9,709,339` | `1.79279482` | `3.02705896` |
| **Kept hybrid** | **`AAAAAASSS`** | **`139,161 ms`** | **`159,391 ms`** | **`10,026,646`** | **`1.76279061`** | **`2.97639811`** |

Delta vs the strongest legal all-attention control: `-0.05772682` BPB.

Delta vs the default same-lane all-attention control: `-0.05845965` BPB.

Delta vs the previous promoted kept result: `-0.03000421` BPB.

This promotion therefore improves raw BPB while also widening the strongest retained all-attention control gap. It also keeps a clearer hybrid identity than the previous public winner by using a `6A/3S` split instead of `7A/2S`.

## Variance / Stability Package

Before promoting a new winner, the previous public winner `AAAAAAASS` was rerun on the same Blackwell lane to test whether its gain over all-attention was stable across seeds.

Retained reruns for `AAAAAAASS` + `SSM_RANK=8` + `ssm_coeff,ssm_log_decay,ssm_d -> float16`:

- seed `2027`: `1.79279482` BPB
- seed `1337`: `1.80535135` BPB
- seed `4242`: `1.80158400` BPB
- mean: `1.79991006`
- stddev: `0.00526106`

Matched same-lane all-attention reruns at `450` steps:

- seed `2027`: `1.82125026` BPB
- seed `1337`: `1.84038899` BPB
- seed `4242`: `1.83073574` BPB
- mean: `1.83079166`
- stddev: `0.00781345`

Mean edge for the previous public winner over its matched default all-attention control package: `-0.03088161` BPB.

The promoted `AAAAAASSS` continuation also has a small rerun package:

- seed `2027`: `1.76279061` BPB
- seed `1337`: `1.77518509` BPB
- mean: `1.76898785`
- stddev: `0.00619724`

## Hybrid Identity Frontier

The structured frontier in this continuation kept favoring more lower-layer exact attention, but not an almost-all-attention collapse:

- `AAASSASSS` + `ssm.* -> float16`: `1.84399667` BPB at the earlier promoted point
- `AAAASASSS` + `ssm.* -> float16`: `1.79647499` BPB
- `AAAASASSS` + `ssm_coeff,ssm_log_decay,ssm_d -> float16`: `1.79647573` BPB
- `AAAAAASSS` + `ssm_coeff,ssm_log_decay,ssm_d -> float16` + `SSM_RANK=4`: `1.79508744` BPB
- `AAAAAAASS` + `ssm_coeff,ssm_log_decay,ssm_d -> float16` + `SSM_RANK=8`: `1.79279482` BPB
- `AAAAAASSS` + `ssm_coeff,ssm_log_decay,ssm_d -> float16` + `SSM_RANK=8` + `520` steps: `1.76279061` BPB

The current best retained point is therefore not the most attention-heavy layout tested. In this one-shard setting, the best frontier point so far keeps a non-trivial three-block SSM tail and beats the strongest retained all-attention control by a meaningful margin.

## Quantization / Export Finding

The recurrent / state-space tensors remain export-sensitive, but this continuation confirmed that only a narrow recurrent-core subset needs `float16`.

On `AAAASASSS` at seed `2027`:

- `ssm.* -> float16`: `1.79647499` BPB, `13,758,841` total bytes
- `ssm_coeff,ssm_log_decay,ssm_d -> float16`: `1.79647573` BPB, `9,715,005` total bytes

The score delta is only `+0.00000074` BPB while saving `4,043,836` bytes.

The kept export policy is therefore:

- legal: yes (`10,026,646` bytes total on the promoted run)
- recurrent-specific rather than global
- effectively score-preserving on the tested frontier point

## Headroom Spending Study

Once the tighter recurrent export policy was in place, the lane still had over `5.9 MB` of legal artifact headroom.

Retained capacity study around the stronger hybrid identity region:

- `AAAAAASSS`, `SSM_RANK=4`, `520` steps: `1.79508744` BPB, `9,692,318` total bytes
- `AAAAAASSS`, `SSM_RANK=8`, `520` steps: `1.76279061` BPB, `10,026,646` total bytes
- `AAAASASSS`, `SSM_RANK=6`, `480` steps: `1.79841119` BPB
- `AAAASASSS`, `SSM_KERNEL_SIZE=96`, `480` steps: `1.79834777` BPB

In this continuation, spending headroom on higher SSM rank in the `AAAAAASSS` region was clearly more effective than spending it on a longer kernel or on preserving more all-attention weights.

## Validity Notes

Passed for this non-record folder:

- Same scorer path for control and hybrid (`train_gpt.py`, `final_int8_zlib_roundtrip_exact`)
- Full official validation split, standard `val_bpb`
- Artifact byte audit under the decimal `16,000,000` byte cap
- No validation-data training
- No evaluation-time downloads or external services
- Quantization/export policy explicitly accounted for in bytes
- Kept run configuration explicitly recorded with `ENABLE_TORCH_COMPILE=0`
- Fixed-predictor labeling explicit; no eval-time adaptation or TTT
- Stronger all-attention control package retained before promotion

Not claimed here:

- full training set usage
- official record-lane legality
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,756`
- Model bytes (`final_model.int8.ptz`): `9,968,890`
- Total bytes: `10,026,646`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `139,161 ms`
- Evaluation time: `159,391 ms`
- Export / serialization / roundtrip overhead: about `5,494 ms`
- End-to-end run duration: `304.05 s`

Strongest legal all-attention control:

- Training time: `133,911 ms`
- Evaluation time: `217,560 ms`
- Export / serialization / roundtrip overhead: about `8,460 ms`
- End-to-end run duration: `359.93 s`

## Exact Command

PowerShell command used for the kept run from the research workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='520'
$env:TRAIN_LOG_EVERY='20'
$env:VAL_LOSS_EVERY='0'
$env:MAX_WALLCLOCK_SECONDS='0'
$env:WARMUP_STEPS='0'
$env:ENABLE_TORCH_COMPILE='0'
$env:SDP_BACKEND='math'
$env:SAVE_RAW_MODEL='0'
$env:FINAL_PREQUANT_EVAL='0'
$env:BLOCK_LAYOUT='AAAAAASSS'
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
- `variance_summary.json`: machine-readable rerun / control package
- `submission.json`: metadata for this non-record folder
