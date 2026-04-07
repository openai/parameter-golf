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
- A Blackwell workstation continuation with stronger legal all-attention controls, explicit artifact-byte audits, and retained hybrid frontier runs.
- An architecture family designed to remain compile-friendly, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAAASASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: four early exact-attention blocks, one mid attention anchor, and a four-block SSM tail
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Kept export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `2027`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `580` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.72653150`
- Primary loss: `val_loss = 2.91517612`
- Model params: `17,094,696`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The strongest retained legal all-attention control in this continuation is the `480`-step artifact-budget-aware control that preserves `tok_emb`, blocks `7-8`, and block `6` attention weights in `float16`.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss | Legality |
|---|---|---:|---:|---:|---:|---:|---|
| Previous promoted hybrid | `AAAAAASSS` | `139,161 ms` | `159,391 ms` | `10,026,646` | `1.76279061` | `2.97639811` | legal |
| Strongest legal all-attention control | `AAAAAAAAA` | `152,342 ms` | `217,439 ms` | `15,803,271` | `1.79187065` | `3.02549853` | legal |
| Illegal all-attention boundary | `AAAAAAAAA` | `167,752 ms` | `201,153 ms` | `16,211,560` | `1.73028291` | `2.92151022` | illegal |
| `6A/3S` headroom spend | `AAAAAASSS` | `158,349 ms` | `163,656 ms` | `10,352,195` | `1.73226675` | `2.92485985` | legal |
| **Kept hybrid** | **`AAAASASSS`** | **`176,889 ms`** | **`130,148 ms`** | **`10,476,226`** | **`1.72653150`** | **`2.91517612`** | **legal** |

Delta vs the strongest legal all-attention control: `-0.06533915` BPB.

Delta vs the previous promoted kept result: `-0.03625911` BPB.

Important legality note:

- the `560`-step all-attention control reached `1.73028291` BPB but is **not** promotable because it exceeded the artifact cap at `16,211,560` bytes
- the kept hybrid therefore beats the strongest **legal** all-attention control actually measured in this continuation

## Variance / Stability Package

Before promoting a new winner, the previous public winner `AAAAAASSS` was rerun on the same Blackwell lane to test whether its gain over the artifact-aware all-attention control survived additional seeds.

Retained reruns for `AAAAAASSS` + `SSM_RANK=8` + `ssm_coeff,ssm_log_decay,ssm_d -> float16`:

- seed `2027`: `1.76279061` BPB
- seed `1337`: `1.77518509` BPB
- seed `4242`: `1.76813040` BPB
- seed `9001`: `1.76407658` BPB
- mean: `1.76754567`
- stddev: `0.00483083`

Matched artifact-aware all-attention reruns at `450` steps:

- seed `2027`: `1.82051743` BPB
- seed `4242`: `1.83000022` BPB
- seed `9001`: `1.82466551` BPB
- mean: `1.82506105`
- stddev: `0.00388142`

Mean edge for the previous public winner over that control package: `-0.05751538` BPB.

The promoted `AAAASASSS` frontier point also has a retained rerun package:

- seed `2027`: `1.72653150` BPB
- seed `1337`: `1.74584230` BPB
- mean: `1.73618690`
- stddev: `0.00965540`

## Stronger Control Frontier

This continuation strengthened the all-attention control package before promoting a new hybrid point.

Retained legal controls on the same Blackwell lane:

- `450` steps, default all-attention baseline: `1.82125026` BPB, `9,444,341` total bytes
- `450` steps, top-two-block fp16 control: `1.82057800` BPB, `14,629,910` total bytes
- `450` steps, top-two-blocks plus block-6 attention fp16: `1.82051743` BPB, `15,642,808` total bytes
- `470` steps, top-two-blocks plus block-6 attention fp16: `1.80064012` BPB, `15,752,677` total bytes
- `480` steps, top-two-blocks plus block-6 attention fp16: `1.79187065` BPB, `15,803,271` total bytes

Retained illegal boundary:

- `560` steps, top-two-blocks plus block-6 attention fp16: `1.73028291` BPB, `16,211,560` total bytes, illegal under the `16,000,000` decimal-byte cap

This legality boundary mattered: the best legal control stayed above the promoted hybrid, while the first all-attention point that nearly erased the gap crossed the byte cap and was therefore not admissible.

## Hybrid Identity Frontier

The structured frontier in this continuation kept the best retained point in the `5A/4S` to `6A/3S` region rather than collapsing to almost-all-attention.

Key retained points:

- `AAAAAASSS`, `520` steps, `SSM_RANK=8`: `1.76279061` BPB
- `AAAAAASSS`, `560` steps, `SSM_RANK=8`: `1.73418039` BPB
- `AAAAAASSS`, `560` steps, `SSM_RANK=12`: `1.73226675` BPB
- `AAAASASSS`, `580` steps, `SSM_RANK=8`: `1.72653150` BPB

The kept point is therefore not the most attention-heavy layout tested. In this one-shard setting, a more clearly hybrid `5A/4S` interleaved-anchor layout beat the previous `6A/3S` public winner and the strongest retained legal all-attention control.

## Quantization / Export Finding

The recurrent / state-space tensors remain export-sensitive, but this continuation kept the narrower recurrent-core allowlist that had already proven score-preserving on the frontier.

Retained policy:

- `ssm_coeff,ssm_log_decay,ssm_d -> float16`
- rest of the model exported with the standard int8+zlib path

The promoted kept run remains comfortably legal at `10,476,226` total bytes.

## SSM Headroom Study

This continuation also spent additional legal bytes on the SSM side before promotion.

Retained `AAAAAASSS` headroom study:

- `SSM_RANK=8`, `560` steps: `1.73418039` BPB, `10,303,341` total bytes
- `SSM_RANK=12`, `560` steps: `1.73226675` BPB, `10,352,195` total bytes

The rank increase was score-positive and remained legal, but the interleaved `AAAASASSS` frontier point still won cleanly on raw BPB.

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
- Stronger legal all-attention control package retained before promotion
- Illegal higher-score all-attention control retained as a boundary but not counted

Not claimed here:

- full training set usage
- official record-lane legality
- H100 or 8xH100 confirmation
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,756`
- Model bytes (`final_model.int8.ptz`): `10,418,470`
- Total bytes: `10,476,226`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `176,889 ms`
- Evaluation time: `130,148 ms`
- Export / serialization / roundtrip overhead: about `5,360 ms`
- End-to-end run duration: `312.40 s`

Strongest legal all-attention control:

- Training time: `152,342 ms`
- Evaluation time: `217,439 ms`
- Export / serialization / roundtrip overhead: about `5,437 ms`
- End-to-end run duration: `375.22 s`

## Exact Command

PowerShell command used for the kept run from the research workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='580'
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
- `variance_summary.json`: machine-readable rerun / control / legality dossier
- `submission.json`: metadata for this non-record folder
