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
- A Blackwell workstation continuation that refreshed the legal all-attention control frontier before promoting a stronger hybrid point.
- A compile-friendly architecture family, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAAASASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: four early exact-attention blocks, one mid attention anchor, and a four-block SSM tail
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Kept transfer: `SmearGate`, retained as a fixed-predictor one-at-a-time transfer
- Kept smear tuning: default `SMEAR_INIT=0.0`
- Kept export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `2027`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `1200` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.59674695`
- Primary loss: `val_loss = 2.69604033`
- Model params: `17,111,592`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The strongest retained legal all-attention control in this continuation remains the `730`-step artifact-budget-aware control that preserves `tok_emb` and the top two attention blocks in `float16`.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss | Legality |
|---|---|---:|---:|---:|---:|---:|---|
| Previous promoted hybrid | `AAAASASSS` | `430,336 ms` | `153,401 ms` | `12,542,838` | `1.61118819` | `2.72042376` | legal |
| Strongest legal all-attention control | `AAAAAAAAA` | `293,093 ms` | `233,089 ms` | `15,979,462` | `1.65376228` | `2.79230833` | legal |
| Nearest byte-cap boundary control | `AAAAAAAAA` | `338,240 ms` | `231,955 ms` | `16,021,806` | `1.65807490` | `2.79959001` | illegal |
| Lower-score but lower-loss illegal control | `AAAAAAAAA` | `385,470 ms` | `237,867 ms` | `16,217,131` | `1.62477053` | `2.74335699` | illegal |
| **Kept hybrid** | **`AAAASASSS`** | **`312,374 ms`** | **`124,259 ms`** | **`13,249,267`** | **`1.59674695`** | **`2.69604033`** | **legal** |

Delta vs the strongest legal all-attention control: `-0.05701533` BPB.

Delta vs the previous promoted kept result: `-0.01444124` BPB.

Important legality note:

- the `740`-step all-attention control crossed the artifact cap at `16,021,806` bytes
- the `800`-step all-attention control reached a lower raw BPB but was even more illegal at `16,217,131` bytes
- both are retained as legality boundaries only and are not admissible as counted controls
- the strongest retained legal control is therefore the `730`-step point at `1.65376228` BPB

## Variance / Stability Package

Before promoting a new winner, the previous public winner `AAAASASSS` + `SSM_KERNEL_SIZE=96` + `SmearGate` was rerun on the same Blackwell lane to test whether its gain survived additional seeds.

Retained reruns for the previous public winner at `1000` steps:

- seed `2027`: `1.61118819` BPB
- seed `1337`: `1.61088286` BPB
- seed `4242`: `1.61544702` BPB
- seed `9001`: `1.62902041` BPB
- mean: `1.61663462`
- stddev: `0.00737503`

Matched `730`-step artifact-aware all-attention reruns:

- seed `2027`: `1.65376228` BPB
- seed `4242`: `1.64550320` BPB
- seed `9001`: `1.65947334` BPB
- mean: `1.65291294`
- stddev: `0.00573482`

Mean edge for the previous public winner over that control package: `-0.03627832` BPB.

The promoted `1200`-step continuation also has a retained rerun package:

- seed `2027`: `1.59674695` BPB
- seed `1337`: `1.60406053` BPB
- mean: `1.60040374`
- stddev: `0.00365679`

Mean edge for the promoted candidate over the refreshed control mean: `-0.05250920` BPB.

## Data / Scale Reality

The biggest realism bottleneck in this local campaign remains unchanged:

- detected local train shards: `1`
- available local shard: `fineweb_train_000000.bin`

This continuation also checked accessible alternate machines before accepting the one-shard limit:

- `vm-ubuntu-pitlab`: reachable, zero visible `fineweb_train_*.bin` shards, no visible `nvidia-smi`
- `ubuntu-dev`: reachable, zero visible `fineweb_train_*.bin` shards, no visible `nvidia-smi`
- `widelab-mac`: reachable, Apple `M4`, zero visible `fineweb_train_*.bin` shards

No additional local or alternate-machine multi-shard continuation path was accessible during this run, so the kept result is still a one-shard non-record Blackwell result.

## Stronger Control Frontier

This continuation kept the legal all-attention control package ahead of the hybrid before promoting a stronger hybrid point.

Retained all-attention controls on the same Blackwell lane with the leaner `tok_emb,blocks.7.,blocks.8. -> float16` policy:

- `600` steps: `1.70809237` BPB, `15,429,515` total bytes, legal
- `700` steps: `1.66501659` BPB, `15,864,516` total bytes, legal
- `730` steps: `1.65376228` BPB, `15,979,462` total bytes, legal
- `740` steps: `1.65807490` BPB, `16,021,806` total bytes, illegal
- `800` steps: `1.62477053` BPB, `16,217,131` total bytes, illegal

This matters for interpretation:

- the public lane is being compared against a materially stronger legal all-attention control than in the previous promotion
- the kept hybrid now clears that strengthened legal control frontier by `0.05701533` BPB

## Export Granularity Study

The recurrent / state-space tensors remain export-sensitive, but the earlier recheck still stands:

On `AAAASASSS`, `SSM_RANK=8`, `580` steps:

- narrow recurrent-core fp16 (`ssm_coeff,ssm_log_decay,ssm_d`): `1.72653150` BPB, `10,476,226` total bytes
- broad `ssm.* -> float16`: `1.72652974` BPB, `14,406,178` total bytes

The broader policy improved score by only `0.00000176` BPB while costing `3,929,952` extra bytes.

The kept export policy therefore remains the narrow recurrent-core allowlist.

## SSM Headroom Study

This continuation carried forward the SSM-side headroom search before the final scale push.

On `AAAASASSS`, `900` steps:

- `SSM_RANK=12`, `SSM_KERNEL_SIZE=64`: `1.62978755` BPB, `12,181,538` total bytes
- `SSM_RANK=16`, `SSM_KERNEL_SIZE=64`: `1.63069080` BPB, `12,245,342` total bytes
- `SSM_RANK=12`, `SSM_KERNEL_SIZE=96`: `1.62958731` BPB, `12,183,567` total bytes

The longer `96`-tap kernel remained slightly score-positive, while further rank expansion to `16` was not.

Scaling that stronger SSM-side point to `1000` steps without any transfer produced:

- `AAAASASSS`, `SSM_KERNEL_SIZE=96`, no SmearGate: `1.62386862` BPB, `12,548,229` total bytes

Scaling the same branch with retained SmearGate to `1200` steps produced the kept result:

- `AAAASASSS`, `SSM_KERNEL_SIZE=96`, `SMEAR_ENABLED=1`, `1200` steps: `1.59674695` BPB, `13,249,267` total bytes

## Fixed-Predictor Transfer Study

This run kept the transfer study strictly fixed-predictor and one-at-a-time.

Positive retained transfer:

- `AAAASASSS`, `SSM_KERNEL_SIZE=96`, `1000` steps, no SmearGate: `1.62386862` BPB
- same branch + `SMEAR_ENABLED=1`: `1.61118819` BPB

`SmearGate` improved score by `0.01268043` BPB and remains part of the kept branch family.

Smear tuning check:

- default `SMEAR_INIT=0.0`: `1.61118819` BPB
- lighter `SMEAR_INIT=-0.5`: `1.61534070` BPB

The lighter smear init was a negative tuning step, so the kept configuration stays on the default smear initialization.

Negative reference transfer:

- `AAAASASSS`, `SSM_RANK=8`, `600` steps, no BigramHash: `1.71464907` BPB
- same branch + `BIGRAM_VOCAB_SIZE=1024`, `BIGRAM_DIM=64`: `1.71975209` BPB

The small fixed-predictor BigramHash side path remains negative evidence in this lane.

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
- Stronger legal all-attention control frontier completed, including legal and illegal byte-boundary points
- SSM-side headroom study completed
- Fixed-predictor transfer study completed
- Alternate-machine realism probe completed
- Illegal higher-score all-attention points retained explicitly and not counted

Not claimed here:

- full training set usage
- official record-lane legality
- H100 or 8xH100 confirmation
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,941`
- Model bytes (`final_model.int8.ptz`): `13,191,326`
- Total bytes: `13,249,267`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `312,374 ms`
- Evaluation time: `124,259 ms`
- Export / serialization / roundtrip overhead: about `4,989 ms`
- End-to-end run duration: `441.62 s`

Strongest legal all-attention control:

- Training time: `293,093 ms`
- Evaluation time: `233,089 ms`
- Export / serialization / roundtrip overhead: about `9,572 ms`
- End-to-end run duration: `535.75 s`

## Exact Command

PowerShell command used for the kept run from the research workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='1200'
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
$env:SSM_KERNEL_SIZE='96'
$env:SSM_RANK='12'
$env:SMEAR_ENABLED='1'
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
