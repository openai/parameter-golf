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
- A Blackwell workstation continuation that refreshed the strongest legal all-attention control package before promoting a stronger hybrid point.
- A compile-friendly architecture family, while the kept run below explicitly used `ENABLE_TORCH_COMPILE=0`.

## Kept Result

- Kept layout: `AAAASASSS`
  - `A` = exact attention block
  - `S` = compile-friendly S4D-style state-space block
- Interpretation: four early exact-attention blocks, one mid attention anchor, and a four-block SSM tail
- SSM core: `S4D-Lin` descendant using learned exponential depthwise conv kernels
- Kept transfer: `SmearGate`, retained as a fixed-predictor one-at-a-time transfer
- Kept attention tuning: default `QK_GAIN_INIT=1.5`
- Kept export policy: keep only `ssm_coeff`, `ssm_log_decay`, and `ssm_d` in `float16`; quantize the rest with the standard int8+zlib path
- Seed: `2027`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `2200` steps on the single available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.50465667`
- Primary loss: `val_loss = 2.54054976`
- Model params: `17,119,784`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data.

The strongest retained legal all-attention control in this continuation is now the leaner `top1blockfp16` family that preserves `tok_emb` and only the top attention block in `float16`, then spends the recovered byte budget on more Blackwell training steps.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss | Legality |
|---|---|---:|---:|---:|---:|---:|---|
| Previous promoted hybrid | `AAAASASSS` | `312,374 ms` | `124,259 ms` | `13,249,267` | `1.59674695` | `2.69604033` | legal |
| Previous strongest legal control | `AAAAAAAAA` | `293,093 ms` | `233,089 ms` | `15,979,462` | `1.65376228` | `2.79230833` | legal |
| **Refreshed strongest legal control** | **`AAAAAAAAA`** | **`442,368 ms`** | **`222,490 ms`** | **`15,993,409`** | **`1.56658161`** | **`2.64510742`** | **legal** |
| Nearest refreshed byte-cap boundary control | `AAAAAAAAA` | `492,617 ms` | `212,123 ms` | `16,006,424` | `1.56591641` | `2.64398426` | illegal |
| **Kept hybrid** | **`AAAASASSS`** | **`609,292 ms`** | **`141,595 ms`** | **`15,260,268`** | **`1.50465667`** | **`2.54054976`** | **legal** |

Delta vs the refreshed strongest legal all-attention control: `-0.06192494` BPB.

Delta vs the previous promoted kept result: `-0.09209028` BPB.

Important legality note:

- the retained legal control is now the `1420`-step `top1blockfp16` point at `1.56658161` BPB
- the nearby `1425`-step control was slightly better on raw BPB but crossed the cap at `16,006,424` bytes
- the older `740`-step and `800`-step `top2blocksfp16` points remain retained legality references only
- all higher-score illegal controls are documentation only and are not admissible as counted controls

## Variance / Stability Package

Before promoting a new winner, the previous public winner `AAAASASSS` + `SSM_KERNEL_SIZE=96` + `SmearGate` at `1200` steps was rerun on the same Blackwell lane to verify that its gain survived more seeds.

Retained reruns for the previous public winner at `1200` steps:

- seed `2027`: `1.59674695` BPB
- seed `1337`: `1.60406053` BPB
- seed `4242`: `1.59435882` BPB
- seed `9001`: `1.60437754` BPB
- mean: `1.59988596`
- stddev: `0.00509915`

The earlier strongest control package at `730` steps was also rerun:

- seed `2027`: `1.65376228` BPB
- seed `4242`: `1.64550320` BPB
- seed `9001`: `1.65947334` BPB
- mean: `1.65291294`
- stddev: `0.00573482`

Mean edge for the previous public winner over that prior control package: `-0.05302698` BPB.

The refreshed strongest control family also has a retained rerun package:

- seed `2027`: `1.56658161` BPB
- seed `1337`: `1.56865945` BPB
- mean: `1.56762053`
- stddev: `0.00146925`

The promoted `2200`-step hybrid continuation has a retained rerun package:

- seed `2027`: `1.50465667` BPB
- seed `1337`: `1.50615600` BPB
- mean: `1.50540634`
- stddev: `0.00106019`

Mean edge for the promoted candidate over the refreshed control mean: `-0.06221419` BPB.

## Data / Scale Reality

The biggest realism bottleneck in this local campaign remains unchanged:

- detected local train shards: `1`
- available local shard: `fineweb_train_000000.bin`

This continuation again checked bounded alternate-machine options before accepting the one-shard limit:

- `vm-ubuntu-pitlab`: reachable, zero visible `fineweb_train_*.bin` shards, no visible `nvidia-smi`
- `ubuntu-dev`: reachable, zero visible `fineweb_train_*.bin` shards, no visible `nvidia-smi`
- `widelab-mac`: reachable, Apple `M4`, zero visible `fineweb_train_*.bin` shards
- `runpodctl`: installed locally but not configured with an API key, so no usable remote H100 lane was available from this workspace

No additional local or alternate-machine multi-shard continuation path was accessible during this run, so the kept result is still a one-shard non-record Blackwell result.

## Refreshed Control Frontier

This continuation materially strengthened the all-attention control package before promoting a new hybrid point.

Retained `top1blockfp16` controls on the same Blackwell lane with `tok_emb,blocks.8. -> float16`:

- `900` steps: `1.61215746` BPB, `14,397,629` total bytes, legal
- `1200` steps: `1.60052451` BPB, `15,371,862` total bytes, legal
- `1400` steps: `1.56884979` BPB, `15,941,727` total bytes, legal
- `1420` steps: `1.56658161` BPB, `15,993,409` total bytes, legal
- `1425` steps: `1.56591641` BPB, `16,006,424` total bytes, illegal

This matters for interpretation:

- the public lane is no longer being compared only against the older `730`-step `top2blocksfp16` baseline
- the kept hybrid now clears a much stronger legal all-attention control by `0.06192494` BPB

## Export Granularity Study

This continuation revisited recurrent export granularity on the stronger `AAAASASSS` branch before committing to more scale.

At `1200` steps on `AAAASASSS`, `SSM_RANK=12`, `SSM_KERNEL_SIZE=96`, `SMEAR_ENABLED=1`:

- narrow recurrent-core fp16 (`ssm_coeff,ssm_log_decay,ssm_d`): `1.59674695` BPB, `13,249,267` total bytes
- topmost full recurrent block fp16 (`ssm_coeff,ssm_log_decay,ssm_d,blocks.8.ssm.`): `1.59674681` BPB, `14,114,480` total bytes

The broader policy was effectively neutral on score while costing `865,213` extra bytes.

The kept export policy therefore remains the narrow recurrent-core allowlist.

## SSM Headroom Study

This continuation then spent the remaining legal headroom on the SSM side before adding any new public architectural claims.

At `1200` steps on `AAAASASSS`:

- `SSM_RANK=12`, `SSM_KERNEL_SIZE=96`: `1.59674695` BPB, `13,249,267` total bytes
- `SSM_RANK=12`, `SSM_KERNEL_SIZE=128`: `1.60650831` BPB, `13,247,332` total bytes
- `SSM_RANK=14`, `SSM_KERNEL_SIZE=96`: `1.59638095` BPB, `13,279,961` total bytes

The longer `128`-tap kernel regressed. A modest rank increase to `14` was slightly positive.

Scaling the stronger rank-14 point on the same lane produced:

- `1800` steps: `1.53097696` BPB, `14,765,396` total bytes
- `2000` steps: `1.51685767` BPB, `15,051,906` total bytes
- `2200` steps: `1.50465667` BPB, `15,260,268` total bytes

The kept result therefore spends the remaining legal budget on recurrent capacity plus more same-lane Blackwell scale, not on more attention.

## Fixed-Predictor Transfer Study

This continuation kept the transfer study strictly fixed-predictor and one-at-a-time.

Retained positive transfer from earlier in the lane:

- `AAAASASSS`, `SSM_KERNEL_SIZE=96`, `1000` steps, no SmearGate: `1.62386862` BPB
- same branch + `SMEAR_ENABLED=1`: `1.61118819` BPB

`SmearGate` improved score by `0.01268043` BPB and remains part of the kept branch family.

Smear tuning check:

- default `SMEAR_INIT=0.0`: `1.61118819` BPB
- lighter `SMEAR_INIT=-0.5`: `1.61534070` BPB

The lighter smear init was negative.

New transfer check on the stronger rank-14 branch:

- `2200`-step rank-14 hybrid, default `QK_GAIN_INIT=1.5`: `1.50465667` BPB
- same branch + `QK_GAIN_INIT=1.7`: `1.53303405` BPB

This QK-gain increase was strongly negative on the stronger branch, so the kept configuration stays on the default attention-gain setting.

Negative reference transfer retained from earlier in the lane:

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
- Refreshed legal all-attention control frontier completed, including legal and illegal byte-boundary points
- SSM-side headroom study completed
- Fixed-predictor transfer study completed
- Alternate-machine realism probe completed
- Local H100 / official-lane feasibility was not possible from the accessible environments during this run

Not claimed here:

- full training set usage
- official record-lane legality
- H100 or 8xH100 confirmation
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,941`
- Model bytes (`final_model.int8.ptz`): `15,202,327`
- Total bytes: `15,260,268`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `609,292 ms`
- Evaluation time: `141,595 ms`
- Export / serialization / roundtrip overhead: about `5,135 ms`
- End-to-end run duration: `756.02 s`

Refreshed strongest legal all-attention control:

- Training time: `442,368 ms`
- Evaluation time: `222,490 ms`
- Export / serialization / roundtrip overhead: about `8,179 ms`
- End-to-end run duration: `673.04 s`

## Exact Command

PowerShell command used for the kept run from the research workspace:

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$env:DATA_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024'
$env:TOKENIZER_PATH='C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\tokenizers\fineweb_1024_bpe.model'
$env:TRAIN_BATCH_TOKENS='32768'
$env:VAL_BATCH_SIZE='262144'
$env:TRAIN_SEQ_LEN='1024'
$env:ITERATIONS='2200'
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
$env:SSM_RANK='14'
$env:PARALLEL_ATTN_BIAS_INIT='1.5'
$env:SMEAR_ENABLED='1'
$env:INT8_FORCE_FLOAT_NAME_PATTERNS='ssm_coeff,ssm_log_decay,ssm_d'
$env:SEED='2027'
$env:RUN_ID='full_anchor_s4d_aaaasasss_rank14_k96_corefp16_smear_2200steps_blackwell_seed2027'
C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\.venv\Scripts\python.exe C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf-ssm-hybrid-research-scale\train_gpt.py
```
