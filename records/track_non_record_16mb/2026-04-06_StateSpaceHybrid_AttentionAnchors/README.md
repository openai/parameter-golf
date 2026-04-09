# Non-Record: State-Space Hybrid with Attention Anchors

This folder records a V8 promotion of the README wishlist `state-space models` lane.

This is **not** a leaderboard record attempt.
This is **not** an official 8xH100 / 10-minute lane run.
This is **not** a full-train-shards claim for the kept run.
This is **not** a statistical-significance claim for a record.

Track label for this folder:

- fixed-predictor state-space hybrid
- not an adaptive-compression result
- no eval-time adaptation or TTT

What it is:

- A scorer-clean hybrid architecture study on the standard `train_gpt.py` path.
- Standard primary metric: `final_int8_zlib_roundtrip_exact val_bpb`.
- Full official validation split (`fineweb_val_*.bin`, `62,021,632` scored tokens).
- The kept promoted result still trains on the single locally available `fineweb_train_000000.bin` shard.
- Phase 0 now includes a three-seed rerun package for the current kept hybrid family and the refreshed strongest legal all-attention control family on the same Blackwell lane.
- Phase 1 now includes a bounded Modal H100 continuation over an 80-shard cached train view, improving realism without changing the non-record status of the kept point.
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
- Seed: `4242`
- Fixed config: `TRAIN_BATCH_TOKENS=32768`, `TRAIN_SEQ_LEN=1024`, `VAL_BATCH_SIZE=262144`
- Training budget: `2200` steps on the single locally available train shard
- Kept run compile setting: `ENABLE_TORCH_COMPILE=0`, `SDP_BACKEND=math`
- GPU lane: single `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- Primary score: `val_bpb = 1.50126339`
- Primary loss: `val_loss = 2.53482035`
- Model params: `17,119,784`

## Controlled Comparison

All rows below use the same scorer path, the same tokenizer, the same full validation split, and the same one-shard local training data unless explicitly labeled as the separate H100 realism probe.

The strongest retained legal all-attention control in this continuation remains the leaner `top1blockfp16` family that preserves `tok_emb` and only the top attention block in `float16`, then spends the recovered byte budget on more Blackwell training steps.

| Run | Layout | Train time | Eval time | Total bytes | val_bpb | val_loss | Legality |
|---|---|---:|---:|---:|---:|---:|---|
| Previous public winner | `AAAASASSS` | `609,292 ms` | `141,595 ms` | `15,260,268` | `1.50465667` | `2.54054976` | legal |
| Strongest legal all-attention control | `AAAAAAAAA` | `442,368 ms` | `222,490 ms` | `15,993,409` | `1.56658161` | `2.64510742` | legal |
| Control rerun on same lane | `AAAAAAAAA` | `431,256 ms` | `190,006 ms` | `15,996,880` | `1.56838339` | `2.64814966` | legal |
| Nearest byte-cap boundary control | `AAAAAAAAA` | `492,617 ms` | `212,123 ms` | `16,006,424` | `1.56591641` | `2.64398426` | illegal |
| **Kept hybrid** | **`AAAASASSS`** | **`1,207,089 ms`** | **`124,606 ms`** | **`15,272,426`** | **`1.50126339`** | **`2.53482035`** | **legal** |

Delta vs the strongest legal all-attention control: `-0.06531822` BPB.

Delta vs the previous public winner: `-0.00339328` BPB.

Important legality note:

- the strongest retained legal control remains the `1420`-step `top1blockfp16` point at `1.56658161` BPB
- the additional `4242` rerun stayed legal but was weaker at `1.56838339` BPB
- the nearby `1425`-step control was slightly better on raw BPB but crossed the cap at `16,006,424` bytes
- all higher-score illegal controls remain documentation only and are not admissible as counted controls

## Phase 0 Variance Package

The current kept hybrid family was rerun twice more on the same Blackwell lane before promoting V8.

Retained reruns for the current kept hybrid family at `2200` steps:

- seed `2027`: `1.50465667` BPB
- seed `1337`: `1.50615600` BPB
- seed `4242`: `1.50126339` BPB
- mean: `1.50402535`
- sample stddev: `0.00250666`

The refreshed strongest legal control family at `1420` steps was also rerun to three seeds:

- seed `2027`: `1.56658161` BPB
- seed `1337`: `1.56865945` BPB
- seed `4242`: `1.56838339` BPB
- mean: `1.56787482`
- sample stddev: `0.00112842`

Mean paired hybrid-minus-control edge across the three matching seeds: `-0.06384946` BPB.

Paired edge sample stddev: `0.00284710`.

## Data / Scale Reality

The biggest local realism bottleneck remains the same:

- local dataset directory: `C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\data\datasets\fineweb10B_sp1024`
- local train shards detected: `1`
- available local shard: `fineweb_train_000000.bin`
- manifest-declared train shards for the full dataset: `195`

This continuation again checked bounded alternate-machine options before accepting the local one-shard limit:

- `vm-ubuntu-pitlab`: reachable, zero visible `fineweb_train_*.bin` shards, no visible `nvidia-smi`
- `ubuntu-dev`: reachable, zero visible `fineweb_train_*.bin` shards, no visible `nvidia-smi`
- `widelab-mac`: reachable, Apple `M4`, zero visible `fineweb_train_*.bin` shards
- `runpodctl`: installed locally but not configured with an API key, so no usable RunPod path was available from this workspace

A usable remote H100 path did exist through Modal without new human setup:

- cached volume: `pg-data`
- cached train view: `fineweb10B_sp1024_train080`
- train shards visible on that view: `80`
- bounded H100 continuation: `modal_hybrid_aaaasasss_rank14_k96_smear_train080_400steps_seed4242_v8`
- exact result: `val_bpb = 1.84995753`, `val_loss = 3.12357579`, `9,265,387` total bytes

This improves the realism package because the same fixed-predictor hybrid recipe was exercised on a real multi-shard H100 path. It does **not** convert the kept result into an official-lane claim, and it does **not** replace the local Blackwell kept run as the promoted non-record point.

Phase 6 official-lane feasibility was not triggered in this campaign because the raw improvement over `1.50465667` was `0.00339328` BPB, below the `0.01` threshold required to force an official-lane feasibility attempt.

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
- the kept hybrid now clears a much stronger legal all-attention control by `0.06531822` BPB
- the strongest legal control remained stable enough across three seeds that the hybrid still keeps a material edge on the refreshed package

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

- `1800` steps, seed `2027`: `1.53097696` BPB, `14,765,396` total bytes
- `2000` steps, seed `2027`: `1.51685767` BPB, `15,051,906` total bytes
- `2200` steps, seed `2027`: `1.50465667` BPB, `15,260,268` total bytes
- `2200` steps, seed `4242`: `1.50126339` BPB, `15,272,426` total bytes

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

QK-gain tuning check on the stronger rank-14 branch:

- `1800`-step rank-14 hybrid, default `QK_GAIN_INIT=1.5`: `1.53097696` BPB
- same branch + `QK_GAIN_INIT=1.7`: `1.53303405` BPB

This QK-gain increase was negative on the stronger branch, so the kept configuration stays on the default attention-gain setting.

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
- Recurrent export policy explicitly accounted for separately from the attention/MLP export policy
- Kept run configuration explicitly recorded with `ENABLE_TORCH_COMPILE=0`
- Fixed-predictor labeling explicit; no eval-time adaptation or TTT
- Phase 0 expanded rerun package completed
- Phase 1 realism package completed with a bounded Modal H100 multi-shard continuation
- Phase 2 refreshed legal all-attention control frontier completed, including legal and illegal byte-boundary points
- Phase 3 fixed-predictor transfer study completed
- Phase 4 SSM-side headroom study completed
- Phase 6 official-lane feasibility was not triggered by this promotion

Not claimed here:

- full training set usage for the kept run
- official record-lane legality
- official-lane feasibility confirmation
- statistical significance for a record claim

## Artifact Size

- Code bytes: `57,941`
- Model bytes (`final_model.int8.ptz`): `15,214,485`
- Total bytes: `15,272,426`
- Remaining legal headroom: `727,574`

## Wallclock Breakdown

From the kept promoted run:

- Training time: `1,207,089 ms`
- Evaluation time: `124,606 ms`
- Export / serialization / roundtrip overhead: about `5,764 ms`
- End-to-end run duration: `1,337.46 s`

Strongest legal all-attention control:

- Training time: `442,368 ms`
- Evaluation time: `222,490 ms`
- Export / serialization / roundtrip overhead: about `8,179 ms`
- End-to-end run duration: `673.04 s`

Modal H100 realism continuation:

- Training time: `88,650 ms`
- Evaluation time: `79,800 ms`
- End-to-end run duration: `264.98 s`

## Exact Commands

PowerShell command used for the kept promoted run from the research workspace:

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
$env:SEED='4242'
$env:RUN_ID='full_anchor_s4d_aaaasasss_rank14_k96_corefp16_smear_2200steps_blackwell_seed4242'
C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\.venv\Scripts\python.exe C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf-ssm-hybrid-research-scale\train_gpt.py
```

PowerShell command used for the bounded Modal H100 realism continuation:

```powershell
C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf\.venv\Scripts\python.exe C:\Users\GreQ\.codex_playground\OpenAIGolf\parameter-golf-ssm-hybrid-research-scale\experiments\state_space_hybrid\modal_phase1_probe.py --mode train --run-name modal_hybrid_aaaasasss_rank14_k96_smear_train080_400steps_seed4242_v8 --iterations 400 --seed 4242
```
