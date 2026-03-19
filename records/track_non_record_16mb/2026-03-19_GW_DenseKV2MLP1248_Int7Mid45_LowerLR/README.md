This folder is a non-record submission for OpenAI's Parameter Golf challenge and the strongest locally repeated clean-rules candidate I could support from my machine.

It targets the official `track_10min_16mb` objective, but it is not a leaderboard claim. The included logs are Apple Silicon proxy runs from `train_gpt_mlx.py`; the intended official path is the bundled PyTorch `train_gpt.py`.

**Why this branch exists**

The local search first found a better use of the parameter budget than the earlier dense `9L / KV4 / MLP_HIDDEN=1120` branch:

- reduce KV heads from `4` to `2`
- reinvest that budget into MLP width, here `MLP_HIDDEN=1248`
- lower the quantizer's small-float passthrough threshold so the `128x512` K/V matrices are actually quantized instead of silently kept in fp16

That last point matters. Without `INT8_KEEP_FLOAT_MAX_NUMEL=65535`, the reduced-KV branch looked smaller in parameter count but not in quantized payload, because the K/V weights landed exactly on the old keep-float threshold and were exported as fp16 passthrough tensors.

**What changed from baseline**

- `NUM_KV_HEADS=2` instead of `4`
- `MLP_HIDDEN=1248`
- lower LR schedule: `MATRIX_LR=0.02`, `SCALAR_LR=0.02`, `TIED_EMBED_LR=0.03`
- grouped `int8+zlib` export
- `INT8_KEEP_FLOAT_MAX_NUMEL=65535` so the reduced-KV weights are quantized
- `INT4_LAYERS=4,5` with `INT4_STEP=2` as a gentler post-quant coarsening pass

The bundled script still supports the adjacent research branches, but this record promotes the `9L / KV2 / 1248 / int7(mid 4,5)` branch because it slightly beat the full-int8 version on the local three-seed mean while also buying substantial extra compression headroom.

**Local evidence**

Standardized Apple M4 proxy recipe:

- `ITERATIONS=20`
- `TRAIN_BATCH_TOKENS=8192`
- `VAL_BATCH_SIZE=8192`
- `VAL_MAX_TOKENS=65536`
- `WARMUP_STEPS=1`
- `WARMDOWN_ITERS=6`
- one local FineWeb train shard, fixed validation slice

Most relevant runs:

| Run | Config summary | final `val_bpb` | Compressed bytes | Decision |
|---|---|---:|---:|---|
| `gw_dense_lowlr_grouped_only_1120_20` | old local dense leader, `KV4`, `MLP_HIDDEN=1120` | `3.33804306` | `7,590,347` | control |
| `gw_dense_lowlr_grouped_only_1120_20_seed42` | same control, second seed | `3.33462423` | `7,602,255` | control |
| `gw_dense_lowlr_grouped_only_1120_20_seed7` | same control, third seed | `3.33175527` | `7,594,036` | control |
| `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20` | `KV2`, `MLP_HIDDEN=1248`, int7 on layers `4,5` | `3.32987150` | `6,753,357` | promote |
| `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20_seed42` | same as above, second seed | `3.33260562` | `6,770,575` | consistency check passed |
| `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20_seed7` | same as above, third seed | `3.31983927` | `6,744,145` | consistency check passed |
| `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr15_20` | same branch, but `0.015 / 0.015 / 0.025` LR schedule | `3.34734563` | `6,204,252` | kill |
| `gw_9l_kv2_mlp1248_qkv_lowlr_20` | same branch, full int8 | `3.33042663` | `7,413,015` | first fallback |
| `gw_9l_kv2_mlp1248_qkv_lowlr_20_seed42` | same as above, second seed | `3.33215260` | `7,430,316` | keep |
| `gw_9l_kv2_mlp1248_qkv_lowlr_20_seed7` | same as above, third seed | `3.32044629` | `7,405,602` | keep |
| `gw_9l_kv2_mlp1248_qkv_int6mid45_lowlr_20` | same branch, int6 on layers `4,5` | `3.33213529` | `6,481,817` | second size fallback |
| `gw_9l_kv2_mlp1248_qkv_int6mid45_lowlr_20_seed42` | same fallback, second seed | `3.33371336` | `6,497,323` | keep |
| `gw_9l_kv2_mlp1248_fp16_qkv_lowlr_20` | same branch, but fp16 tied embed | `3.33098701` | `7,839,257` | kill |
| `gw_9l_kv2_mlp1280_qkv_lowlr_20` | wider follow-up, `KV2`, `MLP_HIDDEN=1280` | `3.33202033` | `7,510,773` | keep as width ceiling |
| `gw_9l_kv1_mlp1312_qkv32_lowlr_20` | `KV1`, matched params, quantized K/V | `3.33300925` | `7,302,970` | kill |
| `gw_10l_kv2_allint8pack_qkv_lowlr_20` | `10L`, `KV2`, all-int8, quantized K/V | `3.33497576` | `7,377,795` | keep as high-risk side branch |
| `gw_10l_kv2_allint8pack_qkv_lowlr_20_seed42` | same as above, second seed | `3.36207825` | `7,358,619` | too unstable |

Three-seed mean for the promoted branch:

```text
9L / KV2 / MLP_HIDDEN=1248 / int7(mid 4,5) mean val_bpb = 3.32743880
```

Comparison points:

```text
9L / KV2 / MLP_HIDDEN=1248 / full-int8 mean val_bpb = 3.32767517
9L / KV4 / MLP_HIDDEN=1120 mean val_bpb = 3.33480752
```

Interpretation:

- reduced KV heads can work in this tiny-model regime, but only if the exporter actually quantizes the smaller K/V matrices
- reallocating those bytes into MLP width beat the earlier denser `KV4` branch on the local proxy
- a gentler int7 fallback on just layers `4,5` slightly beat the full-int8 branch on the local three-seed mean while cutting roughly `0.65MB` from the local compressed artifact
- a milder `0.015 / 0.015 / 0.025` LR schedule on this exact branch did not help locally, despite shrinking the compressed payload
- the harsher int6 fallback on the same layers still exists as the more aggressive size-safety option
- fp16 tied-embedding export did not improve this reduced-KV branch locally and made the artifact materially larger
- pushing KV heads down to `1` did not beat `KV2` even after correcting the quantizer threshold
- the `9L / KV2 / 1248` branch is meaningfully more consistent than the nearby `10L` experiments
- a separate sibling folder tested exact sliding-window eval at strides `64` and `256`; both lost locally, so I am not promoting sliding on this branch without real H100 evidence

Practical size heuristic:

- On this proxy, the int7 branch is about `9.0%` smaller than the full-int8 branch.
- If I scale by the official baseline local-to-public ratio, that would put the full-int8 branch at roughly `16.14MB` and the int7 branch at roughly `14.68MB`.
- That is only a heuristic, not a claim, but it is the clearest reason I now prefer the int7 branch as the default local H100 recommendation.

**Recommended run order**

If the goal is maximum win probability, I would run candidates in this order:

1. `records/track_non_record_16mb/2026-03-19_GW_10L_MixedPrecision_Reference`
   Still the strongest public non-val-only result I found.

2. This folder's `9L / KV2 / MLP_HIDDEN=1248` with `INT4_LAYERS=4,5` and `INT4_STEP=2`
   Strongest locally repeated result from this machine.

3. `records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR`
   Same architecture without the int7 coarsening. Best direct control and first fallback if the gentler coarsening underperforms on H100s.

4. This folder's `9L / KV2 / MLP_HIDDEN=1248` with `INT4_LAYERS=4,5` and `INT4_STEP=4`
   More aggressive size-safety fallback if the gentler variant still lands too close to the cap.

5. `records/track_non_record_16mb/2026-03-18_GW_DensePack1120_LowerLR`
   Previous local leader and best direct control.

6. `records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR`
   Best self-contained local `10L` rerun path, but less stable.

Primary command from this folder:

```bash
RUN_ID=gw_dense_kv2_mlp1248_int7mid45_lowlr \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 \
INT4_STEP=2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

First fallback from the same family:

```bash
RUN_ID=gw_dense_kv2_mlp1248_lowlr \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Second fallback from the same family:

```bash
RUN_ID=gw_dense_kv2_mlp1248_int6mid45_lowlr \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1248 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 \
INT4_STEP=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Third fallback from the same family:

```bash
RUN_ID=gw_dense_kv2_mlp1280_lowlr \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
MLP_HIDDEN=1280 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
WARMDOWN_ITERS=3600 \
SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

**Included files**

- `train_gpt.py`: compact PyTorch candidate with configurable keep-float threshold and mixed-export support
- `train_gpt_mlx.py`: local MLX mirror used for the included proxy logs
- `train.log`: best local proxy log for the promoted `KV2 / 1248 / int7(mid 4,5)` branch
- `train_seed42.log`: second-seed companion log
- `train_seed1337.log`: first-seed companion log
- `train_int6mid45.log`: more aggressive size fallback
- `train_int6mid45_seed42.log`: second-seed validation for the aggressive size fallback
- `train_fp16.log`: fp16 tied-embed ablation on the promoted branch
- `train_kv1.log`: corrected `KV1` matched-budget probe
- `train_kv1_badthreshold.log`: initial `KV1` run before fixing the keep-float threshold
- `experiment_log.md`: append-only experiment ledger
- `submission.json`: staging metadata for this local candidate

**H100 handoff**

- Exact `8xH100` run commands, validation steps, and promotion criteria are in `H100_RUNBOOK.md`.
- That runbook is written for a real leaderboard attempt while staying inside the written rules: no val-only training, no tokenizer edits, no dataset edits.

**Remaining risks**

- No CUDA or `8xH100` verification was possible in this workspace.
- The strongest public result is still a different branch: `10` layers plus mixed `int8/int6` export.
- The local proxy is short and noisy, so the absolute values do not transfer directly to leaderboard numbers.
