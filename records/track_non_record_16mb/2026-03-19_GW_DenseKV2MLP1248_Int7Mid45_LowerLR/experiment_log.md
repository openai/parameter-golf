# Experiment Log

## Controls

### Public PR `#39` reference
- Hypothesis: current public non-val-only frontier is `10` layers plus mixed `int8/int6` export.
- Code diff summary: public record copy only, not rerun locally.
- Command used: published PR `#39` command.
- Wall-clock time: `~600s` on `8xH200`.
- Final `val_loss`: `2.0510`.
- Final `val_bpb`: `1.21474500`.
- Compressed artifact size: `15,880,057` model bytes.
- Decision: keep as the public reference target.

### Previous local leader
- Hypothesis: the strongest repeated local dense branch should remain the baseline to beat.
- Code diff summary: `9L`, `KV4`, `MLP_HIDDEN=1120`, lower LR, grouped export.
- Command used: see `records/track_non_record_16mb/2026-03-18_GW_DensePack1120_LowerLR/experiment_log.md`.
- Wall-clock time: `~21s` per seed on the Apple proxy.
- Final `val_bpb`: `3.33804306`, `3.33462423`, `3.33175527`.
- Compressed artifact size: `7,590,347`, `7,602,255`, `7,594,036`.
- Decision: control.

## Reduced-KV discovery

### Export threshold check
- Hypothesis: reduced-KV branches are not getting full byte savings because the smaller K/V weights sit exactly at the keep-float threshold.
- Code diff summary: inspect the MLX quantizer with `NUM_KV_HEADS=2`.
- Command used: local MLX import check on `train_gpt_mlx.py`.
- Wall-clock time: sub-second.
- Result: with `INT8_KEEP_FLOAT_MAX_NUMEL=65536`, `blocks.0.attn.c_k.weight` and `c_v.weight` are stored as passthrough fp16; with `65535`, they move into the quantized bucket and the raw quantized payload falls from `19,020,096` to `17,714,496` bytes.
- Decision: keep; this fix is required for reduced-KV branches to be real.

## Promoted branch

### `gw_9l_kv2_mlp1248_qkv_lowlr_20`
- Hypothesis: reallocating parameter budget from K/V channels to MLP width can beat the previous local dense leader.
- Code diff summary: `NUM_LAYERS=9`, `NUM_KV_HEADS=2`, `MLP_HIDDEN_DIM=1248`, lower LR, grouped export, `INT8_KEEP_FLOAT_MAX_NUMEL=65535`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `20957ms` training, `1452ms` final roundtrip eval.
- Final `val_loss`: `5.63275710`.
- Final `val_bpb`: `3.33042663`.
- Compressed artifact size: `7,413,015` bytes.
- Decision: promote.

### `gw_9l_kv2_mlp1248_qkv_lowlr_20_seed42`
- Hypothesis: the promoted reduced-KV branch must remain strong on a second seed.
- Code diff summary: same branch, but `SEED=42`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_lowlr_20_seed42 SEED=42 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 WARMDOWN_ITERS=6 NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `20741ms` training, `1443ms` final roundtrip eval.
- Final `val_loss`: `5.63567624`.
- Final `val_bpb`: `3.33215260`.
- Compressed artifact size: `7,430,316` bytes.
- Decision: keep; consistency check passed.

### `gw_9l_kv2_mlp1248_qkv_lowlr_20_seed7`
- Hypothesis: the user asked for consistency, so the promoted branch should be checked with a third seed before it becomes the local lead.
- Code diff summary: same branch, but `SEED=7`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_lowlr_20_seed7 SEED=7 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 WARMDOWN_ITERS=6 NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `20960ms` training, `1457ms` final roundtrip eval.
- Final `val_loss`: `5.61587733`.
- Final `val_bpb`: `3.32044629`.
- Compressed artifact size: `7,405,602` bytes.
- Decision: promote to the new top local candidate. Three-seed mean: `3.32767517`.

### `gw_9l_kv2_mlp1248_fp16_qkv_lowlr_20`
- Hypothesis: the public fp16 tied-embedding export trick may still help after switching to `KV2`.
- Code diff summary: same promoted branch, but `SERIALIZE_TIED_EMBED_FP16=1`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_fp16_qkv_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=1 INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `21607ms` training, `1476ms` final roundtrip eval.
- Final `val_loss`: `5.63370489`.
- Final `val_bpb`: `3.33098701`.
- Compressed artifact size: `7,839,257` bytes.
- Decision: kill. It is slightly worse than the best grouped-only seeds and substantially larger.

## Size fallback

### `gw_9l_kv2_mlp1248_qkv_int6mid45_lowlr_20`
- Hypothesis: a lighter mixed-precision fallback might buy back enough bytes to fit the real cap without giving up much quality.
- Code diff summary: same promoted `KV2 / 1248` branch, but with `INT4_LAYERS=4,5` and `INT4_STEP=4`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_int6mid45_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 INT4_LAYERS=4,5 INT4_STEP=4 \
PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `24722ms` training, `1694ms` final roundtrip eval.
- Final `val_loss`: `5.63564697`.
- Final `val_bpb`: `3.33213529`.
- Compressed artifact size: `6,481,817` bytes.
- Decision: keep as the preferred size-safety fallback.

### `gw_9l_kv2_mlp1248_qkv_int6mid45_lowlr_20_seed42`
- Hypothesis: the lighter mixed-precision fallback needs a second seed before it can be treated as a real H100 backup.
- Code diff summary: same branch, but `SEED=42`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_int6mid45_lowlr_20_seed42 SEED=42 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 WARMDOWN_ITERS=6 NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 INT4_LAYERS=4,5 INT4_STEP=4 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `24469ms` training, `1234ms` final roundtrip eval.
- Final `val_loss`: `5.63831596`.
- Final `val_bpb`: `3.33371336`.
- Compressed artifact size: `6,497,323` bytes.
- Decision: keep. The fallback stayed close to the full branch while buying substantial extra compression headroom.

## Promoted int7 branch

### `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20`
- Hypothesis: a gentler int7-style coarsening on the same middle layers may retain nearly all quality while still buying meaningful size headroom.
- Code diff summary: same promoted `KV2 / 1248` branch, but `INT4_LAYERS=4,5` and `INT4_STEP=2`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 INT4_LAYERS=4,5 INT4_STEP=2 \
PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `22927ms` training, `1786ms` final roundtrip eval.
- Final `val_loss`: `5.63181821`.
- Final `val_bpb`: `3.32987150`.
- Compressed artifact size: `6,753,357` bytes.
- Decision: promote.

### `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20_seed42`
- Hypothesis: the gentler size-safe branch must hold on a second seed before it can displace the full-int8 branch.
- Code diff summary: same branch, but `SEED=42`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20_seed42 SEED=42 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 WARMDOWN_ITERS=6 NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 INT4_LAYERS=4,5 INT4_STEP=2 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `21789ms` training, `1442ms` final roundtrip eval.
- Final `val_loss`: `5.63644243`.
- Final `val_bpb`: `3.33260562`.
- Compressed artifact size: `6,770,575` bytes.
- Decision: keep.

### `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20_seed7`
- Hypothesis: the int7 branch needs a third seed before it can replace the full-int8 branch as the default local recommendation.
- Code diff summary: same branch, but `SEED=7`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr_20_seed7 SEED=7 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 WARMDOWN_ITERS=6 NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 SERIALIZE_TIED_EMBED_FP16=0 \
INT8_KEEP_FLOAT_MAX_NUMEL=65535 INT4_LAYERS=4,5 INT4_STEP=2 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `20778ms` training, `1436ms` final roundtrip eval.
- Final `val_loss`: `5.61485069`.
- Final `val_bpb`: `3.31983927`.
- Compressed artifact size: `6,744,145` bytes.
- Decision: promote to the new top local candidate. Three-seed mean: `3.32743880`, slightly ahead of the full-int8 branch at `3.32767517` while remaining materially smaller.

## Width ceiling

### `gw_9l_kv2_mlp1280_qkv_lowlr_20`
- Hypothesis: the reduced-KV branch may still improve with one more MLP widening step.
- Code diff summary: same branch, but `MLP_HIDDEN_DIM=1280`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1280_qkv_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1280 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `20960ms` training, `1459ms` final roundtrip eval.
- Final `val_loss`: `5.63545254`.
- Final `val_bpb`: `3.33202033`.
- Compressed artifact size: `7,510,773` bytes.
- Decision: keep as the width ceiling and stop the sweep. It did not beat `1248`.

## KV1 matched-budget check

### `gw_9l_kv1_mlp1312_qkv_lowlr_20`
- Hypothesis: if the `KV2` branch works, `KV1` with matched parameter budget might work even better.
- Code diff summary: `NUM_KV_HEADS=1`, `MLP_HIDDEN_DIM=1312`, but still using `INT8_KEEP_FLOAT_MAX_NUMEL=65535`.
- Command used:
```bash
RUN_ID=gw_9l_kv1_mlp1312_qkv_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=1 MLP_HIDDEN_DIM=1312 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `21698ms` training, `1610ms` final roundtrip eval.
- Final `val_loss`: `5.63697305`.
- Final `val_bpb`: `3.33291935`.
- Compressed artifact size: `8,065,171` bytes.
- Decision: invalidate as a size comparison. The `64x512` K/V weights were still fp16 passthrough at this threshold.

### `gw_9l_kv1_mlp1312_qkv32_lowlr_20`
- Hypothesis: `KV1` deserves one fair run with the keep-float threshold lowered enough that its smaller K/V weights are quantized.
- Code diff summary: same `KV1` branch, but `INT8_KEEP_FLOAT_MAX_NUMEL=32767`.
- Command used:
```bash
RUN_ID=gw_9l_kv1_mlp1312_qkv32_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=1 MLP_HIDDEN_DIM=1312 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=32767 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_DenseKV2MLP1248_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `22467ms` training, `1543ms` final roundtrip eval.
- Final `val_loss`: `5.63712510`.
- Final `val_bpb`: `3.33300925`.
- Compressed artifact size: `7,302,970` bytes.
- Decision: kill. It is a valid size improvement, but still worse than the promoted `KV2 / 1248` branch.

## Side branch

### `gw_10l_kv2_allint8pack_qkv_lowlr_20`
- Hypothesis: reduced KV heads might rescue a `10L` all-int8 branch if the smaller K/V weights are actually quantized.
- Code diff summary: `NUM_LAYERS=10`, `NUM_KV_HEADS=2`, all-int8 grouped export, `INT8_KEEP_FLOAT_MAX_NUMEL=65535`.
- Command used:
```bash
RUN_ID=gw_10l_kv2_allint8pack_qkv_lowlr_20 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 \
VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=10 NUM_KV_HEADS=2 MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `25785ms` training, `1600ms` final roundtrip eval.
- Final `val_loss`: `5.64045107`.
- Final `val_bpb`: `3.33497576`.
- Compressed artifact size: `7,377,795` bytes.
- Decision: keep as a higher-risk side branch.

### `gw_10l_kv2_allint8pack_qkv_lowlr_20_seed42`
- Hypothesis: the reduced-KV `10L` branch only matters if it remains stable on a second seed.
- Code diff summary: same branch, but `SEED=42`.
- Command used:
```bash
RUN_ID=gw_10l_kv2_allint8pack_qkv_lowlr_20_seed42 SEED=42 ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 \
VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 \
WARMUP_STEPS=1 WARMDOWN_ITERS=6 NUM_LAYERS=10 NUM_KV_HEADS=2 MATRIX_LR=0.02 SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
PACK_INT8_LAYOUT=grouped \
python records/track_non_record_16mb/2026-03-19_GW_10L_MixedPack_LowerLR/train_gpt_mlx.py
```
- Wall-clock time: `22258ms` training, `1501ms` final roundtrip eval.
- Final `val_loss`: `5.68628956`.
- Final `val_bpb`: `3.36207825`.
- Compressed artifact size: `7,358,619` bytes.
- Decision: kill as a primary branch. It is too unstable compared with the promoted `9L` branch.

## Schedule follow-up

### `gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr15_20`
- Hypothesis: the promoted int7 branch may still improve with a milder `0.015 / 0.015 / 0.025` schedule, even though the same change lost on the full-int8 `KV2 / 1248` branch.
- Code diff summary: same promoted `KV2 / 1248 / int7(mid 4,5)` branch, but `MATRIX_LR=0.015`, `SCALAR_LR=0.015`, and `TIED_EMBED_LR=0.025`.
- Command used:
```bash
RUN_ID=gw_9l_kv2_mlp1248_qkv_int7mid45_lowlr15_20 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=20 TRAIN_BATCH_TOKENS=8192 VAL_BATCH_SIZE=8192 VAL_MAX_TOKENS=65536 \
VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=1 WARMDOWN_ITERS=6 \
NUM_LAYERS=9 NUM_KV_HEADS=2 MLP_HIDDEN_DIM=1248 MATRIX_LR=0.015 SCALAR_LR=0.015 \
TIED_EMBED_LR=0.025 SERIALIZE_TIED_EMBED_FP16=0 INT8_KEEP_FLOAT_MAX_NUMEL=65535 \
INT4_LAYERS=4,5 INT4_STEP=2 PACK_INT8_LAYOUT=grouped OUT_DIR=. \
python train_gpt_mlx.py
```
- Wall-clock time: `21619ms` training, `1178ms` final roundtrip eval.
- Final `val_loss`: `5.66137225`.
- Final `val_bpb`: `3.34734563`.
- Compressed artifact size: `6,204,252` bytes.
- Decision: kill. It buys size but clearly loses on quality.
