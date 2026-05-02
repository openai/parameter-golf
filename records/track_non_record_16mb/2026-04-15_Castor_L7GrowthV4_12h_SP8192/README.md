# L7 growth v4 precursor to PR 2014, 12 hours on RTX 4090, val_bpb 0.9697 pre-quant

This is an archival **non-record** submission package for a 12-hour Castor
pretraining run based on the l7 growth v4 recipe.

I ran this for a personal project, but I think the result is interesting so I decided to share even if we are past the deadline. 

# Main differences with PR 2014
- Max context size was 8k instead of 3k
- I didn't pre compile the context size since the cost of compilation on a 12 hours run is not significant.
- I used a customized LR curve that I didn't include in PR 2014 since it doesn't quantize well
- No EMA
- Dataset used are different and are detailed in the .yaml file included

## Result

The exact logged final metric is:

```text
pre-quantization post-ema val_loss:1.83671792 val_bpb:0.96976490 eval_time:184477ms
final_int6_roundtrip_exact val_loss:1.83671792 val_bpb:0.96976490 skipped_packaging:1
```

Notes:

- `EMA_ENABLED=0` in the config, despite the historical log string saying
  `post-ema`.
- `SKIP_FINAL_PACKAGING=1`, so no final compressed 16MB package was produced.
- Because packaging was skipped, the `final_int6_roundtrip_exact` line should be
  read as a no-packaging roundtrip/check value, not as a produced compressed
  int6 submission artifact.
- The retained full-precision model is 135,431,355 bytes and is intentionally
  not committed to this folder.

## Model And Training Setup

- Parameters: `35,944,536`
- Vocabulary: `8192`, tokenizer `fineweb_8192_bpe.model`
- Layers: `11`
- Model dim: `512`
- Heads: `8`, KV heads: `4`
- MLP multiplier: `4.0`
- Looping: enabled at `0.35`, `loop_start=3`, `loop_end=5`, `num_loops=2`
- Training wallclock cap: `43200s`
- Stopped at step `38707/100000`
- Training batch tokens: `262144`
- Validation batch tokens: `131072`
- Eval context: `8192`
- Eval stride: `4096`
- TTT: enabled, `8` epochs, `32768` chunk tokens, SGD LR `0.005`

Progressive context schedule:

```text
1024@0.200,2048@0.750,4096@0.850,8192@1.000
```

Midrun LR cap schedule:

```text
1.000@0.000,1.000@0.400,0.500@0.400,0.300@0.500,0.180@0.600,0.110@0.700,0.090@0.800,0.070@1.000
```

## Dataset

The run used a pretrain mixture described in
`castor_pretrain_mix_v0.yaml`:

- FineWeb English
- FineWeb2 French
- FineWeb-Edu English
- optional CommitPack code shards

The pretokenized output path in the original run was:

```text
./data/datasets/castor_pretrain_sp8192_v0
```

The tokenizer path was:

```text
./data/tokenizers/fineweb_8192_bpe.model
```

## Reproduction Command

From a workspace that contains the raw data and tokenizer:

```bash
CASTOR_TRAIN_ENV=./configs/train/l7grow_v4_castor_12h.env \
  ./scripts/train_l7grow_v4_castor_12h.sh
```

The wrapper prepares the pretokenized shards if needed, then launches:

```bash
SIMON_ENV_FILE=./configs/train/l7grow_v4_castor_12h.env \
  ./.venv/bin/python -u trainers/l7_grow/train_gpt.py
```


## Included Files

- `train_seed1337.log`: exact historical trainer log
- `l7grow_v4_castor_12h.env`: exact run environment/config
- `castor_pretrain_mix_v0.yaml`: dataset mixture config
- `train_l7grow_v4_castor_12h.sh`: wrapper entrypoint
- `train_l7grow_v4_castor.sh`: underlying Castor launch script
- `train_gpt.py`: Wrapper
- `train_gpt_human.py`: Code
- `env_utils.py`: env-file loader used by the trainer
- `ARTIFACTS.md`: local paths and hashes for retained uncommitted weights
- `submission.json`: metadata for this non-record archive
