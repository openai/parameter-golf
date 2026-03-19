---
title: Experiment Matrix
read_when:
  - You are deciding what to run next on Parameter Golf.
  - You need promotion gates from local smoke to 1xH100 to 8xH100.
  - You want the repo's agreed KPI and run naming scheme.
---

# Experiment matrix

## Scoreboard

Primary KPI:

- `final_int8_zlib_roundtrip_exact val_bpb`

Guardrails:

- `quant_delta_bpb = post_quant_bpb - pre_quant_bpb`
- `total_submission_bytes <= 16000000`
- `step_avg_ms`

Use the parser:

```bash
python3 scripts/pg_lab.py parse-log records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log
python3 scripts/pg_lab.py compare-logs records/track_10min_16mb/2026-03-17_NaiveBaseline records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3
```

## Run naming

Format:

```text
pg_{stage}_{tok}_l{L}_d{D}_h{H}_kv{KV}_m{M}_tb{BT}_sl{SL}_{focus}_s{seed}
```

Examples:

- `pg_base_sp1024_l9_d512_h8_kv4_m2_tb524k_sl1024_ref_s1337`
- `pg_sweep_sp1024_l9_d512_h8_kv4_m2_tb524k_sl1024_qk125_s1337`
- `pg_cmp_sp1024_l9_d512_h8_kv4_m2_tb524k_sl1024_clip9999_s1337`
- `pg_arch_sp1024_l10_d480_h8_kv4_m2_tb524k_sl1024_depth_s1337`

Generate commands:

```bash
python3 scripts/pg_lab.py command --profile cuda-baseline-1x --stage base --focus ref
python3 scripts/pg_lab.py command --profile cuda-baseline-1x --stage sweep --focus qk125 --set QK_GAIN_INIT=1.25
python3 scripts/pg_lab.py command --profile cuda-baseline-1x --stage cmp --focus clip9999 --set INT8_CLIP_PERCENTILE=99.99
```

## Promotion gates

Local smoke:

- command runs
- dataset/tokenizer paths correct
- final metric prints
- no score judgment

1xH100:

- use fixed-iteration or unlimited-wallclock proxy first
- promote only if one of:
- post-quant `val_bpb` improves by `>= 0.003`
- or bytes drop by `>= 100000` with regression `< 0.001`
- or step time improves by `>= 8%` with regression `< 0.002`

8xH100:

- only promote configs that already win on 1xH100
- get 2 clean repeats before burning a final leaderboard run
- reserve finals for configs with `>= 0.005` expected margin or clear systems speed win

## Lanes

### 0. Baseline reproduction

Local smoke:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
python3 scripts/pg_lab.py command --profile mlx-smoke --stage smoke --focus ref
```

Smoke profiles in `scripts/pg_lab.py` cap validation with `VAL_MAX_SEQS=512` so local checks finish quickly.
Do not use `VAL_MAX_SEQS` for any score you plan to submit.

1xH100:

- exact baseline shape
- first with `MAX_WALLCLOCK_SECONDS=0` and shorter iteration proxies
- then full `ITERATIONS=20000`

8xH100:

- exact published baseline
- `MAX_WALLCLOCK_SECONDS=600`
- `VAL_LOSS_EVERY=200`
- `TRAIN_LOG_EVERY=50`

Success:

- step time near published trend
- total bytes near `15863489`
- final post-quant score within about `0.003` on real 8x reproduction

### 1. Low-risk hyper sweeps

Lane A: attention/logit stability

- `QK_GAIN_INIT`: `1.0`, `1.25`, `1.5`, `1.75`
- `LOGIT_SOFTCAP`: `20`, `30`, `40`

Lane B: optimizer split

- `TIED_EMBED_LR`: `0.03`, `0.05`, `0.07`
- `MATRIX_LR`: `0.03`, `0.04`, `0.05`
- `SCALAR_LR`: `0.02`, `0.04`, `0.06`
- `BETA2`: `0.95`, `0.975`, `0.99`
- `MUON_MOMENTUM`: `0.925`, `0.95`, `0.975`

Lane C: schedule

- `WARMUP_STEPS`: `10`, `20`, `40`
- `WARMDOWN_ITERS`: `800`, `1200`, `1600`
- `MUON_MOMENTUM_WARMUP_START`: `0.8`, `0.85`, `0.9`
- `MUON_MOMENTUM_WARMUP_STEPS`: `250`, `500`, `1000`

Lane D: throughput/context

- `TRAIN_SEQ_LEN`: `512`, `768`, `1024`
- hold `TRAIN_BATCH_TOKENS=524288`

Run all sweeps on 1xH100 first. Promote top 2 per lane only.

### 2. Compression-first

No-code-first knobs:

- `QK_GAIN_INIT`
- `LOGIT_SOFTCAP`
- `TIED_EMBED_INIT_STD`
- `GRAD_CLIP_NORM`
- `SCALAR_LR`
- `BETA2`

Compression knobs now exposed in `train_gpt.py`:

- `INT8_CLIP_PERCENTILE`
- `INT8_KEEP_FLOAT_MAX_NUMEL`
- `INT8_KEEP_FLOAT_STORE_DTYPE`
- `INT8_PER_ROW_SCALE_DTYPE`
- `CONTROL_TENSOR_NAME_PATTERNS`
- `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`

Bias:

- shrink `quant_delta_bpb` first
- accept tiny pre-quant regressions if post-quant score wins

Best first rows:

- `QK_GAIN_INIT x LOGIT_SOFTCAP`
- `TIED_EMBED_INIT_STD x GRAD_CLIP_NORM`
- `INT8_CLIP_PERCENTILE` sweep around `99.9`, `99.99`, `99.999`, `99.99984`

### 3. Byte-efficient architecture probes

Probe set:

- `NUM_KV_HEADS`: `2`, `4`, `8`
- `MLP_MULT`: `1`, `2`, `3`
- `L10 D480 H8 KV4 M2`
- `L8 D576 H8 KV4 M2`
- `L12 D384 H8 KV4 M2`
- `L9 D512 H8 KV2 M2`
- `L9 D512 H8 KV4 M1`

Bias:

- fewer KV heads first
- then MLP shrink/expand
- then depth/width trade

### 4. Tokenizer later

Order:

- stabilize `sp1024`
- then try `byte260` with explicit tokenizer support
- then `sp4096`

Tokenizer work only after one `sp1024` branch clearly beats the baseline proxy and the eval pipeline is trusted. The run-command helper auto-generates `sp*` variants; `byte260` still needs an explicit tokenizer path.

## First 8 runs

1. Baseline exact
2. `QK_GAIN_INIT=1.25`
3. `QK_GAIN_INIT=1.75`
4. `LOGIT_SOFTCAP=20`
5. `LOGIT_SOFTCAP=40`
6. `TIED_EMBED_LR=0.03`
7. `MATRIX_LR=0.05`
8. `TRAIN_SEQ_LEN=768`

Why this batch:

- low code risk
- high information
- directly probes compression-friendliness plus throughput
