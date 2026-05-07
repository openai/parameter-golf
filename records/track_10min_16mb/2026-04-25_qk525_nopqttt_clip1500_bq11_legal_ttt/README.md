# Draft Non-Record: QK5.25 + No Pre-Quant TTT + Clip1500 BQ11 + Legal Score-First TTT

Draft PR for compute grant request. Current implementation has passed local, single-H100 proxy, and one 8xH100 2M dryrun. Full 8xH100 multi-seed validation is pending additional compute credits.

Status: draft / compute request. Not claiming final leaderboard record yet. 8xH100 seed0 2M dryrun succeeded; full 2400-iteration and 3-seed validation pending compute.

PR link: https://github.com/openai/parameter-golf/pull/1816

## Current Verified Result

| Run | Seed | Validation tokens | Iterations | Legal TTT BPB | Artifact bytes | Wallclock |
| --- | ---: | ----------------: | ---------: | ------------: | -------------: | --------: |
| h100x8-nopqttt-speed-2m-r2 | 0 | 2,097,152 | 360 | 1.391517 | 15,317,495 | 288.52s |

This is a non-record, in-progress submission. The current seed0 2M dryrun is useful as a compute-request artifact because it verifies the 8xH100 execution path, artifact generation, score-first legal TTT flags, and wallclock margin.

## Method

- SP8192 tokenizer and FineWeb challenge data.
- QK gain initialized to 5.25.
- EMA disabled for the current dryrun.
- Pre-quant TTT disabled.
- GPTQ with int6 matrices and int8 embeddings.
- Matrix SDClip sigma set to 15.0.
- Brotli compression quality set to 11.
- Legal score-first TTT enabled at evaluation time.

## Key Configuration

```text
QK_GAIN_INIT=5.25
EMA_ENABLED=0

PRE_QUANT_TTT_ENABLED=0

GPTQ_CALIBRATION_BATCHES=64
MATRIX_CLIP_SIGMAS=15.0
EMBED_BITS=8
COMPRESSOR=brotli
BROTLI_QUALITY=11

FINAL_SCORE_MODE=legal_ttt_only
POST_QUANT_EVAL_ENABLED=0
SLIDING_WINDOW_ENABLED=0

TTT_ENABLED=1
TTT_LR=0.005
TTT_EPOCHS=2
TTT_CHUNK_TOKENS=32768
EVAL_STRIDE=512
```

## 8xH100 Seed0 2M Dryrun

```text
stage:  stage6_h100x8_nopqttt_speed_2m
method: qk525_nopqttt_clip1500_bq11_legal_s512e2_emaoff_speed
seed:   0
group:  h100x8-nopqttt-speed-2m-r2
```

Observed metrics:

```text
pre_quant_val_bpb:        1.474676
legal_ttt_val_bpb:        1.391517
artifact_bytes:           15,317,495
quantized_model_bytes:    15,135,511
run_wallclock_total_ms:   288,522
wallclock_margin_ms:      311,478
ttt_eval_time_ms:          47,685
```

Legal TTT flags:

```text
ttt_score_before_update:        true
ttt_single_left_to_right_pass:  true
ttt_multi_pass_rescore:         false
ttt_strict_causal_prefix_only:  true
```

## Pending Validation

- Full 2400-iteration 8xH100 run.
- 3-seed 8xH100 validation.
- Final full-validation leaderboard measurement.

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train.log`
- `summary.json`
