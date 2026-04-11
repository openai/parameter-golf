# Experiment Ledger

Use one row per completed run. Keep the ledger append-only.

## Completed Runs

| run_id | seed | knobs | eval_mode | prequant_bpb | postquant_bpb | quant_gap | int6_artifact_bytes | train_time_ms | status | notes |
|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| `wd_v2_smoke_direct` | 1337 | `ITERATIONS=50 MATRIX_LR=0.02 SCALAR_LR=0.03 MUON_WEIGHT_DECAY=0.001 SCALAR_WEIGHT_DECAY=0.0001 EMBED_LR=0.03` | non-sliding | 2.6019 | 2.6360 | 0.0341 | 5424774 | 10302 | smoke-pass | GB10 smoke run, 50 iters, confirms trainer + quant pipeline functional. int8 roundtrip 2.6029. Log: `logs/wd_v2_smoke_direct.txt` |
| `metastack_wd_slide_local_0000` | 1337 | `ITERATIONS=3750 MATRIX_LR=0.0171 SCALAR_LR=0.0698 EMBED_LR=0.0304 MUON_WD=7.96e-5 SCALAR_WD=1.33e-5 WARMDOWN=0.270 MUON_MOM=0.983 QK_GAIN=1.55` | sliding_window_int6 | 1.4248 | 1.4269 (int6) | 0.0021 | 13859986 | 771433 | completed | GB10 full run. Sliding obj=2.4594. int8=1.4249/19.2MB. Quant gap excellent. Log: `logs/metastack_wd_slide_local_0000.txt` |
| `metastack_wd_slide_local_0001` | 1337 | `ITERATIONS=2500 MATRIX_LR=0.0338 SCALAR_LR=0.0384 EMBED_LR=0.0277 MUON_WD=0.0386 SCALAR_WD=0.00585 WARMDOWN=0.270 MUON_MOM=0.983 QK_GAIN=1.55` | roundtrip_int6 (sliding killed) | 1.4997 | 1.5069 (int6) | 0.0072 | 9802478 | 517297 | killed-SIGTERM | GB10 full run. Sliding eval killed at 71% (SIGTERM from batch wrapper). Fell back to roundtrip_int6=1.5069. Much higher WD (485x Muon, 440x scalar). Smaller artifact (9.8 MB) — high WD improves compressibility. Worse bpb than run 0000 (fewer iters + aggressive WD). Log: `logs/metastack_wd_slide_local_0001.txt` |

## Batch #1 Summary

- **Config**: `search_configs/metastack_v2_wd_sliding_local.yaml`
- **Output root**: `search_runs/metastack_v2_wd_sliding_local/`
- **Completed**: 2026-03-20
- **Eval mode**: sliding_window (EVAL_SEQ_LEN=2048, EVAL_STRIDE=256)
- **Runs**: 2/2 (run 0001 sliding eval truncated at 71% by SIGTERM)

**Search space** (11 dimensions, Sobol sampling):

| Knob | Distribution | Min | Max |
|---|---|---|---|
| ITERATIONS | int_uniform (round 250) | 500 | 5000 |
| MATRIX_LR | log_uniform | 0.01 | 0.08 |
| SCALAR_LR | log_uniform | 0.02 | 0.08 |
| TIED_EMBED_LR | log_uniform | 0.01 | 0.08 |
| MUON_MOMENTUM | uniform | 0.90 | 0.995 |
| MUON_MOMENTUM_WARMUP_START | uniform | 0.80 | 0.97 |
| MUON_MOMENTUM_WARMUP_STEPS | int_uniform (round 50) | 250 | 2000 |
| MUON_WEIGHT_DECAY | log_uniform | 1e-5 | 5e-2 |
| QK_GAIN_INIT | uniform | 1.0 | 2.0 |
| SCALAR_WEIGHT_DECAY | log_uniform | 1e-6 | 1e-2 |
| WARMDOWN_FRACTION | uniform | 0.05 | 0.40 |

**Key finding**: Run 0000 (low WD: 7.96e-5 Muon, 1.33e-5 scalar) achieved much better bpb than run 0001 (high WD: 0.039 Muon, 0.006 scalar). However, run 0001 had fewer iterations (2500 vs 3750), so the comparison is confounded. High WD did produce a smaller artifact (9.8 MB vs 13.9 MB) with acceptable quant gap. Neither run is competitive without more iterations and proper comparison to a no-WD baseline.

## Rules

- Record the objective actually used for ranking.
- Keep failed and oversize runs in the same ledger.
- Do not overwrite rows; append corrections as new rows.
- Include the exact evaluation mode.
- Keep negative-result notes brief but specific.
