# H100 Candidate Batch Summary

- Started: `20260501T002059Z`
- Output directory: `/workspace/parameter-golf/records/h100_4x_best_candidate_batch/20260501T002059Z`
- Completed candidates: `4/4`
- Metric: `final_int8_zlib_roundtrip_exact val_bpb` (lower is better)

Best run: `best4x_ttt_disabled_qk525` with `val_bpb=1.26066159`.

## Ranking

| Rank | Candidate | Family | val_bpb | Steps | train tokens | tok/s | step_avg_ms | int8 bytes | Artifact budget |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `best4x_ttt_disabled_qk525` | autoregressive_eval_policy | 1.26066159 | 2000 | 2097152 | 7002878 | 299.47 | 15077948 | within-budget |
| 2 | `best4x_qk525_legal_ttt` | autoregressive | 1.26359539 | 2000 | 2097152 | 7011541 | 299.10 | 15025696 | within-budget |
| 3 | `best4x_parallel_residual_ttt_qk525` | autoregressive | 1.27689523 | 2000 | 2097152 | 7094800 | 295.59 | 15063129 | within-budget |
| 4 | `best4x_dense_optimizer_base` | dense_control_optimizer_tuning | 1.27771744 | 2000 | 2097152 | 7121059 | 294.50 | 15050781 | within-budget |

## Best By Candidate Type

| Type | Completed | Best candidate | Best val_bpb | Median val_bpb |
|---|---:|---|---:|---:|
| autoregressive_eval_policy | 1/1 | `best4x_ttt_disabled_qk525` | 1.26066159 | 1.26066159 |
| autoregressive | 2/2 | `best4x_qk525_legal_ttt` | 1.26359539 | 1.27024531 |
| dense_control_optimizer_tuning | 1/1 | `best4x_dense_optimizer_base` | 1.27771744 | 1.27771744 |

## Best By Quantization

| Quantization | Completed | Best candidate | Best val_bpb | Median val_bpb |
|---|---:|---|---:|---:|
| int8-zlib-artifact | 4/4 | `best4x_ttt_disabled_qk525` | 1.26066159 | 1.27024531 |

## Charts

- `charts/final_val_bpb.svg`
- `charts/speed_vs_bpb.svg`
- `analysis/analysis.md`
- `hypothesis_graph.md`

## Coverage Notes

- Built-in candidates are implementation-backed by modules under `candidates/`; JEPA, text diffusion, and SSM are intentionally absent until native implementations are added.
- To run native implementations, pass `--candidate-file candidates.json`; each candidate can provide `command`, `env`, `family`, and `description`.
