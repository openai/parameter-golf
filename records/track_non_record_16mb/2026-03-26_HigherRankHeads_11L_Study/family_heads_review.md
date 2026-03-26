# Higher-Rank Output Heads Family Study

- Family: `higher_rank_heads`
- Source JSONL: `family_heads.jsonl`
- Log snapshots: `family_logs/heads`
- Runs: `7`
- Current best: `control_standard` at `1.1734 val_bpb`

## Fixed Baseline

11L/512d fixed backbone with EMA, XSA4, SmearGate, NTK-RoPE, FA3 strict, seq2048, Partial RoPE, LN Scale, VE128(9,10), BigramHash, SmearGate, EMA, XSA4, Late QAT, and no unrelated family toggles

## Results

| ID | Variation | val_bpb | Steps | Time (s) | Artifact Est (MB) |
|---:|---|---:|---:|---:|---:|
| 1 | control_standard | 1.1734 | 4415 | 600.1 | - |
| 2 | factorized_r64 | 2.4396 | 4451 | 604.6 | - |
| 3 | factorized_r128 | 1.9227 | 4425 | 600.0 | - |
| 4 | mos_k2_r64 | 2.6167 | 4428 | 600.0 | - |
| 5 | mos_k4_r64 | 2.7112 | 4149 | 600.0 | - |
| 6 | mos_k4_r128 | 2.0898 | 4160 | 600.0 | - |
| 7 | simplex_128 | 4.1069 | 4241 | 600.0 | - |

## Notes

- This report is generated from the private family sweep and is intended as evidence for a later approach-review submission.
- Keep the public claim limited to this family only. Do not mix in unrelated winners from other sweeps.
- If a public submission is warranted, rerun the chosen winner with the artifact-producing path (`SKIP_QUANT=0`) and validate with a compact confirmatory set.
