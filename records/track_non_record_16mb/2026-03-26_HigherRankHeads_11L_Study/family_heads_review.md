# Higher-Rank Output Heads Family Study

- Family: `higher_rank_heads`
- Source JSONL: `family_heads.jsonl`
- Runs: `7`
- Best result: `control_standard` at `1.1734 val_bpb`

## Fixed Baseline

11L/512d fixed backbone with EMA, XSA4, SmearGate, BigramHash, partial RoPE, LN Scale, VE128 on late layers, Late QAT, `seq2048`, Hopper FA3, compiled training, sliding evaluation, and the real quantization/artifact path.

## Results

| ID | Variation | `val_bpb` | Steps | Time (s) | Artifact bytes |
|---:|---|---:|---:|---:|---:|
| 1 | control_standard | 1.1734 | 4415 | 600.1 | 16826913 |
| 2 | factorized_r64 | 2.4396 | 4451 | 604.6 | 16729834 |
| 3 | factorized_r128 | 1.9227 | 4425 | 600.0 | 16918260 |
| 4 | mos_k2_r64 | 2.6167 | 4428 | 600.0 | 16565348 |
| 5 | mos_k4_r64 | 2.7112 | 4149 | 600.0 | 17172588 |
| 6 | mos_k4_r128 | 2.0898 | 4160 | 600.0 | 17943057 |
| 7 | simplex_128 | 4.1069 | 4241 | 600.0 | 10950817 |

## Main Finding

The standard tied head outperformed every tested higher-rank alternative on this frontier-aligned 11L baseline. The simplex head reduced artifact size substantially but at an unusable quality cost. The mixture-softmax variants were both worse in score and, for the larger mixtures, larger in artifact size.
