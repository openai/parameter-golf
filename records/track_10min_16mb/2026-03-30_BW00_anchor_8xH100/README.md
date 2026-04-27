# BW-00 Anchor — int6 SW BPB 1.18616 (seed 444)

**Bandit_Wagon anchor arm.** dim=512, 4F+1C×3, mlp=6.0. Confirms CL3 config on seed 444.

## Result

| Seed | int6 SW BPB | Steps | Size |
|------|:-----------:|------:|------|
| 444  | 1.18616296  | 8052  | 9,095,434 bytes (9.10 MB) |

Hardware: 8×H100 SXM, 600s wallclock.

## Config

- dim=512, 4 flat XSA layers + 1 crawler block × 3 loops
- CRAWLER_MLP_MULT=6.0
- CRAWLER_QUANT_INT8=1 (QAT)
- SKIP_GPTQ=1 (naive int6)
- SKIP_EMA=1
- COMPILE_FULLGRAPH=0
- GQA: 8 heads, 4 KV heads

## Key Numbers

- Pre-quant val_bpb: 1.1983
- final_int6_roundtrip_exact: 1.20983231
- final_int6_sliding_window_exact: **1.18616296**
- Quant delta (roundtrip vs SW): −0.024 (SW benefit)

## vs CL3 Baseline

| Run | Seed | int6 SW BPB |
|-----|------|:-----------:|
| CL3 mean (3-seed) | 1337/42/300 | 1.18742 |
| BW-00 | 444 | **1.18616** |

BW-00 seed=444 beats CL3 mean by 0.00126. Config verified.

## Reproduce

```bash
git checkout TEST_LAB
SEED=444 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run.sh
```
