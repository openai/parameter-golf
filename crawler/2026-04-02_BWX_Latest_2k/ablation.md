# BWX_Latest_2k — Ablation Log

Status: queued/running

Run command:

```bash
SEED=444 NPROC_PER_NODE=4 bash crawler/2026-04-02_BWX_Latest_2k/run_ablation_sequence.sh
```

Summary output:
- `crawler/2026-04-02_BWX_Latest_2k/results/summary_s<seed>_<timestamp>.tsv`

Arms:
- `BWXLT-00`: control (`9F`, naive int6)
- `BWXLT-Q0`: naive int6 on frozen control
- `BWXLT-Q1`: standard GPTQ (128x2048)
- `BWXLT-Q1L`: standard GPTQ-lite (64x1024)
