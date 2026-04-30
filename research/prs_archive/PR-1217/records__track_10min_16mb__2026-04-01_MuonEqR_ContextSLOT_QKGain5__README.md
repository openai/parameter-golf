# Record: MuonEq-R + Context-Only SLOT + QK_GAIN=5.0

**val_bpb: 1.1027** (3-seed mean, std 0.0011) | ~15.80 MB | 8xH100 SXM | ~88.8ms/step | ~6654 steps

Built on PR #1179 (@dexhunter) with three additions:

- **MuonEq-R** (row-normalization before Newton-Schulz) -- from arXiv:2603.28254
- **QK_GAIN_INIT=5.0** -- our hyperparameter sweep (monotonic gains from 1.5 to 5.0)
- **Context-Only SLOT** -- causal variant of SLOT that optimizes delta using only already-scored context tokens

## 3-Seed Results

| Seed | Context-SLOT BPB | TTT BPB | Steps | ms/step | Artifact |
|------|-----------------|---------|-------|---------|----------|
| 1337 | **1.10166** | 1.11008 | 6660 | 88.8 | 15,795,518 |
| 42 | **1.10378** | 1.11206 | 6650 | 88.9 | 15,793,163 |
| 2024 | **1.10271** | 1.11108 | 6653 | 88.9 | 15,796,779 |
| **Mean** | **1.10272 +/- 0.00106** | 1.11107 | 6654 | 88.8 | 15,795,153 |

Beats merged SOTA (PR #1019, 1.1147) by **0.012 BPB** (p << 0.01).

## Reproduction

```bash
pip install brotli
QK_GAIN_INIT=5.0 SLOT_ENABLED=1 SLOT_STEPS=8 SLOT_LR=0.005 SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Training: ~600s. Eval (sliding + context-only SLOT): ~190s. Total: ~13 min end-to-end.
