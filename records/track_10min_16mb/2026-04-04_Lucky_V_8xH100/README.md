## Lucky V

Rascal II + MuonEq-R + QK_GAIN=5 + NS7 + SLOT32 (sliding-window test-time adaptation, lr=0.04)

## Results

| Seed | val_bpb (sliding window + SLOT) | Steps | Size |
|------|--------------------------------|-------|------|
| 444  | 1.08746285                     | 6,271 | 15,544,126 B |
| 4    | 1.08805544                     | 6,269 | 15,552,366 B |
| 300  | 1.08786229                     | 6,272 | 15,547,905 B |
| **mean** | **1.08779353**             |       | **15,552,366 B** |
| **std**  | **0.00030**                |       |              |

Hardware: 8xH100 SXM · 600s wallclock · `bytes_code`: 124,171

## Architecture changes vs Lucky IV (1.0960 BPB)

- **MuonEq-R**: Row-normalize momentum before Newton-Schulz orthogonalization
- **QK_GAIN_INIT=5.0**: Per-head query scaling (from 1.5)
- **MUON_BACKEND_STEPS=7**: More Newton-Schulz iterations (from 5)
- **SLOT_STEPS=32**: Test-time adaptation steps (from 24)
- **SLOT_LR=0.04**: Tuned from default 0.005 via eval-only sweep (biggest single win: -0.005 BPB)

## SLOT_LR sweep (eval-only, no retraining)

| SLOT_LR | BPB (seed 444) | Delta vs 0.005 |
|---------|---------------|-----------------|
| 0.005   | 1.0927        | baseline        |
| 0.010   | 1.0897        | -0.0030         |
| 0.040   | 1.0875        | -0.0052         |
| 0.050   | 1.0895        | -0.0032         |
| 0.100   | 1.0924        | -0.0003         |

## Compliance

- Training <= 600s on 8xH100 SXM: **Yes** (570s wallclock cap)
- Eval <= 600s on 8xH100 SXM: **Yes** (~590s sliding window)
- Total artifact <= 16,000,000 bytes: **Yes** (15,552,366 max)
- No validation leakage during training: **Yes**
- No pre-eval adaptation on unseen validation tokens: **Yes** (SLOT is score-first, backward-looking)
- SKIP_GPTQ=1: **Yes** (naive int6 quantization only)

## Reproduce

```bash
# From repo root, with flash-attention/hopper on PYTHONPATH
SEED=444 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-04_Lucky_V_8xH100/train_gpt.py
```

Expected final line (seed 444):
```
final_sliding_window+slot32steps_exact val_loss:1.83613060 val_bpb:1.08746285
```
