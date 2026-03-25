# Record: 30-Epoch Cosine TTT on LeakyReLU² Stack (3-seed mean val_bpb=1.0781)

**3-seed mean val_bpb: 1.0781** (std=0.0041) | **15.62 MB** artifact | 8xH100 SXM, 600s training + 494s TTT + 96s eval

## Results (8xH100 SXM)

| Seed | Steps | Sliding BPB (s64) | Artifact |
|------|-------|-------------------|----------|
| 1337 | 6,921 | 1.0743 | 15.62 MB |
| 42 | — | 1.0774 | 15.62 MB |
| 7 | — | 1.0825 | 15.62 MB |
| **Mean ± Std** | | **1.0781 ± 0.0041** | |

## Approach

Single change from PR #518: **TTT_EPOCHS=30** (default 30 in PR #518's code). All architecture and training hyperparameters identical to PR #518.

PR #518's architecture: 11L, d=512, 8/4 GQA heads, LeakyReLU(0.5)² MLP 3x, BigramHash(2048), SmearGate, XSA4, Partial RoPE (16 dims), LN Scale, EMA(0.997), SWA, Late QAT, OrthoInit, VE128.

TTT: 30 epochs AdamW (lr=0.0005) with cosine LR decay and per-layer LR groups (3x for mlp.proj, 0.5x for mlp.fc). DDP gradient sync across 8 GPUs.

## Timing (seed 1337)

| Phase | Time |
|-------|------|
| Training (8xH100) | 600s (wallclock capped) |
| TTT (30 epochs) | 494s |
| Sliding window eval (stride=64) | 96s |
| **Total** | **~1190s** |

Training fits in the 10-min cap. TTT + eval = 590s, within the 10-min eval budget.

## Run Command

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are defaults in train_gpt.py.

## Credits

- **PR #518** (architecture + cosine TTT): base submission
- **PR #481** (mrdavtan): Cosine TTT scheduling discovery
- **PR #442** (sjp611): AdamW TTT
- **PR #398** (felipe-parodi): EMA, TTT, XSA foundations

## Test plan

- [x] train_gpt.py compiles (`ast.parse` passes)
- [x] All 3 seeds: artifacts under 16 MB
- [x] 3-seed mean beats verified SOTA (1.1194) by 0.041 BPB
- [x] Training completes in under 10 min on 8xH100
- [x] TTT + eval completes in under 10 min
- [x] PR only adds files to one new folder
