# v2 TTT-only + XSA=3 run — 2026-03-22

## Result: WORSE than baseline
- **final_int6_roundtrip: 1.1982 BPB**
- **final_ttt_sliding: 1.1797 BPB**
- Baseline: 1.1301 BPB

## Why it lost
- XSA_LAST_N=3 used manual matmul attention in last 3 layers (no FA3)
- step_avg: 125.78ms (vs ~100ms without XSA)
- Only completed 4771/9000 steps before 600s wallclock cap
- Undertrained model → TTT couldn't recover

## Config
- XSA_LAST_N=3, D2Z=off, seq_curriculum=off, batch_warmup=off
- TTT v2: lr=0.003, momentum=0.3, epochs=5, cosine_decay, discriminative_lr, wd=0.01
- temp_scaling: optimal T=1.000 (no effect)
- Submission size: 15,922,731 bytes

## Key metrics
```
step:4771/9000 val_loss:1.9572 val_bpb:1.1592 (pre-TTT)
ttt_epoch:5/5 loss:2.0248
final_int6_roundtrip val_bpb:1.19824562
final_ttt_sliding val_bpb:1.17974909
```
