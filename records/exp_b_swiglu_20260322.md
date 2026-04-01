# exp_b SwiGLU — 2026-03-22

## Result: WORSE than baseline
- **final_int6_roundtrip: 1.1570 BPB**
- **final_int6_sliding: 1.1348 BPB**
- Baseline: 1.1301 BPB

## Key metrics
```
step:7062/9000 val_bpb:1.1471 (pre-TTT)
ttt v1: lr=0.002 momentum=0.9 epochs=3
ttt_epoch:3/3 loss:1.9548
final_int6_roundtrip val_bpb:1.15697447
final_int6_sliding_window val_bpb:1.13477217
step_avg:84.97ms
Code size: 69662 bytes
Submission: 17,489,177 bytes (int6+zlib)
```

## Notes
- TTT v1 HURT: 1.1471 → 1.1570
- Sliding window recovered to 1.1348
