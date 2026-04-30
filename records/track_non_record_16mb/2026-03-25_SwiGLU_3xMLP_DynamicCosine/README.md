# Non-Record Submission: SwiGLU + Pure Dynamic Cosine + SWA Excision 

This is a non-record submission documenting the structural optimization of the baseline parameter boundaries by migrating to Llama 3's SwiGLU framework and explicitly locking the dynamic wallclock decay, achieving **1.2005 val_bpb** entirely natively.

The final post-quant score decisively breaks the baseline (1.2244) purely through mathematical density optimizations without relying on Test-Time Training, SmearGate exploits, or aggressive mixed `int5/int6` fragility.

## Summary of Changes

1. **SwiGLU Activation** replacing ReLU² — Vasty superior gradient flow gating mechanism inside the MLP layers.
2. **Dense Cap Expansion (MLP 3x)** — The baseline `MLP_MULT=2` left over 4MB of the 16MB file unutilized. We scaled the hidden state to natively fill the 16.0MB constraint safely.
3. **Double Context Length (2048)** — Boosted local sequence tracking dependencies to capitalize on 8xH100 throughput overhead.
4. **Dynamic Wallclock Cosine Schedule** — Dropped iteration-based LR prediction in favor of natively tracking the physical hardware clock, executing a perfectly smooth Cosine Warmdown curve across the final 40% of the 600-second timer.
5. **SWA Momentum Excision** — Disabled SWA parameter averaging at the buzzer to lock the vastly superior local gradient minimums without backward drag.
6. **Quantization-Aware Immunity** — Embedded Straight-Through Estimators locally inside the matrices to simulate `[-31, 31]` parameter drift natively during the fp32 training passes. 

## Key Discovery: The SWA Penalty

The standard Parameter Golf architecture universally leverages SWA (Stochastic Weight Averaging) as a generalization safety net. However, because our **Dynamic Cosine Schedule** functionally perfectly guides the training loss to an absolute standstill explicitly synchronized with the 10-minute buzzer (hitting 1.1982 BPB raw), we observed the running 4-minute SWA average actively dragging the model backwards conceptually into higher-loss brackets (`1.23+`). 

By natively removing the SWA logic track on line 1092, the script serializes the mathematically deepest checkpoint entirely penalty-free.

## Configuration (The 1.20 Run)

```bash
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_MULT=3 TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048
max_wallclock_seconds=600.0 (40% Bounded Dynamic Cosine Warmdown)
```

**Key Metrics (from terminal logic):**
- `val_bpb` (raw bfloat16 checkout): **1.1978**
- `val_bpb` (post-quant 6-bit zlib): **1.2004**
- Artifact Size: **15,399,277 bytes** 
- Iterations completed: 9,329
- Hardware: 8x H100 SXM, 600s wallclock pure array

## Included Files

- `train_gpt.py` — The strictly optimized unadulterated Native Custom framework
- `train_log.txt` — The 600-second 8xH100 console drop
- `submission.json` — Target metadata format for leaderboard scraping
