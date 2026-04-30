# Variant C: ablation-3 champion + seq_len=2048

## Hypothesis

Doubling sequence length from 1024 to 2048 should improve val_bpb by giving the model longer context to condition predictions on. The #2 leaderboard entry (1.206 BPB) trained at seq_len=2048, confirming that longer context helps language modeling quality. Our SSD architecture handles any sequence length at O(L) cost -- no attention O(L^2) penalty -- so the per-step overhead is manageable.

## Base

ablation-3 (bifurcated A_log init): val_bpb=1.873, 5 min, 1x H100, seq_len=1024.

This was the champion of the 8-ablation sweep. The bifurcated A_log initialization creates 25% induction heads (slow decay, A_log ~ -4.0) and 75% local heads (fast decay, A_log in 0.3-0.6), producing strong specialization pressure.

## Changes

| Parameter | ablation-3 | This variant | Reason |
|-----------|-----------|-------------|--------|
| TRAIN_SEQ_LEN | 1024 | 2048 | Longer context for better predictions. SSD handles this at O(L). |
| MATRIX_LR | 0.03 | 0.02 | Longer sequences need safer LR. Prior sweep showed N=4 seq=2048 LR=0.03 diverged but LR=0.02 was stable. |
| WARMDOWN_ITERS | 1200 | 800 | Fewer total steps at seq=2048 (same batch tokens, 2x tokens per sequence = ~half the steps). 800 is calibrated for the reduced step count. |

## Trade-off analysis

With TRAIN_BATCH_TOKENS=131072 and seq_len=2048, each micro-batch has 64 sequences (vs 128 at seq=1024). Total tokens seen per step is unchanged (131072), but each step processes longer contexts. The wall-clock cost per step increases slightly due to the longer SSD scan (32 chunks vs 16), but the SSD's O(L) scaling means this is a ~2x chunk count increase, not a quadratic blowup.

The key trade-off: fewer total training steps (each step takes longer) vs better per-step learning from longer context. The leaderboard evidence suggests the context advantage wins.

## Decision tree

- **val_bpb < 1.85**: seq=2048 + bifurcated A_log is the winning combo. Scale to 8x H100 for full 10-min run.
- **val_bpb 1.85-1.90**: seq=2048 roughly matches seq=1024. Context length doesn't help at this training budget. Keep seq=1024 for more steps.
- **val_bpb > 1.90**: seq=2048 hurts (fewer steps dominate). Revert to seq=1024.
