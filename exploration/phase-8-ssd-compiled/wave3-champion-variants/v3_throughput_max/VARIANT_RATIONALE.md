# V3 Throughput Max

## Evidence Basis
- **M11 (Batch Size)**: iter-004's loss curve was still steeply decreasing at step 1800 (cutoff). The final 100 steps dropped ~0.005 BPB. More tokens = lower loss, the relationship was not saturating.
- **Final Contenders batch ablation**: batch 262K achieved 1.8305, batch 262K+WD2400 achieved 1.8046 on 1xH100. Larger batch + longer warmdown synergize.
- **GPU Utilization**: iter-004 used 45GB of 80GB available. 786K batch should use ~67GB, well within H100 capacity.
- **Warmdown 3000**: Matches leaderboard SOTA (warmdown=3000 in top submissions). With more total steps expected from larger batch utilization, longer warmdown preserves quality.
- **All V1 evidence (M01, M06-M09)**: Inherited.

## Expected val_bpb Range
- Conservative: 1.27-1.32 (more tokens seen should lower BPB given non-saturating curve)
- Optimistic: 1.24-1.27 (if batch scaling efficiency is near-linear)

## Expected Throughput Impact
- Step time: ~500ms (1.5x longer from 1.5x more tokens per step)
- Steps in 10 min: ~1200 (vs ~1800)
- Total tokens: ~943M (same as iter-004 since batch*steps is similar)
- Net: better gradient estimates per step, same total tokens

## Falsification Threshold
- If val_bpb > 1.35: REJECT. Larger batch hurts convergence (critical batch size exceeded).
- If val_bpb in [1.30, 1.35]: PARTIAL. Batch size is near-neutral, LR may need adjustment.
- If val_bpb < 1.30: STRONG CONFIRM. Better gradient estimates improve convergence.
