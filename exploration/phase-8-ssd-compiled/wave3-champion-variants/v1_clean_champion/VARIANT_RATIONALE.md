# V1 Clean Champion

## Evidence Basis
- **M01 (Vertical State Carry REMOVAL)**: iter-004 achieved 1.3196 BPB by discarding SSD state between depth iterations (`x, _ = block(...)`). Submission v4/v5 added vertical state carry and crashed/diverged at 2.15-2.42 BPB despite identical LRs. Confirmed by source diff: iter-004 line 966 vs submission line 1255.
- **M06 (Weight Decay)**: Leaderboard top-4 all use muon_weight_decay=0.04. Decoupled WD keeps weights smaller, directly improving int4 quantization quality. INFERRED from leaderboard consensus (not ablated in our runs).
- **M07 (Grad Clip)**: v7 uses 0.3 (vs iter-004's 1.0). Tighter clipping prevents gradient spikes during depth recurrence. INFERRED from leaderboard practice.
- **M08 (Sliding Window Eval)**: stride=64 gives ~0.034 BPB free improvement. CONFIRMED from leaderboard submission #9 (1.2244 -> 1.1925 from eval trick alone).
- **M09 (Warmdown)**: Leaderboard SOTA uses warmdown=3000. iter-004 used 1200. 2400 is a conservative push. INFERRED from leaderboard consensus.

## Expected val_bpb Range
- Conservative: 1.28-1.32 (iter-004 baseline + sliding window eval gain)
- Optimistic: 1.25-1.28 (if WD + grad_clip synergize with SSD training dynamics)

## Expected Throughput Impact
- Negligible. All changes are optimizer/eval only, no architecture change.
- Expected: ~1.57M tok/s on 8xH100 (same as iter-004).

## Falsification Threshold
- If val_bpb > 1.35: REJECT. Something in the modifications hurt convergence.
- If val_bpb in [1.28, 1.35]: PARTIAL CONFIRM. Improvements marginal vs iter-004.
- If val_bpb < 1.28: STRONG CONFIRM. WD + grad_clip improve SSD training.
