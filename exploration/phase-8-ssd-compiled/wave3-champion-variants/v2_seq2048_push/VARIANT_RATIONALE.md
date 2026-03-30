# V2 Seq2048 Push

## Evidence Basis
- **M10 (Sequence Length)**: Leaderboard entries at seq=2048 consistently outperform seq=1024. LongContextSeq2048 (1.206) beat the 1024-context baseline (1.224) by 0.018 BPB. Top-4 all use seq=2048.
- **SSD O(L) Advantage**: SSD mixer processes sequences in O(L) via chunk-wise scan (vs transformer O(L^2)). Doubling L from 1024 to 2048 doubles chunk count (16->32) but each chunk is still 64 tokens. Step time increase should be ~30-50%, not 2x.
- **All V1 evidence (M01, M06-M09)**: Inherited.

## Expected val_bpb Range
- Conservative: 1.25-1.30 (seq=2048 gives ~0.02 BPB from more context, offset by fewer steps)
- Optimistic: 1.22-1.25 (if SSD's O(L) advantage means negligible throughput penalty)

## Expected Throughput Impact
- Step time: ~500ms (vs 333ms at seq=1024), ~1.5x slower per step
- Steps in 10 min: ~1200 (vs ~1800)
- Tokens per step: same (524K) but spread over 2048-token sequences
- Net: fewer update steps but each prediction benefits from 2x more context

## Falsification Threshold
- If val_bpb > 1.32: REJECT. Throughput loss from longer sequences outweighs context benefit.
- If val_bpb in [1.28, 1.32]: PARTIAL. Context helps but throughput trade-off is real.
- If val_bpb < 1.28: STRONG CONFIRM. SSD O(L) advantage is real, seq=2048 is strictly better.
