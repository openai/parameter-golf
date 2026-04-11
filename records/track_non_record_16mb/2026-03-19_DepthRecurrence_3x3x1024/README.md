Non-record depth recurrence submission pending H100 validation.

## Approach

3 unique transformer blocks repeated 3 times (effective depth 9) at dim=1536.
Trades architectural diversity for width. Combined with Int6 QAT and NorMuon.

## Configuration

- 3 unique blocks × 3 repeats, dim=1536, 24 heads, 12 KV-heads (GQA 2:1)
- ReLU² MLP (2× expansion), tied embeddings, logit softcap=15
- QAT Int6 STE + NorMuon + sliding window eval (stride=64)
- 51.1M params → ~15.5MB Int6+zlib compressed (97% of 16MB budget)
- Tuned: matrix_lr=0.02, muon_momentum=0.85, warmdown=3000, qk_gain=2.0

## Methodology

Systematically evaluated 6 strategies across 40+ configs. Depth recurrence won.
Added QAT+Int6 (2× more params in same budget), NorMuon (+0.009 BPB), sliding window.

## Local results (Apple Silicon, not H100)

1000-step benchmark: +0.105 BPB over baseline architecture.
Absolute BPB not comparable to leaderboard — needs full 10min H100 run.

## Status

Pending H100 compute credits. Three versioned configs prepared.
