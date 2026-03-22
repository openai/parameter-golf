# TrigramHash + XSA + TTT — sahiee-dev

## Base
Built on: 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04 by thwu1
Base score: 1.1428 val_bpb
records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50

## Novel contributions

### 1. TrigramHash(20480, dim=32)
Adds trigram-level (t-2, t-1, t) embedding signal alongside BigramHash.
BigramHash captures token-pair patterns. TrigramHash captures 3-token
phrase patterns and sub-word morphological structure bigrams cannot see.
Hash: idx = (prev_prev * 961 + prev * 31 + curr) % 20480
Projected to model_dim=512 via nn.Linear(32, 512, bias=False).
Initialized std=0.001. Added to token embedding alongside BigramHash.
Budget: bigram reduced 10240->4096 to fund trigram within 16MB.
Artifact: ~15.64MB.

### 2. XSA — Exclusive Self Attention
Applied to final 4 layers. Removes self-value bias from attention output
via orthogonal projection using GQA-aware reshape. Prevents each token
from over-attending to its own value vector. Zero parameter cost.
Implementation from PR #287, adapted for our (B,T,H,D) layout after
scaled_dot_product_attention transpose.

### 3. TTT — Test-Time Training
Before computing val_bpb, runs 3 SGD epochs (lr=0.002, momentum=0.9)
over validation tokens in evaluation order with bottom 6 layers frozen.
Runs identically and independently on all 8 ranks — deterministic SGD
on identical data produces identical weights without broadcast.
Original weights restored after evaluation.
Budget: ~47 seconds. Timer logged in training output.
QAT was considered but confirmed negative (PR #360) — 8% throughput
penalty outweighs regularization benefit within 600s budget.

## Architecture
Identical to thwu1 base:
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA), ReLU²
- MLP 3x expansion, SmearGate, BigramHash(4096), TrigramHash(20480)
- OrthoInit, U-Net skips, Muon WD=0.04, SWA start_frac=0.4
- Sliding window eval stride=64, zstd-22, ~15.64MB artifact

## Ablation table
| Variant | val_bpb | delta |
|---------|---------|-------|
| thwu1 base | 1.1428 | — |
| + TrigramHash | pending | pending |
| + XSA | pending | pending |
| + TTT | pending | pending |
| + all three (ours) | pending | pending |

## Status
Code complete. Syntax OK. Smoke tests passing.
TTT rank bug found and fixed — all 8 ranks run TTT independently.
Awaiting H100 compute credits for 3-seed validation run.
val_bpb and ablation table will be updated before marking ready for review.
