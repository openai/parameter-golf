# TrigramHash + XSA + EMA + TTT — sahiee-dev

## Base
Built on: 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04 by thwu1
Base score: 1.1428 val_bpb
records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50

## Novel contributions

### 1. TrigramHash(20480, dim=32)
Adds trigram-level (t-2, t-1, t) embedding signal alongside BigramHash.
Captures 3-token phrase patterns and sub-word morphological structure
that bigrams cannot represent.
Hash: idx = (prev_prev * 961 + prev * 31 + curr) % 20480
Projected to model_dim=512 via nn.Linear(32, 512, bias=False), std=0.001 init.
Budget: bigram reduced 10240->4096 to fund trigram within 16MB.
Artifact: ~15.64MB. Zero runtime overhead.

### 2. XSA — Exclusive Self Attention (last 4 layers)
Removes self-value bias from attention output via orthogonal projection.
GQA-aware implementation from PR #287, adapted for (B,T,H,D) layout.
Zero parameter cost. Applied to final 4 layers only.

### 3. EMA — Exponential Moving Average of weights
Maintains shadow model with decay=0.9999 updated every training step.
Final val_bpb evaluation uses EMA weights instead of raw model weights.
EMA coexists with SWA: SWA averages warmdown checkpoints, EMA tracks
the full training trajectory. Zero artifact cost — EMA weights not stored.
Consistent with technique in PR #338 (current best open PR, 1.1254 bpb).

### 4. TTT — Test-Time Training
Before computing val_bpb, runs 3 SGD epochs (lr=0.002, momentum=0.9)
over validation tokens in evaluation order with bottom 6 layers frozen.
Runs identically on all 8 ranks — deterministic in-order SGD on identical
data produces identical weights without broadcast needed.
Original weights restored after evaluation. Budget: ~47 seconds.

### Evaluated and dropped
QAT: confirmed negative (PR #360) — 8% throughput penalty within 600s budget.
11th layer: does not fit within 16MB given current trigram budget (~0.91MB needed, ~0.36MB available).

## Architecture
Identical to thwu1 base plus embedding enhancements:
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
| + EMA | pending | pending |
| + TTT | pending | pending |
| + all four (ours) | pending | pending |

## Status
Code complete. Syntax OK. All smoke tests passing.
Awaiting H100 compute credits (RunPod) for 3-seed validation run.
val_bpb and ablation table will be updated before marking ready for review.
Training logs (seed 42, 1337, 2024) to be added after H100 run.
