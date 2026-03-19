This record captures a non-record depth recurrence submission pending H100 validation.

## Approach

Replace 9 unique transformer blocks with 3 blocks repeated 3 times (effective depth 9). This trades architectural diversity for width: same compute budget allows dim=1024 instead of 512.

## Configuration

- Layout: 3 unique blocks × 3 repeats, dim=1024, 16 heads, 8 KV-heads (GQA 2:1)
- ReLU² MLP (2× expansion), tied embeddings, logit softcap=15
- Tuned hyperparameters: matrix_lr=0.02 (halved due to gradient amplification through repeats), muon_momentum=0.85, muon_backend_steps=7, qk_gain_init=2.0
- 23.1M params → ~14.7MB post-training compressed (92% of 16MB budget)

## Methodology

Systematically evaluated 6 strategies (shape, attention, MLP expansion, depth recurrence, layer sharing, SwiGLU) across 40+ configurations. Depth recurrence won decisively in all comparisons. Then optimized the winning strategy across 5 axes (repeats, width, unique layers, GQA, MLP ratio) and 4 hyperparameters (LR, momentum, softcap, QK-gain).

## Local results (not H100)

Local benchmark (1000 steps, 8K tokens/step on Apple Silicon):
- val_bpb: 1.8698 vs baseline 1.9743 (+0.105 relative improvement)
- Note: absolute BPB not comparable to leaderboard — needs full 10min H100 run

## Status

Pending H100 compute credits for official validation run.
