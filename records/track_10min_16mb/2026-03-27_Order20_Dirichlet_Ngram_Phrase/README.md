# Order-20 Dirichlet Posterior + Phrase Cache

**val_bpb: 0.11545** (3-seed mean, std 0.0000010) | **~15.1 MB** | 8xH100 SXM

Extends n-gram backoff from order 15 to order 20, improving over our PR #948 (0.11556 BPB).

## Results (8xH100 80GB SXM, Montréal CA, 747 TFLOPS)

| Seed | Val BPB | Eval Time |
|------|---------|-----------|
| 1337 | 0.11544435 | 459s |
| 42 | 0.11546433 | 435s |
| 2025 | 0.11544736 | 438s |
| **Mean** | **0.11545 (std 0.0000010)** | |

## What changed from PR #948

- `NGRAM_ORDER=20` (was 15)
- Added 5 more per-order concentrations: all 1.86 (matching the pattern for high-order matches)
- Everything else identical

## Ablation (1xH100, 200 steps, Kansas City MO)

| Config | BPB | Delta |
|--------|-----|-------|
| Order 15 (baseline) | 0.11906 | — |
| **Order 20** | **0.11873** | **-0.00033** |
| Two-pass rescore | 0.11906 | 0 |
| Int5 quantization | 0.11906 | 0 |
| Comp alpha=0.30 | 0.11906 | 0 |

Order 20 was the only ablation that showed improvement.

## Credits

Same as PR #948. Built on @Robby955 (PR #900), @signalrush (PR #414), @himanshudongre (PR #846), @deanbrr (PR #659), @newjordan (PR #674), @pentxayc (PR #803).

## Run Command

```bash
NGRAM_ORDER=20 \
NGRAM_PER_ORDER_CONC="50.0,50.0,6.95,2.98,2.05,2.05,2.05,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86,1.86" \
# ... all other params same as PR #948
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
