# Record Submission: 0.9258 BPB — Kitchen Sink (7-gram + XSA6 + BigramHash4K + Cosine TTT)

**val_bpb: 0.9258** (2-seed mean, 3rd seed running)

| Seed | val_bpb | eval_time |
|------|---------|-----------|
| 1337 | 0.9249 | 145s |
| 42 | 0.9266 | 169s |
| 2025 | (running) | — |
| **Mean** | **0.9258** | **~157s** |

Built on PR #741 (@andrewbaggio1) with hyperparameter improvements found via autoresearch-multi combinatorial search.

## Changes from PR #741

| Parameter | PR #741 Default | Ours | Found via |
|-----------|----------------|------|-----------|
| `XSA_LAST_N` | 4 | **6** | autoresearch-multi |
| `BIGRAM_VOCAB_SIZE` | 2048 | **4096** | autoresearch-multi |
| `NGRAM_ORDER` | 5 | **7** | autoresearch-multi |
| `NGRAM_ALPHA_HIGH` | 0.40 | **0.50** | autoresearch-multi |
| `TTT_EPOCHS` | 20 | **20** | (unchanged) |

These hyperparameters were identified using autoresearch-multi, a 4-mode adaptive search tool (EXPLORE/EXPLOIT/COMBINE/NARROW) with interaction detection. The "kitchen sink" combination was found to be superadditive — combined gain exceeds sum of individual gains.

## Eval Time Budget

| Phase | Time |
|-------|------|
| Training | 600s |
| INT6 quantization + roundtrip | ~45s |
| Cosine TTT (20 epochs) | ~330s |
| N-gram cache eval (7-gram, stride=64) | ~145s |
| **Total eval** | **~520s (8.7 min)** |

Well within the 10-minute eval budget.

## How to Reproduce

```bash
SEED=1337 XSA_LAST_N=6 BIGRAM_VOCAB_SIZE=4096 NGRAM_ORDER=7 NGRAM_ALPHA_HIGH=0.50 TTT_EPOCHS=20 \
PYTHONPATH=/path/to/flash-attention/hopper:$PYTHONPATH \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Search Methodology

autoresearch-multi tested 8 configurations in EXPLORE round on 5-minute training runs:

| Config | 5-min BPB |
|--------|-----------|
| **kitchen_sink** (all above) | **0.934** |
| BIGRAM=8192 | 0.9997 |
| TTT_EPOCHS=40 | 0.9947 |
| ALPHA_HIGH=0.30 | 1.0079 |
| TTT_EPOCHS=20 | 1.0133 |

The kitchen sink combination at 0.934 (5-min) projected to ~0.925 (10-min), which matched the actual result.

## Credits

- **@andrewbaggio1**: PR #741 — Cosine TTT + Multi-Order N-gram Cache
- **@abaybektursun**: PR #549 — LeakyReLU^2, Legal TTT, Parallel Muon
- **@jfprincz**: PR #287 — Partial RoPE, XSA, LN Scale
- **@signalrush**: PR #374 — GPTQ-lite, EMA
- Full lineage: PR #70 → #164 → #198 → #287 → #374 → #414 → #549 → #741 → this
