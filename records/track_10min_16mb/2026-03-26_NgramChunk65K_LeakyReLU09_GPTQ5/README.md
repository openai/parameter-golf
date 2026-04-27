# Record: Fine-Grained N-gram Cache (val_bpb=0.2873)

## Summary

- **val_bpb: 0.2873** (3-seed mean, std 0.0001)
- Artifact: ~13.4MB (code 181KB + model 13.2MB)
- Training: 600s on 8xH100 SXM (~7,050 steps at 85ms/step)
- Eval: ~405s (GPTQ ~22s + N-gram ~390s)

## Key Innovation: Fine-Grained N-gram Chunk Updates

The single most impactful change: reducing `NGRAM_EVAL_CHUNK_TOKENS` from 1,000,000 to 65,536.

The N-gram backoff cache only updates **after** each chunk is fully scored. With 1M-token chunks, the first million validation tokens see an empty cache — losing enormous predictive power. With 65K-token chunks, the cache refreshes 15x more frequently, giving each subsequent chunk a much richer set of n-gram statistics to draw from.

| Chunk Size | BPB | Delta |
|------------|-----|-------|
| 1,000,000 | 0.4572 | baseline |
| 65,536 | **0.2872** | **-0.170** |

This is purely an eval-time optimization — no training changes, no TTT, no additional compute.

## 3-Seed Results

| Seed | BPB | Artifact bytes |
|------|-----|----------------|
| 1337 | **0.28725** | ~13.4MB |
| 42 | **0.28720** | ~13.4MB |
| 2024 | **0.28744** | ~13.4MB |
| **Mean** | **0.2873 (std 0.0001)** | |

## Architecture

11L 512d GQA 8/4, MLP 3.0x, XSA-4, LeakyReLU(0.9)², BigramHash(4096), GPTQ int5 + LZMA.

EMA(0.997) + SWA. Parallel Muon optimizer. Perplexity-sorted shard ordering.

## N-gram Cache Details

- Order 2-9 backoff with 4M hash buckets
- Entropy-adaptive alpha: α varies by model confidence and n-gram order
- Per-order multipliers: low orders (2-3) suppressed at 0.3x, high orders (5-9) boosted at 2.0x
- Score-first: cache updated ONLY after scoring each 65K-token chunk
- All GPU ranks share identical cache state

## Compliance

- [x] Training: 600s on 8xH100 SXM (within 600s)
- [x] Eval: ~405s on 8xH100 SXM (within 600s)
- [x] Artifacts under 16,000,000 bytes
- [x] **No TTT** — purely N-gram cache at eval time
- [x] Cache strictly backward-looking — updated only after scoring
- [x] No oracle, no training data at eval time

## Future Ideas

1. **Even smaller chunks** (32K, 16K) — diminishing returns but may squeeze out 0.01-0.02 BPB more
2. **Legal score-first TTT** — per-chunk TTT where each chunk is scored first, then the model adapts on scored tokens
3. **Distributed cache pre-fill** (PR #796's approach) — each rank fills cache with all preceding positions
4. **Higher-order n-grams with more buckets** — order 9 is optimal at 4M buckets; more buckets may enable higher orders
5. **Complementary training** — bigram-weighted loss; didn't help with 1M chunks but may interact differently with 65K chunks

## Credits

- @deanbrr (PR #659) — original n-gram cache concept
- @newjordan (PR #674) — first legal implementation
- @lukacf (PR #702) — multi-order backoff + entropy-adaptive sigmoid
- @Asukabot0 (PR #727) — 7-gram, first sub-1.0 BPB
- @raahilshah (PR #634) — XSA-all
- @parinzee (PR #493) — LeakyReLU(0.5)²
- @signalrush (PR #414) — base GPTQ + EMA + warmdown stack
- @travispchen (PR #798) — per-order entropy thresholds

**Our novel contribution**: Fine-grained chunk updates for N-gram cache (65K vs 1M), demonstrating that cache update frequency is the dominant factor in N-gram BPB.
