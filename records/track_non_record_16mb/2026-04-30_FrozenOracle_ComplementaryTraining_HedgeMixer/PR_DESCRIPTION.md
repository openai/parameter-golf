# Non-record: Frozen N-gram Oracle + HedgeMixer + SGD TTT

## Summary

A hybrid system that bundles a frozen multi-order n-gram oracle (built offline from FineWeb training tokens, int8 log-probabilities with zstd-22, 3.42 MB compressed on a 10M-token slice) into a single artifact alongside the neural model. The oracle plugs into the existing Hedge mixer at TTT/eval time as additional experts. The submission also includes the SGD TTT switch (PR #967, reported -0.041 BPB) and `LeakyReLU(0.75)²` (PR #977, reported -0.008 BPB) as ancillary changes.

This is a methodology submission, not a record claim. I didn't have 8×H100 access during the cohort. The pipeline runs end-to-end on Kaggle T4×2 NCCL DDP (8L/384d, 13.4M params, 172 training steps, 3,786 TTT chunks, 6.85 MB final artifact, exit 0). The README extrapolates wall-clock to H100×8 from those measurements (around 13 to 17 minutes for the full pipeline at 11L/512d).

## Key contributions

- A frozen multi-order n-gram oracle, packaged as part of the artifact. Standalone offline builder (`build_ngram_oracle.py`, 250 lines, NumPy only): exact unigram, exact bigram, FNV-1a-hashed orders 3 through 8 with bucket counts from 4096 down to 256. Built only from training tokens. Bundled inside the 16 MB cap, designed to address the compliance gap that flagged PR #924.

- HedgeMixer extension. The existing 5-expert mixer (neural + online uni/bi/tri + decay cache) is extended to `5 + |oracle orders|` experts via a single multiplicative-weights update. With no oracle loaded, behavior matches the base.

- Single-artifact format with a 16-byte versioned header (4-byte magic, 1 version byte, 3 reserved, neural and oracle blob lengths). `oracle_len = 0` degrades cleanly to base behavior. Reload uses an in-memory `FrozenNgramOracle.from_bytes` classmethod, no per-rank temp files.

- SGD TTT as a configurable alternative to AdamW (PR #967). `LeakyReLU(0.75)²` configurable per PR #977. Both env-var-gated, both small reviewable diffs.

- Bug fixes. Bucketed `dist.all_reduce` in TTT replaces about 100 per-parameter NCCL launches with one. `index_put_(..., accumulate=True)` replaces a non-deterministic `bi_counts[prev, targets] += 1.0` in HedgeMixer table updates. Inline `loss * weight.mean()` complementary scaling removed (mathematically not equivalent to per-token reweighting).

## Negative results

- Byte-level CTW (`ctw_prototype.py`, depth 8, 262K hash buckets/depth, KT estimator). 2M training bytes + 500K eval bytes from one FineWeb shard:
  - Eval BPB: 6.33 (target < 1.2)
  - Compressed: 21.31 MB (target < 5 MB)
  - Throughput: 16,761 bytes/sec (target > 100K)
  - Verdict: dominated by token-level n-grams at this vocab size. Token-level CTW is the natural follow-up.
- Inline complementary loss scaling. Multiplying scalar-mean CE by `weight.mean()` is not equivalent to per-token reweighting. Removed. Standalone function kept for future inside-graph integration.

## Limitations

- No 8×H100 validation, no 3-seed mean, no competition-scale BPB number from me.
- Oracle build verified on a 100M-token shard (32 s, 4.66 MB) and a 10M-token slice (2.5 s, 3.42 MB). Full 80-shard scan time and final compressed size are extrapolated.
- Complementary training loss is implemented but currently disabled. Inside-graph integration is required for correct per-token weighting; the inline version was wrong (now removed).
- All BPB numbers cited from other PRs (#803: 0.4416, #834: 0.1663, #924: 0.0280, #967: -0.041, #977: -0.008) are from those PRs' authors, not reproduced here.

## Test plan

- [x] Local end-to-end run on RTX 4060 (4L/256d toy, 50 steps, 7.6 MB artifact, exit 0)
- [x] Kaggle T4×2 NCCL DDP run end-to-end (8L/384d, 172 steps, 3,786 TTT chunks, 6.85 MB artifact, exit 0)
- [x] FNV-1a NumPy/Torch equivalence test passes (1000 samples, ctx_len=5, buckets=4096)
- [x] Magic-prefix artifact roundtrip verified (`Header + blobs total: 7,181,197 == 7,181,197: True`)
- [x] All `train_gpt.py` and `build_ngram_oracle.py` files syntax-checked
- [ ] Not done: 8×H100 3-seed validation at competition spec
- [ ] Not done: Full 80-shard oracle build
- [ ] Not done: α-sweep for complementary loss after inside-graph integration
- [ ] Not done: Per-order Hedge weight logging

## Why submit this

I joined late and didn't get 8×H100 access. Rather than fabricate numbers or skip submitting, I'm offering this as a methodology contribution: a clean, modular, reviewable design for hybrid frozen-oracle + neural systems. The README walks through the design, the negative results, the explicit limitations, and the concrete plan for what I'd do with H100 access.

- [README.md](./README.md): technical writeup (~1,300 words) covering the three components, the validated DDP run, the H100 extrapolation, compliance, limitations, related April 2026 references, and reproducing instructions.
- [JOURNEY.md](./JOURNEY.md): process journal documenting the 5-week research arc, the 5 GitHub sweeps, the 6 specialist agents consulted, the strategic pivots, the dead ends with measured numbers, and the day-of-deadline polish loop.
