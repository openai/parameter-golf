# Experiment 116

**Date:** 2026-03-19T18:19:27.407227+00:00
**Result:** REVERTED
**val_bpb:** 1.6001
**Artifact size:** 15,576,132 bytes
**Model params:** 28863584
**Last step:** 299
**Pre-quant val_bpb:** 1.5852
**Quantization gap:** 0.0149
**Propose time:** 197.4s
**Train time:** 287.9s

## Change
Reduced RoPE base frequency from 10000.0 to 500.0. For seq_len=1024 with head_dim=64, the default base=10000 is drastically oversized — the lowest frequency component covers only ~0.11 radians over the full 1024-position sequence, wasting most of the positional encoding capacity. A base of 500 gives the lowest frequency ~1.15 radians of coverage (vs 0.11), making all 32 frequency pairs meaningfully discriminative across the full context. This should improve attention pattern quality by providing richer positional signal, without changing parameter count, artifact size, or step time. This is a standard Phase 1 hyperparameter that hasn't been tested despite being listed as a research priority.

## Diff from previous best
+1 lines / -1 lines (vs current best)
