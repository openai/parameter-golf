# Experiment 8

**Date:** 2026-03-19T18:29:13.163443+00:00
**Lane/Stage:** core/discovery
**Result:** KEPT
**val_bpb:** 1.4981
**Artifact size:** 13,615,691 bytes
**Model params:** 20734552
**Last step:** 377
**Pre-quant val_bpb:** 1.4943
**Quantization gap:** 0.0038
**Eval time:** 13480 ms
**Peak memory:** 12296 MiB
**Gate reason:** improved_val_bpb (1.52467392 -> 1.4981)
**Propose time:** 52.5s
**Train time:** 312.7s

## Change
Increase num_layers from 10 to 11 to add one more transformer block, providing more representational depth within the ~3.5MB of remaining artifact headroom. Prior research showed deeper models (11x448, 12x512) consistently improved BPB, and combining extra depth with the already-optimized matrix_lr=0.08 and warmdown_iters=800 should yield further gains. Expected artifact size ~14.5MB, well under the 16MB cap.

## Diff from previous best
Identical to current best
