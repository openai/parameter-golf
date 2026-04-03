# Hypothesis: RASCAL_III_SLOT_F
Date: 2026-04-01
Track: neural
Parent: neural/2026-03-31_Rascal_III_SLOT/

## What changes (ONE variable only)
True 6-bit packing of int6 quantized weights (PACK_INT6_6BIT=1).

Parent stores int6 values [-31,31] in int8 containers (8 bits per value).
This version packs 4 values into 3 bytes (24 bits for 4x6-bit values).
Lossless — identical weights after dequantization.

## Why
On cu128/torch2.9.1, weight distributions compress differently than cu124.
Need size headroom. Current int8 storage wastes 25% of raw bytes feeding
redundant bits to zstd. True 6-bit packing shrinks the pre-compression
blob by 25%, expected ~1-2MB savings on final compressed artifact.

## Gate target
Identical BPB to parent (packing is post-training, lossless).
Artifact size should be measurably smaller than parent's ~15.44MB.
