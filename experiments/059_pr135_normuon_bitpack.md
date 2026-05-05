# Experiment 059: PR135 + NorMuon + int6 bit-packing

## Status: COMPLETED

## Results
| Metric | Value |
|--------|-------|
| Steps | 7,340 @ 81.7ms/step |
| Artifact | **18,637,766 bytes ❌ (WORSE — bitpack hurts zstd compression!)** |
| Standard eval | 1.1700 BPB |
| Sliding eval | **1.1489 BPB** |

## KEY FINDING
Bit-packing INCREASES artifact size (18.64MB vs 17.85MB without).
Packed 6-bit data has less redundancy for zstd to exploit.
int6-stored-as-int8 compresses better because zstd can exploit the zero'd upper bits.

**CONCLUSION: Don't bit-pack. Keep int6 stored as int8 for better zstd compression.**

The artifact size gap vs PR135 (17.85MB vs 15.16MB) is a torch.save/platform issue, not a packing issue.
Need to test on Runpod (official eval platform) to see actual submission artifact size.
