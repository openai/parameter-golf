# Experiment 085: Partial RoPE 25% + leaky2 — WORSE

## Results
- Standard: 1.1701 (worse than 084's 1.1639)
- Sliding: *pending but standard already worse*
- FLAT+zstd: 15.73MB ✅

## Conclusion
Partial RoPE 25% HURTS by 0.006 BPP. The model needs positional info on all head dims.
**RULED OUT: partial RoPE at 25%. Maybe try 50% or 75% but low priority.**
