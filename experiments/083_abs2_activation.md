# Experiment 083: abs² activation — WORSE than relu²

## Results
- Sliding: 1.1480 (vs 081's 1.1441 with relu²)
- Standard: 1.1694 (vs 081's 1.1653)
- FLAT+zstd: 15.78MB ✅

## Conclusion
abs² is 0.004 BPP WORSE than relu². The X claim was wrong for our setup.
relu²'s zero-gating (suppressing negative values) is beneficial — it acts as a sparsity-inducing mechanism that helps both training quality and weight regularity.
**RULED OUT: abs² activation. Keep relu².**
