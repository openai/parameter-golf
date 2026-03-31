"""
Experiment 05: Compound Attack — Stack All Winners

HYPOTHESIS: Each individual improvement adds 0.005-0.02 BPB.
Stacking compatible improvements should compound:
- Weight sharing saves parameters → use for wider model
- QAT eliminates quantization loss → better final score
- Eval tricks extract more from same model → free BPB
- Better compression → fit even more model

THE MONSTER CONFIG:
1. Weight sharing: 3 unique blocks × 4 repeats = 12 effective layers
2. QAT INT4 with STE
3. Model dim: 1024 (4× wider than baseline possible due to sharing + INT4)
4. Eval at 4096 seq len with NTK-aware RoPE scaling
5. Test-time training for final 0.005-0.01 BPB squeeze
6. Custom ternary encoding for shared blocks (they're identical → trivial compression)

PARAMETER BUDGET:
- Baseline: 9 unique blocks × 512 dim ≈ 3.6M params → 15.8MB (INT8+zlib)
- This: 3 unique blocks × 1024 dim ≈ 6.4M unique params
  BUT: 12 effective layers of compute
  AND: INT4 compression → ~8MB for unique params
  AND: eval tricks add 0+ training cost

This is the "nobody else will think of this" entry.

DEPENDENCIES: Results from experiments 01-04 to know which components work.
"""
