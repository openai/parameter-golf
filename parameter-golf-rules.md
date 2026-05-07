# Parameter Golf Competition Rules

Always follow these constraints:
- Artifact limit: 16,000,000 bytes (code bytes + compressed model bytes)
- Training time: ≤10 minutes on 8×H100 SXM
- No external downloads during evaluation
- No accessing validation data during training
- Test-time training allowed only on tokens already evaluated (not future tokens)
- Compression: use zstd level 22 (not zlib) for best ratio
- Evaluation: sliding window stride=64 is standard
- Submission requires 3 seeds with p<0.01 statistical significance vs SOTA

Known negative results to avoid:
- Loop gates initialized at 1/N (use 1.0 instead)
- QAT symmetric clipping with percentile export (must match exactly)
- TTT on top of XSA/EMA models (redundant mechanisms, hurts by 0.016 bpb)
- Layer sharing at 512-dim scale (costs 0.09 bpb)
