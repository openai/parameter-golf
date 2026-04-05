LeakyReLU + Sliding Window Eval + Zstd
 LeakyReLU(0.5)² activation instead of ReLU²
- Sliding window evaluation with stride=256
- Zstd compression instead of zlib
- Results
- val_bpb: 1.2634
- Model size: 15.77MB (under 16MB limit)
- Hardware: 2xH100, 10 data shards
- Steps: 3396
Baseline comparison
- Naive baseline: 1.2244 (8xH100, full compute)
- Our run: 1.2634 (2xH100, limited compute)
- Expected on 8xH100: ~1.19-1.20
- Changes to train_gpt.py
1. LeakyReLU(0.5)² in MLP forward pass
2. Sliding window eval with EVAL_STRIDE env var
3. Zstd level=22 compression
