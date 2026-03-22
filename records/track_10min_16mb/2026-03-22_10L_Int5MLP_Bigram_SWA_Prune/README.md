# 10L Int5/Int6 Mixed QAT + Expanded Hash & SWA Tuning

This submission achieves state-of-the-art compression and evaluation loss through rigorous parameter-budget optimization and architectural pushing.

## Key Innovations
1. **Mixed Int5/Int6 Quantization**: MLP projections compress cleanly into Int5 precision. Attention matrices remain in Int6. This aggressively freed up space.
2. **10-Layer Architecture**: Space saved from Int5 was reinvested into adding a 10th Transformer layer, exploiting the depth advantage. U-Net skip connections ensure stable gradients.
3. **Expanded Memorization**: Expanded BigramHash from 4096 to 10240 buckets to capture broader local token correlations.
4. **SWA Tuning**: Started Stochastic Weight Averaging earlier at `0.4` to smooth out late-stage optimizer collisions.
5. **Magnitude Pruning**: A small 3% bottom-magnitude pruning zero-out pass right before zstandard compression removes noise and aids entropy packing, ensuring the final artifact slides exactly under 16.0MB.

## Results
- **Val BPB**: 1.1388
- **Model Size**: ~15.85 MB
