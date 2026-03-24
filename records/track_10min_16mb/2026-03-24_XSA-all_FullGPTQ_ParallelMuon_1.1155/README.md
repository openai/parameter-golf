# Record: 11L XSA-all + Full GPTQ + Selective Pruning + Parallel Muon

**val_bpb: 1.1155** (3-seed mean, std 0.0005) | 15.94 MB | 8xH100 SXM, 600s | No TTT

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s64) | Artifact |
|------|-------|---------|--------------------|----------|
| 1337 | 6,923 | 86.7 | **1.1154** | 15,943,135 bytes |
| 1338 | 6,917 | 86.8 | **1.1150** | 15,950,643 bytes |
| 1339 | 6,914 | 86.8 | **1.1160** | 15,939,727 bytes |

**Mean: 1.1155 | Std: 0.0005**

vs merged SOTA (#414 at 1.1228): improvement = 0.0073 nats

## Key Contributions

### 1. XSA on all 11 layers

Standard practice is XSA on the last 4 layers only. We apply XSA (Exclusive Self Attention, arXiv:2603.09078) to all 11 layers, forcing cross-position information mixing from the first layer. Ablation on the same stack: -0.0016 BPP vs XSA-last-4. Zero additional parameters.

### 2. Selective ±1 magnitude pruning

Post-GPTQ, all quantized values of ±1 are sorted by reconstruction error (scale²). The least-impactful ±1 values are zeroed first until the artifact fits the target size. This is more principled than uniform magnitude pruning — it targets only the values whose removal causes the least reconstruction damage, and only in the ±1 bucket where zeroing has minimal impact on the weight distribution.

## Architecture

11L, 512d, 8H/4KV (GQA), LeakyReLU(0.5)² MLP 3x, BigramHash(1536), XSA-all(11), Partial RoPE 16/64, LN Scale, VE128, SmearGate, U-Net skip connections, EMA(0.997), Tight SWA, Full Hessian GPTQ int6 + lzma compression, Parameter Banking + Parallel Muon (batched Newton-Schulz).

## Reproduction

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 zstandard sentencepiece
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
SEED=1337 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Verification

- [x] 3 seeds, all train ≤600s on 8xH100 SXM
- [x] All artifacts ≤16,000,000 bytes (max: 15,950,643)
- [x] No TTT on validation data
- [x] No network calls during evaluation
- [x] Sliding window eval stride=64, consistent across seeds (std=0.0005)

## Credits

- Base model + Parallel Muon: [PR #593](https://github.com/openai/parameter-golf/pull/593) by @abaybektursun
- Full GPTQ: [PR #535](https://github.com/openai/parameter-golf/pull/535) by @raahilshah, [PR #569](https://github.com/openai/parameter-golf/pull/569) by @gowtham0992
- LeakyReLU(0.5)²: [PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #518](https://github.com/openai/parameter-golf/pull/518)
- XSA: arXiv:2603.09078 by Shuangfei Zhai
