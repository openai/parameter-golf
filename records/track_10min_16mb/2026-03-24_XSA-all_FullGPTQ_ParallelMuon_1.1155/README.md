# Record: 11L XSA-all + Full GPTQ + Selective Pruning + Parallel Muon

**val_bpb: 1.1154** (3-seed mean, std 0.0005) | 15.94 MB | 8xH100 SXM, 600s

Two techniques on top of PR #593's Parallel Muon stack.

## Key additions over PR #593

| Change | Impact |
|--------|--------|
| **XSA on all 11 layers** | Standard practice is XSA on last 4. Applying to all layers forces cross-position information mixing from layer 0. -0.0016 BPB vs XSA-last-4 in ablation. Zero new parameters. |
| **Selective ±1 magnitude pruning** | Post-GPTQ, sort ±1 quantized values by reconstruction error (scale²), zero least-impactful first until artifact fits. Targets only values whose removal causes minimal reconstruction damage. |

Everything else from PR #593 carries forward: 11L, 512d, 8H/4KV, LeakyReLU(0.5)² MLP 3x, BigramHash(2048), Partial RoPE 16/64, LN Scale, VE128, SmearGate, U-Net skips, EMA(0.997), Tight SWA, Full Hessian GPTQ int6 + lzma, Parameter Banking + Parallel Muon.

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | Sliding BPB (s64) | Artifact |
|------|-------|---------|--------------------|----------|
| 7 | 6,938 | 86.7 | **1.1153** | 15,937,739 bytes |
| 314 | ~6,930 | 86.7 | **1.1150** | 15,933,191 bytes |
| 2024 | ~6,930 | 86.7 | **1.1159** | 15,928,475 bytes |

**Mean: 1.1154 | Std: 0.0005**

## Requirements

**Flash Attention 3 (Hopper kernel) is required.** The script imports `flash_attn_interface` directly and will fail without it. FA2 is not sufficient — it produces ~100ms/step vs ~87ms, losing ~1,000 training steps and ~0.004 BPB.

```bash
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
python3 -c "from flash_attn_interface import flash_attn_func; print('FA3 OK')"
```

Also requires: `zstandard`, `sentencepiece`

## Run command

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
SEED=7 TARGET_MB=15.9 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Negative results

Techniques tested on this stack that did not help:

| Technique | BPB | Delta | Why |
|-----------|-----|-------|-----|
| Value Residual Learning (linear) | 1.1298 | +0.0012 | Conflicts with VE128 — both inject identity info into deep layers |
| VRL sigmoid gates + TrigramHash | 1.1174 | +0.0020 | Combined overhead costs ~100 steps, net negative |
| Catalytic Residuals | 1.1285 | -0.0001 | Redundant with existing attn_scale/mlp_scale/resid_mix |
| Backout Connection | 1.1291 | +0.0005 | Redundant with U-Net skip connections |
| Gated Attention + XSA-all | 1.1279 | +0.0011 vs XSA-all | 3% step overhead outweighs quality gain |
| Hadamard rotation + GPTQ | 1.1266 | -0.0002 | +0.5MB artifact size, hurts zstd compressibility |
| TrigramHash (zero params) | 1.1237 | +0.0049 | Changes weight distribution, hurts compression |
| BigramHash(8192) | 1.1200 | -0.0068 | Artifact 0.52MB over 16MB budget |
| BigramHash(4096) | 1.1285 | +0.0097 | Artifact 0.52MB over budget, cold cache |
| Stride=32 eval | — | +0.0001 | Negligible at seq2048. Stride=64 already gives 1984 context |
| Temperature scaling (T≠1.0) | — | +0.0002 to +0.003 | Model already well-calibrated; T=1.0 optimal |
| Extended context eval (seq4096) | 1.5695 | catastrophic | RoPE breaks completely beyond training length |
| Checkpoint logit ensemble | — | infeasible | EMA-raw delta is 16.2MB compressed (int8+zstd) |
| Entropy coding (ANS/Huffman) | — | +0.05MB max | lzma already at 99.7% of Shannon entropy limit |
| Magnitude pruning (all ±1) | 1.1341 | +0.015 | Too aggressive — no smooth continuum between threshold=0 and threshold=1 |

## Credits

- Base model + Parallel Muon: [PR #593](https://github.com/openai/parameter-golf/pull/593) by @abaybektursun
- Full GPTQ: [PR #535](https://github.com/openai/parameter-golf/pull/535) by @raahilshah, [PR #569](https://github.com/openai/parameter-golf/pull/569) by @gowtham0992
- LeakyReLU(0.5)²: [PR #493](https://github.com/openai/parameter-golf/pull/493), [PR #518](https://github.com/openai/parameter-golf/pull/518)
- XSA: arXiv:2603.09078
