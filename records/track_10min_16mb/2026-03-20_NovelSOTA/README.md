# Non-record: 11L int5/int6 + XSA + online TTT w/ decay prior (single-run val_bpb=1.1520)

Built on the stack from PRs #198, #180, #162, #164, #265, #254.

## What's new here

- **Pre-Q/K RMSNorm**: extra `rms_norm` on attention input before Q and K projections only (V gets raw input). Motivated by Steinmetz et al. 2025; stabilizes the RoPE-facing path under int5/int6.
- **Online causal TTT with decay prior**: full-weight SGD adaptation during eval, but with a Krause-style decay (`p += λ(p₀ − p)` after each step) to prevent drift. Adapts MLP weights in the last 3 blocks only, following TTT-E2E's finding that attention is unstable to adapt.
- **Reptile meta-learning (last 10%)**: K=1 inner SGD step + Reptile interpolation in the final 10% of training. Teaches the model to be adaptable for eval-time TTT.

## Stack (from prior work)

11L 512d 8h/4kv, MLP 3×, relu², tied fp16 embed, vocab 1024, seq 2048, U-Net skips, SmearGate, BigramHash(10240), OrthoInit + muP, Muon WD=0.04, SWA/200, int5-MLP/int6-attn + zstd-22, XSA in last 3 layers (#265), sliding window stride=64.

## Results

| Seed | val_bpb (TTT+sliding) | val_bpb (roundtrip, non-sliding) | Artifact |
|------|-----------------------|----------------------------------|----------|
| 1337 | 1.1520 | see train.log | 15.1 MB |

Single seed, not a record submission. Posting as a non-record to share the TTT+decay approach.

## Reproduce

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## References

- Krause et al. 2017 (dynamic evaluation / decay prior): arXiv:1709.07432
- Steinmetz et al. 2025 (extra RMSNorm): arXiv:2505.08823
- Sun et al. 2025 (TTT-E2E): arXiv:2512.23675
- Zhai 2026 (XSA): arXiv:2603.09078
- Nichol & Schulman 2018 (Reptile): arXiv:1803.02999
