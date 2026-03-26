# Record: 0.3212 BPB — Complementary N-gram 65K + Int5 GPTQ + LoRA TTT

**Complementary training + Order-9 n-gram eval cache (65K chunks) + Full Hessian GPTQ Int5 + LoRA TTT + Polyak averaging**

**val_bpb: 0.3212** (3-seed mean, std 0.0003) | **~14.9 MB** artifact | 8xH100 SXM, 600s train + ~570s eval

## Results (3 seeds, 8xH100 SXM)

| Seed | Steps | ms/step | val_bpb | Post-quant BPB | Artifact |
|------|-------|---------|---------|----------------|----------|
| 1337 | 5,457 | 101 | **0.3211** | 1.1817 | 14,965,401 bytes |
| 42 | 5,437 | 101 | **0.3210** | 1.1794 | 14,926,117 bytes |
| 2024 | 5,498 | 101 | **0.3216** | 1.1831 | 14,874,853 bytes |
| **Mean** | **5,464** | **101** | **0.3212** | **1.1814** | **14,922,124 bytes** |
| **Std** | **31** | **0** | **0.0003** | **0.0019** | **45,330 bytes** |

## Architecture

- 11 transformer layers, dim=512, GQA 8Q/4KV, head_dim=64
- MLP 3.0x expansion (hidden=1536) with LeakyReLU(0.9) squared
- XSA on last 4 layers (layers 7-10)
- Value Residual Learning on layers 1-10
- Gated Attention with bias=4.0 on all layers
- BigramHash 4096-bucket embedding
- Logit softcap 30.0
- EMA decay 0.997
- ~27.3M parameters

## Key Techniques

### Training
- **Complementary training** (COMPLEMENT_ALPHA=0.50): Downweights bigram-predictable tokens in the loss, making the model deliberately weaker where n-grams are strong. The n-gram cache handles those tokens at eval.
- **Parallel Muon** optimizer with Newton-Schulz5, per-group banking, encoder/decoder LR split (0.025/0.05)
- **WSD learning rate schedule** (75% stable, cosine decay)
- **Late QAT**: Soft-Round quantization-aware training triggered at 85% wallclock

### Quantization
- **Full Hessian GPTQ Int5**: Activation-order column permutation, Cholesky error compensation, 256-batch calibration
- **LZMA compression** (preset 9 extreme): ~14.8MB artifact

### Evaluation (single pass, ~570s)
- **Order-9 n-gram backoff cache**: 4M hash buckets, orders 2-9, entropy-adaptive alpha blending
- **65K-token chunks** (65,536): Cache updates 15x more frequently than standard 1M chunks. Reduces cold-cache penalty on early tokens.
- **Per-order entropy centers + multipliers**: Orders 5-9 boosted 2x, orders 2-3 suppressed 0.3x. Per-order sigmoid centers shift trust toward higher orders.
- **LoRA TTT** (rank 8, Q+V on blocks 9-10): AdamW lr=0.003, Polyak averaging decay=0.998. Adapts model weights causally per chunk.
- **Score-first protocol**: Each chunk scored before cache update (backward-looking compliant).

### What's Novel
- First combination of complementary training + order-9 n-gram cache + 65K chunks + LoRA TTT with Polyak averaging
- Per-order entropy centers combined with per-order multipliers for alpha computation
- Full Hessian GPTQ with Soft-Round QAT (not naive quantization)

## Setup and Run

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git pgolf
cd pgolf
pip install --break-system-packages -r requirements.txt zstandard
python data/cached_challenge_fineweb.py --variant sp1024

# Run (single seed)
SEED=1337 MAX_WALLCLOCK_SECONDS=600 PROG_SEQ_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Compliance

- [x] 3 seeds run on 8xH100 SXM
- [x] All seeds train in <=600s
- [x] All seeds eval in <=600s (~570s)
- [x] Artifact <=16,000,000 bytes (~14.9MB)
- [x] No validation data accessed during training
- [x] TTT is backward-looking (score-first per chunk)
- [x] No network calls during evaluation
- [x] No multi-pass rescoring
- [x] Reproducible from single script with seed

## Credits

Built on techniques from:
- **PR #809** (@quietsmile): Per-order multipliers, entropy-adaptive alpha, order-9 n-gram backoff cache
- **PR #803** (@travispchen): Complementary training (bigram-weighted loss)
- **PR #798** (@travispchen): Per-order entropy centers, drift-free TTT, Polyak averaging
- **PR #840** (@quietsmile): 65K-token chunk size for n-gram eval
- **PR #779** (@lukacf): Integrated TTT + n-gram eval loop concept
- **PR #414** (@signalrush): GPTQ + EMA + warmdown baseline
