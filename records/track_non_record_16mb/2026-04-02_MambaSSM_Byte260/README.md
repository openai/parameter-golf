# Modified Mamba SSM + Byte-Level Tokenization

## Summary

First SSM/Mamba submission to Parameter Golf. Uses a modified Mamba-2 (SSD) architecture with the
"Achilles' Heel" fix from Chen et al. (NeurIPS 2025 Spotlight, arXiv 2509.17514v2), combined
with byte-level tokenization (vocab=260) to minimize embedding overhead.

**Non-record submission**: artifact exceeds 16,000,000 byte cap after int8+zlib compression
(~16.6-16.9MB depending on seed). Would need 11 layers to fit.

## Results

| Seed | val_bpb | val_loss | Steps | Artifact Bytes |
|------|---------|----------|-------|---------------|
| 42   | 1.4807  | 1.0260   | 1987  | 16,653,170    |
| 1337 | 1.4732  | 1.0208   | 1989  | 16,814,037    |
| 314  | 1.4910  | 1.0331   | 1985  | 16,880,791    |
| **Mean** | **1.4816** | **1.0266** | | |

- Hardware: 8x H100 SXM 80GB (IN datacenter)
- Training time: 600s wallclock per seed
- Step avg: ~302ms
- Cost: ~$21.52/hr x ~1hr = ~$22

## Architecture

- **12 Mamba-2 SSD blocks** (d_model=512, d_inner=1024, d_state=64, d_conv=4, headdim=64)
- **Achilles' Heel fix**: Learnable residual bypass around Conv1d, allowing the SSM to access
  raw (pre-convolution) token representations. This addresses the asymmetry bias introduced by
  Mamba's nonlinear convolution.
- **U-Net skip connections**: Encoder-decoder pattern (6+6 layers) with learned skip weights
- **Byte-level tokenization**: 260 vocab (4 special + 256 UTF-8 bytes), tiny embedding table
- **Tied embeddings**, logit softcap (30.0)
- **Pure-PyTorch selective scan**: No custom CUDA kernels; relies on torch.compile
- **Muon optimizer** for matrix params, AdamW for scalars/embeddings/SSM params

## Key Innovation

The paper "Achilles' Heel of Mamba" shows that Mamba's Conv1d + SiLU before the SSM fuses
token information asymmetrically, hurting the model's ability to recognize symmetric patterns.
Their fix — a residual connection bypassing the convolution — restores this capability.

We combine this with byte-level tokenization: at 260 vocab, the embedding table is just 133K
params (vs 524K for sp1024), freeing budget for more layers. Mamba's O(n) complexity handles
the longer byte-level sequences naturally.

## Observations

- **Compression overhead**: The int8+zlib compressed size varies by ~230KB across seeds (same
  architecture, different weight distributions compress differently). At 12 layers the artifact
  is ~650-880KB over the 16MB cap.
- **Competitive gap**: val_bpb 1.48 vs leaderboard best ~1.11. SSMs may need more architectural
  work (better quantization, GPTQ, different compression) to close this gap in the parameter-
  constrained regime.
- **Stable training**: All 3 seeds converge consistently with <0.02 bpb spread.

## References

- Chen et al., "Achilles' Heel of Mamba" (NeurIPS 2025 Spotlight): https://arxiv.org/abs/2509.17514

## Training

```bash
SEED=42 \
DATA_PATH=./data/datasets/fineweb10B_byte260 \
TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
VOCAB_SIZE=260 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
