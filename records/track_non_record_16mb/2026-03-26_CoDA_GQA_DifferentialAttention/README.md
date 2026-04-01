# CoDA-GQA: Constrained Orthogonal Differential Attention for Parameter Golf

## Non-Record Submission

First application of differential attention to the Parameter Golf challenge. Answers OpenAI's interest in novel attention mechanisms.

## Architecture: CoDA-GQA

From the CoDA-GQA-L paper (Maio, 2026): differential attention that sharpens signal by subtracting a gated inhibitory noise stream, where the noise query is produced via learnable orthogonal rotation of the signal query — eliminating the need for a second W_q matrix.

### How It Works

Standard GQA computes: `out = Attn(q, k, v)`

CoDA-GQA computes:
```
q_noise = PairwiseRotate(q, theta)     # orthogonal rotation, ~0 extra params
out_sig = Attn(q, k, v)               # signal attention (same KV)
out_noise = Attn(q_noise, k, v)       # noise attention (same KV)
lambda = sigmoid(Linear(x))           # input-dependent gate (init: sigmoid(-6) ≈ 0.0025)
out = RMSNorm(out_sig - lambda * out_noise)
```

### Parameter Cost
- `theta`: (num_heads, head_dim/2) = (8, 32) = 256 floats per layer × 11 layers = 2,816 params
- `lambda_proj`: Linear(512, 8) = 4,104 params per layer × 11 = 45,144 params
- Total CoDA overhead: ~48K params (0.2% of 27M model)

### Key Properties
- **No second W_q**: noise query derived by rotation, not a separate projection
- **Same KV cache**: both signal and noise attention use identical K, V
- **Smooth on-ramp**: theta=pi/2 and lambda bias=-6 means model starts as standard attention, CoDA activates gradually during training
- **Compatible with FA3**: uses standard SDPA (twice), works with FlashAttention

### Training Stack
11L, 512d, 8H/4KV GQA, **CoDA differential attention**, LeakyReLU(0.5)² MLP 3×, VRL, VE128, BigramHash(2048), XSA4, Partial RoPE 16/64, LN Scale, SmearGate, U-Net skips, EMA(0.997) + Tight SWA, Late QAT, GPTQ-lite int6 + lzma, FA3 Hopper, Muon WD=0.04

### Reproduction
```bash
CODA_ENABLED=1 RUN_ID=coda_test SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 VRL_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Credits
- CoDA-GQA-L: Maio, 2026 (arXiv preprint, Zenodo DOI: 10.5281/zenodo.18804610)
- Differential attention concept: Ye et al., 2024 (arXiv:2410.05258)
- Base model: PR #414 by @signalrush
- VRL: ResFormer (arXiv:2410.17897)
