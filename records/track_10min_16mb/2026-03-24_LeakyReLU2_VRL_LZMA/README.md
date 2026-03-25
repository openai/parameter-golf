# LeakyReLU(0.5)^2 + VRL + lzma — val_bpb 1.1229

val_bpb = 1.1229 (3-seed mean, std 0.0005) | ~15.89 MB | 8xH100 SXM

## 3-Seed Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | val_bpb | Artifact |
|------|----------|-------|---------|----------|
| 1337 | 87.1ms | 6,889 | 1.1234 | 15,887,926 |
| 42 | 88.0ms | 6,818 | 1.1225 | 15,877,570 |
| 2025 | 87.5ms | 6,857 | 1.1228 | 15,890,566 |
| **Mean** | **87.5ms** | **6,855** | **1.1229 (std 0.0005)** | |

## Key Innovations

### LeakyReLU(0.5)^2
One-line activation change delivering ~-0.002 BPB vs standard relu^2:

```python
# relu^2 (standard)
x = torch.relu(self.fc(x)).square()
# leaky relu^2 (this submission)
x = F.leaky_relu(self.fc(x), negative_slope=0.5).square()
```

Preserves negative gradient flow through the MLP. Credit: PR #493 by @parinzee, PR #518 by @sofiabod.

### Value Residual Learning (VRL)
Adds layer 0's raw value output to all subsequent attention layers via learned sigmoid gates (initialized at -1.5, ~18% initial mixing). Combats attention concentration in deep layers per the ResFormer paper (arXiv:2410.17897). Adds only 10 scalar parameters for 11 layers.

### lzma Compression
Switched from zstd-22 to stdlib lzma (preset=6). Compresses 2-5% tighter on quantized weights, recovering ~300-500KB of artifact headroom. This enabled restoring MLP from 2.875x back to 3.0x and BigramHash from 1536 back to 2048 without exceeding 16MB. No external dependencies required.

## Training Architecture

PR #414 base stack with additions:

- 11L, 512d, 8H/4KV (GQA), **LeakyReLU(0.5)^2** MLP 3x
- BigramHash(2048), XSA4, Partial RoPE 16/64, LN Scale 1/sqrt(i+1)
- **VRL** (Value Residual Learning, sigmoid-gated, all layers)
- VE128 (Shared Value Embedding, layers 9-10)
- SmearGate, OrthoInit, U-Net skips (5 enc, 6 dec)
- EMA(0.997) + Tight SWA (scale < 0.2)
- Late QAT (STE, threshold 0.15)
- GPTQ-lite int6 + **lzma** compression
- FlashAttention 3 (Hopper native)
- Muon WD=0.04, warmdown=3500, batch=786K tokens

## Reproduction

```bash
cd /workspace
git clone https://github.com/anthony-maio/parameter-golf.git
cd parameter-golf && git checkout submission/reproduce-414

# FA3 Hopper kernels (required, ~60 min build)
git clone https://github.com/Dao-AILab/flash-attention.git /workspace/flash-attention
cd /workspace/flash-attention/hopper && MAX_JOBS=8 pip install --no-build-isolation .

# Download data
cd /workspace/parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Train (replace SEED as needed)
RUN_ID=seed1337 SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 VRL_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 \
records/track_10min_16mb/2026-03-24_LeakyReLU2_VRL_LZMA/train_gpt.py
```

## Credits

- LeakyReLU(0.5)^2: PR #493 by @parinzee, PR #518 by @sofiabod
- VRL: ResFormer paper (arXiv:2410.17897), PR #569 by @gowtham0992
- Base model: PR #414 by @signalrush
- XSA: PR #287 by @jfprincz
- Competition infrastructure: OpenAI, RunPod
