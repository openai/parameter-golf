# 10L Int5/Int6 + BigramHash(10240) + SmearGate + SWA Boost

**LLMAdvisor.ai — powered by HighSignal™**

## Score

**Best val_bpb = 1.14638** (seed=1337, SWA boost config, 8×H100 SXM, 600s wallclock)

| Run | Seed | SWA Config | Steps | val_loss | val_bpb |
|-----|------|-----------|-------|----------|---------|
| SWA boost | 1337 | every=30, frac=0.50 (49 ckpts) | 7,372 | 1.93562 | **1.14638** |
| Standard | 1337 | every=50, frac=0.36 (21 ckpts) | 7,376 | 1.93571 | 1.14644 |
| WD=2250 | 2024 | every=50, frac=0.36 | 7,387 | 1.93680 | 1.14709 |

Artifact size: **15,736,555 bytes** (263,445 bytes under the 16MB limit)

## Run Command

```bash
# Setup (once)
pip install sentencepiece zstandard huggingface_hub
python cached_challenge_fineweb.py --variant sp1024

# Train + evaluate (best config: SWA boost)
SEED=1337 SWA_EVERY=30 SWA_START_FRAC=0.50 \
  torchrun --nproc_per_node=8 --standalone train_gpt.py
```

Default env vars reproduce the standard run. Override `SEED`, `SWA_EVERY`, and `SWA_START_FRAC` for the SWA boost config above.

## Approach

### 1. Mixed Int5/Int6 Quantization
- **Int5 [-16,15]** for MLP weights — saves ~1.86MB vs uniform int6, funding the 10th layer
- **Int6 [-32,31]** for attention weights — precision-sensitive
- **FP16** for tied embeddings and last-layer key projections
- **zstd level 22** compression

### 2. BigramHash(10240, dim=128)
- Hash consecutive token pairs into 10,240-bucket embedding table (dim=128)
- Projected to model dim=512 via learned linear — captures local token-pair context

### 3. SmearGate
- Learned per-dimension gate blending current + previous token embeddings
- Initialized near-identity for stable early training

### 4. SWA Density Sweep
- **SWA boost**: every=30 steps, start_frac=0.50 → 49 averaged checkpoints (best: 1.14638)
- **Standard**: every=50 steps, start_frac=0.36 → 21 averaged checkpoints (1.14644)
- Denser SWA collection provides marginal but consistent improvement

### 5. Reduced Batch Size (622,592 tokens)
- 75% of the standard 786K batch → ~81ms/step (vs ~117ms at full batch)
- ~7,370 training steps in 600s wallclock (vs ~5,100)
- More steps overcomes slightly noisier gradients

## Architecture

| Parameter | Value |
|-----------|-------|
| Layers | 10 |
| Model dim | 512 |
| Heads | 8 (4 KV heads, GQA) |
| MLP hidden | 1536 (3× expansion) |
| Activation | relu² |
| Vocab size | 1024 (sp1024 BPE) |
| Embeddings | Tied input/output, FP16 |
| Init | Orthogonal with muP-scaled outputs |
| Skip connections | U-Net style |

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer (matrix) | Muon, lr=0.02, momentum=0.99 |
| Optimizer (embed/scalar) | AdamW, lr=0.02 |
| Weight decay | 0.04 (decoupled) |
| Batch size | 622,592 tokens |
| Sequence length | 2,048 |
| Warmup | 20 steps |
| Warmdown | 3,000 iters |
| Gradient clipping | 0.3, 3% magnitude pruning |
| SWA (boost) | start_frac=0.50, every=30 steps |
| Wall clock cap | 600 seconds |

## Evaluation

- Sliding-window evaluation with stride=64
- BPB = (val_loss / ln(2)) × (tokens / bytes)
- Eval time: ~259 seconds

## Hardware

- 8× NVIDIA H100 SXM GPUs
- RunPod cloud instance
- DDP training with torchrun

## Acknowledgments

Built on the SOTA techniques from [thwu1](https://github.com/KellerJordan/parameter-golf/tree/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50) (1.14276 BPB) and [Raahil Shah](https://github.com/KellerJordan/parameter-golf/tree/main/records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA) (1.1458 BPB). Key adaptations: reduced batch size for faster step throughput, SWA density sweep (every=30/frac=0.50 vs every=50/frac=0.40), and PyTorch version auto-detection for GQA compatibility.
