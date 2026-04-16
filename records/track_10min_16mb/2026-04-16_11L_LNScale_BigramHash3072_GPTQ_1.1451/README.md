# 11L + LN Scale + BigramHash 3072×112 + GPTQ + EMA

**val_bpb: 1.1451** (sliding window, stride=256)  
**Artifact size: 15.79 MB**  
**Hardware: 8×H100 SXM, 600s**

## Results

| Metric | Value |
|--------|-------|
| val_bpb (sliding window) | 1.14508063 |
| val_bpb (non-sliding) | 1.16721158 |
| val_loss | 1.93342259 |
| Artifact size | 15.79 MB |
| Steps completed | 7333 / 10 min |
| Step avg | ~82ms |
| Peak VRAM | 17.4 GB / GPU |

## Training log

```
step:1000  val_bpb:1.3569
step:2000  val_bpb:1.2941
step:3000  val_bpb:1.2713
step:4000  val_bpb:1.2554
step:5000  val_bpb:1.2354
step:6000  val_bpb:1.2046
step:7000  val_bpb:1.1686
step:7333  val_bpb:1.1588  ← wall-clock cap (600s)
GPTQ post-training → 1.1672 (non-sliding) / 1.1451 (sliding window)
```

## Key changes from baseline

- **LN Scale**: `attn_scale` and `mlp_scale` initialized to `1/sqrt(layer_idx+1)` instead of 1.0. Deeper layers contribute less initially, improving gradient flow.
- **BigramHash 3072×112**: Larger bigram hash table (3072 buckets) with smaller per-bucket dim (112) for same parameter budget. Improves short-range context modelling.
- **Full Hessian GPTQ**: Post-training quantization using Hessian-weighted column-wise int6 quantization. Calibration data generated autoregressively from the trained model (64 sequences, seq_len=2048).
- **XSA all 11 layers**: Cross-position self-attention applied to all layers (not just last 4).
- **LeakyReLU(0.5)²**: MLP activation `leaky_relu(x, 0.5)^2` instead of ReLU.
- **Muon momentum=0.99**, warmup_steps=0 (full momentum from step 1).
- **EMA** decay=0.997, applied before GPTQ.

## Architecture

- 11 layers, model_dim=512, 8 heads / 4 KV heads (GQA)
- MLP multiplier 3×, seq_len=2048
- NTK RoPE (base=10000), SmearGate, tied embeddings
- Vocab size 1024 (SentencePiece)

## Hyperparameters

```
ITERATIONS=20000  MAX_WALLCLOCK_SECONDS=600  WARMDOWN_ITERS=3000
TRAIN_BATCH_TOKENS=524288  EVAL_STRIDE=256
BIGRAM_VOCAB_SIZE=3072  BIGRAM_DIM=112
GPTQ_ENABLED=1  GPTQ_N_SEQS=64
MUON_MOMENTUM=0.99  MUON_MOMENTUM_WARMUP_STEPS=0
MUON_WD=0.04  GRAD_CLIP_NORM=0.3
EMA_ENABLED=1  EMA_DECAY=0.997
```

## Reproduction

```bash
torchrun --nproc_per_node=8 --standalone train_gpt.py
```

With env vars as listed above. Requires SentencePiece tokenizer and FineWeb10B dataset (sp1024 variant).
