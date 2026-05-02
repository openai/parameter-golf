# Record: 11L LeakyReLU² + XSA-all + Full GPTQ + SLOT(64) + AR-calib + BigramHash(3072,112)

**val_bpb: 0.6951** (SLOT, 64 steps) | **sliding_window_bpb: 1.1316** | **artifact: 15.69 MB**

## Results

| Metric | Seed 1337 |
|--------|-----------|
| Pre-quant val_bpb | 1.0508 |
| Int6+lzma roundtrip val_bpb | 1.1549 |
| Sliding window (stride=64) val_bpb | 1.1316 |
| **SLOT (64 steps) val_bpb** | **0.6951** |
| Training steps | 5,684 |
| Training time | 595s |
| Step avg | 104.69 ms |
| Artifact size | 15,690,779 bytes |

> **Note:** This is a single-seed result. SLOT eval takes 825s on 8×H100 SXM, which exceeds the 600s eval budget. I plan to follow up with a reduced SLOT_STEPS configuration (46 steps, ~593s) and 3-seed validation. Submitting now for visibility and feedback.

## Architecture

- 11 layers, 512 model dim, 8 heads, 4 KV heads (GQA)
- **LeakyReLU(0.5)² MLP** with 3x expansion (1536 hidden)
- **XSA (Exclusive Self Attention) on all 11 layers**
- QK-Gain initialization: 4.0
- Partial RoPE: 16/64 dims
- U-Net skip connections (5 encoder + 6 decoder)
- SmearGate + **BigramHash (3072 buckets, 112d)**
- VE128 shared value embedding on layers 9-10
- LN scale: 1/sqrt(layer+1)
- Logit softcap: 30.0
- **Focal loss: γ=1.0**

## Training

- Parallel Muon: lr=0.025, momentum 0.92→0.99 over 1500 steps, WD=0.04
- AdamW: embed lr=0.035, scalar lr=0.025, WD=0.04
- EMA decay=0.997 + SWA every 50 steps
- Late QAT at LR scale < 0.15
- **Sqrt warmdown** over 3500 iterations
- Batch: 786,432 tokens, seq_len=2048
- 8×H100 SXM, 104.69 ms/step, 5684 steps in 595s

## Quantization

- **Full Hessian GPTQ** (Cholesky error compensation + column reordering)
- **AR self-generated calibration**: model generates 64×2048 tokens at temp=0.8 after training
  - No training data leakage; calibration matches model's inference distribution
- Multi-percentile clip search (0.999, 0.9995, 0.9999, 0.99999, 1.0)
- Int6 for MLP+attention weights, Int8 for embeddings
- **lzma preset=9** compression

## Evaluation

- Sliding window (stride=64, seq_len=2048)
- **SLOT (Score-First Test-time Learning)**:
  - Per-window learned delta + logit bias optimized via 64 AdamW steps
  - Cosine LR schedule: 0.010 → 0.001
  - **Warmstart=0.85**: optimizer state inherited 85% from adjacent window
  - Model weights completely frozen during eval

## Key Innovations

1. **AR Self-Generated GPTQ Calibration**: Instead of using training data for Hessian collection, the model generates its own calibration tokens autoregressively. This prevents data leakage and provides better distribution match for quantization.

2. **SLOT with Warmstart**: Score-First test-time learning where each eval window warm-starts from the previous window's optimized delta (α=0.85). This dramatically accelerates convergence since adjacent windows share context.

3. **Full Hessian GPTQ with Cholesky Compensation**: Complete GPTQ implementation with column reordering by Hessian diagonal and Cholesky-based error compensation, producing significantly better int6 quantization than simple percentile clipping.

## Reproducing

```bash
# On RunPod 8×H100 SXM
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
pip install sentencepiece numpy

SEED=1337 DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-03_11L_LeakyReLU2_XSA-all_GPTQ-AR_SLOT64_0.6951/train_gpt.py
```

All hyperparameters are set via environment variables with defaults in the script.
