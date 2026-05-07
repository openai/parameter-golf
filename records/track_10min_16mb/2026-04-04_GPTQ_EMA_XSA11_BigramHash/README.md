# GPTQ + EMA + XSA-all + BigramHash3072

**Score: 1.1220 val_bpb** (sliding window, stride=32, seq_len=2048)

## Architecture

- 11 layers, d=512, MLP 3x with LeakyReLU² activation
- GQA: 8 query heads, 4 KV heads (head_dim=64)
- Tied input/output embeddings (vocab=1024)
- BigramHash embedding: 3072 buckets × 112 dims with learned projection to model_dim
- SmearGate: learned token-level blending with previous position
- U-Net skip connections (5 encoder + 6 decoder layers with 5 skip weights)
- Value Embedding: shared VE table (1024×128) injected at layers 9,10
- XSA (Exclusive Self-Attention) on all 11 layers
- Partial RoPE: 16 of 64 head dims use rotary embeddings
- LN Scale: per-layer normalization scaling (1/sqrt(layer_idx+1))
- Logit softcap: 30.0

## Training

- 8xH100 SXM, PyTorch 2.11, SDPA flash backend
- Muon optimizer for weight matrices (lr=0.025, momentum=0.99, NS steps=5)
- AdamW for embeddings (lr=0.035) and scalars (lr=0.025)
- Weight decay: 0.04 (Muon and Adam)
- Batch: 786,432 tokens/step, seq_len=2048, grad_accum=1
- Warmup: 20 steps (with state reset)
- Warmdown: 3200 iterations (wallclock-based linear decay)
- EMA: decay=0.997, applied as final weights
- SWA: collected every 50 steps when LR scale < 0.2 (EMA outperformed SWA)
- Late QAT: STE-based INT6 fake quantization activated at LR scale < 0.15
- Total: 5,851 steps in 600s (~102.5ms/step)

## Quantization & Compression

- Mixed INT6/INT8 quantization:
  - MLP + attention weights: INT6 ([-31,31] range) via GPTQ with Hessian-based error propagation
  - Embeddings: INT8 per-row quantization
  - Control tensors (scales, gains, gates): FP32 passthrough
  - Small tensors (<65536 params): FP16 passthrough
- GPTQ calibration: 64 batches from training data, Cholesky-based Hessian inverse, block_size=128
- Compression: zstd level 22
- Final artifact: 15,877,304 bytes (15.1 MB)

## Evaluation

- Sliding window: stride=32, seq_len=2048
- NTK-aware RoPE scaling for sequence length extrapolation
- Compiled eval with torch.compile(dynamic=False, fullgraph=True)

## Key Results

| Metric | Value |
|--------|-------|
| Pre-quant EMA val_bpb | 1.1412 |
| Post-quant roundtrip val_bpb | 1.1457 |
| **Sliding window val_bpb (stride=32)** | **1.1220** |
| Sliding window val_bpb (stride=64) | 1.1220 |
| Training steps | 5,851 |
| Step avg | 102.5ms |
| Peak memory | 23,629 MiB |

## Command

```bash
NCCL_IB_DISABLE=1 NO_FA3=1 \
RUN_ID=v3_submit \
torchrun --standalone --nproc_per_node=8 train_gpt_v3.py
```
