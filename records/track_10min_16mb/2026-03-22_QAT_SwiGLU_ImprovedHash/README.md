# 10L QAT + SwiGLU + BigramHash(10240) + SWA(frac=0.4)

**val_bpb: TBD** (pending 3-seed validation on 8xH100)

## Run Command

```bash
bash prepare.sh
bash eval/eval.sh
# With specific seed
SEED=42 bash eval/eval.sh
```

## Key Improvements over SOTA (1.14276 BPB)

### QAT (Quantization-Aware Training)
- STE-based QAT during training matches quantization at inference
- Int5 clip [-16,15] for MLP weights, Int6 clip [-32,31] for attention
- Eliminates the quantization gap (estimated 0.006-0.01 BPB)

### SwiGLU Activation
- Replaces relu^2 with SiLU-gated linear unit (gate * up with silu)
- Hidden dim = 2/3 * mlp_mult * model_dim (parameter-matched)
- Standard in modern LLMs (Llama, Mistral, Gemma)

### Architecture (unchanged from SOTA base)
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

### Training Hyperparameters
- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- Sliding window eval: stride=64

Built on SOTA submission by @thwu1.
