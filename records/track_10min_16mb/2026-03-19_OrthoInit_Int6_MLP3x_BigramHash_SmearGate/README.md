# OrthoInit + Int6 MLP3x + BigramHash + SmearGate

## Score: val_bpb = 1.1539 (sliding window, stride=64)

## Approach

Six orthogonal improvements stacked on the baseline 9-layer, 512-dim GPT:

### 1. Orthogonal + muP-scaled Weight Initialization
- All large weight matrices initialized with orthogonal init (gain=1.0)
- Output projections (attn.proj, mlp.proj) scaled by `1/sqrt(2 * num_layers)` following muP
- Accelerates early convergence — the model starts closer to a well-conditioned point, giving Muon a head start

### 2. Int6 Mixed Quantization + zstd-22
- Per-row int6 quantization ([-32,31]) on MLP and attention weight matrices
- FP16 passthrough for tied embeddings and last 2 layers' Key projections (quantization-sensitive)
- zstd level 22 compression (better ratio than zlib-9 on int6 data)

### 3. 3x MLP Expansion
- MLP hidden dimension 1536 (3x model_dim), up from baseline 1024 (2x)
- Budget freed by int6 quantization pays for the extra parameters

### 4. Tuned Optimizer Hyperparameters
- `matrix_lr=0.02, scalar_lr=0.02, tied_embed_lr=0.03` (halved from baseline)
- `muon_momentum=0.99` with warmup from 0.92 over 1500 steps
- `warmdown_iters=3000`, `grad_clip_norm=0.3`
- `AdamW` with `weight_decay=0.01` for embedding/scalar params
- `beta1=0.9, beta2=0.95`

### 5. SmearGate + Bigram Hash Embedding
- SmearGate: learned gate blending each token's embedding with the previous token's (~512 params)
- Bigram Hash: 4096-bucket hash table (dim=128, projected to 512) injecting token-pair info

### 6. Training + Evaluation Setup
- `train_seq_len=2048, train_batch_tokens=786432`
- Sliding window evaluation with stride=64 at 2048-token windows

## Configuration

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

- Training: 7201 steps in 600s (83.33ms/step)
- Model params: 22,368,841
- Pre-quant: `val_bpb: 1.1696`
- Int6+zstd roundtrip: `val_bpb: 1.1748`
- **Sliding window (stride=64): `val_bpb: 1.1539`**
- Artifact: 15,162,375 bytes (under 16MB by 837,625 bytes)
