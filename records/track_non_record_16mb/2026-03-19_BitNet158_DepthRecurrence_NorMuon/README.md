# BitNet b1.58 + Depth Recurrence + NorMuon

**Track**: Non-record (unlimited compute — single RTX 3060 12GB, ~3h training)

## Results

| Metric | Value |
|---|---|
| val_bpb (pre-quantization) | **1.4866** |
| val_bpb (ternary roundtrip) | **1.7510** |
| val_loss (pre-quant) | 2.4786 |
| val_loss (post-quant) | 2.9193 |
| Submission size | **3.78 MB** / 16 MB |
| Training time | ~3h (10,000 steps, single GPU) |
| Tokens seen | ~328M (32768 tok/step × 10k steps) |

## Architecture

**BitNet b1.58** with depth recurrence and U-Net skip connections.

| Component | Value |
|---|---|
| Unique transformer blocks | 4 |
| Recurrence factor | 3 (each block run 3× per forward pass) |
| Effective depth | 12 layers |
| Model dim | 640 |
| Heads / KV heads | 8 / 4 (GQA) |
| Head dim | 80 |
| MLP multiplier | 2 (squared ReLU activation) |
| Vocab size | 1024 (sp1024 tokenizer) |
| Tied embeddings | Yes |
| Total params (unique) | 12,138,272 |
| Ternary params (unique) | 11,468,800 |

### Key techniques

- **BitNet b1.58**: ternary weights {-1, 0, +1} via absmean quantization, int8 absmax activations per-token, STE for backprop
- **Depth recurrence + U-Net skips**: 4 unique blocks run 3× = 12 effective layers. Encoder (passes 0-5) pushes to skip stack, decoder (passes 6-11) pops with learnable `skip_weights[6×640]`
- **Ternary packing**: 2 bits/weight (4 values/byte) + zlib → 3.74 MB model vs ~47 MB raw fp32
- **NorMuon optimizer**: Muon with per-neuron row-wise RMS normalization after Newton-Schulz orthogonalization (instead of uniform scaling)
- **Sequence length warmup + YaRN**: geometric warmup 128→1024 over 2000 steps with NTK-aware RoPE base scaling
- **Sliding window evaluation**: stride = seq_len // 2, skipping cold-start tokens for each non-first window
- **Cosine LR schedule**: linear warmup (100 steps) → constant → cosine cooldown to 0.1× min_lr over last 2000 steps
- **QK-norm**: RMSNorm on Q and K before RoPE application
- **Logit softcapping**: tanh-based cap at 30.0

### Additional architectural details

- **`resid_mix`**: learnable parameter `[2, dim]` per block that linearly mixes the current hidden state `x` with the original embedding `x0` at each recurrence pass: `x = mix[0]*x + mix[1]*x0`. Initialized to `[1, 0]` (identity), allows the model to learn how much "fresh" input signal to inject at each depth step — key to making depth recurrence work well.
- **`q_gain`**: learnable per-head scale applied to Q after RoPE, initialized to `qk_gain_init=1.5`. Compensates for the effect of QK-norm on attention logit scale.
- **`attn_scale` / `mlp_scale`**: learnable per-dim scaling of the attention and MLP residual outputs, initialized to 1. Allows the model to modulate contribution of each sublayer independently.
- **`proj` zero-init**: the output projection of both attention and MLP is zero-initialized, so each block starts as an identity at step 0.

## Serialization

Weights are packed as 2 bits/weight (ternary {-1→00, 0→01, 1→10}, 4 weights/byte) then zlib-compressed at level 9. Non-ternary params (embeddings, norms, skip weights) stored as float16.

Final sizes:
- Ternary+zlib model: 3,736,878 bytes
- Code (train_gpt.py): 45,677 bytes
- **Total: 3,782,555 bytes (3.78 MB)**

## Training command

```bash
RUN_ID=bitnet_4uniq_3rec_640d \
NUM_UNIQUE_LAYERS=4 \
RECURRENCE_FACTOR=3 \
MODEL_DIM=640 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
TRAIN_BATCH_TOKENS=32768 \
ITERATIONS=10000 \
GRADIENT_CHECKPOINTING=0 \
TERNARY_COMMIT_LAMBDA=0 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Notes

The quantization gap (pre-quant 1.4866 → post-quant 1.7510, Δ=+0.264 BPB) indicates the QAT does not fully converge the latent weights toward {-1,0,+1}. Subsequent runs add a ternary commitment loss to address this.

The 3.78 MB submission size leaves significant headroom under the 16 MB limit — the ternary packing allows fitting a model ~4× larger than equivalent int8+zlib approaches.
