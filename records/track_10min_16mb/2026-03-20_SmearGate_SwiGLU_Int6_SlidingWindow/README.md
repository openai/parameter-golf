# SmearGate + BigramHash + Int6 + SWA + U-Net Skips + Sliding Window

## Score: val_bpb = 1.1518 (seed 1337, sliding window stride=64, post int6+zstd quantization)

Trained on 8×H100 SXM in 600 seconds. 15.2MB artifact (int6+zstd-22).

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=1337)
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All parameters are set as defaults in `train_gpt.py`. No env vars needed.

## Results

| Eval Method | val_loss | val_bpb |
|-------------|----------|---------|
| Pre-quantization | 1.9841 | 1.1751 |
| Int6 roundtrip (non-overlapping) | 2.0027 | 1.1861 |
| **Int6 sliding window (stride=64)** | **1.9448** | **1.1518** |
| Int6 TTT LoRA (rank=8) | 1.9476 | 1.1535 |

Single seed (1337). Additional seeds available on request.

## Approach

Eight techniques stacked on the baseline 9-layer, 512-dim GPT:

### 1. Per-Row Int6 Quantization + zstd-22 Compression
MLP and attention weight matrices quantized to int6 ([-32, 31]) with per-row scaling. Tied embeddings and the final layer's key projection remain in fp16 (quantization-sensitive). zstd at level 22 compresses int6 data efficiently, yielding a 15.2MB artifact from 105MB raw model.

### 2. 3× MLP Expansion with Relu²
MLP hidden dimension increased from 1024 (2×) to 1536 (3×), enabled by byte savings from int6 quantization. Activation is `relu(x)²` (squared ReLU), which promotes sparsity and compresses well.

### 3. 11 Layers (vs 9 baseline)
Int6 compression frees enough bytes for 2 additional transformer layers. More depth improves representation at the cost of fewer training steps per wall-clock second.

### 4. SmearGate
A learned gate blending each token's embedding with the previous token's embedding after layer norm, providing lightweight bigram-level context at the input. Adds ~512 parameters.

### 5. BigramHash Embedding
A 2048-bucket hash table (dim=128, projected to 512) mapping adjacent token pairs to learned embeddings via `(prev_token * 31 + curr_token) % 2048`. Adds ~262K parameters. Complements SmearGate with an additive bigram signal.

### 6. U-Net Skip Connections
The 11 layers are split into encoder (first 5) and decoder (last 5) halves, with learned skip connections from encoder layer i to decoder layer (N-1-i). Each skip weight is a per-dimension scaling parameter. This helps gradient flow in deeper models.

### 7. Muon Optimizer with Weight Decay
Muon with decoupled weight decay WD=0.04 for matrix parameters. Momentum warmup from 0.92 to 0.99 over 1500 steps. AdamW with WD=0.04 for embedding and scalar parameters. Weight decay regularizes magnitudes, improving int6 quantization quality.

### 8. Stochastic Weight Averaging (SWA)
SWA every 200 steps during warmdown (7 snapshots averaged). Produces smoother weight distributions that quantize better and generalize.

### Evaluation: Sliding Window (stride=64)
Standard non-overlapping evaluation scores 1.1861. Sliding window at stride=64 gives each token more context, yielding 1.1518 — a 0.034 BPB improvement from evaluation alone.

### TTT LoRA (Test-Time Training)
Rank-8 LoRA adapters fine-tuned per validation chunk (256 tokens, lr=0.01). Achieves 1.1535, slightly worse than sliding window for this model — the base model is strong enough that TTT's marginal gains don't justify the overhead.

## Architecture

| Parameter | Value |
|-----------|-------|
| num_layers | 11 |
| model_dim | 512 |
| num_heads | 8 |
| num_kv_heads | 4 (GQA) |
| mlp_mult | 3 (hidden=1536) |
| mlp_activation | relu² |
| vocab_size | 1024 |
| bigram_vocab_size | 2048 |
| bigram_dim | 128 |
| tie_embeddings | True |
| logit_softcap | 30.0 |
| rope_base | 10000.0 |
| model_params | 26.8M |

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| train_seq_len | 1024 |
| train_batch_tokens | 524,288 |
| iterations (max) | 20,000 |
| actual_steps | 9,906 (600s wallclock cap) |
| ms_per_step | 60.57 |
| warmup_steps | 20 |
| warmdown_iters | 3,000 |
| matrix_lr | 0.025 |
| scalar_lr | 0.025 |
| tied_embed_lr | 0.035 |
| embed_lr | 0.6 |
| muon_momentum | 0.99 (warmup from 0.92) |
| muon_weight_decay | 0.04 |
| adam_weight_decay | 0.04 |
| grad_clip_norm | 0.3 |
| swa_every | 200 steps |
| swa_snapshots | 7 |
| eval_stride | 64 |

## Artifact Size

| Component | Bytes |
|-----------|-------|
| Model (int6+zstd-22) | 15,144,475 |
| Code (train_gpt.py) | 58,040 |
| **Total** | **15,202,515** |
| Limit | 16,000,000 |

## Differences from PR #162 (raahilshah)

This submission was developed independently and shares some techniques with PR #162 (SmearGate, BigramHash, int6, SWA) but differs in:

- **Relu² activation** vs unreported in #162 (assumed relu²)
- **11 layers** vs 9 layers — we trade seq_len=1024 (vs 2048) for 2 extra layers
- **BigramHash 2048** vs 4096 buckets
- **SWA every 200** vs every 50 steps (fewer but later snapshots)
- **U-Net skip connections** with learned per-dimension weights
- **TTT LoRA evaluation** included as alternative eval method
- **seq_len=1024** vs 2048 — shorter sequences allow more steps (9906 vs 7379) in the same wall-clock budget
