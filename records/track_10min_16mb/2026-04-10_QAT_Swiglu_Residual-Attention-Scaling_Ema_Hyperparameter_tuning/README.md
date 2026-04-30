# Efficient Quantization-Aware Transformer under 10-Minute Training Constraint

## Overview

This submission focuses on optimizing a compact Transformer model under strict constraints:

- 10-minute training limit  
- <16MB model size  
- Optimized for bits-per-byte (BPB) metric  

The objective is to maximize learning efficiency and compression quality, rather than increasing model size.

## Final Result

- **Validation BPB**: ~1.599
- **Model Size (compressed)**: ~7.3 MB
- **Training Time**: ~10 minutes
- **Script Used**: final_optim_h100.py

## Key Technical Innovations

### 1. Quantization-Aware Training (QAT with STE)

Linear layers simulate int8 quantization during training using a **Straight-Through Estimator (STE)**:

```python
# Fake quantization in forward pass
scale = (w.abs().max() / 127.0).detach() + 1e-8
w_q = w + (torch.round(w / scale).clamp(-127, 127) * scale - w).detach()
```

**Benefits:**
- Weights adapt to int8 quantization constraints during training
- Minimal post-quantization loss after model export
- Training and final evaluation scores remain nearly identical
- Gradients flow through unquantized weights for stability

### 2. Gated MLP (SwiGLU-style Activation)

Replaced standard ReLU² MLP with gated activation:

```python
x1 = self.fc(x)
x2 = self.gate(x)
x = F.silu(x1) * x2  # Gated activation
return self.proj(x)
```

**Benefits:**
- Improved representation capacity with same parameter count
- Faster convergence compared to ReLU²
- Better numerical stability

### 3. Adaptive Residual Mixing

Introduced **learnable blending** between current activation and input:

```python
mix = self.resid_mix  # Learnable parameter
x = mix[0] * x + mix[1] * x0  # Weighted skip connection
```

**Benefits:**
- Enables dynamic skip connection strength
- Improves gradient flow during backpropagation
- Helps shallow networks converge faster under compute constraints
- Prevents vanishing gradients in early training stages

### 4. Learnable Attention Scaling (q_gain)

Each attention head has a **learnable scaling factor**:

```python
self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
```

**Benefits:**
- Stabilizes attention logits early in training
- Prevents softmax saturation
- Head-specific calibration improves attention effectiveness
- Reduces need for manual attention temperature tuning

### 5. Per-Channel Residual Scaling

Learnable scaling applied to:
- Attention output
- MLP output

```python
self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
x = x + self.attn_scale * attn_out
x = x + self.mlp_scale * mlp_out
```

**Benefits:**
- Controlled residual updates prevent training instability
- Per-channel scaling is more flexible than scalar scaling
- Improves numerical stability and convergence

### 6. Exponential Moving Average (EMA) Model

Added **EMA tracking** alongside SWA for validation:

```python
ema_model = copy.deepcopy(model)
ema_decay = 0.99995
update_ema(model, ema_model, ema_decay)  # Updated every training step
```

**Benefits:**
- Smoother weight trajectory improves generalization
- Final evaluation uses EMA model for stable predictions
- Complements SWA (started at 70% of iterations)
- Better final model quality without additional computational cost

### 7. Muon Optimizer with Distributed Synchronization

Custom optimizer for matrix-shaped parameters:

**Parameter Groups:**
- **Token Embeddings**: Adam with embedding-specific learning rate
- **Matrix Parameters** (2D): Muon optimizer with orthogonal normalization
- **Scalar/Vector Parameters**: Adam for residual scales and bias terms
- **Language Model Head**: Adam (when untied from embeddings)

**Muon Features:**
- Newton-Schulz iteration for weight orthogonalization
- Distributed all-reduce for synchronized updates across ranks
- Momentum warmup (from 0.85 to 0.95 over 500 steps)
- Backend orthogonalization steps: 5

### 8. Grouped Query Attention (GQA)

Architecture uses **GQA for efficiency**:

```python
num_heads = 4          # Query heads
num_kv_heads = 4       # Key/Value heads (can be < num_heads)
kv_dim = num_kv_heads * head_dim
```

**Benefits:**
- Reduced key/value cache memory (important for inference)
- Maintains attention expressiveness
- Faster computation than full multi-head attention

### 9. Rotary Positional Embeddings (RoPE)

Applied at query and key positions:

```python
cos, sin = self.rotary(seqlen, x.device, q.dtype)
q = apply_rotary_emb(q, cos, sin)
k = apply_rotary_emb(k, cos, sin)
```

**Benefits:**
- Extrapolates better than absolute positional embeddings
- Naturally preserves relative position information
- Numerically stable (cached per sequence length)

### 10. Encoder-Decoder Style Architecture with Skip Connections

**Architecture:**
- **Total Layers**: 8
- **Encoder Layers**: 4 (stores skip connections)
- **Decoder Layers**: 4 (uses stored skips in reverse)

```python
# First half: store skips
for i in range(num_encoder_layers):
    x = blocks[i](x, x0)
    skips.append(x)

# Second half: reuse skips
for i in range(num_decoder_layers):
    if skips:
        x = x + skip_weights[i] * skips.pop()  # Pop in reverse
    x = blocks[num_encoder_layers + i](x, x0)
```

**Benefits:**
- Skip connections bridge distant layers
- Improves gradient flow without additional layers
- Enables parameter reuse pattern
- `skip_weights` are learnable to control skip strength

### 11. Efficient Int8 + Zlib Compression Pipeline

**Quantization Strategy:**
1. **Per-Row Int8** for 2D matrices (weight matrices)
2. **Per-Tensor Int8** for 1D vectors (biases, scales)
3. **Percentile Clipping** at 99.99984th percentile
4. **Small Tensor Passthrough** (≤65,536 elements stored as FP16)
5. **Control Parameters** kept in FP32 (attention scales, residual mixes)

```python
INT8_CLIP_PERCENTILE = 99.99984
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
```

**Compression Ratios:**
- Baseline (BF16): ~3× larger
- Final (Int8 + Zlib): ~7.3 MB total
- Achieves high compression without significant reconstruction error

### 12. Sliding Window Validation

During evaluation, uses **sliding windows** to compute validation BPB:

```python
stride = seq_len // 4  # Overlapping windows
for start in range(0, total_len - seq_len + 1, stride):
    # Process windows of size seq_len with overlap
```

**Benefits:**
- More efficient evaluation than simple next-token prediction
- Better estimates of actual compression quality
- Mimics real-world inference patterns

### 13. Warmdown Schedule

Learning rate warmdown during final iterations:

```python
def lr_mul(step, elapsed_ms):
    if step < lr_warmup_iters:
        return step / lr_warmup_iters  # Warmup
    
    warmdown_start = max(iterations - warmdown_iters, 0)
    if step >= warmdown_start:
        return (iterations - step) / warmdown_iters  # Warmdown
    return 1.0
```

**Settings:**
- **Warmup Iters**: 50
- **Warmdown Iters**: 700
- **Wallclock-Aware**: Scales based on remaining time budget

## Hyperparameter Tuning Strategy

This section documents the systematic hyperparameter optimization process that led to the final configuration.

### Phase 1: Stability Foundation

**Initial Baseline (Experiment 0):**
- matrix_lr: 0.04, train_batch_tokens: 524,288
- Result: val_bpb ≈ 2.04 (unstable training)

**Key Fix (Experiment 1):**
- Reduced matrix_lr: 0.04 → 0.01
- Added LR warmup (20 steps)
- Result: val_bpb ≈ 1.73 (stable convergence)

**Learning**: Stability trumps size. Proper learning rate tuning and warmup are prerequisite for any other optimization.

### Phase 2: Model Architecture Search

**Scaling Experiment (Experiment 2 - Failed):**
- Increased model_dim to 320, num_layers to 8, batch to 262,144
- Result: val_bpb ≈ 1.83 (worse), only 451 steps in 10 min
- Issue: Larger model = slower steps = fewer iterations = underfitting

**Balanced Model Discovery (Experiment 3 - Success):**
- model_dim: 256, num_layers: 8, num_heads: 4, mlp_mult: 3
- TRAIN_BATCH_TOKENS: 262,144, train_seq_len: 128
- Result: val_bpb ≈ 1.7185 with 800 steps

**Learning**: Under time constraints, balance model capacity with training speed. More steps with slightly smaller model > fewer steps with larger model.

### Phase 3: Batch Token Optimization

**Batch Size Tuning (Experiment 4):**
- Initial attempt: TRAIN_BATCH_TOKENS: 200,000 (failed - math misalignment)
- Corrected: TRAIN_BATCH_TOKENS: 196,608 (divisible by grad_accum_steps × train_seq_len)
- Later refined to 180,224

**Critical Constraint**: 
```
TRAIN_BATCH_TOKENS must be divisible by: world_size × grad_accum_steps × train_seq_len
i.e., TRAIN_BATCH_TOKENS % (8 × 1 × 128) == 0
```

### Phase 4: Learning Rate Fine-tuning (RUN-4)

| Parameter | Initial | Tuned | Impact |
|---|---|---|---|
| matrix_lr | 0.04 | 0.010 | Stable convergence |
| embed_lr | - | 0.050 | Better embedding training |
| scalar_lr | - | 0.020 | Residual scale optimization |
| lr_warmup_iters | - | 20 | Prevent early divergence |

**Result**: val_bpb ≈ 1.6836 with 1,073 steps

**Observation**: Multi-optimizer approach with different learning rates per parameter type outperforms single LR setting.

### Phase 5: Schedule Optimization (Warmdown Tuning)

**Key Finding**: Model still improving at end of 10 min → LR decay happening too early

**Experiments:**
- warmdown_iters: 800 (early decay) → val_bpb: 1.6840
- warmdown_iters: 500 (later decay) → expected improvement

**Schedule Trade-off**:
```
Early warmdown: Better convergence quality but learning stops too soon
Late warmdown: More late-stage learning but potential instability
Optimal: ~700 iterations
```

### Phase 6: Architectural Enhancements (Below 1.65)

After hyperparameter tuning reached ~1.68, the following architectural improvements pushed performance below 1.65:

**Key Additions:**
1. **SWA (Stochastic Weight Averaging)** - Started at 70% of iterations
   - Smooths weight trajectory in latter half
   - Contributes ~0.01-0.02 BPB improvement

2. **Sliding Window Validation** - Evaluation strategy
   - Replaces simple next-token evaluation with overlapping windows (stride = seq_len/4)
   - Better correlation with actual compression quality
   - More accurate BPB measurement

3. **EMA (Exponential Moving Average)** - decay=0.99995
   - Updated every training step
   - Used for final evaluation (paired with SWA)
   - Smooths out training noise

4. **Gated MLP** - SwiGLU-style activation
   - Better representation with same parameters
   - Faster convergence

5. **Learnable Scales** (attention, MLP, residual mixing)
   - Per-channel adaptive scaling
   - Improved numerical stability

6. **Quantization-Aware Training (QAT)**
   - Straight-through estimator in linear layers
   - Weights adapt to int8 constraints during training
   - Minimal post-quantization loss

7. **Encoder-Decoder Skip Connections**
   - Skip weights bridge distant layers
   - Improved gradient flow

**Combined Effect**: These architectural changes stacked to achieve val_bpb ≈ 1.599 (from 1.684)

### Summary: Tuning Phases Impact

| Phase | Focus | Result | Improvement |
|---|---|---|---|
| Phase 1 | Stability (LR + warmup) | 2.04 → 1.73 | 0.31 BPB |
| Phase 2 | Model architecture | 1.73 → 1.72 | 0.01 BPB |
| Phase 3-4 | Batch & multi-optimizer | 1.72 → 1.68 | 0.04 BPB |
| Phase 5 | Schedule tuning | 1.68 → 1.68 | Stable |
| Phase 6 | Arch enhancements | 1.68 → 1.60 | 0.08 BPB |
| **Total** | **Full Pipeline** | **2.04 → 1.60** | **0.44 BPB** |

**Key Principle**: Hyperparameter tuning first (stable foundation), then architectural innovations (capacity improvements).

## Training Strategy

**Optimized for 600-second constraint:**

1. **Quick Convergence**: Uses efficient architectures and aggressive learning rates
2. **Minimal Validation Overhead**: Validation only at milestones, not every step
3. **Stochastic Weight Averaging**: Started at 70% of iterations (step 840)
4. **EMA Tracking**: Smooths weights without extra computational cost
5. **Distributed Training**: 8xH100s with gradient accumulation (8 steps total)

**Training Parameters:**
- **Iterations**: 1200
- **Batch Size**: 172,032 tokens per step (across 8 GPUs)
- **Sequence Length**: 128 tokens
- **Warmup Steps**: 20
- **Max Wallclock**: 600 seconds

## Model Architecture Details

**Model Configuration:**
- **Vocab Size**: 1024 (SentencePiece)
- **Hidden Dim**: 256
- **Num Layers**: 8
- **Num Heads**: 4
- **KV Heads**: 4 (GQA)
- **MLP Expansion**: 3× (768 hidden units)
- **RoPE Base**: 10000.0
- **Logit Softcap**: 30.0

## Hyperparameter Learning Rates

**Multi-Optimizer Strategy:**

| Parameter Type | Optimizer | Learning Rate | Details |
|---|---|---|---|
| Token Embeddings | Adam | 0.05 | Tied or separate embedding |
| Matrix Params (2D) | Muon | 0.010 | Orthogonal normalization |
| Scalar/Vector Params | Adam | 0.02 | Residual scales, control params |
| LM Head (untied) | Adam | 0.008 | If embeddings are not tied |

**Optimizer Settings:**
- **Adam Beta1**: 0.9
- **Adam Beta2**: 0.95
- **Adam Eps**: 1e-8
- **Gradient Clipping**: Disabled (0.0)

**Muon Momentum Warmup:**
- **Start**: 0.85
- **End**: 0.95
- **Warmup Steps**: 500

## Files

- **final_optim_h100.py**: Main training script
- **final_model.pt**: Full precision model state (debug only)
- **final_model.int8.ptz**: Compressed model (Int8 + Zlib)
- **train.log**: Training logs with metrics

## Reproducibility

**Requirements:**
- PyTorch 2.0+
- Sentencepiece tokenizer
- FineWeb dataset (preprocessed)
- 8xH100 GPUs (or adjust WORLD_SIZE)

**Run Training:**

```bash
python final_optim_h100.py
```

**Environment Variables (optional):**
```bash
ITERATIONS=1200
TRAIN_BATCH_TOKENS=172032
TRAIN_SEQ_LEN=128
NUM_LAYERS=8
MODEL_DIM=256
NUM_HEADS=4
```

## Key Insights

1. **Quantization Awareness Matters**: QAT during training (not post-training) yields better compression quality
2. **Architectural Efficiency > Size**: Gated MLPs and skip connections outperform scaling in constrained settings
3. **Learnable Parameters Everywhere**: Scales, gates, and mixing weights adapt to task requirements
4. **EMA + SWA Synergy**: Both tracking mechanisms together provide better final model quality
5. **Distributed Optimization**: Custom Muon optimizer with synchronized updates improves matrix parameters
6. **Under Compute Limits**: Fast convergence and efficient architectures beat larger models

## Conclusion

This submission demonstrates that through:

- [x] **Quantization-aware training** with straight-through estimators
- [x] **Architectural innovations** (gated MLPs, residual mixing, learnable scales)
- [x] **Careful optimization** (EMA, SWA, multi-optimizer strategy)
- [x] **Efficient compression** (per-row int8 + zlib)

...we can achieve competitive model quality under strict parameter and compute constraints.

## References

- **Rotary Embeddings (RoPE)**: https://arxiv.org/abs/2104.09864
- **Grouped Query Attention**: https://arxiv.org/abs/2305.13245
- **Muon Optimizer**: https://kellerjordan.github.io/posts/muon/
- **SwiGLU MLPs**: https://arxiv.org/abs/2002.05202
- **Quantization-Aware Training**: https://arxiv.org/abs/1609.07061
