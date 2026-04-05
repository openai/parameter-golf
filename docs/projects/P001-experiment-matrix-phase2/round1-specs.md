# Round 1: Formal Specifications

## R1-1: LeakyReLU(0.5)² + 11L + 3x MLP

### Mathematical Spec
```
f(x) = (leaky_relu(x, 0.5))²
     = x²           if x >= 0
     = 0.25·x²      if x < 0
```
MLP forward: `output = down_proj(leaky_relu(up_proj(x), 0.5)²)`

### Interface
- **Input**: MLP.forward(x: Tensor[B, T, 512]) → Tensor[B, T, 512]
- **Env vars**: `NUM_LAYERS=11`, `MLP_MULT=3`
- **Code change**: Replace `torch.relu(self.fc(x))` with `F.leaky_relu(self.fc(x), negative_slope=0.5)` in MLP.forward()

### Parameter Budget
| Component | Baseline (9L, 2x) | R1-1 (11L, 3x) |
|-----------|-------------------|-----------------|
| Embedding (tied) | 512×1024 = 524K | 524K |
| Per-layer attn | 512×512×4 = 1.05M | 1.05M |
| Per-layer MLP | 512×1024×2 = 1.05M | 512×1536×2 = **1.57M** |
| Per-layer total | 2.10M | 2.62M |
| All layers | 9×2.10M = 18.9M | 11×2.62M = **28.8M** |
| Total model | ~19.4M | ~29.3M |

Note: 29.3M float params → ~29.3MB in fp32. After int8+zlib compression: ~8-10MB (within 16MB limit based on SOTA submissions using same config).

### Initialization
- `up_proj`: orthogonal, gain=1.0 (matches baseline CastedLinear default)
- `down_proj`: zero-init, then scale by `1/sqrt(2*num_layers)` = `1/sqrt(22)` ≈ 0.213
- `down_proj._zero_init = True` (existing baseline convention)

### Verifiable DoD
1. `assert isinstance(model.blocks[0].mlp, MLP)` — MLP class exists
2. `x_neg = torch.tensor([-1.0]); assert F.leaky_relu(x_neg, 0.5).square().item() == 0.25` — activation math correct
3. `assert len(model.blocks) == 11` — layer count
4. `assert model.blocks[0].mlp.fc.weight.shape == (1536, 512)` — MLP hidden dim = 3×512
5. Model runs forward pass without error on shape `[2, 1024]` input
6. `final_model.int8.ptz` < 16MB after quantization

---

## R1-2: BigramHash 3072 Embedding

### Mathematical Spec
```
bigram_hash(t) = XOR(36313 × token[t], 27191 × token[t-1]) mod (V_bg - 1)    for t ≥ 1
bigram_hash(0) = V_bg - 1                                                       (sentinel)

embedding(tokens) = embed(bigram_hash(tokens)) × scale
output = token_embedding + bigram_embedding                                      (additive)
```
Where `V_bg = 3072` (bigram vocab size), `D_bg = 128` (bigram dim), `D = 512` (model dim).

When `D_bg ≠ D`: `embedding(tokens) = proj(embed(bigram_hash(tokens))) × scale`

### Interface
- **New class**: `BigramHashEmbedding(bigram_vocab_size=3072, bigram_dim=128, model_dim=512)`
- **Integration point**: After `tok_emb(input_ids)`, before `F.rms_norm()`
  ```python
  x = self.tok_emb(input_ids)
  x = x + self.bigram(input_ids)   # additive
  x = F.rms_norm(x, (x.size(-1),))
  ```
- **Env vars**: `BIGRAM_VOCAB_SIZE=3072`, `BIGRAM_DIM=128`

### Parameter Budget
| Component | Params |
|-----------|--------|
| `embed.weight` [3072, 128] | 393,216 |
| `proj.weight` [128, 512] | 65,536 |
| `scale` [1] | 1 |
| **Total** | **458,753** (~0.46M, +1.6% of model) |

### Initialization
- `embed.weight`: **zeros** (`nn.init.zeros_`)
- `proj.weight`: **zeros** (`nn.init.zeros_`)
- `scale`: **0.05** (`nn.Parameter(torch.tensor(0.05, dtype=torch.float32))`)
- Net effect at init: bigram embedding contributes exactly **zero** to the model output. It learns from zero.

### Optimizer Assignment
- `embed.weight` → Adam with `token_lr` (same as token embedding)
- `proj.weight` → Adam with `scalar_lr` (NOT Muon)
- `scale` → Adam with `scalar_lr`

### Hash Constants
- Bigram: 36313, 27191 (primes)
- Trigram: 36313, 27191, 51497 (primes) — NOT used in R1-2
- Modulus: `bigram_vocab_size - 1` = 3071 (NOT 3072 — last slot is sentinel)
- Arithmetic: int32 (cast tokens before hashing to avoid overflow)

### Verifiable DoD
1. `assert model.bigram.embed.weight.shape == (3072, 128)`
2. `assert model.bigram.proj.weight.shape == (128, 512)`
3. `assert model.bigram.embed.weight.sum() == 0.0` — zero-init
4. `assert model.bigram.scale.item() == pytest.approx(0.05)` — scale init
5. `tokens = torch.tensor([[5, 10, 15]]); h = model.bigram.bigram_hash(tokens); assert h[0,0] == 3071` — sentinel at position 0
6. `assert h[0,1] == (36313*10 ^ 27191*5) % 3071` — hash formula
7. `assert 0 <= h.min() and h.max() <= 3071` — all indices in bounds
8. Forward pass: `model.bigram(tokens).shape == (1, 3, 512)` — correct output shape

---

## R1-3: XSA (Cross-Sequence Attention)

### Mathematical Spec
```
v̂ = v / ||v||₂                                    (L2-normalize per KV head, dim=-1)
y_out = y - (y · v̂) · v̂                           (Gram-Schmidt: remove v-parallel component)
```
GQA-aware: reshape y from `[B,T,H,D]` to `[B,T,Hkv,group,D]` where `group = H/Hkv`. Each group of query heads shares one KV head's `v̂`.

### Interface
- **New method**: `CausalSelfAttention._xsa_efficient(y, v) → y_corrected`
- **New attribute**: `self.use_xsa: bool` (default False, set True by GPT.__init__ for last N layers)
- **Integration point**: After attention computation, before output projection
  ```python
  y = attention(q, k, v)
  if self.use_xsa:
      y = self._xsa_efficient(y, v)
  y = y.reshape(bsz, seqlen, dim)
  return self.proj(y)
  ```
- **Env var**: `XSA_LAST_N=4` (last 4 of 11 layers)

### Tensor Shapes (B=batch, T=1024, H=8, Hkv=4, D=64)
```
y      [B, T, 8, 64]      → input (attention output)
v      [B, T, 4, 64]      → input (value vectors)
y_g    [B, T, 4, 2, 64]   → reshape for GQA grouping
vn     [B, T, 4, 1, 64]   → normalized v, unsqueezed for broadcast
proj   [B, T, 4, 2, 64]   → parallel component
output [B, T, 8, 64]       → reshape back
```

### Important: Tensor Layout Mismatch
**Baseline uses `[B, H, T, D]` (transposed), SOTAs use `[B, T, H, D]`.**
- If keeping baseline's SDPA: must transpose before XSA, transpose back after
- OR: switch to flash_attn_3_func which outputs `[B, T, H, D]` natively

### Parameter Budget
- **Zero additional parameters** — pure computation on existing tensors
- One boolean flag per attention layer

### Verifiable DoD
1. `assert model.blocks[7].attn.use_xsa == True` — last 4 layers (indices 7-10) have XSA
2. `assert model.blocks[6].attn.use_xsa == False` — earlier layers don't
3. **Orthogonality test**: For any XSA-enabled layer, after forward pass:
   ```python
   y_out = xsa_efficient(y, v)
   v_norm = F.normalize(v, dim=-1)
   # Reshape y_out to [B,T,Hkv,group,D]
   dot = (y_out_grouped * v_norm.unsqueeze(-2)).sum(dim=-1)
   assert dot.abs().max() < 1e-5  # output is orthogonal to v
   ```
4. When `use_xsa=False`: output equals input (no modification)
5. Gradient flows through XSA (no detach)

---

## R1-4: U-Net Skip Connections

### Mathematical Spec
```
Encoder (layers 0..4):
  for i in 0..4:
    x = Block_i(x, x0)
    skips.push(x)

Decoder (layers 5..10):
  for i in 0..5:
    if skips not empty:
      x = x + skip_weight[i] ⊙ skips.pop()     (⊙ = element-wise, broadcast over [B,T])
    x = Block_{5+i}(x, x0)
```
Skip connections are LIFO: encoder layer 4 → decoder layer 5, encoder layer 3 → decoder layer 6, etc.

### Interface
- **New parameter**: `self.skip_weights: nn.Parameter` shape `[num_skips, model_dim]`
- **Modified forward()**: Split loop into encoder/decoder halves with skip stack
- `num_encoder_layers = num_layers // 2 = 5`
- `num_decoder_layers = num_layers - 5 = 6`
- `num_skip_weights = min(5, 6) = 5`

### Parameter Budget
| Component | Params |
|-----------|--------|
| `skip_weights` [5, 512] | **2,560** |

Negligible (~0.01% of model).

### Initialization
- `skip_weights`: **ones** (`torch.ones(5, 512, dtype=torch.float32)`)
- At init, skips act as identity additions: `x = x + 1.0 * skip`

### Optimizer Assignment
- `skip_weights` → Adam with `scalar_lr` (NOT Muon)

### Verifiable DoD
1. `assert model.skip_weights.shape == (5, 512)`
2. `assert (model.skip_weights == 1.0).all()` — ones init
3. `assert model.num_encoder_layers == 5 and model.num_decoder_layers == 6`
4. **Skip connectivity test**: Run forward with hooks, verify encoder layer 4 output appears in decoder layer 5 input (via the skip addition)
5. **Gradient test**: `model.skip_weights.grad is not None` after backward
6. Forward pass produces same shape output as before

---

## R1-5: Value Residual Propagation

### Mathematical Spec (SOTA #1 variant — sigmoid gate)
```
Layer 0:  v₀ = v                           (capture raw v from first layer)
Layer i (i>0):
  α = sigmoid(vrl_alpha)                    (learnable scalar gate)
  v_mixed = v + α · v₀                     (additive residual)
  attention uses v_mixed instead of v
```

### Alternative (SOTA #2 — lambda mix)
```
Layer 0:  v₀ = v
Layer i:
  v_mixed = λ[0] · v₀ + λ[1] · v          (unconstrained linear combination)
```

### Recommended: SOTA #1 (sigmoid gate) — simpler, one fewer parameter

### Interface
- **New attribute**: `Attention.value_residual: bool`
- **New parameter**: `Attention.vrl_alpha: nn.Parameter` shape `[1]`
- **GPT-level**: Store `v0` from first layer's raw v, pass to all subsequent layers
- **Env var**: `VALUE_RESIDUAL=1`

### Tensor Shapes
- `v0`: `[B, T, Hkv, D]` = `[B, T, 4, 64]`
- `vrl_alpha`: `[1]` (scalar)
- `sigmoid(0) = 0.5` → starts at 50% mixing

### Parameter Budget
| Component | Params |
|-----------|--------|
| `vrl_alpha` per layer × 11 layers | **11** |

Negligible.

### Initialization
- `vrl_alpha`: **zeros** → `sigmoid(0) = 0.5` → equal mix of v and v₀

### Capture Point
- `v₀` is captured from layer 0, AFTER value projection and reshape to `[B,T,Hkv,D]`, BEFORE RoPE
- `v₀` is NOT updated during decoder layers

### Optimizer Assignment
- `vrl_alpha` → Adam with `scalar_lr`

### Verifiable DoD
1. `assert model.blocks[0].attn.vrl_alpha.item() == 0.0` — zero init
2. `assert torch.sigmoid(model.blocks[0].attn.vrl_alpha).item() == pytest.approx(0.5)` — 50% mix at init
3. **v₀ capture test**: Hook first layer, verify v₀ is stored and has shape `[B, T, 4, 64]`
4. **Mixing test**: Hook layer 5, verify v_mixed = v + 0.5 * v₀ (at init, before training)
5. **No v₀ update in decoder**: v₀ remains constant across all layers after capture
6. `model.blocks[5].attn.vrl_alpha.grad is not None` after backward

---

## Cross-Cutting: Integration Order Verification

After implementing all R1 features, verify the full forward pass order:

```
1. tok_emb(input_ids)                           → [B, T, 512]
2. + bigram(input_ids)                          → [B, T, 512]  (R1-2)
3. rms_norm()                                   → [B, T, 512]
4. x0 = x                                       (capture embedding residual)
5. Encoder loop (layers 0-4):
   a. Block_i(x, x0) with:
      - Attention: q,k,v projections
      - v₀ capture at layer 0               (R1-5)
      - Value residual mix at layers 1-4     (R1-5)
      - Flash attention / SDPA
      - XSA if use_xsa                      (R1-3, only if layer in last 4)
      - Output projection
      - LeakyReLU(0.5)² MLP                 (R1-1)
   b. skips.push(x)                          (R1-4)
6. Decoder loop (layers 5-10):
   a. x = x + skip_weight[i] * skips.pop()  (R1-4)
   b. Block_i(x, x0) with same as above
7. Final norm → output head → logits
```
