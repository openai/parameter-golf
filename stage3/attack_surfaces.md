# Stage 3: Attack Surface Analysis of train_gpt.py

Line-by-line analysis of the 1372-line training script. Every modification that could improve val_bpb is catalogued with its mechanism, estimated impact, and implementation complexity.

The script has 7 major sections. Each is an independent attack surface.

---

## UPDATE 2026-03-24: Stage 3 Pivots to Original Hypotheses

Stage 2_1 now owns the entire community technique playbook (LeakyReLU^2, EMA, MuonWD, XSA, GPTQ, Partial RoPE, LN Scale, curriculum). Stage 3 no longer duplicates those.

Instead, stage3 tests **5 original hypotheses derived from cross-domain mechanism transfer**. Each attacks a surface no community PR has touched. The catalog below remains valid as a reference, but the active stage3 slots are:

| Slot | Patch | Target Surface (below) | Source Domain | Target Lines |
|------|-------|----------------------|---------------|-------------|
| H1 | `zloss` | §7 Loss function | PaLM stabilization | 723-724 |
| H2 | `adaptive_ns_steps` | §1 Optimizer internals | Control theory (gain scheduling) | 96-109, 153 |
| H3 | `nuclear_norm` | §1 Weight structure (NEW surface) | Rate-distortion theory | 852-884 (matrix_params) |
| H4 | `weight_perturbation` | §1 Landscape geometry (NEW surface) | Langevin dynamics | 1032-1034 (post-step) |
| H5 | `grad_centralization` | §1 Gradient preprocessing (NEW surface) | ICCV 2020 (Yong et al.) | 1030-1031 (pre-step) |
| H6 | `zloss` + `nuclear_norm` | Cross-surface compound | — | — |

Three of these target the optimizer section (§1) but attack completely different aspects: H2 changes the computation budget of Newton-Schulz, H4 perturbs weights after the step, H5 preprocesses gradients before the step. They don't overlap.

### New attack surfaces not in the original catalog

The original catalog missed several exploitable surfaces:

| Surface | Description | Stage 3 Hypothesis |
|---------|-------------|-------------------|
| **Loss function regularization** | Soft penalties on the output distribution beyond cross-entropy. The model has logit softcapping but no differentiable partition-function penalty. | H1 (z-loss) |
| **Optimizer computation budget** | Newton-Schulz iteration count is fixed at 5. The quality/cost tradeoff changes over training phases. | H2 (adaptive NS) |
| **Weight matrix spectral structure** | WD penalizes Frobenius norm (magnitude). Nuclear norm penalizes sum of singular values (rank). Different compression properties. | H3 (nuclear norm) |
| **Loss landscape geometry** | Sharp vs flat minima have different quantization robustness. Explicit noise can bias toward flat regions. | H4 (weight perturbation) |
| **Gradient DC component** | The mean of each gradient tensor is a global shift signal. Removing it (centralization) may improve convergence by focusing on differential structure. | H5 (gradient centralization) |

---

## 1. Optimizer (lines 96-175): Muon + Adam Split

### Current State
- Muon for 2D matrix params in transformer blocks
- Adam for embeddings, scalars, skip weights
- Newton-Schulz orthogonalization with configurable steps (default 5)
- Momentum with linear warmup
- No weight decay on Muon params
- No per-row normalization (plain Muon, not NorMuon)

### Attack Surfaces

**A1. NorMuon (replace Muon)**
- Lines 103-116: `zeropower_via_newtonschulz5` does standard Newton-Schulz
- NorMuon adds per-row normalization of the update after Newton-Schulz: `X = X / X.norm(dim=1, keepdim=True)`
- This is ~5 lines of code change in the function
- Community evidence: +0.005-0.01 BPB (#89, #122, #156, #173)
- Risk: low — drop-in replacement, same optimizer interface

**A2. Muon Weight Decay**
- Line 172: `p.add_(g, alpha=-lr)` — no WD term
- Add `p.mul_(1 - wd * lr)` before the update
- Need new env var `MUON_WEIGHT_DECAY` and wire it through
- Community evidence: +0.003-0.01 BPB. Best values 0.02-0.04 (#60, #162, #179, #180)
- Mechanism: reduces weight norms → smaller quantization error → better post-quant BPB
- Risk: low — one-line change in optimizer step

**A3. Adam Weight Decay for Scalars**
- Line 1111-1116: `optimizer_scalar` is plain Adam with no weight_decay
- Could add `weight_decay=0.01` to scalar Adam
- Community evidence: #162 uses `AdamW WD=0.01` for embeddings/scalars
- Risk: low

**A4. Adaptive Muon Backend Steps**
- Line 81: `muon_backend_steps = 5` (default)
- Already env-var configurable. We tested 7, no clear win.
- Diminishing returns territory. Skip.

---

## 2. Weight Initialization (lines 706-712): _init_weights

### Current State
```python
def _init_weights(self):
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)  # std=0.005
    for module in self.modules():
        if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
            nn.init.zeros_(module.weight)
```
- Tied embedding: tiny normal init (std=0.005)
- Output projections (attn.proj, mlp.proj): zero init (via `_zero_init = True`)
- All other CastedLinear weights: **default PyTorch init** (kaiming_uniform)
- No orthogonal init
- No muP scaling

### Attack Surfaces

**B1. Orthogonal Init**
- Replace default kaiming_uniform with `nn.init.orthogonal_` for all CastedLinear weights that aren't zero-inited
- Targets: `c_q`, `c_k`, `c_v` (attention), `fc` (MLP up-projection)
- Lines 582-585 (CastedLinear creation), Line 706-712 (_init_weights)
- Community evidence: +0.003-0.005 BPB (#135, #162, #164)
- Mechanism: better conditioned initial weight matrices → Muon converges faster in early steps → more useful learning in the 600s budget

**B2. muP Output Scaling**
- Scale output projection weights (attn.proj, mlp.proj) by `1/sqrt(2 * num_layers)` instead of zero
- Lines 586, 623: `self.proj._zero_init = True` — currently fully zeroed
- Change: init to orthogonal scaled by `1/sqrt(2*num_layers)` instead of zeros
- Or: keep zero init but scale the residual connection
- Community evidence: paired with ortho init in all top submissions

**B3. Overtone Spectral Init**
- SVD power-law spectrum shaping for embedding init
- Used by #60 (notapplica, merged SOTA)
- Impact unclear vs ortho init. Lower priority.

---

## 3. Architecture (lines 504-744): Transformer Modules

### Current State
- RMSNorm (no learnable scale)
- CastedLinear: fp32 weights, bf16 compute
- Rotary embeddings with configurable base (default 10000)
- GQA: 8 heads, 4 KV heads
- relu^2 MLP (relu then square)
- U-Net skip connections (encoder-decoder with learnable skip_weights)
- Logit softcapping (tanh, cap=30)
- resid_mix: learned residual mixing between current hidden state and initial embedding

### Attack Surfaces

**C1. SmearGate**
- Insert between embedding and first block (after line 714-715: `x = self.tok_emb(input_ids); x = F.rms_norm(...)`)
- ~512 params: one learnable vector, sigmoid gate
- `gate = sigmoid(self.smear_gate); x = gate * x + (1-gate) * shift(x, 1)`
- Needs: new parameter, roll operation for previous token
- Community evidence: +0.005 BPB (#102, #135, #162, #164, #170, #180)
- Risk: low — tiny parameter count, simple forward pass change

**C2. BigramHash Embedding**
- 4096-bucket hash table, dim=128, projected to model_dim=512
- Hash current and previous token IDs to bucket, look up embedding, add to token embedding
- ~524K extra params (~2MB in fp32, ~0.5MB in int6)
- Community evidence: +0.005 BPB (#135, #162, #164, #180)
- Risk: medium — needs hash function, projection layer, and careful quantization handling
- Size concern: 524K params at int6 = ~393KB. Fits in budget.

**C3. 10th Layer**
- Line 64: `num_layers = int(os.environ.get("NUM_LAYERS", 9))`
- Already configurable! Just set `NUM_LAYERS=10`
- But: 10th layer adds ~2.4M params → ~1.8MB at int6 → need to reclaim space
- Mixed int5/int6 (#180) saves 1.86MB, exactly enough
- Or: MuonWD compresses weights enough to fit (#179 fits 11 layers with WD=0.038)
- Community evidence: +0.005-0.01 BPB (#60, #150, #179, #180)

**C4. FlashAttention 3**
- Lines 996-999: Explicitly enables flash_sdp, disables others
- Currently uses PyTorch's SDPA which dispatches to FA2
- FA3 requires direct `from flash_attn import flash_attn_func` and replacing the `F.scaled_dot_product_attention` call
- Community evidence: ~10ms/step faster → ~15% more steps in 600s
- Risk: low — drop-in for the attention call, needs flash-attn package

**C5. MLP Hidden Dimension**
- Line 68: `mlp_mult = int(os.environ.get("MLP_MULT", 2))`
- Already configurable. We use `MLP_HIDDEN=1536` (3x) in SOTA stack.
- Non-integer multipliers possible: #107 uses MLP_HIDDEN=1488 (2.91x)
- Could squeeze a few more params if size budget is tight

**C6. SwiGLU Activation**
- Line 625-627: relu^2 MLP (`relu(fc(x)).square()`)
- SwiGLU would change to: `gate = sigmoid(fc_gate(x)); x = gate * silu(fc(x))`
- Adds a second up-projection (fc_gate)
- Community evidence: neutral to negative (#81, #163). relu^2 seems fine for this scale.
- Skip.

---

## 4. Quantization / Export (lines 287-430): int8 + zlib

### Current State
- Per-row int8 quantization for 2D tensors, per-tensor for 1D
- Clip at 99.99984th percentile
- fp16 storage for small tensors (< 65536 elements)
- zlib level 9 compression
- No int6 support in the base script (our SOTA stack patches this via env vars)
- No STE / QAT during training

### Attack Surfaces

**D1. STE Int6 QAT (Quantization-Aware Training)**
- During forward pass, fake-quantize weights to int6 range [-31, 31] with per-row scaling
- Use straight-through estimator for backward pass (gradient passes through the quantization)
- Modify `CastedLinear.forward` (line 518):
  ```python
  def forward(self, x):
      w = self.weight.to(x.dtype)
      if self.training:
          # Fake int6 quantization
          scale = w.abs().amax(dim=1, keepdim=True) / 31.0
          w_q = (w / scale).round().clamp(-31, 31) * scale
          w = w + (w_q - w).detach()  # STE
      return F.linear(x, w, self.bias)
  ```
- Community evidence: reduces quant gap from ~0.015 to ~0.002 BPB (#63, #65, #89, #128, #156)
- Risk: low-medium — well-understood technique, ~10 lines of code

**D2. zstd-22 Compression**
- Line 1312: `quant_blob = zlib.compress(quant_raw, level=9)`
- Replace with `zstandard.ZstdCompressor(level=22).compress(quant_raw)`
- Saves ~0.5MB vs zlib-9 on int6 data
- This extra space can fund more params or a denser model
- Community evidence: universal in top submissions
- Risk: trivial — `pip install zstandard`, one-line change

**D3. Mixed Int5/Int6 Quantization**
- Int5 [-16, 15] for MLP weights — 3 zero high bits per byte → zstd compresses at 1.88x
- Int6 [-32, 31] for attention weights
- Saves 1.86MB → funds a 10th layer
- Only #180 uses this (current best at 1.1453)
- Risk: medium — needs per-tensor quantization policy, careful testing

**D4. fp16 Tied Embedding Passthrough**
- Line 380: small tensors (< 65536 elements) already kept in fp16
- But tok_emb.weight is 1024 * 512 = 524,288 elements → gets quantized to int8
- Need to add explicit passthrough for tok_emb
- Already supported via `INT8_KEEP_TOK_EMB_FP16=1` env var in our SOTA stack
- Community evidence: ~0.01 BPB improvement. Universal in top submissions.

**D5. Late-K fp16 Passthrough**
- Keep last N layers' key projections in fp16
- Already supported via `INT8_KEEP_LAST_KV_LAYERS_FP16=2` in our SOTA stack
- Marginal additional gain on top of STE QAT

---

## 5. Learning Rate Schedule (lines 1156-1165): lr_mul

### Current State
```python
def lr_mul(step, elapsed_ms):
    # Wallclock-aware warmdown
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = warmdown_iters * step_ms
    remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0
```
- Linear warmdown based on remaining wallclock time
- No warmup (momentum warmup exists, but LR starts at full)
- `warmdown_iters` default 1200, our SOTA uses 3000

### Attack Surfaces

**E1. LR Warmup**
- Currently no LR warmup (line 1156 returns 1.0 when not in warmdown)
- The "warmup_steps" (line 1169) is for torch.compile JIT warmup, not LR warmup
- Adding a short LR warmup (100-200 steps, linear 0→1) could help stability
- Community evidence: mixed. Most top submissions don't use LR warmup.
- Risk: low but likely marginal

**E2. Cosine Warmdown**
- Current warmdown is linear
- Cosine warmdown: `0.5 * (1 + cos(pi * progress))` where progress = fraction through warmdown
- Could help SWA by providing smoother weight trajectories
- Community evidence: not tested by community (everyone uses linear)
- Risk: low, speculative

**E3. Higher Base LR**
- #179 uses matrix_lr=0.025 (vs our 0.02) with 11L + WD=0.038
- WD acts as implicit LR scaling → can push LR slightly higher
- Env-var only change if MuonWD is wired

---

## 6. Training Loop (lines 1199-1288): Main Loop

### Current State
- Grad accumulation: 8 // world_size steps
- Wallclock-aware stopping
- No gradient clipping by default (env var exists but default is 0.0)
- No SWA / checkpoint averaging
- No MTP auxiliary loss
- Validation every 1000 steps (configurable)

### Attack Surfaces

**F1. SWA (Stochastic Weight Averaging)**
- After warmdown begins, save model state_dict every N steps
- At the end, average all saved checkpoints
- Implementation: ~20 lines — accumulator dict, periodic snapshot, final average
- Best frequency: every 50 steps (#180) or every 200 steps (#162)
- Community evidence: +0.002-0.005 BPB (#89, #122, #162, #180)
- Mechanism: averaged weights are smoother → less quantization damage
- Risk: low — well-understood, no impact on training dynamics until final average

**F2. MTP (Multi-Token Prediction) Auxiliary Loss**
- Add a second linear head that predicts token at position t+2 (or t+1 from a shifted position)
- Auxiliary loss added to main loss with a coefficient (e.g., 0.1)
- Head is excluded from the exported artifact
- Community evidence: +0.003 BPB (#88)
- Risk: medium — adds parameters and compute during training, need to balance coefficient

**F3. Gradient Clipping**
- Line 87: `grad_clip_norm = 0.0` (disabled by default)
- Our SOTA stack doesn't use it either
- #114 found 0.3 optimal for seq2048
- Should be env-var: `GRAD_CLIP_NORM=0.3`
- Community evidence: +0.003 for seq2048 (#114)
- Risk: trivial — already wired, just needs the right value

---

## 7. Eval Policy (lines 226-286, 746-955): Validation + TTT

### Current State
- Standard non-overlapping eval at train_seq_len
- TTT LoRA eval at the end (after quant roundtrip)
- No sliding window eval during training (only final)
- Sliding window is done post-hoc via EVAL_STRIDE env var in our SOTA stack

### Attack Surfaces

**G1. Stride=64 vs Stride=256**
- Our SOTA uses stride=256 (from #114's recommendation)
- Most top submissions use stride=64
- #114 argued 256 is slightly better (1.1574 vs 1.1579) — but this is within noise
- The top 5 submissions all use stride=64
- Pure env-var change: `EVAL_STRIDE=64`
- Risk: trivial, needs to verify eval time stays reasonable

**G2. Doc-Isolated Eval**
- Currently eval streams tokens continuously across document boundaries
- Resetting context at BOS tokens gives ~0.011 BPB improvement (#77)
- Independent of sliding window — can stack
- Requires modifying `eval_val` function (lines 226-286)

**G3. TTT LoRA Tuning**
- Lines 89-94: TTT hyperparameters are env-var configurable
- Default: rank=8, lr=0.01, chunk_size=256, eval_seq_len=1024, batch_size=64
- Could tune: higher rank, different LR, larger eval_seq_len
- Community evidence: +0.003 total (#77), mostly from doc-isolation + sliding window, TTT itself is ~0.003 on top

---

## 8. Data / Tokenizer (lines 432-501)

### Current State
- Sequential shard reading, deterministic streaming
- 1024-token BPE vocabulary
- No data augmentation, no curriculum

### Attack Surfaces

**H1. Larger Vocabulary**
- 2048 or 4096 vocab with corresponding tokenizer
- #122 got 1.1585 with 2048 vocab (but needed to drop to 8 layers)
- Tradeoff: bigger embedding + better tokenization vs fewer layers
- Risk: high — needs new tokenizer, new data shards, careful size budget analysis

**H2. Data Shuffling**
- Current loader reads shards sequentially
- Shuffling shards per epoch could reduce ordering bias
- Likely negligible impact at this scale
- Skip.

---

## Priority Matrix

### Tier 1: High confidence, easy implementation
| ID | Technique | Est. BPB | Lines Changed | Needs Code Edit |
|----|-----------|----------|---------------|-----------------|
| A2 | Muon Weight Decay | 0.003-0.01 | ~5 | Yes |
| D1 | STE Int6 QAT | 0.002 | ~15 | Yes |
| F1 | SWA | 0.002-0.005 | ~25 | Yes |
| B1 | Orthogonal Init | 0.003-0.005 | ~10 | Yes |
| D2 | zstd-22 | ~0.002 (via size savings) | ~3 | Yes |
| F3 | Grad Clip 0.3 | 0.003 | 0 (env var) | No |
| G1 | Stride=64 | 0.002 | 0 (env var) | No |

### Tier 2: High confidence, moderate implementation
| ID | Technique | Est. BPB | Lines Changed | Needs Code Edit |
|----|-----------|----------|---------------|-----------------|
| A1 | NorMuon | 0.005-0.01 | ~10 | Yes |
| C1 | SmearGate | 0.005 | ~20 | Yes |
| C2 | BigramHash | 0.005 | ~40 | Yes |
| C4 | FlashAttention 3 | 0.002 (via speed) | ~15 | Yes |
| C3 | 10th Layer + int5 | 0.005-0.01 | ~30 | Yes |

### Tier 3: Speculative / lower priority
| ID | Technique | Est. BPB | Lines Changed | Needs Code Edit |
|----|-----------|----------|---------------|-----------------|
| F2 | MTP Auxiliary Loss | 0.003 | ~30 | Yes |
| G2 | Doc-Isolated Eval | 0.011 | ~30 | Yes |
| B2 | muP Output Scaling | 0.002 | ~5 | Yes |
| A3 | Adam WD for Scalars | 0.001 | ~2 | Yes |
| E3 | Higher Base LR | 0.002 | 0 (env var) | No |

### Tier Skip: Not worth pursuing
| ID | Technique | Reason |
|----|-----------|--------|
| C6 | SwiGLU | Neutral in community tests |
| H2 | Data Shuffling | Negligible |
| A4 | Backend Steps | Already tested, no win |
| E1 | LR Warmup | Most top submissions don't use it |
| H1 | Larger Vocab | High risk, needs new tokenizer pipeline |

---

## Recommended Implementation Order

```
Phase 1 (env-var only, no code):
  GRAD_CLIP_NORM=0.3
  EVAL_STRIDE=64

Phase 2 (easy code, one-at-a-time, screen each):
  Muon WD (A2)           — 5 lines, optimizer step
  STE Int6 QAT (D1)      — 15 lines, CastedLinear.forward
  SWA (F1)               — 25 lines, training loop
  OrthoInit (B1)          — 10 lines, _init_weights
  zstd-22 (D2)            — 3 lines, serialization

Phase 3 (medium code, screen combinations):
  NorMuon (A1)            — 10 lines, Newton-Schulz function
  SmearGate (C1)          — 20 lines, new module + forward
  BigramHash (C2)         — 40 lines, new module + forward
  FA3 (C4)                — 15 lines, attention forward

Phase 4 (advanced, full decision runs):
  10L + int5/int6 (C3+D3) — 30 lines, architecture + export
  MTP (F2)                — 30 lines, auxiliary head
  Doc-isolated eval (G2)  — 30 lines, eval function

Each phase screens against control before moving to the next.
Total estimated gain if everything composes: ~0.025-0.035 BPB
Target: 1.163 → 1.128-1.138
```

---

## 9. Gaps Identified by Second-Pass Review

The first pass focused on community-proven techniques. This section covers attack surfaces that are real code-level opportunities but were missed in the initial analysis.

### Infrastructure / Throughput

**I1. torch.compile mode (line 1075)**
- Current: `torch.compile(base_model, dynamic=False, fullgraph=True)` — no mode specified (default)
- `mode="max-autotune"` enables CUDA graph capture and autotuning of kernel selection on H100
- `mode="reduce-overhead"` captures CUDA graphs to eliminate kernel launch overhead
- Either could yield 5-15% throughput → 5-15% more training steps in 600s
- Zero-risk change — same model, just faster execution

**I2. Warmup steps waste wallclock (lines 1169-1193)**
- Default `WARMUP_STEPS=20` runs 20 full training steps just to JIT-compile, then throws away the weights and optimizer state
- torch.compile only needs 1-2 traces to JIT, not 20 gradient steps
- Each saved warmup step recovers ~one real training step of wallclock (~55-83ms each)
- Reducing to `WARMUP_STEPS=5` or even `WARMUP_STEPS=1` recovers 15-19 extra training steps
- Env-var change: trivial

**I3. Muon buffer allocation every step (lines 146-147)**
- `updates_flat = torch.zeros(total_params, ...)` allocates a fresh multi-MB bf16 tensor every step
- Pre-allocating once as `self._updates_flat` and calling `.zero_()` each step eliminates per-step allocation overhead and reduces memory fragmentation
- Micro-optimization but compounds across 10K+ steps

**I4. No data prefetching (lines 493-501, 1246)**
- Each training step waits for `next_batch` before computing
- Double-buffering: call `next_batch` for step N+1 during step N's backward+optimizer
- Could hide 0.5-1ms per step → ~5-10s over a full run → ~100 extra steps

### Model Architecture / Numerics

**I5. relu^2 creates quantization-unfriendly weight distributions (lines 625-627)**
- `torch.relu(self.fc(x)).square()` produces quadratically growing activations
- The MLP output projection `self.proj` receives extremely spiky inputs → outlier weight rows → poor quantization
- A soft clamp on the squared activation (`x.square().clamp(max=C)`) would bound dynamic range
- This is *why* MLP weights need int5 in #180 — they quantize worse than attention weights
- Orthogonal to STE QAT, could stack

**I6. RMSNorm has no learnable scale (lines 507-513)**
- Current RMSNorm is just normalization, no affine parameter
- Adding `self.weight = nn.Parameter(torch.ones(dim))` costs 512 params per norm (~9K total for 9 layers, ~18 bytes in int8)
- Standard in LLaMA and most modern architectures
- Gives per-dimension gain control after normalization

**I7. resid_mix[1] initialized to zero — slow to activate (line 647)**
- `torch.stack((torch.ones(dim), torch.zeros(dim)))` means embedding residual starts disconnected
- Gradient through zero gate is weak → slow activation of the x0 shortcut
- Initializing mix[1] to 0.1 gives immediate partial access to embedding residual
- One-line init change

**I8. q_gain is per-head, not per-head-dim (line 587)**
- `torch.full((num_heads,), qk_gain_init)` — one scalar per head
- Per-dimension: shape `(num_heads, head_dim)` = 512 params total
- Could selectively amplify/suppress RoPE frequency components
- Minimal param cost, speculative benefit

### Loss / Regularization

**I9. No label smoothing (line 743)**
- Standard hard-target cross-entropy
- `F.cross_entropy(..., label_smoothing=0.1)` softens targets
- Produces less confident predictions → smaller logit magnitudes → flatter weight distributions → better quantization
- Also acts as regularizer for generalization
- One-argument change

**I10. Logit softcap tanh is expensive (line 738)**
- `self.logit_softcap * torch.tanh(logits / self.logit_softcap)` on full (batch, seq, vocab) tensor
- At vocab=1024, seq=2048, batch=~32: ~67M elements through tanh every forward pass
- `x.clamp(-softcap, softcap)` is mathematically similar but cheaper
- Could use tanh only at eval, clamp at training
- Small throughput gain

### Eval / Budget

**I11. Validation during training pollutes GPU cache (lines 1208-1229)**
- Val eval at `VAL_LOSS_EVERY=1000` runs full validation set through the model
- Time is excluded from wallclock budget, but GPU L2 cache is polluted
- Next few training steps after val eval are slightly slower due to cold caches
- Setting `VAL_LOSS_EVERY=0` for competition runs eliminates this
- Env-var change: trivial

**I12. torch.save metadata overhead (lines 1308-1316)**
- `torch.save` uses pickle + ZIP internally — adds ~10-20KB of metadata per artifact
- At the 16MB boundary, that is ~13-26K int6 parameters wasted on format overhead
- Custom binary format (concatenated tensor bytes + small header) would reclaim this
- Marginal but real

### Summary: Second-Pass Priority

| Priority | Gap | Lines | Mechanism | Effort |
|----------|-----|-------|-----------|--------|
| High | I1: compile mode | 1075 | 5-15% more steps | 1 line |
| High | I2: reduce warmup | 1169 | 15-19 extra steps | env var |
| Medium | I5: relu^2 clamp | 625-627 | better MLP quant | 2 lines |
| Medium | I9: label smoothing | 743 | regularization + quant | 1 arg |
| Low | I6: learnable RMSNorm | 507-513 | standard arch | 5 lines |
| Low | I7: resid_mix init | 647 | faster convergence | 1 line |
| Low | I3: Muon buffer reuse | 146-147 | micro-optimization | 5 lines |
| Low | I4: data prefetching | 493-501 | hide latency | 20 lines |
| Low | I11: disable mid-train val | 1208 | cache coherence | env var |
| Low | I12: custom serialization | 1308 | ~20KB savings | 30 lines |

---

## UPDATE 2026-03-23: Leaderboard Intelligence & New Attack Surfaces

The landscape has shifted dramatically since the original analysis. The merged SOTA moved from 1.1458 to 1.1233, and the open no-TTT frontier is now 1.1171. Our gap has widened from 0.018 to 0.046.

### Leaderboard Snapshot

#### Merged Records
| Rank | Score | Author | Key Techniques |
|------|-------|--------|----------------|
| 1 | 1.1233 | signalrush (PR #414) | 11L, GPTQ-lite, EMA(0.997), XSA4, Partial RoPE 16/64, LN Scale, VE128, Late QAT@0.15, warmdown3500, FA3, SmearGate, BigramHash(2048), MuonWD=0.04, int6+zstd-22 |
| 2 | 1.1248 | jfprincz (PR #315) | 11L, Partial RoPE 16/64, LN Scale, EMA, XSA4 |
| 3 | 1.1271 | jfprincz (PR #287) | 11L, XSA4, EMA, Int6 MLP3x, WD=0.04 |
| 4 | 1.1307 | unnir (PR #265) | 11L, Efficient Partial XSA (3 layers), FA3, SWA120 |
| 5 | 1.1428 | thwu1 (PR #180) | 10L, Mixed int5/int6, BigramHash(10240), SWA(0.4), WD=0.04 |
| 6 | 1.1458 | raahilshah (PR #162) | Int6 MLP3x, SmearGate, BigramHash, OrthoInit, MuonWD, SWA |
| 7 | 1.1502 | aruniyer (PR #86) | 11L, MLP3x, Int6 QAT, zstd-22, WD=0.04, sliding eval |
| 8 | 1.1556 | aquariouseworkman (PR #65) | SmearGate, BigramHash, 3x MLP, int6 STE QAT, sliding eval |
| 9 | 1.1631 | **us** | SOTA stack + batch 524K |

#### Open PRs (frontier, not yet merged)
| Score | PR | Author | Key Techniques | TTT? |
|-------|-----|--------|----------------|------|
| 1.0523 | #573 | Sarimsaljook | 11L XSA4, Multi-Pass Streaming Score-First TTT (3 trajectories) | Yes |
| 1.0698 | #581 | teddyoweh | 11L Sidecar48 + Enhanced TTT (cosine LR, 20 epochs) | Yes |
| 1.0916 | #555 | ymrohit | 11L Shared Sparse Sidecar + EMA + AdamW TTT | Yes |
| 1.1100 | #595 | LoquiAuris | 10L + SWA + Standard TTT | Yes |
| 1.1160 | #557 | hypery11 | 10L + Batched LoRA TTT (non-record) | Yes |
| 1.1164 | #576 | cmcdnd | 33.6M params, Int5 GPTQ, MHA 8/8, MLP 3.5x, XSA-all, LeakyReLU², BigramHash 8192, VE128, Early QAT + 2% pruning, EMA 0.997, TTT + T=0.98 | Yes |
| 1.1171 | #593 | abaybektursun | **Full GPTQ, LeakyReLU(0.5)², Parallel Muon, Parameter Banking, 11L, BigramHash(1536), XSA4, Partial RoPE, LN Scale, VE128, EMA(0.997). NO TTT** | No |
| 1.1175 | #569 | gowtham0992 | 11L VRL, LeakyReLU², Full GPTQ, 2% pruning, QAT clip 0.9995. No TTT | No |
| 1.1178 | #589 | RoyiRa | Late Soft-Round QAT + Score-First Backward-Looking TTT (SGD, cosine LR, 3 epochs) | Yes |
| 1.1208 | #587 | newjordan | XSA on all 11 layers, GPTQ block_size=64 percdamp=0.002 | Minimal |
| 1.1215 | #578 | newjordan | GPTQ + Early QAT + Legal TTT | Yes |
| 1.1354 | #562 | bigbag | 10L, TrigramHash, LeakyReLU(0.5)², VRL, Gated Attention, XSA4, MATRIX_LR=0.03, SWA 27 ckpts, TTT 22ep AdamW cosine | Yes |
| 1.1365 | #586 | EaCognitive | 11L, Hadamard Rotation, VE128, cuDNN SDPA (not FA3) | No |
| 1.1476 | #592 | Skytuhua | 12L Int5-MLP BigramHash10K EMA | No |

---

### NEW High-Signal Techniques

#### N1. GPTQ (Full Hessian) — CRITICAL, HIGHEST PRIORITY
Post-training quantization using Hessian-aware error compensation instead of simple clip-and-round.

Algorithm (based on IST-DASLab/gptq, ICLR 2023):
1. Collect H = X^T X per layer via 256 calibration batches
2. Column reordering by descending Hessian diagonal (actorder)
3. Block-wise Cholesky error compensation: propagate quantization error to remaining columns using H^{-1}
4. Per-row scale via 5-percentile search within the Cholesky framework

**Impact**: Turns quantization from a quality LOSS (~0.015 BPB penalty) into a quality GAIN. PR #593 gets pre-quant 1.1386 → post-GPTQ **1.1171** = +0.0215 BPB improvement from quantization alone.

**Cost**: ~60-85s eval time for Hessian collection + quantization. Well within 10min eval budget.

**Evidence**: PR #593 (1.1171), #569 (1.1175), #576 (1.1164), #587 (1.1208), #578 (1.1215). Every no-TTT submission below 1.1220 uses Full GPTQ.

**Verdict**: THE single highest-impact change available. Our current simple int6 clip loses ~0.02 BPB vs GPTQ. Must implement.

#### N2. GPTQ-lite (Clip Percentile Search) — HIGH
Lightweight alternative to full GPTQ. Try 5 clip percentiles (0.999, 0.9995, 0.9999, 0.99999, 1.0) per weight row, pick the one minimizing reconstruction MSE.

**Impact**: -0.0006 BPB over fixed clip. Zero training cost, minimal eval cost.

**Evidence**: PR #414 (merged SOTA, 1.1233) uses this.

**Verdict**: If full GPTQ is too complex, this is the cheap fallback. But full GPTQ is strictly better.

#### N3. LeakyReLU(0.5)^2 — HIGH, MUST HAVE
Replace `relu(x).square()` with `F.leaky_relu(x, 0.5).square()`. Preserves negative gradient flow.

**Impact**: Consistent -0.003 BPB across multiple submissions.

**Evidence**: PR #493 (first discovery), #569, #593, #576, #562 all use it. Every submission below 1.1200 uses LeakyReLU².

**Implementation**: One-line change in MLP forward.

**Verdict**: Trivial to implement, well-proven. No reason not to include. Supersedes I5 (relu^2 clamp).

#### N4. EMA (Exponential Moving Average) — HIGH
Maintain an exponential moving average of model weights during training (decay=0.997). Use EMA weights for export instead of (or combined with) SWA.

**Impact**: -0.0006 BPB vs SWA alone (from PR #414 ablation).

**Evidence**: PR #414 (merged SOTA), PR #287, #315 all use EMA. EMA is replacing SWA as the default.

**Implementation**: ~15 lines. Shadow copy of state_dict updated every step.

**Verdict**: Small but consistent gain, easy to implement. Standard in top submissions.

#### N5. XSA (Cross-Sequence Attention) — HIGH, MUST HAVE
During attention, concatenate fixed cross-sequence key-value pairs to the KV cache. Gives the model access to broader context.

**Impact**: XSA on last 4 layers: ~-0.005 BPB. XSA on all 11 layers: additional -0.0006 (PR #587).

**Evidence**: Universal in top submissions. Every submission below 1.1300 uses XSA.

**Verdict**: Important technique, moderate implementation complexity (~30 lines in attention forward).

#### N6. Value Residual Learning (VRL) — MEDIUM-HIGH
From ResFormer (arxiv:2410.17897). Save layer 0's value projection output, add it to all subsequent layers' V via learned sigmoid gates.

**Impact**: -0.015 BPB in PR #562 (10L), -0.005 in PR #569 (11L with GPTQ).

**Evidence**: PR #569 (1.1175, no TTT), #562 (1.1354 with TTT).

**Verdict**: Novel and impactful. Moderate implementation (~20 lines).

#### N7. Partial RoPE 16/64 — MEDIUM
Apply RoPE to only 16 of 64 head dimensions. Remaining dims use no positional encoding.

**Impact**: ~-0.002 BPB (from PR #315 ablation).

**Evidence**: PR #315 (1.1248), #414 (1.1233).

#### N8. LN Scale — MEDIUM
Per-layer learned scaling factor for residual connections. Depth-scaled.

**Impact**: ~-0.002 BPB, mainly for 11L.

**Evidence**: PR #315, #414.

#### N9. VE128 (Value Embeddings) — MEDIUM
Separate 128-dim value embedding lookup, projected to model dim. Independent value representation.

**Impact**: ~-0.002 BPB when space budget allows.

**Evidence**: PR #414, #586, #593 all use VE128. GPTQ compression makes this feasible.

#### N10. Parameter Banking + Parallel Muon — MEDIUM
Parameter Banking: 3D weight tensors for weight sharing across layers. Parallel Muon: optimize all banks simultaneously.

**Impact**: Faster step time (83.3ms vs ~88ms) → ~350 extra training steps in 600s.

**Evidence**: PR #399 (original), #593 (1.1171).

**Verdict**: Systems optimization. Complex but impactful.

#### N11. Score-First TTT — HIGH (if pursuing TTT lane)
Split validation into chunks. Score each chunk first (inference_mode), then train on already-scored tokens. Legal per competition rules.

Variants:
- Basic backward-looking: -0.003 BPB (PR #589)
- Multi-pass streaming (3 trajectories, min NLL per token): -0.070 BPB (PR #573)
- Batched LoRA TTT (32 seq/GPU): ~500x faster than chunk-based (PR #562)
- Post-TTT temperature calibration T=0.98: -0.003 BPB free (PR #576)

**Verdict**: TTT is the path to sub-1.10 scores. Multi-pass is strongest.

#### N12. Late Soft-Round QAT — LOW-MEDIUM
Replace hard STE rounding with temperature-controlled soft-round near end of training. Bin-aware gradient signal.

**Impact**: -0.001 to -0.002 BPB over standard STE QAT.

**Evidence**: PR #589 (1.1178).

#### N13. TrigramHash — LOW-MEDIUM
Extend BigramHash to 3-token context. Reuses BigramHash embedding table via different XOR hash. Zero extra parameters.

**Impact**: ~-0.002 BPB on top of BigramHash (from PR #562).

#### N14. 2% Magnitude Pruning — LOW
After quantization, zero out smallest 2% of weights. Improves zstd compressibility.

**Impact**: ~-0.001 BPB (via space savings).

**Evidence**: PR #569 (1.1175).

#### N15. Hadamard Rotation — INFORMATIONAL ONLY
Walsh-Hadamard rotation before quantization for more uniform weight distributions.

PR #586 found: "Hadamard rotation and GPTQ are substitutes, not complements at int6." If using GPTQ, skip this.

#### N16. Dead QAT Code Bug — CAUTION
PR #586 discovered: "Late QAT (STE fake-quantization) was dead code in all prior work due to instance vs class attribute shadowing." Removing the dead QAT guard yields 7% throughput gain.

If we implement QAT, must verify it's actually active. Check for Python attribute shadowing.

---

### REVISED Priority Matrix (2026-03-23)

#### Tier 0: Critical (>0.005 BPB each, universal in frontier)
| ID | Technique | Est. BPB | Notes |
|----|-----------|----------|-------|
| D6/N1 | GPTQ (Full Hessian) | +0.020 | Every no-TTT sub below 1.12 |
| C7/N5 | XSA4 | +0.005 | Every sub below 1.13 |
| N3 | LeakyReLU(0.5)² | +0.003 | Every sub below 1.12 |
| C3 | 11 Layers | +0.005-0.010 | Standard everywhere |

#### Tier 1: High confidence (0.002-0.005 BPB each)
| ID | Technique | Est. BPB | Notes |
|----|-----------|----------|-------|
| A2 | Muon WD=0.04 | +0.005 | Universal |
| N4 | EMA(0.997) | +0.003 | Replacing SWA |
| N7 | Partial RoPE 16/64 | +0.002 | In merged SOTA |
| N8 | LN Scale | +0.002 | In merged SOTA |
| N9 | VE128 | +0.002 | In merged SOTA, needs GPTQ headroom |
| D2 | zstd-22 | +0.002 | Universal |
| D4 | fp16 Embedding | +0.002 | Universal |
| G1 | Stride=64 eval | +0.002 | Universal |
| E4 | Warmdown 3500 | +0.002 | Merged SOTA |

#### Tier 2: Medium confidence
| ID | Technique | Est. BPB | Notes |
|----|-----------|----------|-------|
| N6 | VRL | +0.005 | Novel, 2 submissions |
| C1 | SmearGate | +0.003 | Mid-tier, not in very top |
| C2 | BigramHash | +0.003 | In merged SOTA but not critical |
| B1 | Orthogonal Init | +0.003 | Some top submissions |
| F3 | Grad Clip 0.3 | +0.002 | Valid |
| E3 | MATRIX_LR=0.03 | +0.002 | PR #562 |
| N13 | TrigramHash | +0.002 | One submission, zero extra params |
| N14 | 2% Pruning | +0.001 | Trivial |

#### Tier 3: TTT Lane (separate eval-time budget)
| ID | Technique | Est. BPB | Eval Time |
|----|-----------|----------|-----------|
| N11a | Score-First Backward-Looking TTT | +0.003-0.007 | ~300s |
| N11b | Multi-Pass Streaming TTT (3 traj) | +0.070 | ~500s |
| N11c | Post-TTT Temperature T=0.98 | +0.003 | Free |
| N11d | Batched LoRA TTT (32 seq/GPU) | +0.005 | ~400s |

#### Tier Skip (updated)
| ID | Technique | Reason |
|----|-----------|--------|
| A1 | NorMuon | Top submissions don't use it |
| B2 | muP Output Scaling | Not used by any top sub |
| C6 | SwiGLU | Neutral/negative |
| N15 | Hadamard Rotation | Substitute for GPTQ, not complement |
| I9 | Label Smoothing | Not used by top subs |
| F2 | MTP Auxiliary Loss | Not used by top subs |
| C4 | FlashAttention 3 | PR #586: worse compression than cuDNN SDPA |

---

### Revised Implementation Order (2026-03-23)

The no-TTT frontier stack (PR #593, 1.1171) looks like:
```
11L, 512d, 8H/4KV
LeakyReLU(0.5)² MLP 3x
BigramHash(1536), XSA4, Partial RoPE, LN Scale, VE128
EMA(0.997) + Tight SWA
MuonWD=0.04, Parallel Muon
Full Hessian GPTQ int6 + lzma
Sliding eval stride=64
```

#### Phase 1: Foundation (env-var + trivial patches)
```
NUM_LAYERS=11
MUON_WD=0.04
GRAD_CLIP_NORM=0.3
EVAL_STRIDE=64
WARMDOWN_ITERS=3500
zstd-22
fp16 embedding passthrough
```

#### Phase 2: High-impact code patches (Tier 0)
```
LeakyReLU(0.5)²          — 1 line in MLP forward
EMA(0.997)                — 15 lines, shadow state_dict
XSA4                      — ~30 lines, KV cache extension in attention
GPTQ (full Hessian)       — ~100 lines, post-training quantization
```

#### Phase 3: Architecture refinements (Tier 1)
```
Partial RoPE 16/64        — ~15 lines, modify rotary embedding
LN Scale                  — ~10 lines, per-layer residual scaling
VE128                     — ~20 lines, separate value embedding
VRL                       — ~20 lines, layer-0 V residual
```

#### Phase 4: TTT (if pursuing sub-1.10)
```
Score-First TTT           — ~80 lines, chunk-wise eval→train
Multi-Pass (3 traj)       — ~40 lines on top of TTT
Post-TTT T=0.98           — 1 line
```

#### Estimated composite (no TTT):
```
Our current:     1.1631
+ Phase 1:       ~1.140  (WD, 11L, grad clip, stride, zstd, warmdown)
+ Phase 2:       ~1.118  (LeakyReLU², EMA, XSA4, GPTQ)
+ Phase 3:       ~1.112  (Partial RoPE, LN Scale, VE128, VRL)
```

#### Estimated composite (with TTT):
```
After Phase 3:   ~1.112
+ Phase 4:       ~1.050  (multi-pass streaming TTT + T=0.98)
```

---

### Key Strategic Observations (2026-03-23)

1. **GPTQ is the single biggest unlock.** Worth +0.020 BPB — more than all other individual techniques combined. Every no-TTT submission below 1.1220 uses it. Must be first priority.

2. **The merged SOTA stack (PR #414) is the template.** Combines 11L + XSA4 + Partial RoPE + LN Scale + VE128 + EMA + GPTQ-lite + Late QAT + MuonWD + zstd-22. We should adopt wholesale rather than building incrementally.

3. **LeakyReLU² is a free lunch.** One-line change, -0.003 BPB, universal in frontier. Supersedes our I5 (relu^2 clamp) analysis.

4. **TTT is a separate game.** Gap between best no-TTT (1.1171) and best TTT (1.0523) is 0.065 BPB — enormous. Multi-pass streaming score-first is the strongest variant.

5. **SmearGate and BigramHash are fading from the very top.** Present in merged SOTA (PR #414) but the no-TTT frontier (PR #593, #569) succeeds without heavy reliance on them. XSA + VRL + GPTQ carry more weight.

6. **FA3 may be a trap.** PR #586 found FA3 compresses 1.8% worse than cuDNN SDPA. Also found that "removing dead QAT guard from CastedLinear.forward() yields 7% throughput gain" — cuDNN SDPA + no dead code may beat FA3.

7. **Parameter Banking + Parallel Muon is an orthogonal systems win.** Reduces step time ~5ms → ~350 extra training steps. Used by PR #593.

---

## APPENDIX: Code-Level Intelligence from Merged Leaderboard (2026-03-23)

Extracted from actual train_gpt.py files in each PR's records folder. These are EXACT implementations, not summaries from PR descriptions.

### Common Hyperparameter Shifts (all top submissions vs our baseline)

| Param | Our Baseline | Top Submissions | Notes |
|-------|-------------|-----------------|-------|
| NUM_LAYERS | 9 | 11 | Universal above rank 5 |
| MLP_MULT | 2 (int) | 3.0 (float) | Float allows fractional expansion |
| TRAIN_SEQ_LEN | 1024 | 2048 | 2x longer context |
| TRAIN_BATCH_TOKENS | 524,288 | 786,432 | +50% batch |
| MATRIX_LR | 0.04 | 0.02-0.025 | Lower LR with WD |
| SCALAR_LR | 0.04 | 0.02-0.025 | Lower LR with WD |
| TIED_EMBED_LR | 0.05 | 0.03-0.035 | Lower LR |
| MUON_MOMENTUM | 0.95 | 0.99 | Higher momentum |
| MUON_MOMENTUM_WARMUP_START | 0.85 | 0.92 | Higher start |
| MUON_MOMENTUM_WARMUP_STEPS | 500 | 1500 | 3x longer warmup |
| GRAD_CLIP_NORM | 0.0 | 0.3 | Enable clipping |
| WARMDOWN_ITERS | 1200 | 3000-3500 | Much longer warmdown |
| VAL_LOSS_EVERY | 1000 | 4000 | Reduce validation overhead |
| EVAL_STRIDE | N/A | 64 | Sliding window eval |

### Exact Code: SmearGate (from PR #162, lines 572-581)

```python
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev
```
- 512 params. Init to zeros (sigmoid(0)=0.5, starts as 50/50 blend).
- Applied after tok_emb + bigram + RMSNorm, before first transformer block.

### Exact Code: BigramHashEmbedding (from PR #162, lines 584-608)

```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod  # first position → boundary bucket
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```
- Default: vocab=4096 (PR #162) or 10240 (PR #180, 2048 in PR #414), dim=128, proj to 512.
- Params: embed (V×128) + proj (128×512) + scale (1). ~590K for V=4096.
- All zero-initialized. Scale=0.05 initially.
- Added to x after tok_emb, before RMSNorm.

### Exact Code: XSA / Cross-Sequence Attention (from PR #265, lines 625-636)

```python
def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
    """Subtract self-value projection via GQA-aware reshape."""
    B, T, H, D = y.shape
    Hkv = v.size(-2)
    group = H // Hkv
    y_g = y.reshape(B, T, Hkv, group, D)
    vn = F.normalize(v, dim=-1).unsqueeze(-2)  # [B, T, Hkv, 1, D]
    proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
    return (y_g - proj).reshape(B, T, H, D)
```
- Math: `y_out = y - (y · v̂) × v̂` where v̂ is L2-normalized value.
- Removes the component of attention output that is aligned with the value.
- Applied post-attention, only on last N layers (default XSA_LAST_N=4).
- Activation: `self.use_xsa = True` set by GPT.__init__ for deep layers.

### Exact Code: Partial RoPE (from PR #315, lines 596-606)

```python
def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    rd = cos.size(-1) * 2  # rope_dims
    if rd < x.size(-1):  # Partial RoPE
        x_rope, x_pass = x[..., :rd], x[..., rd:]
        half = rd // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rot, x_pass), dim=-1)
    # Full RoPE fallback
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
```
- Default ROPE_DIMS=16 (of 64 head dims). Only 16 dims get positional encoding.
- Remaining 48 dims pass through unchanged → act as position-agnostic features.

### Exact Code: LN Scale (from PR #315, line 739)

```python
# In Block.__init__:
self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

# In Block.forward:
attn_out = self.attn(self.attn_norm(x) * self.ln_scale_factor)
x = x + self.attn_scale * attn_out
x = x + self.mlp_scale * self.mlp(self.mlp_norm(x) * self.ln_scale_factor)
```
- Layer 0: scale=1.0, Layer 1: 0.707, Layer 5: 0.408, Layer 10: 0.302.
- Prevents deep layers from dominating. Applied to norm output before attn/mlp.

### Exact Code: EMA (from PR #287/315, lines ~1340-1465)

```python
# Init:
ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
ema_decay = 0.997

# Update every training step:
with torch.no_grad():
    for name, t in base_model.state_dict().items():
        ema_state[name].mul_(ema_decay).add_(t.detach().float(), alpha=1.0 - ema_decay)

# Apply at end of training (before quantization):
avg_state = {name: t.to(dtype=base_model.state_dict()[name].dtype) for name, t in ema_state.items()}
base_model.load_state_dict(avg_state, strict=True)
```
- Float32 shadow copy updated every step. Applied before export.
- Decay=0.997 means ~333-step effective window.

### Exact Code: Late QAT (from PR #315/414)

```python
# In CastedLinear:
class CastedLinear(nn.Linear):
    _qat_enabled: bool = False
    def forward(self, x):
        w = self.weight.to(x.dtype)
        if CastedLinear._qat_enabled and self.training and w.ndim == 2:
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / 31.0).clamp_min(1.0 / 31.0)
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]), -32, 31) * scale[:, None]).to(x.dtype)
            w = w + (w_q - w).detach()  # STE
        return F.linear(x, w, self.bias)

# Activation trigger in training loop:
if args.late_qat_threshold > 0 and scale < args.late_qat_threshold and not CastedLinear._qat_enabled:
    CastedLinear._qat_enabled = True
```
- Class-level flag `_qat_enabled` toggles globally for all CastedLinear instances.
- Activates when LR scale drops below threshold (default 0.15 in PR #414, 0.1 in PR #315).
- CAUTION: PR #586 found this can be dead code if instance attribute shadows class attribute.

### Exact Code: Muon Weight Decay (from PR #162, lines 162-167)

```python
wd = group.get("weight_decay", 0.0)
curr = 0
for p in params:
    g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
    if wd > 0:
        p.data.mul_(1.0 - lr * wd)  # Decoupled WD BEFORE update
    p.add_(g, alpha=-lr)
    curr += p.numel()
```
- Decoupled (AdamW-style): multiply weights by `(1 - lr * wd)` before the Muon update.
- Default WD=0.04 in all top submissions.

### Exact Code: SWA (from PR #162/265, lines ~1107-1149)

```python
# Collection (during warmdown when scale < start_frac):
if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
    if swa_state is None:
        swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
        swa_count = 1
    else:
        for name, t in base_model.state_dict().items():
            swa_state[name] += t.detach().cpu()
        swa_count += 1

# Application:
avg_state = {name: (tensor / swa_count).to(dtype=current_state[name].dtype)
             for name, tensor in swa_state.items()}
base_model.load_state_dict(avg_state, strict=True)
```
- Accumulates on CPU. Arithmetic mean of checkpoints.
- PR #162: swa_start_frac=0.5, swa_every=50.
- PR #180: swa_start_frac=0.4, swa_every=50.
- PR #265: swa_every=200 (but runtime likely SWA_EVERY=10 for 120 ckpts).
- EMA is replacing SWA in newer submissions (PR #287+).

### Exact Code: Mixed int5/int6 Quantization (from PR #180, lines 334-376)

```python
def quantize_intN_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        row_max = t32.abs().amax(dim=1)
        scale = (row_max / clip_range).clamp_min(1e-12).to(torch.float16)
        q = torch.clamp(torch.round(t32 / scale.float()[:, None]), -(clip_range+1), clip_range).to(torch.int8)
        return q, scale
    # ... 1D fallback

# In mixed_quantize_int6:
clip = 15 if cat == "mlp" else 31  # int5 for MLP, int6 for attention
q, s = quantize_intN_per_row(t, clip_range=clip)
```
- Int5: clip_range=15, values in [-16, 15]. Better zstd compression (3 zero MSBs per byte).
- Int6: clip_range=31, values in [-32, 31]. Standard for attention.
- Saves ~1.86MB → funds 10th/11th layer.

### Exact Code: Full GPTQ (from PR #569, lines 213-280)

```python
def quantize_int6_gptq(weight, hessian=None, clip_range=31, block_size=128):
    t32 = weight.float()
    if t32.ndim != 2 or hessian is None:
        return _quantize_int6_percentile(t32, clip_range)
    rows, cols = t32.shape
    H = hessian.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    damp = 0.01 * torch.mean(torch.diag(H))
    H[torch.arange(cols), torch.arange(cols)] += damp

    # ActOrder: sort columns by descending Hessian diagonal
    perm = torch.argsort(torch.diag(H), descending=True)
    inv_perm = torch.argsort(perm)
    W = t32[:, perm].clone()
    H = H[perm][:, perm]

    # Cholesky inverse
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    # Block-wise quantization with error compensation
    best_q, best_scale, best_err = None, None, float('inf')
    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:
        # Per-row scale via percentile search
        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        sf = s.float()
        Q = torch.zeros_like(W, dtype=torch.int8)
        W_work = W.clone()
        for i1 in range(0, cols, block_size):
            i2 = min(i1 + block_size, cols)
            W1 = W_work[:, i1:i2].clone()
            Q1 = torch.zeros(rows, i2-i1, dtype=torch.int8)
            Err1 = torch.zeros(rows, i2-i1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            for i in range(i2-i1):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)
                Q1[:, i] = q
                err = (w - q.float() * sf) / d
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err
            Q[:, i1:i2] = Q1
            if i2 < cols:
                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
        recon = Q.float() * sf[:, None]
        mse = (W - recon).pow(2).mean().item()
        if mse < best_err:
            best_q, best_scale, best_err = Q, s, mse
    best_q = best_q[:, inv_perm]  # Undo column permutation
    return best_q, best_scale
```

Hessian collection (from PR #569, lines 683-722):
```python
def collect_hessians(base_model, train_loader, args, device, grad_accum_steps, num_batches=256):
    hessians = {}
    hooks = []
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            param_name = name + ".weight"
            cols = module.weight.shape[1]
            hessians[param_name] = torch.zeros(cols, cols, dtype=torch.float32, device='cpu')
            def make_hook(pname):
                def hook_fn(module, input, output):
                    x = input[0].detach().float()
                    if x.ndim == 3:
                        x = x.reshape(-1, x.shape[-1])
                    hessians[pname] += (x.T @ x).cpu()
                return hook_fn
            h = module.register_forward_hook(make_hook(param_name))
            hooks.append(h)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(num_batches):
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            base_model(x, y)
    for h in hooks: h.remove()
    for name in hessians:
        H = hessians[name]
        H /= num_batches
        damp = 0.01 * torch.diag(H).mean().clamp_min(1e-6)
        H += damp * torch.eye(H.shape[0])
    return hessians
```
- Forward hooks on CastedLinear to collect H = X^T X on CPU.
- 256 calibration batches. Damping: 0.01 * mean(diag(H)).
- GPTQ runs AFTER EMA is applied, so it quantizes the smoothed weights.

### Exact Code: VRL / Value Residual Learning (from PR #569, lines 599-665)

```python
# Init:
self.vrl_enabled = num_layers > 1
if self.vrl_enabled:
    self.vrl_alphas = nn.ParameterList([
        nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for _ in range(num_layers - 1)
    ])

# Forward: compute layer-0's raw V output once
v0_raw = None
if self.vrl_enabled:
    blk0 = self.blocks[0]
    mix0 = blk0.resid_mix.to(dtype=x0.dtype)
    x_in0 = mix0[0][None, None, :] * x0 + mix0[1][None, None, :] * x0
    v0_raw = blk0.attn.c_v(blk0.attn_norm(x_in0) * blk0.ln_scale_factor)

# In each subsequent layer:
if i > 0 and v0_raw is not None:
    alpha = torch.sigmoid(self.vrl_alphas[vrl_idx].to(dtype=x.dtype))
    v_res = alpha * v0_raw

# In attention:
v = self.c_v(x)
if v_residual is not None:
    v = v + v_residual  # Add layer-0's V, gated by learned alpha
```
- num_layers-1 scalar params (initialized to 0.0 → sigmoid=0.5).
- Layer 0's V projection output is computed once, then mixed into all subsequent layers.
- Zero throughput cost (one extra c_v call + addition per layer).

### Exact Code: LeakyReLU² (from PR #569, line 499)

```python
def forward(self, x):
    return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())
```
- One-line change: `F.relu(...)` → `F.leaky_relu(..., negative_slope=0.5)`.
- Preserves negative gradient flow through MLP.

### Exact Code: Orthogonal Init (from PR #162, lines 673-685)

```python
def _init_weights(self) -> None:
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
    num_layers = len(self.blocks)
    for name, module in self.named_modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
                if ".proj." in name or name.endswith(".proj"):
                    with torch.no_grad():
                        module.weight.mul_(1.0 / math.sqrt(2 * num_layers))
```
- Orthogonal init for all 2D linear weights with both dims ≥ 64.
- Output projections (.proj) additionally scaled by 1/sqrt(2*num_layers).

### Exact Code: Value Embeddings (from PR #414, lines 562-577)

```python
class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```
- Separate embedding table for attention values (VE_DIM=128, projected to model_dim).
- Applied only at selected layers (default VE_LAYERS="9,10").
- Reinjected into v before attention: `v = self.c_v(x) + v_embed`.

### Exact Code: Parallel Muon with Banking (from PR #593)

PR #593 is the most complex implementation. Key differences from standard Muon:

1. **Parameter Banking**: All layer weights stored as 3D tensors:
   ```python
   self.qo_bank = nn.Parameter(torch.empty(2*num_layers, model_dim, model_dim))
   self.kv_bank = nn.Parameter(torch.empty(2*num_layers, kv_dim, model_dim))
   self.mlp_up_bank = nn.Parameter(torch.empty(num_layers, mlp_dim, model_dim))
   self.mlp_down_bank = nn.Parameter(torch.empty(num_layers, model_dim, mlp_dim))
   ```
   Each block gets weight slices: `qo_bank[i]` = Q weights for layer i.

2. **Batched Newton-Schulz**: Supports (B,M,N) input for parallel bank processing.

3. **Three-phase pipeline**:
   - Phase 1: `launch_reduce_scatters()` — async RS for all banks immediately after backward
   - Phase 2: Local NS5 on shards + async all-gather (overlaps with next bank)
   - Phase 3: Wait for all-gather, apply updates

   This overlaps communication with computation → 83.3ms/step vs ~88ms baseline.

### Exact Code: NTK-aware RoPE (from PR #265/315)

```python
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        rd = self.rope_dims
        inv_freq = 1.0 / (base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))

    def forward(self, seq_len, device, dtype):
        rd = self.rope_dims
        if seq_len > self.train_seq_len:
            scale = seq_len / self.train_seq_len
            new_base = self.base * (scale ** (rd / (rd - 2)))  # NTK-by-parts
            inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, device=device) / rd))
        else:
            inv_freq = self.inv_freq.to(device)
```
- NTK scaling: `new_base = base × (seq_len/train_seq_len)^(dim/(dim-2))`.
- Automatically extends positional encoding for eval_seq_len > train_seq_len.
- Combined with partial RoPE (ROPE_DIMS=16): only 16 of 64 dims get this encoding.

### Exact Code: Multi-Pass Streaming TTT (from PR #573, lines 1388-1480)

```python
ttt_passes = 3  # default
ttt_lr = 0.0005

# Pass 0: inference only (score-first)
# Pass 1-2: score then train with shifted data orderings

for pass_idx in range(ttt_passes):
    # Circular shift for trajectory diversity
    shift = (pass_idx * len(positions)) // ttt_passes
    positions = positions[shift:] + positions[:shift]

    # Score first (inference_mode)
    with torch.inference_mode():
        logits = eval_model.forward_logits(x)
        nll = F.cross_entropy(logits.reshape(-1, V).float(), y.reshape(-1), reduction="none")
    best_nll[indices] = torch.minimum(best_nll[indices], nll)  # min across passes

    # Train after scoring (only pass > 0)
    if pass_idx > 0:
        cos_mul = 0.5 * (1 + math.cos(math.pi * batch_idx / total_batches))
        for g in ttt_opt.param_groups:
            g["lr"] = g["initial_lr"] * cos_mul
        loss = eval_model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(eval_model.parameters(), 1.0)
        ttt_opt.step()
```
- Per-layer LR groups: MLP.proj 3x, MLP.fc 0.5x, other 1x.
- AdamW optimizer, cosine decay per batch, grad clip 1.0.
- min(NLL) across all 3 passes per token position → best of 3 adaptation trajectories.
- Total eval time: ~500s (within 10min eval budget).

### Key Corrections to Earlier Analysis

1. **SmearGate init**: Gate initialized to ZEROS (not ones). sigmoid(0)=0.5, so starts as 50/50 blend, not disabled.

2. **VE128 NOT in PR #315**: Despite PR description mentioning it, code analysis confirms Value Embeddings are NOT present in PR #315. First appears in PR #414.

3. **LN Scale formula**: `1/sqrt(layer_idx + 1)`, NOT `1/sqrt(num_layers)`. Layer 0 gets 1.0, layer 10 gets 0.302.

4. **GPTQ runs AFTER EMA**: The Hessian collection and GPTQ quantization happen after EMA weights are applied. This means GPTQ is quantizing the smoothed (EMA) weights, not the raw training weights.

5. **SWA vs EMA evolution**: PR #162/#180/#265 use SWA (checkpoint averaging). PR #287+ use EMA (exponential moving average). PR #414 uses both EMA + "Tight SWA" (SWA during final warmdown only).

6. **Parameter Banking is PR #593 only**: Other submissions (including merged SOTA #414) use standard per-layer CastedLinear modules. Banking is a PR #593/Parallel Muon innovation.

7. **FA3 layout change**: FA3 requires [B, T, H, D] tensor layout (vs baseline's [B, H, T, D]). This changes the reshape/transpose in attention forward.

---

## UPDATE 2026-04-06: Competition Has Accelerated Dramatically

The frontier has moved far beyond our last analysis. Merged SOTA is now **1.1147** (PR #1019), and open frontier is **1.079 BPB** (no SLOT) / **0.709 BPB** (with SLOT). Our gap is now **0.048 BPB** to merged SOTA.

### Current Merged Leaderboard (as of 2026-04-06)

| Rank | Score | PR | Key Techniques |
|------|-------|----|----------------|
| 1 | 1.1147 | #1019 | AR Self-Gen GPTQ, XSA-all 11 layers, BigramHash 3072×112, Parallel Muon, LeakyReLU², EMA, no TTT |
| 2 | 1.1194 | #549 | LeakyReLU², Legal Score-First TTT, Parallel Muon |
| 3 | 1.1233 | #414 | 11L, GPTQ-lite, EMA, XSA4, Partial RoPE, LN Scale, VE128, Late QAT |
| ... | ... | ... | ... |
| ~20 | 1.1631 | ours | SOTA stack + batch 524K |

### Open Frontier PRs (2026-04-06)

| Score | PR | Key Techniques | TTT? | SLOT? |
|-------|-----|----------------|------|-------|
| 0.7094 | #1376 | SLOT-24 + Pre-quant TTT | Yes | Yes |
| 1.0791 | #1423 | SP8192, Pre-Quant TTT, QK-Gain 5.0, Depth Recurrence, MuonEq-R | Yes | No |
| 1.0795 | #1416 | SP8192, Pre-Quant TTT on PR #1394 base | Yes | No |
| 1.0800 | #1408 | dTTT 10ep, BigramHash 3072×112, GPTQ, XSA-all, QK-Gain 5.0 | Yes | No |
| 1.0801 | #1420 | Triple Loop (17 virtual layers), Fused MLP Kernels, Parallel Residuals, N-gram Tilt | Yes | No |
| 1.0856 | #1394 | SP8192, GPTQ Embeddings, Depth Recurrence, MuonEq-R, SDClip, Brotli. **No TTT** | No | No |
| 1.0898 | #1399 | Pre-Quant TTT + ETLB (Eval-Time Logit Bias) | Yes | No |
| 1.0913 | #1415 | SP4096, 3-Layer Recurrence, GPTQ Embeddings, SDClip, ETLB. No TTT | No | No |
| 1.0925 | #1421 | Depth Recurrence + EMA Tuning (0.9965) | Yes | No |
| 1.1020 | #1392 | SP4096, Depth Recurrence, Parallel Residuals, QK-Gain, Brotli. No TTT | No | No |
| 1.1158 | #1410 | LatentMask TTT, GPTQ, Product-Key Bigram, Brotli | Yes | No |

### NEW Techniques (high-signal, 2026-04-06)

#### R1. Depth Recurrence — CRITICAL, NOW UNIVERSAL IN SUB-1.10

Loop a subset of transformer layers multiple times, sharing parameters across iterations. Creates "virtual depth" from fewer physical parameters.

**How it works** (from PR #1394, #1420):
- Layers 4-5 are designated as the "recurrence core"
- During forward pass, these layers are executed 2-3 times (NUM_LOOPS=2 or 3)
- With 11 physical layers + NUM_LOOPS=2 on layers 4-5: encoder becomes `[0,1,2,3,4,5,4,5]`, decoder `[4,5,6,7,8,9,10]` = 15 virtual layers
- With NUM_LOOPS=3 (PR #1420): `[0,1,2,3,4,5,4,5,4,5]` = 17 virtual layers
- Activation threshold: enable looping after 35-50% of training (to allow warmup without recurrence)

**Impact**: ~-0.015 BPB from depth recurrence alone.

**Trade-off**: Each loop iteration costs throughput (~-200 steps in 600s per extra loop). Triple loop (17 virtual layers) is optimal; quadruple (19 virtual) loses too many steps.

**Quantization interaction**: Shared weights are quantized once, but quantization error compounds through N repeats. Solutions: Noisy QAT (PR #363), GPTQ with careful calibration.

**Evidence**: Universal in every no-TTT submission below 1.10 (PR #1394, #1392, #1415, #1420).

#### R2. SP4096/SP8192 Tokenizers — HIGH

Larger vocabularies than SP1024. Better per-byte compression = lower BPB at same perplexity.

**SP4096**: Allows MLP 4x (wider MLPs). ~3.32 bytes/token avg. Used by PR #1392, #1415.
**SP8192**: Even more tokens per byte. Used by PR #1394, #1416, #1423. Current no-TTT frontier (1.086).

**BPB is tokenizer-agnostic**: Total byte count of validation set is identical regardless of tokenizer. More tokens (SP1024) or fewer tokens (SP4096/8192) — bytes sum is the same.

**Trade-off**: Larger embedding table. SP8192 embedding is 8192×512 = 4M params. GPTQ on embeddings (int8) makes this fit.

#### R3. Pre-Quant TTT — HIGH (for TTT lane)

Adapt full-precision EMA model on validation data BEFORE GPTQ quantization. Adapted weights are baked into the artifact — zero eval-time overhead.

**From PR #1416**: TTT gives -0.034 BPB on SP8192 base (post-EMA 1.1019 → post-TTT 1.0682 → post-GPTQ+sliding 1.0795).

**Key details**:
- 6 epochs AdamW on EMA model
- Freeze first 2 blocks, adapt last 9
- Cosine LR decay from 0.0005
- Score-first compliant (each chunk scored before training)
- Time: ~112s within 10min eval budget

**vs traditional TTT**: Pre-quant TTT bakes adaptation into the model permanently. No eval-time adaptation needed. Cleaner separation.

#### R4. SDClip (Standard Deviation Clip) — MEDIUM-HIGH

Choose quantization clip threshold based on row standard deviation instead of percentile search.

**From PR #1394**: Instead of trying 5 clip percentiles and picking lowest MSE, directly use `clip = K * std(row)` where K is tuned. Interacts better with compression because it produces a more predictable quantized value distribution.

**Key insight from PR #1394**: "An int5 quantized network can actually compress smaller than an int4 one if the int5 quantization uses a much wider clip range." The effectiveness of brotli depends on entropy of quantized values, and SDClip optimizes this entropy.

#### R5. GPTQ on Embeddings — MEDIUM

Apply GPTQ (int8) to the embedding matrix too, not just weight matrices. Frees ~0.5-1MB of artifact space.

**From PR #1394**: Standard approach keeps embeddings in fp16. GPTQ int8 on embeddings saves space with minimal quality loss because embeddings are already low-rank.

#### R6. MuonEq-R (Row-Normalized Muon) — MEDIUM

From PR #1217/@bigbag. Row normalization in Muon optimizer. Different from the original NorMuon — applies normalization to equalize row norms of the update.

**Evidence**: Used in PR #1394 (1.086), #1415 (1.091), #1423 (1.079). Universal in current top stack.

#### R7. QK-Gain 5.0 — LOW-MEDIUM

Increase q_gain initialization from default (varies) to 5.0. One env var change.

**From PR #1217**: Validated by @bigbag. +0.0004 BPB improvement (PR #1423 over #1416).

**Evidence**: Used in PR #1413, #1415, #1420, #1423.

#### R8. Parallel Residuals — MEDIUM

GPT-J style: attention and MLP both read from the same pre-residual input, outputs summed in parallel.

**From PR #1420**: Faster forward pass (+68 training steps). Also helps quantization (less interference between attention and MLP during GPTQ calibration).

**Applied to deep layers only** (layers 7-10 in PR #1420).

#### R9. Brotli Compression — LOW-MEDIUM

Replacing lzma/zstd. Better compression ratio on quantized weight data in some configurations.

**From PR #1392, #1394, #1410**: `brotli` level 11 used by several top submissions.

#### R10. ETLB (Eval-Time Logit Bias) — LOW

Optimize a bias vector `b ∈ R^vocab` added to output logits during sliding window evaluation. ~-0.002 BPB.

**From PR #1399**: Document-level token frequency patterns. Zero-overhead at training time.

#### R11. Fused MLP Kernels — LOW (high complexity)

Triton TMA forward + CUTLASS EVT backward. ~10% throughput gain → +127 training steps.

**From PR #1420**: Fuses `leaky_relu(fc(x), 0.5).square()` into single Triton kernel, eliminates 403MB intermediate from HBM. Backward fuses `(grad_out @ proj.weight) * act_grad` into CUTLASS epilogue.

**Verdict**: High engineering effort. Only worthwhile at the very top of the leaderboard.

#### R12. Product-Key Bigram — LOW-MEDIUM

Factored `embed_prev(1024,512) * embed_cur(1024,512)` — zero hash collisions, no projection layer.

**From PR #1410**: Cleaner than hash-based BigramHash. More parameter-efficient.

#### R13. SLOT (Sparse Latent Optimization at Test-time) — SEPARATE TRACK

Per-sample delta + logit bias, 24 AdamW steps per sample during eval. Gets 0.7094 BPB.

**From PR #1376**: Per-sample `delta [bsz,1,512]` + `logit_bias [bsz,1,1024]`, 24 AdamW steps, stride=96.

**Verdict**: Fundamentally different approach. Massive improvement but requires significant eval-time compute. Competition may split into SLOT vs non-SLOT tracks.

### REVISED Leaderboard Targets (2026-04-06)

```
Our current:        1.1631

Merged SOTA:        1.1147  (PR #1019)
Open no-TTT:        1.0856  (PR #1394)
Open with TTT:      1.0795  (PR #1416)
Open with SLOT:     0.7094  (PR #1376)

Gap to merged SOTA: 0.048
Gap to no-TTT:      0.078
```

### REVISED Priority Stack (2026-04-06)

The current winning no-TTT stack (PR #1394, 1.086) looks like:
```
SP8192 vocab (not SP1024!)
11L, 512d, 8H/4KV, MLP 4x
Depth Recurrence (loop layers 4-5 twice → 15 virtual layers)
MuonEq-R (row-normalized Muon)
LeakyReLU(0.5)²
XSA on all 11 layers
BigramHash 3072×112
QK-Gain 5.0
EMA(0.997)
Sigmoid-gated U-Net skips
SDClip quantization
GPTQ on embeddings (int8)
Full Hessian GPTQ int6 (AR self-gen calibration)
Brotli compression
Sliding eval stride=64
```

With Pre-Quant TTT added (PR #1416): 1.086 → 1.080.

### What We Must Do

1. **Adopt PR #1394 as template** — not PR #414 (which is now 2 generations behind).
2. **Depth Recurrence is the biggest unlock we're missing** — loop layers 4-5, giving 15 virtual layers from 11 physical.
3. **SP4096 or SP8192 tokenizer** — SP1024 is no longer competitive. This is a fundamental limitation.
4. **SDClip replaces percentile-based quantization** — better compression-aware clipping.
5. **GPTQ on embeddings** — free space savings.
6. **MuonEq-R** — replaces plain Muon.
7. **Pre-Quant TTT** (if pursuing TTT lane) — bakes adaptation into artifact.

### Key Strategic Shift

The competition has moved into a fundamentally different regime:
- **Tokenizer matters now**: SP8192 > SP4096 > SP1024. Our SP1024 baseline is ~0.02-0.03 BPB behind just from tokenizer choice.
- **Virtual depth via recurrence**: 15-17 virtual layers from 11 physical. This is ~-0.015 BPB.
- **Compression-aware quantization**: SDClip + Brotli > percentile + zstd/lzma.
- **GPTQ everywhere**: Not just weights — embeddings too.
- **Pre-Quant TTT**: The cleanest form of TTT — baked into artifact, no eval-time overhead.
