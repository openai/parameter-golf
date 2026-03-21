# Mamba/SSM Research for Parameter Golf

**Date:** 2026-03-20
**Status:** Research complete, NOT recommended for implementation
**TL;DR:** Mamba is theoretically attractive (30% fewer params/block = more layers in 16MB) but practically blocked by speed concerns, quality uncertainty at 17M scale, and high implementation risk. The attention baseline is a known quantity; Mamba is a gamble.

---

## 1. Package Availability: `mamba-ssm`

### Prebuilt Wheels: YES, available for our stack
- **Our environment:** PyTorch 2.9.0, CUDA 13.0, Linux x86_64
- **mamba-ssm 2.2.4+** publishes prebuilt wheels for CUDA 13.0.1 + PyTorch 2.9 (confirmed in `.github/workflows/publish.yaml`)
- Install command: `pip install mamba-ssm[causal-conv1d] --no-build-isolation`
- The `--no-build-isolation` flag is REQUIRED to use existing CUDA-enabled PyTorch
- Also needs `causal-conv1d>=1.4.0` for the fused depthwise conv1d kernel
- Wheel naming: `mamba_ssm-2.2.4+cu13torch2.9cxx11abiFALSE-cp3XX-cp3XX-linux_x86_64.whl`

### Pure PyTorch Alternative (NO custom kernels)
If `mamba-ssm` fails to install, two pure-PyTorch implementations exist:
1. **`johnma2006/mamba-minimal`** — single-file Mamba-1, sequential for-loop scan. ~342 lines.
2. **`PeaBrane/mamba-tiny`** — uses `torch.logcumsumexp` (Heinsen sequence) for numerically stable parallel scan. ~49-line scan function.
3. **Official `ssd_minimal.py`** — Mamba-2 chunk-based SSD in pure PyTorch (~78 lines). Uses einsum + cumsum, no CUDA kernels.

### ⚠️ CRITICAL SPEED WARNING
The pure PyTorch implementations are **5-20x slower** than the CUDA kernel versions:
- Mamba-1 sequential scan: O(L) serial for-loop over seq_len=1024. Cannot be parallelized on GPU.
- Mamba-2 SSD minimal: chunk-based matmul (better than Mamba-1) but still much slower than fused Triton kernel.
- GitHub issue #389: "mamba2 training speed is very very very slow" — users report pure PyTorch SSD is 3-5x slower.
- The CUDA kernels are the **core contribution** of the Mamba paper — without them, Mamba loses its speed advantage.

**Verdict:** Must use `mamba-ssm` package with CUDA kernels. Pure PyTorch is a non-starter for training within 10-minute budget.

---

## 2. Minimal Code Changes to Swap Attention for Mamba

### Architecture Mapping

| Current (Transformer Block) | Mamba Replacement |
|-----|-----|
| `CausalSelfAttention` (Q,K,V,proj + RoPE + Flash Attention) | `Mamba2` module from `mamba_ssm` |
| `MLP` (fc + relu² + proj) | **REMOVED** — Mamba block includes its own gating (SiLU + residual gate) |
| `Block.forward(x, x0)` → attn_norm → attn → + scale → mlp_norm → mlp → + scale | `MambaBlock.forward(x)` → norm → Mamba2 → + residual |

### Key Changes Needed

```python
# 1. NEW: MambaBlock class (replaces Block)
from mamba_ssm import Mamba2

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2, headdim=64):
        super().__init__()
        self.norm = RMSNorm()
        self.mamba = Mamba2(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
        )
        self.scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        # NOTE: resid_mix with x0 (U-Net skip input) may not apply — 
        # Mamba doesn't need positional info from x0 since it has its own 
        # sequential state. Test with and without.

    def forward(self, x, x0=None):
        # Optional: mix with x0 for U-Net skip connection compatibility
        out = self.mamba(self.norm(x))
        return x + self.scale.to(dtype=x.dtype)[None, None, :] * out

# 2. REMOVE: Rotary embedding (RoPE) — Mamba has no attention, no Q/K
#    The conv1d in Mamba provides local position info.
#    The SSM recurrence provides global position info.

# 3. MODIFY: GPT.__init__ — replace Block list with MambaBlock list
#    Remove Rotary, remove attention-specific params

# 4. MODIFY: forward() — simpler loop (no x0 needed if removing U-Net)

# 5. MODIFY: _classify_param() — new categories for Mamba params:
#    - in_proj.weight, out_proj.weight → "matrix" (Muon)
#    - conv1d.weight → "matrix" or "scalar" (small, could go either way)  
#    - A_log, D, dt_bias → "scalar" (Adam)
#    - norm.weight → "scalar" (Adam)

# 6. KEEP: Everything else (embedding, bigram, quantization, eval, TTT)
```

### What Stays the Same
- Tokenizer (SentencePiece BPE, vocab=1024)
- Embedding + tied LM head
- Quantization pipeline (int6 mixed + zstd-22)
- BigramHash embedding
- SmearGate (redundant with Mamba's conv1d but harmless)
- Training loop, DDP, optimizer structure
- Sliding window eval (still works — just run forward pass on windows)
- TTT post-training

### What Gets Removed
- `CausalSelfAttention` class entirely
- `Rotary` / `apply_rotary_emb` (no attention = no RoPE)
- `_ROPE_FRACTION`, `ROPE_BASE` hyperparams
- Flash Attention dependency
- XSA (Exclusive Self-Attention) — N/A for Mamba

### Estimated Code Diff Size
- ~150 lines removed (attention, rotary)
- ~50 lines added (MambaBlock class)
- ~30 lines modified (GPT init, forward, param classification)
- Net: ~70 lines smaller

---

## 3. Parameter Count Implications

### Per-Block Comparison (dim=512)

| Component | Attention (GQA 8:4) | Mamba-2 (expand=2, d_state=64) |
|-----------|---------------------|-------------------------------|
| Sequence mixing | 786,440 (Q,K,V,proj,gain) | 1,653,424 (in_proj, conv1d, SSM, out_proj) |
| MLP | 1,572,864 (3x relu²) | **0** (included in Mamba block) |
| Block overhead | 2,048 (scales, resid_mix) | 1,024 (scale only) |
| **TOTAL PER BLOCK** | **2,361,352** | **1,654,448** |
| **Ratio** | 1.00x | **0.70x (30% smaller!)** |

### Model-Level Comparison (fits in 16MB artifact)

| Config | Params | Layers | Est. Artifact | vs Baseline |
|--------|--------|--------|---------------|-------------|
| **Baseline** (9L attn+3xMLP, dim=512) | 21.9M | 9 | ~15.5MB | — |
| Mamba-2 (9L, dim=512) | 15.6M | 9 | ~9.7MB | -28% params |
| Mamba-2 (11L, dim=512) | 18.9M | 11 | ~11.8MB | -14% params, +2 layers |
| **Mamba-2 (13L, dim=512)** | **22.2M** | **13** | **~13.9MB** | **+1% params, +4 layers** |
| Mamba-2 (15L, dim=512) | 25.5M | 15 | ~15.9MB | +16% params, +6 layers |
| Mamba-2 (11L, dim=576) | 23.7M | 11 | ~14.8MB | +8% params, wider+deeper |
| Mamba-2 (9L, dim=640) | 23.9M | 9 | ~14.9MB | +9% params, much wider |

### Sweet Spot Recommendations
1. **Conservative:** 13L × dim=512 (22.2M params, ~13.9MB) — same param count as baseline, 44% more layers
2. **Aggressive:** 15L × dim=512 (25.5M params, ~15.9MB) — 16% more params, 67% more layers
3. **Wide:** 9L × dim=640 (23.9M params, ~14.9MB) — same depth, 25% wider

### Key Insight
Mamba blocks are 30% smaller than attention+MLP blocks because they combine sequence mixing AND feature transformation in one module (the SiLU gate + expand/contract acts as the "MLP"). This means **4-6 extra layers for free** within the 16MB budget.

---

## 4. Optimizer & Pipeline Compatibility

### Muon Optimizer: PARTIALLY compatible

| Mamba Parameter | Shape | Muon or Adam? | Notes |
|----------------|-------|---------------|-------|
| `in_proj.weight` | (2320, 512) | **Muon** ✅ | 2D matrix, largest param |
| `out_proj.weight` | (512, 1024) | **Muon** ✅ | 2D matrix |
| `conv1d.weight` | (1152, 1, 4) | **Adam** | 3D tensor, Muon needs 2D |
| `conv1d.bias` | (1152,) | **Adam** | 1D |
| `A_log` | (16,) | **Adam** | 1D scalar |
| `D` | (16,) | **Adam** | 1D scalar |
| `dt_bias` | (16,) | **Adam** | 1D scalar |
| `norm.weight` | (1024,) | **Adam** | 1D scalar |

**~98% of Mamba params are in in_proj + out_proj (2D matrices)**, so Muon handles the bulk. The conv1d weight is 3D but could be reshaped to 2D for Muon (squeeze the middle dim=1). This is a minor issue.

### Quantization Pipeline: FULLY compatible ✅
- `in_proj.weight` and `out_proj.weight` → int6 per-row quantization (same as attention matrices)
- `conv1d.weight` → int6 (small, ~6KB)
- Scalar params (A_log, D, dt_bias) → fp16 passthrough (tiny, <1KB)
- No architectural barriers to int5 MLP-equivalent quantization on `in_proj` (it's the "expansion" layer)

### TTT (Test-Time Training): COMPATIBLE ✅
- TTT runs SGD on the dequantized model weights
- Mamba's weights are standard `nn.Parameter`s — no special handling needed
- The `forward_logits()` function just needs to run the Mamba blocks instead of attention

### SWA/LAWA: COMPATIBLE ✅
- Weight averaging operates on state_dict — architecture-agnostic

### Sliding Window Eval: ⚠️ DIFFERENT BEHAVIOR
- Transformer attention benefits from overlapping windows (more context per scored token)
- Mamba maintains hidden state across the sequence — overlapping windows would **reset** the hidden state at each window start, losing the advantage of Mamba's theoretically infinite context
- Better approach for Mamba: **single long sequence eval** (concatenate validation data, run once)
- NTK-RoPE scaling is N/A (no RoPE in Mamba)

---

## 5. Has Anyone Tried Mamba/SSM for Parameter Golf?

### Answer: **NO.** Zero submissions use Mamba or any SSM variant.

Searched:
- All 42+ PRs on `openai/parameter-golf` — no mentions of "mamba", "ssm", or "state space"
- Code search in the repo — zero results
- `andrewgcodes/parameter-golf` fork (5 PRs) — all attention-based

### Why Nobody Has Tried It (Likely Reasons)
1. **Installation friction:** `mamba-ssm` requires CUDA compilation or specific prebuilt wheel matching. Most competitors use the stock `train_gpt.py` which only needs PyTorch.
2. **Unknown quality at 17M scale:** All Mamba benchmarks are at 130M-7B scale. Nobody knows if it works at our tiny scale.
3. **Competition is transformer-centric:** The baseline IS a transformer. Most competitors iterate on the baseline rather than replacing it entirely.
4. **Short context:** Our seq_len=1024 is where attention is FAST (O(n²) isn't bad at n=1024). Mamba's linear scaling advantage only kicks in at longer sequences.
5. **Mamba's known weaknesses:** Recall, copying, in-context learning — all tasks where the fixed-size hidden state is a bottleneck. Language modeling on general web text (FineWeb) exercises these capabilities.

### This Means: High Risk, High Reward
- First-mover advantage if it works
- But nobody avoiding it is also a signal — the experienced competitors all stick with attention

---

## 6. Recommended Config for Experimental Run

### Primary Config: 13L Mamba-2, dim=512
```
NUM_LAYERS=13
MODEL_DIM=512
D_STATE=64        # SSM state dimension (Mamba-2 default)
D_CONV=4          # Depthwise conv kernel (Mamba default)
EXPAND=2          # Expansion factor (Mamba default, gives d_inner=1024)
HEADDIM=64        # Per-head dim (Mamba-2 default)
NGROUPS=1         # Groups for B/C (default)
```

**Rationale:**
- 13 layers vs baseline's 9 = 44% more depth at same param count (22.2M ≈ 21.9M baseline)
- Fits comfortably at ~13.9MB (2.1MB headroom)
- Standard Mamba-2 hyperparams — don't optimize SSM config until we know base quality
- dim=512 matches baseline for fair comparison

### Secondary Config (if primary works): 15L Mamba-2, dim=512
- 67% more layers, 16% more params
- ~15.9MB — tight but fits with int5 MLP trick on in_proj

### Training Hyperparams (keep from current best)
```
MATRIX_LR=0.02        # Same as PR#39 optimized
SCALAR_LR=0.02
TIED_EMBED_LR=0.03
MUON_MOMENTUM=0.95
WARMDOWN_ITERS=3600
LOGIT_SOFTCAP=30
SEQ_LEN=1024
BATCH_TOKENS=524288
```

### What to Watch For
1. **Training loss at step 100:** Should be < 4.0. If > 5.0, something is fundamentally broken.
2. **Step speed:** Should be < 60ms/step on 8×H100 with CUDA kernels. If > 100ms, pure PyTorch fallback may have kicked in.
3. **Val BPB at step 500:** Must be < 1.55 to be competitive (baseline: 1.48 at step 500)
4. **Gradient norms:** Mamba A_log gradients can explode — monitor closely.

---

## 7. Critical Risk Assessment

### 🔴 HIGH RISKS

1. **Quality at 17M scale is UNKNOWN**
   - All published Mamba results are at 130M+ params
   - The Mamba paper's smallest model is 130M — no data at our 17-22M scale
   - At small scale, Mamba's fixed-size state (d_state=64 → 64 numbers per head to summarize entire history) may be catastrophically insufficient
   - Transformers at this scale already struggle — Mamba may struggle more

2. **Training speed with CUDA kernels is uncertain**
   - Mamba-2 SSD kernel speed depends on headdim, d_state, and sequence length
   - At seq_len=1024, attention with FlashAttention is already very fast (~2ms)
   - Mamba's conv1d + SSM scan may not be faster at this short sequence length
   - GitHub issue #657 ("Is mamba slower than transformer?") — at small model sizes and short sequences, Mamba can be SLOWER

3. **torch.compile compatibility**
   - The baseline heavily relies on `torch.compile(fullgraph=True)` for speed
   - `mamba_ssm` CUDA kernels may not be compatible with `torch.compile` graph capture
   - This could force `fullgraph=False` (slower) or manual kernel calls
   - Unknown territory — would need to test

4. **Mamba's recall weakness**
   - FineWeb validation includes diverse text requiring recall of earlier context
   - Mamba's finite state means it MUST forget some information
   - At d_state=64, each head stores 64 floats — is that enough to recall rare tokens?
   - Literature: "Repeat After Me: Transformers are Better than SSMs at Copying" (Kempner, Harvard)

### 🟡 MEDIUM RISKS

5. **U-Net skip connections may not help**
   - Skip connections in baseline compensate for vanishing gradients in deep attention networks
   - Mamba has different gradient dynamics (recurrence, not attention) — skips may be useless or harmful

6. **Quantization behavior unknown**
   - Mamba's A_log parameter is in log-space — int6 quantization may destroy it
   - Need to keep A_log, D, dt_bias in fp16 (tiny cost, ~96 params/layer)
   - in_proj and out_proj should quantize normally

7. **SmearGate redundancy**
   - SmearGate blends adjacent token embeddings — Mamba's conv1d already does this
   - Keeping SmearGate is harmless but wasteful

### 🟢 LOW RISKS

8. **Package installation** — prebuilt wheels exist for our exact stack
9. **Optimizer compatibility** — Muon works on 2D matrices, which dominate Mamba params
10. **Code complexity** — actually simpler than attention (fewer components)

---

## 8. Recommendation: SHOULD WE DO THIS?

### Verdict: **EXPERIMENTAL ONLY — Not for submission**

**Arguments FOR:**
- 30% smaller blocks → 44-67% more layers in 16MB budget
- Nobody else is trying it → first-mover advantage
- Mamba's linear complexity could enable longer eval sequences (2048, 4096) without memory issues
- No RoPE = no positional encoding limitations
- Conceptually elegant: SSM is a more parameter-efficient inductive bias for sequence modeling

**Arguments AGAINST:**
- Zero data on Mamba quality at 17M scale — we'd be the first
- Short seq_len=1024 removes Mamba's main speed advantage over attention
- Mamba's recall weakness may hurt BPB on general web text
- torch.compile compatibility is a gamble
- High implementation cost for uncertain payoff
- Our current approach (attention + SwiGLU + training tricks) is producing validated improvements

**If we do run the experiment:**
1. Use **1×A100 prototyping** first (not 8×H100) — cheap screening
2. Run for **2K steps** as baseline comparison
3. Compare val_bpb at step 500 against our known baseline (1.4805)
4. If Mamba is within 0.02 BPB, proceed to full-scale test
5. If worse by > 0.05 BPP, abandon immediately

### Implementation Time Estimate
- Script modification: ~2 hours
- Package installation + debugging: ~1 hour  
- 2K-step screening run: ~30 minutes (1×H100)
- **Total: ~3.5 hours for screening, before any full-scale run**

### Alternative: Hybrid (Mamba + Attention)
A safer bet: replace SOME layers with Mamba, keep 1-2 attention layers for recall.
- Example: 8 Mamba + 1 Attention (last layer) = 16.3M params, ~10.2MB
- The attention layer handles recall/copying, Mamba handles efficient sequence processing
- Jamba (AI21) uses this approach at scale with success
- But: still has all the implementation risks, plus more complexity

---

## 9. References

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-2: Transformers are SSMs (SSD)](https://arxiv.org/abs/2405.21060)
- [state-spaces/mamba GitHub](https://github.com/state-spaces/mamba) — Official implementation
- [johnma2006/mamba-minimal](https://github.com/johnma2006/mamba-minimal) — Pure PyTorch Mamba-1
- [PeaBrane/mamba-tiny](https://github.com/PeaBrane/mamba-tiny) — logcumsumexp scan
- [On the Tradeoffs of SSMs and Transformers](https://goombalab.github.io/blog/2025/tradeoffs/)
- [Repeat After Me: Transformers > SSMs at Copying](https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/)
- [Jamba: Hybrid Transformer-Mamba](https://arxiv.org/abs/2403.19887)
- [An Empirical Study of Mamba-based Language Models](https://arxiv.org/abs/2406.07887)
- [Exploring Limitations of Mamba in COPY and CoT](https://arxiv.org/abs/2410.03810)
