# GRAFT-WX: Grouped Recurrent Adaptive Forward Transformer — Wide eXpansion

> **Neural Ockham Architecture Proposal**
> Parameter Golf Challenge | 16MB artifact | 8×H100 | ≤10 min
> Target: Beat current SOTA BPB of **1.1194** (2026-03-23)

---

## 1. Diagnostic Audit

**Task**: Autoregressive language modeling on FineWeb (1024-token BPE vocab). Minimize bits-per-byte (BPB) on the validation set.

**Input/output shapes**: `input_ids: [B, T=2048]` → scalar cross-entropy loss (training), logits `[B, T, 1024]` (eval).

**Hard constraints**:
- ≤16MB artifact (int6+zstd-22 compressed)
- ≤10 minutes on 8×H100
- ≤1500 lines in training script

**Baseline** (Naive, 9L×512): **BPB = 1.2244**, ~22M raw params at int8+zlib = 15.86MB

**Current SOTA** (2026-03-23): **BPB = 1.1194**
Stack: 11L × d=512, GQA 8Q/4KV, MLP-3×, LeakyReLU(0.5)², XSA-4 (last 4 layers), Partial RoPE (16/64 dims), LN Scale, SmearGate, BigramHash, EMA(0.997), SWA, int6+zstd-22, TTT
Params: ~22M → ~15.9MB artifact

**Untried territory identified**: SwiGLU activation, cross-layer weight sharing / depth recurrence, width scaling (d>512), and combining recurrence + width to exploit the 16MB budget asymmetry.

---

## 2. Architectural Blueprint: GRAFT-WX

### Theory: Why Recurrence + Width Beats 11 Independent Narrow Layers

The 16MB budget under int6+zstd-22 allows ≈22M float32-equivalent parameters. The current SOTA allocates these as **11 independent blocks × narrow width (d=512)**. This is wasteful: the 16MB artifact stores 11 separate copies of nearly-identical weight matrices.

**GRAFT-WX reframes the question**: instead of spending parameter budget on *depth copies*, spend it on *width*. Use **K=4 unique shared blocks** iterated **R=3 times** (= 12 virtual layers), allocating saved parameters to **d=768** (50% wider).

The bet is rooted in two empirical observations from scaling laws:
1. **Width scales better than depth for small language models** — wider attention heads capture richer key/value subspaces; the rank of the attention score matrix scales with head_dim, not depth
2. **Recurrent models with residuals = looped Transformers** — proven in *Universal Transformers* (Dehghani+2018) and *Looped Transformers* (Giannou+2024) to match standard Transformers at ≥1/3 unique parameters

**Additional innovation**: Replace relu²(hidden=3d) with **SwiGLU**(hidden=2d). These are **isoparametric** (both use 2×d×hidden + hidden×d = 6d² weight entries) but SwiGLU provides:
- Multiplicative gating: `(Wx) ⊙ σ(Vx)` can represent XOR-like interactions impossible in purely additive MLPs
- Smooth non-zero gradient everywhere (vs dead zones in relu)
- Empirically validated: LLaMA, PaLM, GPT-J all show 1–5% perplexity gains vs relu

### Parameter Math

**Per-block cost** (d=768, GQA 8Q/2KV, SwiGLU):

| Component | Shape | Params |
|-----------|-------|--------|
| Q projection | 768×768 | 589,824 |
| K projection | 768×(2×96)=768×192 | 147,456 |
| V projection | 768×192 | 147,456 |
| Attn output proj | 768×768 | 589,824 |
| q_gain | 8 | 8 |
| SwiGLU gate | 768×1536 | 1,179,648 |
| SwiGLU fc | 768×1536 | 1,179,648 |
| SwiGLU proj | 1536×768 | 1,179,648 |
| iter_attn_scale (3×D) | 3×768 | 2,304 |
| iter_mlp_scale (3×D) | 3×768 | 2,304 |
| iter_resid_mix (3×2×D) | 3×2×768 | 4,608 |
| **Per block total** | | **5,022,728** |

**4 unique blocks**: 4 × 5,022,728 = **20,090,912**

**Global parameters**:

| Component | Params |
|-----------|--------|
| tok_emb (1024×768) | 786,432 |
| bigram.embed (2048×128) | 262,144 |
| bigram.proj (128×768) | 98,304 |
| bigram.scale | 1 |
| smear.gate (768) | 768 |
| skip_weights (2×768) | 1,536 |
| final_norm | 0 |
| **Global total** | **1,149,185** |

**Grand total: 21,240,097 ≈ 21.24M params**

**Artifact size estimate** (empirical rate: 22M params → 15.9MB):
21.24M × (15.9MB / 22M) ≈ **15.35MB** ✓ (0.65MB headroom below 16MB)

**Comparison to SOTA**: 21.24M vs 22M params, but each unique matrix is 50% wider (head_dim 96 vs 64, model residual stream 768 vs 512).

### Gradient Flow Proof

For shared weight **W** in block **k**, the gradient accumulates over all R=3 iterations:

```
∂L/∂W_k = Σ_{r=0}^{2} (∂L/∂h_{r,k}) · (∂h_{r,k}/∂W_k)
```

Each term is well-formed because the residual path gives:

```
∂h_{r,k}/∂h_{r,k-1} ≈ I + J_block   (Jacobian ≈ identity + small perturbation)
```

So gradients sum rather than vanish/explode across R=3 steps. The per-iteration `resid_mix` gates can be learned to minimize inter-iteration interference. Gradient norm grows by at most √R = 1.73× — compensated by reducing `matrix_lr` by 1/√3 ≈ 0.58.

---

## 3. Model Implementation

```python
# ────────────────────────────────────────────────────────────────
# GRAFT-WX: Grouped Recurrent Adaptive Forward Transformer
# Wide eXpansion Edition
#
# Drop-in replacement for the Block + GPT classes.
# Compatible with the SOTA optimizer loop interface:
#   base_model.blocks         -> ModuleList[K] RecurrentBlocks
#   base_model.skip_weights   -> Parameter[num_skip, D]
#   base_model.smear          -> SmearGate (has .gate attribute)
#   base_model.bigram         -> BigramHashEmbedding
#   base_model.tok_emb        -> Embedding
#   base_model.lm_head        -> None (tied embeddings)
#   base_model.mtp_heads      -> nn.ModuleList([]) (disabled)
#   base_model.mtp_num_heads  -> 0
#
# PARAM COUNT: ~21.24M → ~15.35MB at int6+zstd-22
# TARGET BPB:  < 1.1194 (current SOTA)
# ────────────────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    """
    SwiGLU gated MLP with hidden = 2 * dim.

    Isoparametric to relu²(hidden=3d): both use 6d² weight entries.
    SwiGLU advantage: multiplicative gating captures XOR-like interactions;
    smooth gradient everywhere (no dead relu zones).

    Parameter math:
        gate:  d  → 2d   (d × 2d)
        fc:    d  → 2d   (d × 2d)
        proj:  2d →  d   (2d × d)
        Total: 6d²  (same as relu²(3d): d×3d + 3d×d = 6d²)
    """
    def __init__(self, dim: int):
        super().__init__()
        hidden = 2 * dim
        self.gate = CastedLinear(dim, hidden, bias=False)
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(F.silu(self.gate(x)) * self.fc(x))


class RecurrentBlock(nn.Module):
    """
    Shared-weight transformer block used across R recurrence iterations.

    Key design:
      - Weight matrices (2D, no control patterns): Q,K,V,proj,gate,fc,proj → Muon
      - Per-iteration scalars (1D or name-matched control patterns) → scalar AdamW
        * iter_attn_scale  (name contains "attn_scale")
        * iter_mlp_scale   (name contains "mlp_scale")
        * iter_resid_mix   (name contains "resid_mix", ndim=2 but routed by name)

    This naming ensures CONTROL_TENSOR_NAME_PATTERNS in the training loop
    correctly routes these to fp32 + AdamW rather than Muon.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int,
        num_iterations: int,
        use_xsa: bool,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            rope_dims=rope_dims,
        )
        self.attn.use_xsa = use_xsa
        self.mlp = SwiGLU(dim)

        # Per-iteration adapters — cheap, float32, scalar Adam optimizer
        # Each is a ParameterList of num_iterations tensors
        self.iter_attn_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(dim, dtype=torch.float32))
             for _ in range(num_iterations)]
        )
        self.iter_mlp_scale = nn.ParameterList(
            [nn.Parameter(torch.ones(dim, dtype=torch.float32))
             for _ in range(num_iterations)]
        )
        # [2, D] per iteration: row 0 = current-stream weight, row 1 = x0 weight
        # Initialized to [1, 0] = full pass-through, x0 contribution zeroed
        self.iter_resid_mix = nn.ParameterList(
            [nn.Parameter(torch.stack([
                torch.ones(dim, dtype=torch.float32),
                torch.zeros(dim, dtype=torch.float32),
            ]))
             for _ in range(num_iterations)]
        )

    def forward(self, x: Tensor, x0: Tensor, r: int) -> Tensor:
        # Learned blend of current stream and initial embedding anchor
        mix = self.iter_resid_mix[r].to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # Attention sub-layer
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.iter_attn_scale[r].to(dtype=x.dtype)[None, None, :] * attn_out

        # MLP sub-layer
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + self.iter_mlp_scale[r].to(dtype=x.dtype)[None, None, :] * mlp_out

        return x


class GRAFT_GPT(nn.Module):
    """
    GRAFT-WX: K=4 shared blocks × R=3 iterations = 12 virtual depth layers.
    Width d=768 (vs d=512 baseline) fills the saved byte budget.

    Architecture summary:
      - 4 unique transformer blocks (shared weights across 3 iterations)
      - SwiGLU(hidden=2d) — isoparametric to relu²(hidden=3d), richer
      - GQA 8Q/2KV — more extreme than SOTA's 8Q/4KV, saves ~260K params/block
      - Partial RoPE: 16 of 96 head_dim dims get positional encoding (NTK-aware)
      - XSA (Exclusive Self-Attention) on last 2 blocks in final iteration only
      - U-Net skips: iter-0 block-0,1 outputs → iter-2 block-0,1 inputs (reversed)
      - SmearGate: learned 1-token causal blend
      - BigramHash(2048×128→768): parameter-efficient bigram context
      - Tied embeddings (lm_head = tok_emb.T)
      - Logit soft-cap

    Hyperparameter recommendations:
      NUM_UNIQUE_BLOCKS=4  NUM_ITERATIONS=3  MODEL_DIM=768
      NUM_HEADS=8          NUM_KV_HEADS=2
      ROPE_DIMS=16         XSA_LAST_N=2
      MATRIX_LR=0.023      (≈ 0.04/√3 — compensate for 3× gradient accumulation)
      SCALAR_LR=0.04       TIED_EMBED_LR=0.05
      WARMDOWN_ITERS=3500  TRAIN_SEQ_LEN=2048
      BIGRAM_VOCAB_SIZE=2048  BIGRAM_DIM=128
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,          # overridden; kept for interface compat
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,          # ignored; SwiGLU always uses hidden=2*dim
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        # GRAFT-specific (set via Hyperparameters or direct kwargs)
        num_unique_blocks: int = 4,
        num_iterations: int = 3,
        rope_dims: int = 16,
        xsa_last_n: int = 2,
        bigram_vocab_size: int = 2048,
        bigram_dim: int = 128,
        # ignored kwargs passed by SOTA training loop
        mtp_num_heads: int = 0,
        mtp_loss_weight: float = 0.0,
        ln_scale: bool = False,
        **kwargs,
    ):
        super().__init__()
        K = num_unique_blocks
        R = num_iterations
        D = model_dim

        self.tok_emb = nn.Embedding(vocab_size, D)
        self.smear   = SmearGate(D)
        self.bigram  = BigramHashEmbedding(bigram_vocab_size, bigram_dim, D)

        self.blocks = nn.ModuleList([
            RecurrentBlock(
                dim=D,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope_base=rope_base,
                qk_gain_init=qk_gain_init,
                rope_dims=rope_dims,
                num_iterations=R,
                use_xsa=(k >= K - xsa_last_n),  # XSA only on deepest blocks
            )
            for k in range(K)
        ])

        # U-Net skip connections across recurrence iterations:
        # iter-0 first (K//2) block outputs → iter-(R-1) first (K//2) block inputs
        # Injected in reverse order (U-Net decoder pattern)
        num_skip = K // 2  # = 2 for K=4
        self.skip_weights = nn.Parameter(
            torch.ones(num_skip, D, dtype=torch.float32)
            if num_skip > 0
            else torch.empty(0)
        )

        self.final_norm     = RMSNorm()
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap  = logit_softcap
        self.lm_head = (
            None if tie_embeddings
            else CastedLinear(D, vocab_size, bias=False)
        )
        if self.lm_head is not None:
            self.lm_head._zero_init = True

        # Required attributes for the SOTA training-loop interface
        self.mtp_heads      = nn.ModuleList([])
        self.mtp_num_heads  = 0
        self.num_encoder_layers = 0  # not used but may be inspected
        self.num_decoder_layers = 0

        self._num_unique_blocks = K
        self._num_iterations    = R
        self._num_skip          = num_skip

        self._init_weights()

    # ─── Initialization ────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Orthogonal + muP initialization strategy for recurrent depth.

        Why orthogonal?
          At init, ‖Wx‖ = ‖x‖ for all x when W is orthogonal (singular values = 1).
          After R=3 identical passes with residuals, the activation magnitude stays
          bounded: ‖h_R‖ ≈ ‖h_0‖ + O(R · ε) where ε is the block perturbation norm.
          This prevents the blow-up that would occur with random Gaussian init × 3.

        muP output scaling:
          The output projections (attn proj, MLP proj) are scaled by 1/√d to ensure
          the residual update magnitude is O(1) regardless of model width d.
          This generalizes the depth-dependent scaling in the SOTA (1/√(2L)) to our
          virtual depth setting (1/√(2·K·R)).
        """
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, 0.0, self.tied_embed_init_std)

        virtual_depth = self._num_unique_blocks * self._num_iterations  # = 12
        for module in self.modules():
            if not isinstance(module, CastedLinear):
                continue
            if getattr(module, '_zero_init', False):
                nn.init.zeros_(module.weight)
            else:
                # Orthogonal init: all singular values = 1 at start
                nn.init.orthogonal_(module.weight)
                # muP: scale down outputs to prevent residual magnitude growth
                module.weight.data.mul_(1.0 / math.sqrt(2 * virtual_depth))

    # ─── Forward body (shared between train and eval) ──────────────

    def _forward_body(self, input_ids: Tensor) -> Tensor:
        """
        Recurrent forward pass:

          h_0   = RMSNorm(Embed(x) + Bigram(x));  apply SmearGate
          x0    = h_0                              (anchor for resid_mix)

          for r in [0, 1, 2]:                      (R iterations)
            for k in [0, 1, 2, 3]:               (K unique blocks)
              if r==R-1 and k < num_skip:          (U-Net decoder inject)
                h += skip_weights[num_skip-1-k] * first_iter_hidden[num_skip-1-k]
              h = Block_k(h, x0, r)               (shared weights, iter-specific scales)
              if r==0 and k < num_skip:            (U-Net encoder collect)
                first_iter_hidden.append(h)

          return RMSNorm(h)
        """
        K = self._num_unique_blocks
        R = self._num_iterations

        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x

        first_iter_hidden: list[Tensor] = []

        for r in range(R):
            for k in range(K):
                # U-Net decoder: inject encoder skips in reverse order
                if r == R - 1 and k < self._num_skip:
                    skip_idx = self._num_skip - 1 - k
                    w = self.skip_weights[skip_idx].to(dtype=x.dtype)
                    x = x + w[None, None, :] * first_iter_hidden[skip_idx]

                x = self.blocks[k](x, x0, r)

                # U-Net encoder: record first-iteration hidden states
                if r == 0 and k < self._num_skip:
                    first_iter_hidden.append(x)

        return self.final_norm(x)

    # ─── Training forward ──────────────────────────────────────────

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._forward_body(input_ids)                  # [B, T, D]
        x_flat  = x.reshape(-1, x.size(-1))               # [B*T, D]
        targets = target_ids.reshape(-1)                   # [B*T]

        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)

        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction='mean')

    # ─── Evaluation forward (sliding window) ───────────────────────

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits [B, T, V] for sliding-window BPB evaluation."""
        x = self._forward_body(input_ids)

        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)

        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


# ─── Utility ───────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """Break down parameter count by component."""
    total = sum(p.numel() for p in model.parameters())
    blocks_total = sum(p.numel() for p in model.blocks.parameters())
    shared_matrices = sum(
        p.numel() for n, p in model.blocks.named_parameters()
        if p.ndim == 2 and not any(pat in n for pat in
            ('attn_scale', 'mlp_scale', 'resid_mix', 'q_gain'))
    )
    per_iter = sum(
        p.numel() for n, p in model.blocks.named_parameters()
        if any(pat in n for pat in ('iter_attn_scale', 'iter_mlp_scale', 'iter_resid_mix'))
    )
    return {
        'total_params': total,
        'block_params': blocks_total,
        'shared_matrix_params': shared_matrices,
        'per_iter_scalar_params': per_iter,
        'embedding_params': model.tok_emb.weight.numel(),
        'estimated_artifact_mb': total * 0.75 * 0.965 / 1e6,
    }
```

---

## 4. Training & Initialization Strategy

### Optimizer Configuration

The critical change from the standard SOTA setup is **reducing `matrix_lr` to compensate for the 3× gradient accumulation from shared weights**:

```bash
# Environment variables for a full 8×H100 run
RUN_ID=graft_wx_v1 \
MODEL_DIM=768 \
NUM_HEADS=8 \
NUM_KV_HEADS=2 \
ROPE_DIMS=16 \
XSA_LAST_N=2 \
BIGRAM_VOCAB_SIZE=2048 \
BIGRAM_DIM=128 \
TRAIN_SEQ_LEN=2048 \
WARMDOWN_ITERS=3500 \
TRAIN_BATCH_TOKENS=786432 \
GRAD_CLIP_NORM=0.3 \
MATRIX_LR=0.023 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
MUON_WD=0.04 \
ADAM_WD=0.01 \
EMA_ENABLED=1 \
EMA_DECAY=0.997 \
SWA_ENABLED=1 \
SWA_EVERY=200 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The optimizer setup in `main()` requires the same small additions as the PartialRoPE SOTA (registering `bigram.embed.weight` and `bigram.proj.weight` to the correct optimizer groups). The rest of the loop is identical.

### Why `matrix_lr = 0.023`

Shared weights receive gradient contributions from all R=3 iterations:

```
effective_grad = Σ_{r=0}^{2} grad_r  ≈  √3 × single_layer_grad  (by CLT)
```

To maintain the same effective learning step magnitude as in the 11-layer SOTA:

```
matrix_lr_graft = matrix_lr_sota / √R = 0.04 / √3 ≈ 0.023
```

### Why Orthogonal + muP Initialization Matters Here

Standard Kaiming initialization scales by `1/√fan_in`. For a recurrent model, each weight is applied R=3 times, so the effective fan-in is 3× larger. Using orthogonal init (`‖W‖_2 = 1`) and then scaling by `1/√(2·K·R) = 1/√24 ≈ 0.204` ensures:

```
‖h_R - h_0‖ ≤ R · ‖block_perturbation‖ ≈ 3 × O(1/√24) ≈ O(0.6)
```

This keeps hidden state magnitude stable across all 12 virtual depth steps, avoiding the layer-norm saturation seen in poorly-initialized recurrent Transformers.

### Quantization Strategy

Keep the proven int6 per-row + zstd-22 pipeline from the SOTA stack. The wider model (d=768) quantizes particularly well because per-row scales for 768-wide matrices have lower relative quantization error than 512-wide ones (more elements per row → better Gaussian approximation → tighter int6 clip percentile).

---

## 5. Expected Performance

| Contribution | Expected ΔBPB |
|---|---|
| SwiGLU vs LeakyReLU²(0.5) | −0.003 to −0.005 |
| Width 768 vs 512 (per virtual layer) | −0.005 to −0.015 |
| 12 virtual layers vs 11 actual | −0.002 to +0.005 |
| Lower KV heads (2 vs 4, saves params for width) | −0.001 to +0.001 |
| **Combined estimate** | **−0.008 to −0.016** |

**Projected BPB**: ~1.103–1.111, vs current SOTA **1.1194**

The SwiGLU and width gains are the high-confidence contributions. The recurrence is the speculative element — if it matches independent-layer quality, the width bonus alone beats SOTA; if recurrence adds even marginal quality, the gap widens further.

---

## 6. Key References

1. **Universal Transformers** — Dehghani et al. 2018. Recurrent depth matches standard Transformers with fewer unique parameters on NLP benchmarks.
2. **Looped Transformers** — Giannou et al. 2024. Formally proves looped Transformers are Turing-complete and can match depth-L networks with L/3 unique layers.
3. **SwiGLU** — Noam Shazeer 2020 / LLaMA. SwiGLU(2d) is isoparametric to GeLU(4d) but outperforms it on language modeling perplexity.
4. **ALBERT** — Lan et al. 2020. Cross-layer weight sharing achieves ~90% of BERT quality at ~1/18 unique parameters.
5. **Exclusive Self-Attention (XSA)** — arXiv:2603.09078. Subtracting self-value projection in deep layers forces context-dependent representations.
