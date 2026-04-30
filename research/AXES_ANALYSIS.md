# Parameter Golf: Axes of Work — `AXES_ANALYSIS.md`

> A deep structural read of every merged record in `records/track_10min_16mb/` against the NaiveBaseline. Organized by orthogonal axis of work (not chronology). Every entry shows the *actual code hunk* from the winning submission. ⚠ markers flag hidden/subtle details the README doesn't make obvious.

## How to read this

- **One axis = one section.** Within each axis, records are listed chronologically.
- **Code hunks are verbatim** from `sota_analysis/records_normalized/<name>.py` (AST-canonicalized so one-liner-compressed records diff cleanly).
- **"Baseline" = `2026-03-17_NaiveBaseline`.** This is the stock GPT-like reference implementation. Anything present there is NOT flagged as novel anywhere in this doc — it's already-built infrastructure.
- **⚠ means "subtle / easy to miss".** The README and commit title usually describe the headline change; ⚠ points to the guard conditions, init choices, ordering, dead code, and interactions that actually matter in practice.
- **Some "records" change zero source lines** — they're hyperparameter overrides via env vars. Those are noted as "env-var only" and do not introduce code hunks.

## Timeline of merged records

Each row is a directory inside `records/track_10min_16mb/` on `origin/main`. BPB is the 3-seed (or noted) mean post-quantization val_bpb. Column "Δ" is the delta from the immediately preceding record on the frontier lineage.

| Date        | Record                                                            | BPB (post-quant) | Δ vs prior | Author(s)                 |
| ----------- | ----------------------------------------------------------------- | ---------------- | ---------- | ------------------------- |
| 2026-03-17  | `NaiveBaseline`                                                   | **1.2172**       | —          | reference                 |
| 2026-03-17  | `LoRA_TTT`                                                        | 1.1950           | −0.0222    | LoRA-TTT pioneer          |
| 2026-03-18  | `LowerLR`                                                         | 1.2183           | +0.0011    | env-only                  |
| 2026-03-18  | `LongContextSeq2048`                                              | 1.2005           | −0.0167    | env-only (seq_len=2048)   |
| 2026-03-18  | `FP16Embed_WD3600`                                                | 1.2197           | +0.0025    | fp16 embed passthrough    |
| 2026-03-19  | `TrainingOptSeq4096`                                              | 1.1980           | −           | env-only                  |
| 2026-03-19  | `10L_MixedPrecision`                                              | 1.2129           | −          | post-hoc int4-step + prune|
| 2026-03-19  | `SlidingWindowEval`                                               | 1.2172           | −          | first sliding-window eval |
| 2026-03-19  | `WarmdownQuantization`                                            | 1.2154           | −          | parameterized bits + WD   |
| 2026-03-19  | `smeargate_orthoinit_muonwd`                                      | 1.1556           | −          | SmearGate + BigramHash    |
| 2026-03-19  | `SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`                   | 1.1748           | −          | overtone embed init       |
| 2026-03-19  | `Seq2048_FP16Emb_TunedLR`                                         | 1.1598           | −          | int6 QAT always-on        |
| 2026-03-19  | `MixedQuant_Int6Int8_SlidingWindow`                               | 1.1630           | −          | int6/int8 mixed           |
| 2026-03-19  | `MLP3x_QAT_Int6_SlidingWindow`                                    | 1.1502           | −          | MLP 3x + int6 QAT         |
| 2026-03-20  | `Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`                      | 1.1458           | −          | unified stack             |
| 2026-03-20  | `10L_Int5MLP_MuonWD04_SWA50`                                      | 1.1428           | −          | int5 MLP                  |
| 2026-03-20  | `11L_EfficientPartialXSA_FA3_SWA120`                              | 1.1307           | −0.0151    | XSA + FA3                 |
| 2026-03-20  | `11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`                             | 1.1271           | −0.0036    | EMA introduced            |
| 2026-03-21  | `11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`                         | 1.1248           | −0.0023    | partial RoPE + LN scale   |
| 2026-03-22  | `11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`                    | 1.1233           | −0.0015    | GPTQ-lite + VE            |
| 2026-03-23  | `LeakyReLU_LegalTTT_ParallelMuon`                                 | 1.1194           | −0.0039    | legal TTT + parallel Muon |
| 2026-03-24  | `74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon`                   | 1.1570           | (branch)   | ternary (dead branch)     |
| 2026-03-25  | `ValCalib_GPTQ_XSA_BigramHash3072` (old SOTA)                     | 1.1147           | −0.0047    | AR-calib GPTQ + XSA-all   |
| 2026-03-31  | `ParallelResiduals_MiniDepthRecurrence`                           | 1.1063           | −0.0084    | parallel resid + depth-rec|
| 2026-04-01  | `Vocab4096_MLPMult4_WD085`                                        | 1.0979           | −0.0084    | SP4096 + mult=4 + WD      |
| 2026-04-03  | `MuonEqR_DepthRecurrence_WD090_AllInt6`                           | 1.0912           | −0.0067    | MuonEq-R                  |
| 2026-04-04  | `SP4096_DepthRecurrence_ParallelResid_MuonEqR`                    | 1.0897           | −0.0015    | SP4096 consolidated       |
| 2026-04-05  | `SP8192_GPTQ-Embeddings_SDClip_Loop45x2`                          | 1.0856           | −0.0041    | SDClip + GPTQ emb         |
| 2026-04-06  | `SP8192_HessianSDClip_ProgressiveRecurrence` *(non-record)*       | 1.0835           | −          | Hessian-aware SDClip      |
| 2026-04-06  | `SP8192_QK5_LegalTTT_1.0828`                                      | 1.0828           | −0.0028    | QK-gain=5 + legal TTT     |
| 2026-04-08  | `SP8192_ParallelResid_ScoreFirstTTT`                              | 1.0822           | −0.0006    | parallel+TTT              |
| 2026-04-09  | `SP8192_3LayerRecur_ParResid_QK525_LegalTTT`  **(current SOTA)**  | **1.0810**       | −0.0012    | 3-layer rec, QK=5.25      |

**Total gain over baseline**: 1.2172 → 1.0810 = **−0.1362 BPB** (−11.2% relative).

## Baseline cheat-sheet

Everything in this list is already in `2026-03-17_NaiveBaseline/train_gpt.py`. Do not confuse these with innovations:

**Architecture**
- 9-layer GPT, `model_dim=512`, `num_heads=8`, `num_kv_heads=4` (GQA), `mlp_mult=2`, tied embeddings, vocab 1024 (SP1024).
- **U-Net skip connections**: `num_encoder_layers = num_layers // 2`, `skip_weights` is a learned per-dim tensor of shape `[num_skip_weights, model_dim]`. Encoder pushes to LIFO stack, decoder pops and adds `skip_weights[i] * popped_skip`.
- **Per-block**: `attn_norm`, `mlp_norm` (RMSNorm, no weight), `attn_scale`, `mlp_scale` (learnable per-dim fp32 ones), `resid_mix` (mixes block input `x` with initial embedding `x0`, init `[[1...],[0...]]`).
- **Attention**: QK RMSNorm, per-head `q_gain` scale (init 1.5, learnable), full RoPE (all head dims), causal SDPA with GQA.
- **MLP**: `relu(x).square()` (ReLU²), no gating, no bias, zero-init output proj.
- **Logit softcap**: `30 * tanh(logits / 30)`.

**Optimizer**
- Muon (Newton-Schulz 5 iters) for all 2D block matrices. Post-step gradient rescale `max(1, g.size(0)/g.size(1))**0.5` after NS.
- Muon momentum warmup: 0.85 → 0.95 over 500 steps.
- Adam (plain, no WD) for token embedding, head (if untied), skip weights, and all scalar/low-dim params.
- Gradient clipping default 0 (OFF).

**Training**
- `bfloat16` autocast forward, fp32 for low-dim params and `CastedLinear` weights (via `restore_low_dim_params_to_fp32` + a `.float()` sweep on `CastedLinear`). Masters live in fp32.
- **Warmup-then-reset**: runs 20 warmup steps, saves pre-warmup state, restores it after warmup to let JIT compile outside the 600s clock.
- Wallclock-aware linear LR warmdown over last `warmdown_iters` steps.
- No EMA, no SWA.

**Data**
- `TokenStream` / `DistributedTokenLoader` — sequential per-rank slice of concatenated shards (`fineweb_train_*.bin`). One pass through data in order.
- `load_validation_tokens` → one contiguous val buffer.
- Tokenizer = SP1024 (SentencePiece BPE, vocab=1024). Pre-tokenized shards distributed via HuggingFace; tokenizer model file ≈ 50 KB is counted against the artifact budget.

**Eval**
- Simple non-overlapping validation: reshape tokens into `[N, seq_len]`, mean cross-entropy. No sliding window. No stride. No TTT.

**Quantization** (end-of-training, post-training, outside the 600s)
- `quantize_state_dict_int8`: per-row int8 with percentile clipping `INT8_CLIP_Q = 99.99984%`.
- `INT8_KEEP_FLOAT_MAX_NUMEL = 65536` — tensors with ≤65K elements skip quantization and are stored as fp16 (for the embedding this isn't enough — `tok_emb` with vocab=1024, dim=512 = 524288 elements, far over the threshold, so it gets int8-quantized by default in baseline).
- Control tensors matched by `CONTROL_TENSOR_NAME_PATTERNS = ('attn_scale', 'mlp_scale', 'resid_mix', 'q_gain', 'skip_weight', ...)` are kept as fp32 even if under the size threshold.
- `zlib` level-9 compression on the serialized state dict.

## Axes

The 8 axes below partition the space of techniques. Most records span multiple axes; the "primary" axis is the dominant contribution, "supporting" axes list the secondary changes.

1. [Architecture](#axis-1-architecture)
2. [Attention](#axis-2-attention)
3. [Quantization](#axis-3-quantization)
4. [Optimizer](#axis-4-optimizer)
5. [Training dynamics](#axis-5-training-dynamics)
6. [Data & tokenizer](#axis-6-data--tokenizer)
7. [Eval-time adaptation](#axis-7-eval-time-adaptation)
8. [Compression](#axis-8-compression)

At the end:
- [Cross-cutting observations](#cross-cutting-observations) — known interactions, dead branches, composition notes
- [Notable failures](#notable-failures) — what didn't work and why
- [Reading guide](#reading-guide)

---

## Axis 1: Architecture

*What lives inside the model: layer structure, residual patterns, skip connections, recurrence, auxiliary input features, MLP design.*

**Axis at a glance.** Baseline is already a U-Net-ish stack with `resid_mix` and per-block control scales. The evolution went: (1) wider, deeper (9L→10L→11L, MLP 2x→3x→4x), (2) auxiliary bigram signal injected (BigramHash, SmearGate, ValueEmbedding), (3) LN scale per-layer depth weighting, (4) skip connections gated by a learned sigmoid, (5) **depth recurrence** (reuse middle layers for extra virtual depth), (6) **parallel residuals** (GPT-J style: attention and MLP read from same input), (7) extension to 3-layer recurrence. The capstone SOTA has 11 physical layers but 17 virtual layers (three copies of layers 3-5 in the middle).

### 1.1 Width & depth

These are mostly *hyperparameter* changes — the baseline `GPT` class handles any `num_layers`, `model_dim`, `mlp_mult`. But some records introduced the `MLP_HIDDEN` override to let the hidden size be independent of a `model_dim` multiple (useful when FP16 embed passthrough spends MLP budget).

#### `2026-03-18_FP16Embed_WD3600` — `mlp_hidden` override

```python
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, mlp_hidden: int = 0):
        super().__init__()
        hidden = mlp_hidden if mlp_hidden > 0 else mlp_mult * dim
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
```

⚠ `mlp_hidden = 0` is the sentinel for "fall back to `mlp_mult * dim`". To get multiplier behavior after this change, you must *not* set `MLP_HIDDEN`. Used as `MLP_HIDDEN=992` to recoup the ~500KB budget eaten by keeping tok_emb at fp16.

*Subsequent records* keep this pattern. By `2026-04-01_Vocab4096_MLPMult4_WD085`, `mlp_mult` becomes a float (4.0, not 4) and is the canonical knob again.

### 1.2 Embedding initialization tricks

#### `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` — overtone-spectrum embed init

```python
nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
with torch.no_grad():
    U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
    target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
    self.tok_emb.weight.data = U * target_S[None, :] @ V
```

Reshapes the embedding's singular-value spectrum to a power law `S_k ~ k^{-1/2}`. The stated intuition: the embedding is essentially the vocabulary basis, and the desired "overtone" structure gives a controlled spectral decay.

⚠ This is a one-shot init only — no constraint is enforced during training. By step 500 the Adam-driven updates will have destroyed the exact spectrum; what remains is the *initial condition* having biased the optimization basin. This idea did not propagate to frontier records; the later frontier just uses `normal_(std=0.005)` and relies on warmup-reset + higher LR for the embedding to explore freely.

#### `2026-03-19_SlidingWindow_FP16Emb_...` — phase-transition `resid_mix` init

```python
for i, block in enumerate(self.blocks):
    with torch.no_grad():
        phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
        block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
        block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])
```

Early blocks get more of `x0` (original embedding), late blocks get more of `x` (the running residual). A sigmoid interpolation over depth with steepness 3.

⚠ Also a one-shot init. Didn't propagate. Frontier records keep baseline init `[[1...],[0...]]` (identity for `x`, zero for `x0`) and let training learn whatever per-dim mix it wants.

### 1.3 Auxiliary input features (SmearGate, BigramHash, ValueEmbedding)

These sit at the input of the network and inject additional per-position signal beyond the token embedding. None cost much parameter budget; all three survive into the current SOTA lineage (with variations).

#### `2026-03-19_smeargate_orthoinit_muonwd` — SmearGate (first form)

```python
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.full((dim,), 3.0, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate).to(dtype=x.dtype)
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return g * x + (1.0 - g) * x_prev
```

Per-dim learned sigmoid that blends each token's embedding with the *previous* token's (shifted-by-one causal blend). Zero-pad at position 0 to keep causality.

⚠ `gate` is initialized at **3.0**, not 0. `sigmoid(3.0) ≈ 0.953`, so at init the blend is 95% current / 5% previous — a near-identity that preserves baseline behavior. Training nudges this away. Later records (03-20 Int6_SmearGate) re-init to 0 (`sigmoid(0) = 0.5`, equal blend) which changes the direction of adaptation.

#### `2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` — SmearGate (0-init form, same formula)

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

⚠ Note the swap: here `(1-g)*x + g*x_prev`. Compared to the earlier form `g*x + (1-g)*x_prev`, this inverts the meaning of `gate`. With `gate` init=0, the blend is 0.5/0.5 in both parameterizations, but as `gate` grows positive in this version the model pulls *toward* the previous token, while in the earlier version it pulled *toward* the current token. These are the same function of `g` with different conventions; a sign flip of the gate recovers one from the other.

#### `2026-03-19_smeargate_...` — BigramHash (first form)

```python
class BigramHash(nn.Module):
    def forward(self, input_ids: Tensor) -> Tensor:
        bsz, seqlen = input_ids.shape
        prev_ids = torch.cat([torch.zeros(bsz, 1, ...), input_ids[:, :-1]], dim=1)
        h = ((prev_ids.long() * 92821 + input_ids.long()) % self.num_buckets).long()
        return self.proj(self.table(h))
```

Multiplicative hash `h = (prev * 92821 + curr) % num_buckets` indexes a learned table (init ones per README; dims: `table` = `[buckets, bigram_dim]`, `proj` = `[bigram_dim, model_dim]`). The output is **added to** the token embedding before the first block.

⚠ `92821` is just a large prime. No structure in the hash — it's a uniform-mix of bigram identity into `num_buckets` buckets. The initial version uses 4096 buckets at dim=128.

#### `2026-03-20_Int6_..._BigramHash_...` — BigramHash (XOR form, zero-init scale)

```python
def bigram_hash(self, tokens: Tensor) -> Tensor:
    t = tokens.to(torch.int32)
    mod = self.bigram_vocab_size - 1
    out = torch.empty_like(t)
    out[..., 0] = mod  # sink bucket for position 0
    out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
    return out.long()
```

XOR variant with two different primes (36313, 27191). Position 0 always maps to `mod` (reserved sink bucket).

⚠ Table is **zero-initialized** in this version (vs `ones` in the earlier), and the bigram contribution is gated by a learned scalar `scale` initialized to **0.05** (see `2026-03-23_LeakyReLU_...` version below). Combined, the bigram signal starts at ~0 and grows only if the gradient says so — a conservative introduction.

#### `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` — `BigramHashEmbedding` with `scale=0.05`

```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size, bigram_vocab_size, bigram_dim, model_dim):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.table = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.table.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
```

⚠ *Three* pieces are initialized to zero or near-zero: `table.weight` (zeros), `proj.weight` (zeros), `scale` (0.05). Even if the hash collides badly or the training dataset has weird bigram frequencies, the network cannot be disturbed at init. This exemplifies a recurring PG pattern: aux modules start at the identity transformation.

⚠ `table.weight` is placed in the **token optimizer group** (gets `tied_embed_lr`, not `matrix_lr`). `proj.weight` is in `matrix_params` (Muon). `scale` is in `scalar_params` (AdamW). This three-way split per aux module happens because each tensor has different geometry and dynamics.

Between 03-23 and the current SOTA, the BigramHash bucket count was tuned: 2048 → 3072 (03-23, −0.0009 BPB), and dim was reduced 128 → 112 (03-25) to stay under 16 MB with 3072 buckets.

#### `2026-03-22_11L_EMA_GPTQ-lite_..._1.1233` — ValueEmbedding (VE)

```python
class ValueEmbedding(nn.Module):
    """Reinject token identity into attention values at specific layers."""
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.normal_(self.embed.weight, std=0.01)
        self.proj = CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
```

Usage: at layers listed in `ve_layers` (default `"9,10"` for the last two blocks), the raw `v` vector *before* reshape into heads is augmented with `scale * proj(embed(input_ids))`. Re-injects token identity into the value stream, compensating for information loss through attention routing.

⚠ Unlike BigramHash, the `embed.weight` here is **normally initialized** (std=0.01), not zero. The `proj.weight` is zero-init and `scale` is 0.1 — so the raw signal is non-zero but the projection to model_dim is initially null. This is a middle ground: the network starts with non-trivial "content" ready to flow but gated by the proj layer.

⚠ VE is a **shared** module (one table, one proj) across all VE layers. Per-layer differentiation comes only from a separate `ve_layer_scales` tensor (one scalar per VE layer, init 1.0). No per-layer feature specialization at the embedding level — only per-layer gain.

### 1.4 LN scale (per-depth normalization damping)

#### `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` — `ln_scale_factor`

```python
self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

# ... in Block.forward:
s = self.ln_scale_factor
attn_out = self.attn(self.attn_norm(x) * s)
x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * s)
```

Per-layer scalar multiplier (no learnable parameter) on the *output of RMSNorm*, decaying as `1/√(layer_idx+1)`. Layer 0 uses ×1.0, layer 10 uses ×√(1/11) ≈ 0.302.

⚠ `ln_scale_factor` is a Python float on the Block object, **not** a `nn.Parameter`. Zero parameter overhead, but also not adaptable — the schedule is hardcoded. The same scalar is applied to both the pre-attention and the pre-MLP norm output within the same block.

⚠ Interpretation: as the residual stream accumulates variance through depth, the RMSNorm output is always unit-variance per position, but the subsequent `attn_scale` / `mlp_scale` learnable gains (per-dim) have to pick the right magnitude. Pre-scaling by `1/√(depth)` is a depth-dependent prior that nudges later layers toward smaller update contributions — a form of Pre-LN residual stability.

### 1.5 Skip-gate (sigmoid-gated U-Net skips)

#### `2026-03-31_ParallelResiduals_MiniDepthRecurrence` — skip_gates introduced

```python
self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) if h.skip_gates_enabled else None

# ... in forward decoder loop:
if self.skip_gates is not None:
    g = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
    x = torch.lerp(scaled_skip, x, g)  # lerp(a, b, t) = a + t*(b-a)
else:
    x = x + scaled_skip
```

A per-layer per-dim sigmoid gate interpolates linearly between "take only the scaled skip" (gate=0) and "ignore the skip entirely" (gate=1).

⚠ `skip_gates` is **zero-initialized**, so `sigmoid(0) = 0.5`. At init the output is an equal LERP: `0.5 * scaled_skip + 0.5 * x`. Compare to baseline, which adds `scaled_skip` to `x` — so baseline corresponds to `skip_gate` → −∞ (gate=0 in the `lerp(a, b, gate)` convention → `x = scaled_skip`, which is additive when the previous `x = 0`... actually this is a different semantics than baseline entirely). The skip gate replaces the additive formulation, so the network has to relearn whether to keep the skip.

⚠ Inherited by all subsequent frontier records (Apr 01, 03, 04, 05, 06, 08, 09). Always on (`SKIP_GATES_ENABLED=1`).

### 1.6 Parallel residuals (GPT-J style)

#### `2026-03-31_ParallelResiduals_MiniDepthRecurrence` — first parallel-resid implementation (two-lane form)

```python
def _parallel_block(self, virtual_idx, lane0, lane1, x0, q_w, k_w, v_w, out_w, up_w, down_w, ...):
    attn_read = self._mix_with_x0(lane0, x0, block.resid_mix)
    attn_out = block.attn_scale * block.attn(block.attn_norm(attn_read) * block.ln_scale_factor, ...)
    resid = self.parallel_resid_lambdas[physical_idx, 0].to(dtype=lane0.dtype)
    post  = self.parallel_post_lambdas[physical_idx, 0].to(dtype=lane0.dtype)
    lane0 = resid * lane0 + post[0] * attn_out   # attn writes to attn lane
    lane1 = resid * lane1 + post[1] * attn_out   # attn writes to mlp lane
    mlp_read = self._mix_with_x0(lane1, x0, block.resid_mix)
    mlp_out = block.mlp_scale * block.mlp(block.mlp_norm(mlp_read) * block.ln_scale_factor, ...)
    resid = self.parallel_resid_lambdas[physical_idx, 1].to(dtype=lane0.dtype)
    post  = self.parallel_post_lambdas[physical_idx, 1].to(dtype=lane0.dtype)
    lane0 = resid * lane0 + post[0] * mlp_out
    lane1 = resid * lane1 + post[1] * mlp_out
    return (lane0, lane1)
```

This is the baroque, two-lane version. **Four learned scalars per block** (`parallel_post_lambdas[i, j, 2]` — layer, sublayer=attn/mlp, lane=0/1 target). Attention and MLP each live in their own "lane" (`lane0`, `lane1`), and each sublayer *writes back to both lanes* with independent coefficients. The `parallel_resid_lambdas` scale each lane's identity contribution.

⚠ `parallel_resid_lambdas` init = `1.1 ** 0.5 ≈ 1.049` (slightly > 1). `parallel_post_lambdas` init = 1.0 (all four write-backs full strength at init). At the end of the parallel region, `lane0` and `lane1` are merged hardcoded 50/50: `(lane0 + lane1) * 0.5`.

⚠ The training discovery per the README: "MLP almost never writes back to the attention lane (weights near 0)". So the learned `parallel_post_lambdas[*, 1, 0]` values drift toward zero, confirming that attention and MLP want to occupy *different* subspaces of the residual stream. This retroactively validates the two-lane design — but also suggests it's unnecessary complexity.

#### `2026-04-04_SP4096_DepthRecurrence_ParallelResid_MuonEqR` — two-lane form with single `lane_merge`

```python
if is_parallel_mode:
    block = self.blocks[phys_idx]
    mix = block.resid_mix.to(dtype=lane0.dtype)
    attn_in = mix[0][None, None, :] * lane0 + mix[1][None, None, :] * x0
    attn_out = block.attn(block.attn_norm(attn_in) * block.ln_scale_factor, v_embed=ve)
    lane0 = attn_in + block.attn_scale.to(dtype=attn_in.dtype)[None, None, :] * attn_out
    mlp_in = block.mlp_norm(lane1) * block.ln_scale_factor
    mlp_out = block.mlp(mlp_in)
    lane1 = lane1 + block.mlp_scale.to(dtype=lane1.dtype)[None, None, :] * mlp_out
# ... at end of parallel region:
m = self.lane_merge.to(dtype=lane0.dtype)
x = m * lane0 + (1 - m) * lane1
```

⚠ `lane_merge` is a single learned scalar (`nn.Parameter(torch.tensor(0.5))`) placed in `CONTROL_TENSOR_NAME_PATTERNS` → AdamW scalar group. Not per-dim, not per-layer. Starts 50/50 and drifts.

⚠ This is a substantial simplification from the 03-31 "four scalars per block" form: only **one** scalar for the entire parallel region, and each sublayer writes only to its own lane during the region. The cross-lane write-backs of 03-31 are gone.

#### `2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence` — **the clean `block.parallel=True` form** (current frontier)

```python
def forward(self, x: Tensor, x0: Tensor) -> Tensor:
    mix = self.resid_mix.to(dtype=x.dtype)
    x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
    attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
    if self.parallel:
        mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out + self.mlp_scale.to(dtype=x_in.dtype)[None, None, :] * mlp_out
    else:
        x_out = x_in + self.attn_scale.to(dtype=x_in.dtype)[None, None, :] * attn_out
        x_out = x_out + self.mlp_scale.to(dtype=x_out.dtype)[None, None, :] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
    return x_out
```

This is what the current SOTA uses. Both attention and MLP read `x_in` (same pre-residual activation); their outputs are summed directly into `x_out`. **No lane_merge, no lambdas, no two-lane state.** Just a flag.

⚠ Configured via `parallel_residual_start`: for `i in range(parallel_residual_start, num_layers): blocks[i].parallel = True`. In the current SOTA, `parallel_residual_start=7`, so layers 7-10 run in parallel mode.

⚠ **Interaction with depth recurrence**: the recurred layers (3, 4, 5 in the SOTA) have `blocks[3..5].parallel = False`, so even if those physical layers occupy virtual positions ≥ 7, they run sequentially. This is enforced because `parallel` is a property of the *physical block*, not of the virtual position.

### 1.7 Depth recurrence (virtual layers via layer reuse)

#### `2026-03-31_ParallelResiduals_MiniDepthRecurrence` — first introduction

```python
def set_recurrence_active(self, active):
    self._recurrence_active = bool(active) and bool(self.recur_layers)
    if self._recurrence_active:
        self.v2p = self._v2p_recur
        self.num_encoder_layers = self._enc_recur
        self.num_decoder_layers = self._dec_recur
    else:
        self.v2p = self._v2p_no_recur
        self.num_encoder_layers = self._enc_no_recur
        self.num_decoder_layers = self._dec_no_recur
    if self._recurrence_active and (not was_active) and self.repeat_mlp:
        self._sync_repeat_mlp_from_base()
```

`v2p` is the virtual-to-physical layer index map. With `recur_layers=[4, 5]`, the off-state map is `[0..10]` (identity for 11 layers), and the on-state map inserts one extra copy of `[4, 5]`, giving `[0,1,2,3,4,5,4,5,6,7,8,9,10]` — 13 virtual layers from 11 physical. Activated at `recur_start_step=3000` (~54% of training).

⚠ `_sync_repeat_mlp_from_base()` runs exactly once on the transition (inactive→active). It copies the current MLP weights from the base blocks into the optional untied `RepeatMLPWeights` slots. Without this warm-start the repeat pass would use zero weights and collapse.

⚠ **Optional untied MLP** for the repeated pass (`repeat_untie_mlp ∈ {'none', 'down', 'full'}`). Modes:
  - `'none'` → fully shared (default, cheapest)
  - `'down'` → separate MLP down-projection only
  - `'full'` → both up and down untied

```python
class RepeatMLPWeights(nn.Module):
    def __init__(self, dim, mlp_mult, mode):
        mlp_dim = int(mlp_mult * dim)
        self.fc   = CastedLinear(dim, mlp_dim, bias=False) if mode == 'full' else None
        self.proj = CastedLinear(mlp_dim, dim, bias=False) if mode in ('full', 'down') else None
```

The 03-31 record runs with `mode='full'` for layers `[4, 5]`. The current SOTA (04-09) runs with `mode='none'` (fully shared) — simpler, and the README claims comparable.

#### `2026-04-03_MuonEqR_DepthRecurrence_WD090_AllInt6` — cleaner virtual-layer construction

```python
def _get_virtual_layers(self):
    if self._recurrence_active and self.recur_layers:
        return (list(range(self._repeat_cutoff))
              + self.recur_layers
              + list(range(self._repeat_cutoff, len(self.blocks))))
    return list(range(len(self.blocks)))
```

With `recur_layers=[4, 5]` and 11 blocks, `_repeat_cutoff = max([4, 5]) + 1 = 6`, yielding `[0,1,2,3,4,5] + [4,5] + [6,7,8,9,10]` = `[0,1,2,3,4,5,4,5,6,7,8,9,10]`. Same sequence as 03-31 via a different generator.

⚠ U-Net split uses `len(virtual_layers) // 2`. With 13 virtual layers → encoder 6, decoder 7. That means the *last* decoder layer never receives a skip (`if skips: x = x + skip_weights[i] * skips.pop()` skips when the list is empty). This asymmetry is baseline-inherited but now has a larger footprint because of the expanded virtual layer count.

#### `2026-04-05_SP8192_GPTQ-Embeddings_SDClip_Loop45x2` — static 2-loop pattern

```python
if h.num_loops > 0:
    loop_seg = list(range(h.loop_start, h.loop_end + 1))
    all_indices = list(range(h.loop_start))
    for _ in range(h.num_loops + 1):
        all_indices.extend(loop_seg)
    all_indices.extend(range(h.loop_end + 1, h.num_layers))
    num_enc = len(all_indices) // 2
    self.encoder_indices = all_indices[:num_enc]
    self.decoder_indices = all_indices[num_enc:]
```

With `loop_start=4, loop_end=5, num_loops=2`: `loop_seg=[4,5]`, `all_indices = [0,1,2,3] + [4,5] + [4,5] + [4,5] + [6,7,8,9,10] = [0,1,2,3,4,5,4,5,4,5,6,7,8,9,10]` — 15 virtual layers, layer pair 4-5 appears **three** times.

⚠ **Recurrence forward now gated by `looping_active`**, a separate boolean; `num_loops > 0` at construction determines *potential* loop structure, but the loop is only traversed if the flag is on. This lets the model train as 11-layer, flip on at `enable_looping_at=0.5`, then train as 15-virtual-layer for the rest.

⚠ **Skip weight shape depends on whether looping is used**: `num_skip_weights = min(len(encoder_indices), len(decoder_indices))`. For the 15-virtual version, that's `min(7, 8) = 7`. For the non-looped 11L version, `min(5, 6) = 5`. So the `skip_weights` parameter has a different shape in the looped vs non-looped builds. You cannot migrate a checkpoint between them without padding.

#### `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` — 3-layer recurrence (current SOTA)

```python
loop_start = int(os.environ.get('LOOP_START', 3))
loop_end   = int(os.environ.get('LOOP_END', 5))
num_loops  = int(os.environ.get('NUM_LOOPS', 2))
enable_looping_at = float(os.environ.get('ENABLE_LOOPING_AT', 0.35))
```

`loop_seg = [3, 4, 5]`, `all_indices = [0,1,2] + [3,4,5] + [3,4,5] + [3,4,5] + [6,7,8,9,10]` = 17 virtual layers. Encoder = first 8 = `[0,1,2,3,4,5,3,4]`, decoder = last 9 = `[5,3,4,5,6,7,8,9,10]`. `num_skip_weights = min(8, 9) = 8`.

⚠ **Earlier activation**: `enable_looping_at=0.35` (not 0.5). At 35% of wallclock ≈ 205s, the model flips from 11L to 17-virtual-layer and trains there for the remaining 65% (which includes the full warmdown). Combined with `warmdown_frac=0.72`, the recurrence activation happens ~40s *after* warmdown begins — the model adapts to higher effective depth while the LR is already decaying.

⚠ **Which layers to recur?** The README says layer 3 is at the early/mid Hessian-trace boundary (early blocks ≈10× the trace of middle blocks). Extending the loop set from `[4,5]` to `[3,4,5]` picks up a layer whose importance is neither trivial nor critical — suitable for recurrence because the extra passes help but don't destabilize.

⚠ **Gradient flow under recurrence**: the repeated physical layer receives gradients from *every* virtual position where it appears. In the SOTA, layer 4 appears 4 times (positions 4, 7, 9, 12 in `all_indices`). Its weights see 4× the gradient signal per step compared to layer 0 (which only appears once). The effective learning rate for recurred layers is higher, which is why `muon_wd=0.095` was tuned — to counteract the larger total gradient magnitude.

### 1.8 Layer-0 attention disabled

#### `2026-03-31_ParallelResiduals_MiniDepthRecurrence`

```python
if self.disable_layer0_attn and phys_idx == 0:
    # Layer 0: MLP only, no attention.
    ...
```

Layer 0 runs MLP only. Its attention weights (`blocks.0.attn.c_q`, `c_k`, `c_v`, `proj`) are *not allocated* in the bank (the function `_drop_disabled_layer0_attn_unbanked` strips them from the state dict before GPTQ).

⚠ Parameter savings are redirected to wider auxiliary features (BigramHash scale, VE, etc.). Layer 0 still has `attn_norm` and `attn_scale` parameters, but they are unused (dead branches). The `attn_norm` module is constructed but never called — a kept-around artifact for state-dict compatibility.

⚠ This choice did **not** propagate to current SOTA — the 04-09 SOTA has `disable_layer0_attn` disabled. The hypothesis "layer 0 is a feature extractor, not a transformer block" was reversed when SP8192 + depth recurrence made earlier layers more valuable.

### 1.9 MLP activation: LeakyReLU²

#### `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`

```python
def forward(self, x: Tensor, up_w: Tensor, down_w: Tensor) -> Tensor:
    x = F.leaky_relu(F.linear(x, up_w.to(x.dtype)), negative_slope=0.5)
    return F.linear(x.square(), down_w.to(x.dtype))
```

Baseline MLP is `torch.relu(self.fc(x)).square()` (ReLU then square). This replaces `relu` with `leaky_relu(_, negative_slope=0.5)`. Negative inputs no longer go to zero — they're halved, and the square operation turns them into positive contributions.

⚠ **Output is still non-negative** (`x.square()` after `leaky_relu`). But the *gradient through the activation* for negative inputs is no longer zero: `d(leaky_relu(x, 0.5))/dx = 0.5` for negative `x`, so `d(x^2 @ W)/d(input)` includes a non-zero contribution from negative pre-activations. This keeps gradient flow through the entire MLP input without breaking the "non-negative hidden" invariant.

⚠ README ablation gives **−0.0021 BPB post-TTT** for this change alone — the biggest single contribution in that record. Inherited by all subsequent frontier records.

### 1.10 Factored embedding / embed dim bottleneck

#### `2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon` (dead branch)

```python
vocab_size = 8192
model_dim  = 768
embed_dim  = 254   # bottleneck: 8192 × 254 in fp8
# ... with a separate `embed_proj: CastedLinear(254, 768)` up-projecting
```

Factored embedding: instead of `[vocab, model_dim] = [8192, 768]` (6.3M fp8 ≈ 6.3MB), use `[vocab, embed_dim] + [embed_dim, model_dim] = [8192, 254] + [254, 768]` (2.3MB total in fp8). The bottleneck saves ~4MB.

⚠ **Does not propagate to current frontier**. The SP8192 frontier records keep untied-dimension embedding (`[8192, 512]`) and rely on GPTQ + SDClip to fit within 16MB. The ternary branch's factored embedding was a budget-management strategy for the wider 768d model; when the frontier settled on 512d, the factoring was unnecessary.

### Architecture axis — summary

- **What's in the current SOTA** (`2026-04-09`): 11 physical blocks with `mlp_mult=4`, LeakyReLU² activation, per-layer `ln_scale_factor = 1/√(layer+1)`, BigramHash (3072×112 XOR form), SmearGate, ValueEmbedding, **17 virtual layers** via 3-layer recurrence of `[3, 4, 5]`, **parallel residuals** on layers 7-10 via the clean `block.parallel` form, sigmoid-gated skips with `torch.lerp`. No layer-0 disable. No factored embedding. No one-shot embedding init.
- **Dead branches**: overtone spectral embed init, phase-transition `resid_mix` init, layer-0 attn disable, factored embedding (ternary branch).
- **Biggest single architectural win**: LeakyReLU² (−0.0021 BPB). Second: parallel residuals (−0.002 to −0.003 post-TTT). Third: depth recurrence (−0.002 to −0.003, more at 3 layers than 2).

*(Remainder of document follows: Axes 2-8, cross-cutting, failures, reading guide. See section "Axis 2: Attention" next.)*

<!-- REMAINING AXES GO HERE - PENDING USER FEEDBACK ON FORMAT -->
