# H-Net on Parameter Golf — Implementation Scope

Companion doc to the investigation PR. Details the proposed H-Net variant targeted for the 16 MB / 10-min-on-8×H100 budget, milestone gates, and failure modes.

Reference: Hwang et al., *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, arXiv:2507.07955 (Jul 2025).

## 1. Architecture at a glance

```
bytes  →  byte-encoder  →  dynamic chunker  →  main network  →  byte-decoder  →  per-byte logits
   |         (thin,          (Wq, Wk;           (deep,                ( thin,
   |         D_enc=256)      boundary_prob      D_main=512)           D_enc=256)
   |                         + STE + EMA)           ^
   |                                                |
   └─ byte emb (256 × D_enc, ≈66K params) ──────────┘
```

The byte encoder/decoder are thin (2 layers each) transformers at D_enc=256. The main network is essentially bigbag's current SP8192 stack (11 layers, D_main=512) but now consumes / produces a *chunk* stream, not a SP8192-token stream. The dynamic chunker decides which encoder outputs are chunk boundaries.

## 2. Parameter budget (int6 quantized + Brotli, targeting 16 MB)

| component | params (≈) | quantized bytes (≈) | notes |
|---|---:|---:|---|
| byte embedding (256 × D_enc) | 66 K | 60 KB (int8 like SP8192) | vs. 2.1 M / 2 MB for SP8192 |
| byte encoder (2 × transformer blocks at D_enc=256, 8 heads/4 KV, MLP 3×) | 1.5 M | 1.2 MB | tied layer weights with decoder would halve this |
| chunker (Wq + Wk, both D_enc × D_enc) | 130 K | 100 KB | unchanged from paper |
| main network (11 L × D_main=512 × GQA × MLP 4×, same as bigbag) | 22 M | 14 MB | inherits SP8192-stack fusions: parallel residuals, 3-layer loop, QK-Gain, GPTQ+SDClip |
| byte decoder (1 × block at D_enc=256) | 0.8 M | 0.6 MB | can be tied with encoder layers if tight |
| **total** | **~24.5 M** | **~16 MB** | some slack for upsampler + final LN + skip weights |

At 24–26 M params we're slightly under bigbag's 35.9 M (they spend ~4 MB on the SP8192 token embedding we eliminated). The saved budget goes into either (a) a deeper main network, (b) a wider byte encoder, or (c) an explicit upsampler with dedicated params.

## 3. Dynamic-chunker design (copy directly from paper)

Per-position boundary probability:

$$
p_t = \tfrac{1}{2} \Big( 1 - \frac{q_t^{\top} k_{t-1}}{\Vert q_t \Vert \, \Vert k_{t-1} \Vert} \Big)
$$

where $q_t = W_q \hat{x}_t$, $k_t = W_k \hat{x}_t$ and $\hat{x}_t$ is the byte-encoder output at position $t$.

Downsampling: select encoder outputs where $b_t = \mathbf{1}[p_t > 0.5]$. Discard the rest.

Smoothing for gradient flow (EMA):

$$
\bar{z}_t = P_t \hat{z}_t + (1 - P_t) \bar{z}_{t-1},
$$

and a Straight-Through Estimator rounds $P_t$ to 1 in forward but preserves real-valued gradients.

## 4. Losses

$$
\mathcal{L} = \mathcal{L}_{\text{AR}} + \alpha \, \mathcal{L}_{\text{ratio}}, \quad \alpha = 0.03
$$

$\mathcal{L}_{\text{AR}}$: standard per-byte autoregressive cross-entropy (full 256-token vocab; no SentencePiece).

$\mathcal{L}_{\text{ratio}}$: encourages a target compression ratio $r \approx 3.5$ (matches SP8192's effective bytes/token on English FineWeb). The paper's form:

$$
\mathcal{L}_{\text{ratio}} = \big( r \cdot F - G \big)^2, \quad F = \tfrac{1}{T}\sum_t b_t, \quad G = \tfrac{1}{T}\sum_t p_t
$$

(F is the actual fraction of chunk boundaries, G is the mean boundary probability.)

## 5. Sequence lengths and compute

SP8192 baseline: seq_len = 2048 tokens ≈ 2048 × 3.5 ≈ 7200 bytes per sequence.

Byte-level H-Net at equivalent data coverage: seq_len_bytes = 7200.

- Byte encoder runs on 7200 bytes per sequence. FA3 at 7200 seq is fine (memory-bound, ~2× slower than 2048).
- Main network runs on ~2000 chunks per sequence (after ~3.5× compression). Matches current SP8192 compute budget.
- Byte decoder runs on 7200 bytes again.

Net forward FLOPs roughly doubled vs. SP8192 (the extra byte encoder + decoder passes). This is the main compute risk — we may need to reduce main-network depth (11 → 9 or 10 layers) to fit the 10-min budget. Milestone 1 measures this.

## 6. Milestone plan (aligned with $500 dev grant scope)

### Milestone 1 — hierarchical stack with **fixed** chunker ( ≈ $60 GPU)

Before learning the chunker, prove the hierarchical stack trains. Fix boundaries at every $r$-th byte (stride-3 or stride-4 deterministic). Train byte-encoder + main-network + byte-decoder end-to-end at 1× H100 scale (reduced iters, SP8192-equivalent vocab).

**Gate**: val_bpb better than a byte-level transformer of the same total param count (simple byte LM, no hierarchy). If worse, the upsampler / reinjection path is broken and we fix before learning the chunker.

### Milestone 2 — learned chunker with EMA + STE ( ≈ $120 GPU)

Drop the fixed chunker. Add Wq, Wk + ratio loss. Watch for:
- Degenerate collapse (all boundaries or no boundaries).
- Chunker "freezing" (boundary positions stop moving once set).
- Ratio-loss mis-specification.

**Gate**: non-degenerate boundary distribution (F ∈ [0.2, 0.4]) and val_bpb ≤ Milestone 1 value.

### Milestone 3 — full 16 MB submission ( ≈ $200 GPU)

Turn on all the SP8192-stack bells: parallel residuals on main, 3-layer depth loop, MuonEq-R, GPTQ+SDClip int6, Brotli-11. 3-seed mean on 8× H100 SXM.

**Gate**: artifact < 16 MB, eval < 10 min, val_bpb competitive with the best non-record SP8192 results (~1.10 BPP). Hitting the SP8192-record threshold (1.016 as of 2026-04-14) is aspirational and not required for the PR to be a creative contribution.

### Milestone 4 — ablations ( ≈ $120 GPU)

- Compression-ratio sweep (r ∈ {2, 3, 3.5, 4, 5}).
- Byte-encoder depth (1 / 2 / 3 layers).
- Chunker variants (cosine sim vs. small MLP, see paper §E).

Published as an update to this PR, not a separate PR.

## 7. Failure modes and abort criteria

| risk | likelihood | mitigation / abort |
|---|---|---|
| Chunker collapses (always / never chunk) | medium | EMA + STE + ratio loss (paper's recipe). If still collapses at M2, try Gumbel-softmax variant before aborting. |
| Scale too small for hierarchy to help | medium-high | Paper only shows ≥680M. If M1 byte-stack is worse than a plain byte transformer at 25 M params, H-Net may simply not work this small — abort and pivot grant remainder to documenting the negative result. |
| Byte-level compute exceeds 10 min | medium | Reduce main-network depth from 11 → 9; or add FA3 to byte encoder too. |
| SOTA stack (depth recurrence, parallel residuals) incompatible with chunk stream | low | They operate on the main-network token dimension, should transfer directly. |
| Training instability from joint loss | low-medium | Warmup α from 0 to 0.03 over first 500 steps. |

## 8. Why this is the right bet now

- **Unclaimed** on the repo's explicit Requests-for-PRs list.
- **Complements** rather than competes with current SOTA (GDN-Hybrid, varlen + fused MLP, parallel residuals, adaptive TTT). The current SOTA stacks all start from SP8192 or SP4096; H-Net attacks the tokenizer axis they can't touch.
- **Concrete payoff** from the paper: 3.5–4× effective byte compression + ~4× better data efficiency on code / non-Latin / DNA. FineWeb has heavy code fragments; exactly the regime H-Net helps most.
- **Within our budget**: $500 grant covers 4 milestones. No milestone individually requires more than ~$200.

## 9. Prior art we're NOT duplicating

- `#1548 dljr-github` "Frozen Random Backbone + LoRA Adapters" — a different adapter-on-random-backbone idea, not H-Net.
- `#973 mrbese` "38-token structured alphabet + BPE" — a fixed alternative tokenizer, not learned chunking.
- `#1312 / #1480 / #1581` JEPA submissions — a different hierarchical idea (representation learning, not dynamic chunking).
- `#1582 / #1596` masked-diffusion submissions — orthogonal to tokenization.

No open PR has attempted learned byte-level chunking as of 2026-04-14.
