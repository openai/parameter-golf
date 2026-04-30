# Behemoth GEN-2: The Orthogonal Temporal Router

- Goal: Improve Behemoth GEN1-V12 by unifying its three core mechanisms into a coherent architectural thesis.
- Parent: `models/Behemoth/history/v12` (GEN1 final, gradient explosion fix)

## Thesis

The residual stream is a bandwidth-limited communication channel. Standard transformers waste this bandwidth through representation collapse (redundant features) and struggle to route information across long temporal horizons.

The Orthogonal Temporal Router addresses these sequential bottlenecks through a three-stage mechanism: **Filter, Diversify, and Compress**.

## The Three Stages

### 1. Filter: Adaptive Depth Gating

Before computation occurs, the pre-normed gate evaluates the residual stream. If the current features are sufficient, it routes shallow embedding context (`x_0`) forward, preventing over-processing and preserving gradient flow.

- Gate logits clamped to `(-10.0, 10.0)` and pre-normed with `RMSNorm`
- Acts as a dynamic parameter allocator: early layers process, late layers can skip
- Zero-init weight plus bias `2.0` means the gate starts open (`sigmoid(2.0) ~= 0.88`), defaulting to standard processing

### 2. Diversify: Orthogonal Branch Forcing

When the block computes, OBF acts as a repelling force between the Attention and MLP branches. By explicitly penalizing cosine similarity (`lambda = 0.003`), Attention specializes in token-to-token routing while MLP specializes in feature transformation. This packs the residual stream with maximum, non-redundant information.

- `aux_obf_weight = 0.003`
- Delayed start at step `750`, allowing early gradients to organize the network before enforcing orthogonality
- `O(D)` complexity: mean-pool across batch and sequence, then cosine similarity on the direction vectors

### 3. Compress and Broadcast: Macro Pyramid Self-Distillation

Because the residual stream is now densely packed with orthogonal features, standard causal attention degrades over long contexts. The Macro sidechannel steps in at the encoder/decoder boundary to compress these dense features into temporal summaries. It distills them and safely injects them back into future tokens using heavily stabilized cross-attention, improving long-range credit assignment.

- Interval summaries extracted from the last token per interval
- Student-teacher distillation: student sees only past intervals and predicts the current interval
- Cross-attention with causal mask: tokens attend only to past summaries
- Critical stabilization: `RMSNorm` before all QKV projections and before injection
- Gated injection with clamped gate values `(-6.0, 6.0)`
- Higher learning rates become viable because `RMSNorm` stabilizes the context injections

## Architecture Diagrams

### Figure 1: The Orthogonal Routing Block (Micro View)

```text
       [Input: x, x_0]
              |
      +---------------+
      |  RMSNorm(x)   |  <-- Stabilization
      +-------+-------+
              |
    +---------+---------+
    | Linear + Clamp/Sig|  <-- Adaptive Depth Gate
    +---------+---------+
              | (Gate decides split between x and x_0)
              v
         [ x_in ]
              |
      +-------+-------+
      |               |
  +---v---+       +---v---+
  | Attn  |<--X-->|  MLP  |  <-- Orthogonal Branch Forcing (OBF)
  +---+---+ Repel +---+---+      Penalizes cosine similarity
      |               |
      +-------+-------+
              v
        [ Output: x ]
```

### Figure 2: The Macro-Temporal Highway (Macro View)

```text
 [ Dense Orthogonal Features from Encoder ]
                   |
                   v
        +---------------------+
        | Interval Summarizer |  <-- Extracts last token per interval
        +----------+----------+
                   |
      Teacher +----+----+ Student
     (Current)|         | (Shifted + Predicted)
              v         v
    +-----------+     +-----------+
    | Distill   |<--->| Distill   |  <-- MSE loss matches student to teacher
    | Proj/Pred |     | Proj/Pred |
    +-----------+     +-----+-----+
                            |
            +---------------+---------------+
            |       Cross-Attention         |
            |  Tokens Attend to Summaries   |
            +---------------+---------------+
                            |
                    +-------+-------+
                    | RMSNorm(ctx)  |  <-- Critical variance stabilization
                    +-------+-------+
                            v
              [ Gated Injection into x ]
                            |
                            v
                  [ To Decoder Blocks ]
```

## Implementation Reference

### The Stabilized Block (Adaptive Depth + OBF)

```python
class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        qk_gain_init: float,
        adaptive_depth: bool = True,
        obf_enabled: bool = True,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.adaptive_depth = adaptive_depth
        self.obf_enabled = obf_enabled
        if self.adaptive_depth:
            self.depth_gate = nn.Linear(dim, 1, bias=True)
            nn.init.zeros_(self.depth_gate.weight)
            nn.init.constant_(self.depth_gate.bias, 2.0)

    def forward(self, x: Tensor, x0: Tensor) -> tuple[Tensor, Tensor]:
        # 1. FILTER: Adaptive Depth Gate
        if self.adaptive_depth:
            x_norm_for_gate = F.rms_norm(x, (x.size(-1),))
            gate_logits = self.depth_gate(x_norm_for_gate).clamp(-10.0, 10.0)
            gate = torch.sigmoid(gate_logits)
            branch_input = gate * x + (1.0 - gate) * x0
        else:
            branch_input = x

        attn_out = self.attn(self.attn_norm(branch_input))
        mlp_out = self.mlp(self.mlp_norm(branch_input))

        # 2. DIVERSIFY: Orthogonal Branch Forcing
        obf_loss = torch.zeros((), device=x.device, dtype=torch.float32)
        if self.training and self.obf_enabled:
            a_dir = attn_out.float().mean(dim=(0, 1))
            m_dir = mlp_out.float().mean(dim=(0, 1))
            obf_loss = F.cosine_similarity(
                a_dir.unsqueeze(0), m_dir.unsqueeze(0), dim=-1
            ).abs().squeeze()

        x_out = (
            branch_input
            + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
            + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        )
        return x_out, obf_loss
```

### The Stabilized Temporal Highway

```python
def _apply_macro_sidechannel(self, x: Tensor) -> tuple[Tensor, Tensor]:
    B, L, D = x.shape
    interval = self.macro_interval
    n_intervals = L // interval

    # Summarizer
    raw_summaries = x.reshape(B, n_intervals, interval, D)[:, :, -1, :].contiguous()

    # Distillation (student predicts teacher)
    shifted = torch.cat(
        [
            torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
            raw_summaries[:, :-1, :],
        ],
        dim=1,
    )
    student = shifted + self.macro_pred(F.rms_norm(shifted, (D,)))

    # 3. COMPRESS AND BROADCAST
    token_interval = torch.arange(L, device=x.device) // interval
    macro_pos = torch.arange(n_intervals, device=x.device)
    mask = macro_pos.unsqueeze(0) <= token_interval.unsqueeze(1)

    # Critical stabilization: pre-norm before attention projections
    x_norm = F.rms_norm(x, (D,))
    student_norm = F.rms_norm(student, (D,))

    q = self.macro_q(x_norm)
    k = self.macro_k(student_norm)
    v = self.macro_v(student_norm)

    scores = torch.bmm(q, k.transpose(-1, -2)) * (self.macro_xattn_dim ** -0.5)
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    context = torch.bmm(attn, v)

    # Critical stabilization: safe injection
    context_norm = F.rms_norm(context, (D,))
    gate_val = self.macro_gate.to(dtype=x.dtype).clamp(-6.0, 6.0)
    x = x + torch.sigmoid(gate_val)[None, None, :] * context_norm

    # Distillation MSE loss
    distill_loss = torch.zeros((), device=x.device, dtype=torch.float32)
    if self.training and self.macro_distill_weight > 0.0 and n_intervals > 2:
        proj = self.macro_distill_proj(F.rms_norm(raw_summaries, (D,)))
        distill_loss = F.mse_loss(
            self.macro_distill_pred(proj[:, :-1, :]),
            proj[:, 1:, :].detach(),
        )

    return x, distill_loss
```

### The Integration Logic (Forward Pass)

```python
# Inside GPT._forward_impl:
obf_loss_total = torch.zeros((), device=x.device, dtype=torch.float32)

# Collect micro OBF loss
for i in range(self.num_encoder_layers):
    x, block_obf_loss = self.blocks[i](x, x0)
    obf_loss_total = obf_loss_total + block_obf_loss

# Apply macro highway
x, distill_loss = self._apply_macro_sidechannel(x)

for i in range(self.num_decoder_layers):
    x, block_obf_loss = self.blocks[self.num_encoder_layers + i](x, x0)
    obf_loss_total = obf_loss_total + block_obf_loss

obf_loss_total = obf_loss_total / len(self.blocks)
return x, distill_loss, obf_loss_total
```

### Training Loop Loss Composition

```python
ce_loss = F.cross_entropy(logits.float(), targets, reduction="mean")

# Apply OBF strictly after step 750 with weight 0.003
if self.training and step >= 750:
    ce_loss = ce_loss + 0.003 * obf_loss_total

if self.training and self.macro_distill_weight > 0.0:
    ce_loss = ce_loss + self.macro_distill_weight * distill_loss

return ce_loss
```

## Design Decisions

### What is kept from GEN1-V12

- Macro Pyramid Self-Distillation (the main architectural asset, now with higher-LR headroom from RMSNorm stabilization)
- Adaptive Depth Gating (logits clamped to `(-10.0, 10.0)`, pre-normed)
- Orthogonal Branch Forcing (OBF weight increased to `0.003`, delayed start at step `750`)
- Encoder-decoder split with U-Net skip connections
- Full int4 QAT from step 1 with STE
- The critical v12 gradient fix: `RMSNorm` before all macro QKV projections

### What is removed from GEN1

- BigramHash
- SWA
- SmearGate

Removing these local-context injection mechanisms frees up the 16 MB budget and allows a cleaner evaluation of the Orthogonal Temporal Router as the sole architectural innovation.

## Next Steps

1. Implement the unified `train_gpt.py` with the three-stage architecture.
2. Smoke test on `1xH100` (2-minute cap) and verify no NaNs and a decreasing loss.
3. Run a 5-minute `1xH100` experiment to establish baseline BPB for GEN-2.
4. Sweep learning rate, since RMSNorm stabilization should unlock higher LR than GEN1.
5. Run a full 10-minute `8xH100` experiment and measure real BPB.
6. Compare against GEN1-V12 at matched training budget.
