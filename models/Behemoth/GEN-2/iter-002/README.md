# Behemoth GEN2-v2: Orthogonal Temporal Router

> Revised blueprint. Changes from v1 are marked as `[Changed]`.
>
> Reference: [iter-001 README](../iter-001/README.md)

## Philosophy

**Filter -> Diversify -> Compress.** Treat the residual stream as a bandwidth-limited channel. Do not waste computation (Filter), maximize information density (Diversify), and solve long-range dependency handling (Compress).

## Structural Change: `BlockAttentionResidual` Replaces the U-Net Split `[Changed]`

The fixed encoder/decoder boundary at layer 5 is the one rigid decision in an otherwise adaptive architecture. `BlockAttentionResidual` makes depth routing learned and continuous, which means:

- The macro sidechannel is freed from its role as an encoder-decoder bridge and becomes purely a temporal compression mechanism.
- Skip connections are replaced by learned softmax attention over all completed depth levels.
- The model can route different tokens through different effective depths.

```text
GEN2 forward

x0 = embed + bigram + rms_norm
completed = [x0]

Group 0 (layers 0-1)
  x = block[0](x, x0)
  x = block[1](x, x0)
completed.append(x)

Group 1 (layers 2-3)
  x = block_attn(completed, x)   # depth routing
  x = block[2](x, x0)
  x = block[3](x, x0)
completed.append(x)

Group 2 (layers 4-5)
  x = block_attn(completed, x)
  x = block[4](x, x0)
  x = block[5](x, x0)
completed.append(x)

Macro fires here (after group 2)
  x, distill = macro_sidechannel(x)

Groups 3-4 (layers 6-9)
  x = block_attn(completed, x)
  ... layers 6-9 with depth routing ...

x = block_attn(completed, x)     # final route
x = final_norm(x)
```

**Why macro fires after group 2 (layer 5):** representations are rich enough to summarize by that point, while five layers still remain to use the injected context. This keeps the depth placement from v12's encoder/decoder boundary, but makes the skip structure around it adaptive.

## Pillar 1: Filter - Adaptive Depth Gating

Unchanged from v1.

```python
class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        mlp_mult,
        rope_base,
        qk_gain_init,
        group_size=128,
    ):
        super().__init__()
        self.norm = RMSNorm()  # [Changed] single norm
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, group_size
        )
        self.mlp = MLP(dim, mlp_mult, group_size)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        # Adaptive Depth Gate
        self.depth_gate = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.depth_gate.weight)
        nn.init.constant_(self.depth_gate.bias, 2.0)  # sigmoid(2) = 0.88

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        # FILTER
        x_norm = F.rms_norm(x, (x.size(-1),))
        gate = torch.sigmoid(self.depth_gate(x_norm).clamp(-10.0, 10.0))
        branch_input = gate * x + (1.0 - gate) * x0

        normed = self.norm(branch_input)  # [Changed] one call
        attn_out = self.attn(normed)      # both branches read
        mlp_out = self.mlp(normed)        # the same normed input

        return (
            branch_input
            + self.attn_scale.to(x.dtype)[None, None, :] * attn_out
            + self.mlp_scale.to(x.dtype)[None, None, :] * mlp_out
        )
```

### Changes from v1

- `attn_norm` and `mlp_norm` merge into a single `self.norm`. They computed the same function on the same tensor, so this saves 10 `rms_norm` kernel launches per forward pass.
- The block now returns `Tensor` instead of `tuple[Tensor, Tensor]`. OBF is moved out of the block, which keeps the compiled graph cleaner.

## Pillar 2: Diversify - Weight-Space OBF `[Changed]`

The mean-pool activation OBF is replaced with a weight-space penalty. Instead of averaging outputs into a single direction and measuring cosine similarity, the new version checks whether the attention output projection and MLP output projection write into overlapping directions of the residual stream.

```python
# Computed every 50 steps in the training loop, not in forward().
# Zero runtime cost during forward pass.
def compute_obf_loss(base_model: GPT) -> Tensor:
    """Penalize overlap between attention and MLP output column spaces."""
    loss = torch.zeros((), device=next(base_model.parameters()).device)
    for block in base_model.blocks:
        # Both branches write into the same R^dim residual stream, so compare
        # column spaces along dim=0 (the output axis).
        a_cols = F.normalize(block.attn.proj.weight.float(), dim=0)
        m_cols = F.normalize(block.mlp.w_down.weight.float(), dim=0)

        # Cross-Gram of residual-space write directions.
        cross = a_cols.transpose(0, 1) @ m_cols  # (dim, hidden)
        loss = loss + cross.pow(2).mean()

    return loss / len(base_model.blocks)
```

Training loop integration:

```python
if step >= args.obf_start_step and step % args.obf_every == 0:
    obf_loss = compute_obf_loss(base_model)
    (args.obf_weight * obf_loss).backward()
```

### Why this is better

`proj.weight` and `w_down.weight` both write into the same `R^dim` residual stream, so their column spaces are the comparable objects. If those column spaces overlap, the branches are writing redundant information into the same residual directions. The squared Frobenius norm of the cross-Gram matrix directly penalizes that overlap.

This removes activation dependence, avoids the mean-pool approximation, and runs every 50 steps outside the compiled graph, so the other 49 steps pay zero throughput cost.

## Pillar 3: Compress - Streamlined Macro Sidechannel `[Changed]`

```python
def _apply_macro_sidechannel(self, x: Tensor) -> tuple[Tensor, Tensor]:
    B, L, D = x.shape
    interval = self.macro_interval
    n_intervals = L // interval

    raw_summaries = x.reshape(B, n_intervals, interval, D)[:, :, -1, :].contiguous()

    # [Changed] No macro_pred. Shift raw summaries directly.
    # macro_pred was 262K params on a near-zero gradient path.
    shifted = torch.cat(
        [
            torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
            raw_summaries[:, :-1, :],
        ],
        dim=1,
    )

    # Causal cross-attention
    token_interval = torch.arange(L, device=x.device) // interval
    macro_pos = torch.arange(n_intervals, device=x.device)
    mask = macro_pos.unsqueeze(0) <= token_interval.unsqueeze(1)

    x_norm = F.rms_norm(x, (D,))
    s_norm = F.rms_norm(shifted, (D,))

    q = self.macro_q(x_norm)
    k = self.macro_k(s_norm)
    v = self.macro_v(s_norm)

    scores = torch.bmm(q, k.transpose(-1, -2)) * (self.macro_xattn_dim ** -0.5)
    scores = scores.masked_fill(~mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    context = torch.bmm(attn, v)

    context = F.rms_norm(context, (D,))
    gate_val = self.macro_gate.to(x.dtype).clamp(-6.0, 6.0)
    x = x + torch.sigmoid(gate_val)[None, None, :] * context

    # [Changed] Distillation reads post-enrichment summaries.
    distill_loss = torch.zeros((), device=x.device, dtype=torch.float32)
    if self.training and self.macro_distill_weight > 0.0 and n_intervals > 2:
        enriched = x.reshape(B, n_intervals, interval, D)[:, :, -1, :].contiguous()
        proj = self.macro_distill_proj(F.rms_norm(enriched, (D,)))
        teacher = proj[:, 1:, :].detach()
        student = self.macro_distill_pred(proj[:, :-1, :])
        distill_loss = F.mse_loss(student, teacher)

    return x, distill_loss
```

### Changes from v1

- `macro_pred` is removed, saving 262K parameters. Raw shifted summaries go directly through Q/K/V.
- Distillation now reads post-enrichment summaries, so the distill head trains on what the decoder actually sees.
- Level 2 pyramid is removed for now. It added about 8 ms per step for unclear benefit and can be reintroduced after the base architecture proves out.

## Depth Routing: `BlockAttentionResidual` `[New]`

```python
class BlockAttentionResidual(nn.Module):
    """Learned depth routing replacing the U-Net encoder/decoder split."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.proj = nn.Linear(model_dim, 1, bias=False)
        self.norm = RMSNorm()

    @torch.compiler.disable
    def forward(self, blocks: list[Tensor], current: Tensor) -> Tensor:
        sources = [*blocks, current]
        scores = torch.stack(
            [self.proj(self.norm(src)).squeeze(-1) for src in sources],
            dim=0,
        )
        attn = torch.softmax(scores, dim=0)

        out = torch.zeros_like(current)
        for i, src in enumerate(sources):
            out = out + attn[i].unsqueeze(-1) * src
        return out
```

`512` parameters total. This replaces `5x SkipGate` (`248K` parameters) plus the fixed encoder/decoder split. `torch.compiler.disable` forces `fullgraph=False`, which costs about `2 to 5%` throughput, but removes the SkipGate compute and recovers that parameter budget.

## Forward Pass Integration

```python
def _forward_impl(self, input_ids):
    emb_w = fake_quantize_4bit(self.tok_emb.weight, self.group_size)
    x = F.embedding(input_ids, emb_w)

    if self.bigram is not None:  # [Changed] kept, smaller
        x = x + self.bigram(input_ids)

    x = F.rms_norm(x, (x.size(-1),))
    x0 = x  # [Changed] no SmearGate

    completed = [x0]
    macro_done = False

    for group_idx in range(self.n_groups):  # 5 groups of 2
        if group_idx > 0:
            x = self.block_attn(completed, x)

        for j in range(self.block_size):  # 2 layers per group
            x = self.blocks[group_idx * self.block_size + j](x, x0)

        completed.append(x)

        # Macro fires once after the middle group.
        if group_idx == self.macro_after_group and not macro_done:
            x, distill_loss = self._apply_macro_sidechannel(x)
            macro_done = True

    x = self.block_attn(completed, x)  # final depth route
    return x, distill_loss
```

## What's Kept, What's Cut, What's Changed

| Component | v1 (your blueprint) | GEN2-v2 | Rationale                                               |
| --- | --- | --- |---------------------------------------------------------|
| Adaptive Depth Gate | Yes | Yes (unchanged) | Proven, low cost                                        |
| OBF | Mean-pool cosine in forward | Weight-space Frobenius every 50 steps | Zero runtime cost, measures real overlap                |
| Macro Level 1 | Yes, with `macro_pred` | Yes, without `macro_pred` | 262K dead params removed                                |
| Macro Level 2 | Yes | Cut for now | About 8 ms per step for unverified benefit              |
| Macro distillation | Pre-enrichment summaries | Post-enrichment summaries | Aligns signal with decoder input                        |
| U-Net split + skips | Fixed boundary + scalar/gated | `BlockAttentionResidual` | Adaptive depth routing, `512` vs `248K` params          |
| BigramHash | Cut entirely | Kept at `(1536, 64)` | Proven `-0.005 BPB`, only `130K` params at small config |
| SmearGate | Cut | Cut | Marginal value                                          |
| SWA | Cut | Replace with EMA (`0.997`) | `-0.003 BPB`, zero GPU cost                             |
| `aux_obf_probe` | Cut | Cut | OBF moved to weight-space                               |
| Dual `rms_norm` per block | `attn_norm + mlp_norm` | Single `self.norm` | Identical computation, saves 10 kernel launches         |
| Surprisal loss | Batch-mean normalized | EMA-normalized | Stable gradient scale across batches                    |
| `torch.compile` | `fullgraph=True` | `fullgraph=False` | Required by `BlockAttentionResidual`                    |

## Parameter Budget

- `v12`: `34,756K` params -> `15,987K` compressed
- `GEN2-v2`: `~33,500K` params (estimated)
- Removed: `macro_pred (262K)`, `Level 2 macro (400K)`, `SkipGate x5 (248K)`, `SmearGate (512)`, large `BigramHash` (`1.24M` saved from shrink)
- Added: `BlockAttentionResidual (512)`, small `BigramHash (130K)`
- Headroom: about `1.5 to 2 MB` freed, enough room for an 11th layer if needed
- Throughput target: about `85 to 90 ms/step` (vs. v12 at `108 ms/step`)
- Step target: about `6,600 to 7,000` in `600s` (vs. v12 at `5,577`)

This keeps the three pillars intact: Filter, Diversify, Compress. The surrounding structure is simpler and cleaner.

## OBF Hyperparameters

Both OBF timing and weight should be hyperparameters. A fixed start at `750` is probably too late for `1xH100` iteration runs, where total step count is only around `1,100`. That leaves too little time for the penalty to noticeably reshape the weight matrices.

A more practical default is to start OBF after roughly `10 to 15%` of total convergence, exposed as tunable integers and floats:

```python
obf_start_step = int(os.environ.get("OBF_START_STEP", 500))
obf_weight = float(os.environ.get("OBF_WEIGHT", 0.003))
obf_every = int(os.environ.get("OBF_EVERY", 50))
```

Then:

```python
if step >= args.obf_start_step and step % args.obf_every == 0:
    obf_loss = compute_obf_loss(base_model)
    (args.obf_weight * obf_loss).backward()
```

Defaulting to `500` instead of `750` gives roughly 600 steps of OBF pressure on `1xH100` and thousands more on `8xH100`, while keeping all three knobs easy to override for ablations.
