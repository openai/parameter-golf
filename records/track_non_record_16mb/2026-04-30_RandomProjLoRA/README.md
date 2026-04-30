# Random Projection + Low-Rank Adapter MLPs

**Author:** Ammar Falah ([@Ammar12Falah](https://github.com/Ammar12Falah))
**Track:** Non-Record (Requested PR — "Learning adapters on random linear maps")
**Date:** 2026-04-30

## TL;DR

I replace the MLP linear layers of the baseline GPT with the parameterization
`W = R(seed) + α · BA`, where `R` is a frozen Gaussian regenerated from an
8-byte seed at construction time, and `BA` is a low-rank learned adapter of
rank `r`. Only the seed-derived buffer and the adapter `(A, B)` are part of
the model.

This is one of the techniques OpenAI listed under **Requests for PRs**
("Learning adapters on random linear maps") and to the best of my knowledge
had no implementation in either the record or non-record tracks at the time
of submission. The submission is intended as a clean reference
implementation and a small two-rank ablation, not a SOTA attempt.

**Headline result:** rank=64, **val_bpb = 1.4135** (single seed, sp1024
baseline tokenizer, 600 s wallclock cap on 1× H100).

## Hypothesis

Random-projection theory (Johnson–Lindenstrauss, Rahimi–Recht random
features) suggests that high-dimensional Gaussian linear maps preserve
geometry well enough that a network can recover much of a fully-trained MLP
layer's behavior on top of a frozen random projection, given a small
learned correction. If true, the parameter cost of MLP weights — typically
the dominant term in a small parameter-constrained transformer — can be cut
substantially with sub-linear quality loss.

The challenge's 16 MB artifact cap makes this hypothesis directly testable:
any byte freed by replacing dense MLP weights with `seed + adapter` can be
spent on more layers, larger `d_model`, larger vocab, or longer training.

## Method

### The module

```python
class RandomLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, seed, alpha=1.0, bias=False):
        super().__init__()
        # Frozen random projection (non-persistent buffer; reproducible from seed).
        g = torch.Generator(device="cpu").manual_seed(seed & 0x7FFFFFFF)
        R = torch.randn(out_features, in_features, generator=g) / math.sqrt(in_features)
        self.register_buffer("R", R, persistent=False)
        # Learned low-rank adapter, LoRA-style init: A kaiming, B zero.
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.R) + self.alpha * F.linear(F.linear(x, self.A), self.B)
```

Compute is two cheap matmuls (`xA^T`, then `(xA^T)B^T`) instead of forming
a dense `BA` and adding it to `R`. Memory cost during training is dominated
by the frozen `R` buffer, but `R` is not part of the optimizer state, so
optimizer memory scales only with the much smaller `(A, B)`.

### What's replaced and what isn't

- **Replaced:** every `CastedLinear` inside every `MLP` block (`fc` and
  `proj`) is swapped for a `RandomLoRALinear` of the same shape, with two
  distinct seeds per block (`seed = 10000 + 2·layer_idx + {0,1}`).
- **Untouched:** attention QKV/O projections, token embedding, RMSNorm
  weights, scalar parameters (`q_gain`, `attn_scale`, `mlp_scale`,
  `resid_mix`, `skip_weights`), and the tied LM head. Attention weights
  are smaller in absolute terms, are more directly involved in
  routing/structure, and were left dense to isolate the effect of the MLP
  intervention. Extending the technique to attention is noted as future
  work below.

### Initialization at zero perturbation

Standard LoRA init: `A ~ Kaiming-uniform`, `B = 0`. At step zero the network
sees pure random projection (`B @ A = 0`), so the adapter starts as a
no-op and learns its way out. Training was stable at both ranks tested
with no NaN events.

### Compute graph compatibility

`R` is registered as a non-persistent buffer (regenerated from seed at
module construction). `A` and `B` are fp32 `nn.Parameter`s in line with
the existing `CastedLinear` pattern, cast to bf16 at matmul time. This
keeps the module fully compatible with the existing
`torch.compile(fullgraph=True)` pipeline, the Muon/Adam optimizer split
(both `A` and `B` are 2-D matrix params and route to Muon), and the
existing int8+zlib post-training quantization export path.

## Setup

- **Hardware:** 1× H100 PCIe (Runpod, official Parameter Golf template).
- **Vocab:** sp1024 (baseline tokenizer, unchanged).
- **All other hyperparameters:** baseline defaults from `train_gpt.py`
  (9 layers, `d_model=512`, 8 heads, 4 KV heads, MLP mult 2, tied
  embeddings, seq len 1024, 524k tokens/step, 10-min wallclock cap).
- **Optimizer:** unchanged from baseline (Muon for matrix params at
  lr=0.04, Adam for embeddings/scalars).
- **Single seed (1337) per configuration.** No averaged runs; this is a
  non-record reference implementation, not a record attempt requiring
  `p < 0.01` significance.
- **Wallclock cap:** standard 600 s. Both runs hit the cap before
  reaching the nominal 20,000 iterations and stopped early.

## Results

### Rank ablation (single seed, 10-min wallclock cap, sp1024)

| Configuration             | val_bpb (roundtrip) | val_loss | params      | steps reached | artifact (int8+zlib) |
|---|---|---|---|---|---|
| Baseline (leaderboard)    | 1.2244              | —        | —           | —             | —                    |
| Random+LoRA, **rank=16**  | 1.4472              | 2.4436   | 8,065,096   | 1,236         | 6.87 MB              |
| Random+LoRA, **rank=64**  | **1.4135**          | **2.3867** | **9,392,200** | 1,209       | 8.93 MB              |

### Observations

- **Adapter capacity matters.** Going from rank=16 to rank=64 cut val_bpb
  by 0.034 nats (1.4472 → 1.4135). The adapter is doing meaningful work
  on top of the random projection — this is not pure noise, the network
  is genuinely using the extra rank to correct `R`.
- **Both configurations fit comfortably under the 16 MB cap.** The larger
  artifact (rank=64) at 8.93 MB still leaves ~7 MB of headroom that future
  work could spend on more layers or larger `d_model`.
- **The submission does not match the dense-MLP baseline (1.2244).** This
  is the central honest finding: a frozen random projection in MLP weights
  costs roughly 0.19 nats relative to a fully-trained dense MLP at this
  scale (rank=64). The compression-vs-quality tradeoff at small param
  budgets is real and not free, but it is not catastrophic — the
  random-projection MLP captures most of the dense-MLP behavior with a
  much smaller adapter than would be required for a from-scratch low-rank
  factorization. The trend from rank=16 to rank=64 suggests further rank
  scaling may continue to close the gap.

### Why I expected better and what's actually happening

The motivating intuition was that the released parameter budget (from
replacing dense MLPs with `seed + adapter`) could be redirected toward
*more capacity elsewhere* — more layers, larger vocab, etc. — and that
the combined effect could close the gap with the dense baseline. **In
this submission I did not exercise that redirection.** I held the
architecture identical to baseline and only swapped the MLP linears,
isolating the effect of the parameterization itself. The result is
therefore a clean measurement of the *cost* of the random-projection MLP
under fixed architecture, not an optimization of total quality. Future
work below addresses this explicitly.

## Discussion

### What worked

- **Training is stable** at both ranks. Zero-init `B` produces clean loss
  curves with no NaN events and the standard initial loss spike around
  step 2 (consistent with the dense baseline on this codebase).
- **The parameterization is fully compatible** with the existing infra:
  `torch.compile(fullgraph=True)`, Muon/Adam split, int8+zlib
  quantization export. No special-casing required.
- **Compression accounting matches theory.** For an MLP `(d, 4d)` block
  at rank `r=64`, the adapter stores `r·(d + 4d) = 5rd = 163,840` params
  vs the naive `4d² = 1,048,576` params, a ~6.4× reduction *per matrix*
  in parameter count.

### What didn't work / open issues

- **`R` is currently serialized.** A cleaner version would store *only*
  the 8-byte seed and regenerate `R` at load time, which would actually
  realize the theoretical compression benefit on the artifact byte
  count. The current submission keeps `R` in the saved checkpoint
  (`persistent=False` was set on the buffer but the surrounding export
  logic still pulls weights through the state_dict path). This is the
  single most impactful follow-up.
- **No headroom redirection.** As noted above, the freed budget was not
  redeployed into a larger architecture, so the experiment measures only
  the *cost* of the parameterization, not its potential *upside*.
- **No attention coverage.** Attention V/O projections are likely safe
  candidates for the same treatment; QK is more sensitive but worth
  trying with a smaller alpha.
- **Two ranks only, single seed.** Non-record submission, so no
  statistical-significance bar applies, but more ranks (rank=128, 256)
  and a 2nd seed would have been useful and were omitted only due to
  the deadline.

### What I would do with more time

1. **True seed-only storage.** Modify the export path so `R` is stripped
   from the saved state and regenerated at load. This is the version
   that actually delivers on the parameter-golf premise.
2. **Spend the freed budget.** With true seed-only storage, scale up
   `d_model`, `mlp_mult`, or `num_layers` and re-measure.
3. **Continue the rank sweep** to find where adapter capacity saturates.
   The rank=16 → rank=64 trend suggests there's room above rank=64.
4. **Structured random matrices.** Replace dense Gaussian `R` with
   Hadamard / FastFood / orthogonal random projections. Same theoretical
   compression, faster forward pass, and possibly better-conditioned
   adapters.
5. **Random-projection attention.** Apply to V/O first (safer), then QK
   with an alpha schedule.
6. **Joint search over `(rank, init_scale, α)`.** I held `α=1.0` and used
   He-init for `R`. A short sweep would likely move val_bpb measurably.

## Reproducibility

```bash
# 1. Clone and download data (sp1024 variant)
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# 2. Run the headline rank=64 result (or substitute 16 for ablation)
RUN_ID=lora_r64 RANDOM_LORA_RANK=64 \
    torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Environment variables added:

- `RANDOM_LORA_RANK` (int, default 0 = disabled / dense baseline).
  Setting > 0 enables the RandomLoRA MLP swap at that rank.
- `RANDOM_LORA_ALPHA` (float, default 1.0). Scaling factor on the
  adapter contribution.

Trained on 1× H100 PCIe via Runpod, official Parameter Golf template
image. Each run hit the 600 s wallclock cap before completing 20,000
iterations.

## Acknowledgements

Built on the OpenAI Parameter Golf baseline (`train_gpt.py`) and adapts
the code structure from `modded-nanogpt`. The random-projection-plus-adapter
direction is one of OpenAI's Requests for PRs in the challenge README.
