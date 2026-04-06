# Non-record: Parallel Residuals + Hessian-Aware SDClip (3-seed mean 1.08354 BPB)

**val bpb: 1.08354** (3-seed mean, std=0.00050)

Not a record. This is a small 3-seed experiment over [PR #1394](https://github.com/openai/parameter-golf/pull/1394) on my runs, but not enough evidence for a statistical claim — the seed count is too small for confidence. Posting because the changes are zero-cost, reproducible, and may be useful to others trying out different techniques. 

| Seed | Steps | Pre-quant BPB | Post-quant BPB | **Sliding BPB** | Artifact |
|-|-|-|-|-|-|
| 1337 | 5178 | 1.08765 | 1.09959 | **1.08301** | 15,976,275 |
| 42 | 5180 | 1.08816 | 1.10013 | **1.08363** | 15,978,439 |
| 3141 | 5182 | 1.08872 | 1.10044 | **1.08399** | 15,979,649 |
| **Mean** | | 1.08818 | 1.10005 | **1.08354** | 15,978,121 |

## Changes

Three zero-cost modifications on top of [PR #1394](https://github.com/openai/parameter-golf/pull/1394), adding zero extra parameters or bytes:

### 1. Parallel Residuals (Layers 7+)

GPT-J style parallel attention+MLP ([Wang & Komatsuzaki, 2021](https://github.com/kingoflolz/mesh-transformer-jax)) for the last 4 layers. Both attention and MLP read from the same input and their outputs are added in parallel:

```
# Parallel (layers 7-10):
x_out = x + attn_scale * Attn(norm(x)) + mlp_scale * MLP(norm(x))

# Sequential (layers 0-6, unchanged):
h = x + attn_scale * Attn(norm(x))
x_out = h + mlp_scale * MLP(norm(h))
```

I expected parallel residuals to reduce interference between attention and MLP during GPTQ calibration. Pre-quant BPB barely moved, but the quantization gap tightened across all 3 seeds, which made this the most useful change in practice.

### 2. Hessian-Aware SDClip

I used GPTQ's existing Hessian diagonal as a cheap importance signal to slightly modulate SDClip thresholds by row:

$$c_i = k \cdot \text{std}(\text{row}_i) \cdot [1 + \lambda(\text{row\_importance}_i - 1)], \quad \lambda = 0.175$$

where row importance is derived from Hessian-weighted magnitude. The effect is small but directionally useful at $\lambda = 0.175$; higher $\lambda$ hurt compression. I initially used $\lambda = 0.30$ but found $\lambda = 0.175$ is consistently better across seeds — both lower BPB and smaller artifact. Higher $\lambda$ reduces rounding error but increases entropy, which makes Brotli compression less effective.

### 3. Progressive Recurrence

Depth recurrence split into two phases: first loop enabled at 50% of training, second at 65%. The split points were not optimized — 50% matches the original and 65% was a single manual choice. Enabling both loops at once causes a sharper loss spike; splitting gives the model time to adapt to each additional pass before adding the next.

## Hessian Analysis (Cross-Seed)

Hessian diagnostics from 3 seeds, 67 matrices each:

- **Group-level traces** (early/loop/mid/late blocks): $r=0.997$ across seeds
- **Per-matrix traces**: $r=0.994$
- **Per-row importance**: $r=0.12$ (noise)

Importance hierarchy: early blocks (30x trace of late blocks) >> loop >> mid >> late. Per-row importance is too noisy to be a reliable signal, but group-level traces are very stable across seeds. This suggests per-group clip allocation could be a useful direction.

## Future Directions

Several ideas I'd like to explore with more compute time:

- **Per-group clip allocation**: Non-uniform $k$ across layer groups, using the stable group-level trace hierarchy as a guide.
- **Output-Hessian weighting**: Using backward-pass gradients for output-side row importance rather than input-side alone.
- **More seeds**: 3 seeds is not enough for strong statistical claims. I'd want 5+ to be confident about the gap vs PR #1394.
- **YAQA**: I like the idea of the paper ([arXiv:2505.22988](https://arxiv.org/abs/2505.22988)), but I couldn't get a working backward pass for it. I think maybe it could be adapted for the parameter golf problem in an interesting way. I also like the math in Mousse ([arXiv:2603.09697](https://arxiv.org/abs/2603.09697)), but exploiting curvature in small LMs seems tough.

## Run Command

```bash
HESSIAN_CLIP_LAMBDA=0.175 LOOP_PHASE2_AT=0.65 PARALLEL_RESIDUAL_START=7 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt_sweep.py
```

## Requirements

Flash Attention 3 (Hopper) required. SP8192 BPE tokenizer trained on FineWeb 10B (sentencepiece BPE, 8192 vocab).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install --no-cache-dir \
  "https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
pip install -r requirements.txt
```

## Compliance (Track A — Fixed Predictor)

- No TTT, SLOT, n-gram cache, or eval-time adaptation
- GPTQ calibration within training budget
- Standard autoregressive sliding-window eval (stride=64)


## Credits

Learned from and inspired by [PR #1394](https://github.com/openai/parameter-golf/pull/1394) (@clarkkev) — SDClip, depth recurrence, and GPTQ embedding quantization ideas. Parallel residuals from GPT-J ([Wang & Komatsuzaki, 2021](https://github.com/kingoflolz/mesh-transformer-jax)). Additional credits: PR #1204 (@msisovic, depth recurrence), PR #1217 (@bigbag, MuonEq-R), PR #1019 (@abaybektursun, previous SOTA).
