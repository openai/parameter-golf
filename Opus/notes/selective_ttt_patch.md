# Selective-Param TTT — Patch Design

**Goal:** Replace the current "TTT updates every parameter" behavior with "TTT updates only the parameters we choose," gated by a new env var `TTT_PARAM_FILTER`.

## Why

The SOTA's TTT (line 343 of `Opus/code/train_gpt_base.py`):

```python
ttt_params=[p for p in base_model.parameters()]
for p in ttt_params: p.requires_grad_(True)
optimizer = SGD(ttt_params, lr=h.ttt_lr, momentum=h.ttt_momentum)
```

Trains **every parameter**, including the int6-quantized matrices that have already been GPTQ-rounded. Two problems:

1. **Fighting the quantization grid.** Most matrix parameters live on the int6 grid after GPTQ. SGD updates push them off the grid, but the artifact gets re-quantized at evaluation time anyway... except it doesn't — `eval_val_ttt` operates on the dequantized fp32 model and never re-quantizes. So every TTT step is *erasing* part of the careful GPTQ rate-distortion work for that chunk's downstream tokens.
2. **High variance, low signal.** Most of the parameter mass is in `c_q/c_k/c_v/proj` and `fc/proj`. Updating all of them on a 32K-token chunk × 3 epochs is a tiny amount of data per parameter — most of those updates are noise.

The hypothesis: training **only the small fp32 control surface** (`q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`, `skip_gates`) gives the model knobs to re-balance attention/MLP contribution per chunk *without* corrupting the matrix weights. That's a few-thousand-float surface, well-suited to 32K tokens of training.

## Param accounting

From `notes/sota_architecture.md`:

| Class | Pattern in name | Total floats | Quantized? |
|-------|-----------------|--------------|------------|
| `tok_emb` | `tok_emb` | 4.19M | int8 |
| Attention matrices | `.attn.c_q/c_k/c_v.weight`, `.attn.proj.weight` | 7.2M | int6 |
| MLP matrices | `.mlp.fc.weight`, `.mlp.proj.weight` | 23.1M | int6 |
| `q_gain` | `q_gain` | 88 | fp32 (control) |
| `attn_scale` | `attn_scale` | 5,632 | fp32 (control) |
| `mlp_scale` | `mlp_scale` | 5,632 | fp32 (control) |
| `resid_mix` | `resid_mix` | 11,264 | fp32 (control) |
| `skip_weights` | `skip_weights` | ~8K | fp32 (control) |
| `skip_gates` | `skip_gates` | ~8K | fp32 (control) |
| **Control total** | (`CONTROL_TENSOR_NAME_PATTERNS`) | **~38K** | fp32 |

Adapting 38K floats vs 34M floats is a 1000× reduction in update surface.

## The filter options

`TTT_PARAM_FILTER` env var, default `all` (matches current SOTA exactly):

| Value | Selects | Rationale |
|-------|---------|-----------|
| `all` | every param | baseline (matches PR #1493) |
| `scales` | params matching `CONTROL_TENSOR_NAME_PATTERNS` | minimal fp32 surface, hypothesis-1 |
| `scales+embed` | scales + `tok_emb` | scales plus int8 token embedding |
| `last_n_layers:K` | params in blocks `[N-K, N)` | concentrate adapt on top |
| `attn_only` | params with `.attn.` in name | attention-only |
| `mlp_only` | params with `.mlp.` in name | MLP-only |

Default is `all` so reproducing the SOTA needs no env-var change.

## Implementation

Two minimal changes to `train_gpt_base.py`:

### 1. Add the hyperparameter

Add to the `Hyperparameters` class:

```python
ttt_param_filter = os.environ.get('TTT_PARAM_FILTER', 'all')
```

### 2. Replace the param-collection line in `eval_val_ttt`

Original (line 343):
```python
ttt_params=[p for p in base_model.parameters()]
```

Replacement: select via filter, freeze the rest, restore at the end.

```python
named = list(base_model.named_parameters())
filt = h.ttt_param_filter
if filt == 'all':
    ttt_params = [p for _, p in named]
elif filt == 'scales':
    ttt_params = [p for n, p in named
                  if any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
elif filt == 'scales+embed':
    ttt_params = [p for n, p in named
                  if 'tok_emb' in n
                  or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
elif filt.startswith('last_n_layers:'):
    K = int(filt.split(':', 1)[1])
    L = len(base_model.blocks)
    keep = {f'blocks.{i}.' for i in range(L - K, L)}
    ttt_params = [p for n, p in named if any(n.startswith(k) for k in keep)]
elif filt == 'attn_only':
    ttt_params = [p for n, p in named if '.attn.' in n]
elif filt == 'mlp_only':
    ttt_params = [p for n, p in named if '.mlp.' in n]
else:
    raise ValueError(f"Unknown TTT_PARAM_FILTER: {filt}")

# Freeze everything not selected so backward pass is cheaper.
selected = set(id(p) for p in ttt_params)
for n, p in named:
    p.requires_grad_(id(p) in selected)
```

The existing line 376 `for p in base_model.parameters(): p.requires_grad_(True)` already restores all gradients post-TTT, so callers see no behavior change.

## Risks

- **Frozen attention matrices may not be enough surface for adaptation.** If `scales` underperforms `all`, fall back to `scales+embed` or `last_n_layers:3`.
- **Gradient all-reduce list shrinks** — fewer params to all_reduce, faster TTT (good).
- **Quantized matrices stored as int but loaded as float** — TTT updates on `all` actually update the dequantized float values that GPTQ produced. Selective TTT with `scales` only changes if the matrix grad path is now skipped — `requires_grad=False` skips both autograd record and parameter update, so we save backward compute too.

## Validation

A spot-check script (`Opus/scripts/validate_param_filter.py`) instantiates the model with random weights and verifies each filter selects the expected parameter names. Runs locally on CPU.
