- So, first of all, this is to train a LM that is under 16MB artifact (we'll get to this later) and this should be under 10 minutes 8xH100s.
- This will be tested by compression on FineWeb validation set and it is tokenizer-agnostic, bits per byte (not sure what this means either, so we'll get to this as well).
- Very very similar to Modded-NanoGPT speedrun.
- Problem is definitely going to be parameter contrainted like in NanoGPT-speedrun so we need to be cheeky about the approach (can take inspiration from NanoGPT tricks people have already used).
- It is like L(N) optimization problem, where we have to optimize the lowest loss given a fixed number of N parameters, without thinking about data, compute, steps, or arch (?? why not this).
- Now WTF is this 16MB artifact? So this artifact is computed as "code bytes" + "compressed model bytes" (as in code generated binary + final model size??). Also 16MB is 16,000,000 total bytes and no other interpretation.
- Restrictions on evaluation are interesting. They are saying that eval shouldn't take more than 10mins on 8xH100 + we have 10mins of training time. Eval are done at varying seq_len. And for the most important part, we are allowed to access training data, yes it is allowed, during eval, if and only if we pay for those 'bits' in the 16MB limit (this is very interesting coz they are literally saying go MF wild on this so prolly TT-compute, TT-training & other tricks??).

-----------------------------------------------------------------------------
### Compute Form Responses.

to start with experiment, there are some obv problems to address:

- make the model give higher probability for next token (can do this by adding recurrence or a engram-mech where same tokens are hashed).
- less tokens means less tokens per byte, making for a better val_bpb. (using a diff tokenizer)
- since we have to have limited size of artifact, instead of having 9 diff blocks each with their own weight (which is redundant, if the output remains ~ same). So using only 2-3 shared weights (stored with a correction-term, which should be more parameter-efficient).
- quanti is also very obv here as the current storage is row-scale with int8 entries in row . we can change the not-so imp ones with int4, imp ones with fp16 & rest can be int8.
-----------------------------------------------------------------------------

### train_gpt.py

- 9 transformer blocks at width 512
- 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
- vocab size 1024, seq_len 1024, tied embeddings
- 524,288 train tokens per step for 20,000 iterations with a ~10min cap

---

# Complete Walkthrough of `train_gpt.py`

## Lines 1-5: Module Docstring

```python
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points ...
Hard stop: ... never are longer than 1500 lines.
"""
```

**What:** A guard-rails docstring for contributors.
**Why:** This file lives in a "parameter golf" challenge repo — participants compete to train the best small GPT under strict size/time budgets. The docstring prevents the baseline script from becoming an unreadable monolith. The 1500-line hard cap is a social contract, not enforced by code.

---

## Lines 7-28: Imports

```python
from __future__ import annotations
```

**What:** Defers evaluation of type annotations to strings.
**Why:** Lets you write `tuple[int, ...]` and `Tensor | None` without Python 3.9 crashing — the annotations are never actually evaluated at import time. Cheap forward-compat trick.

```python
import copy, glob, io, math, os, random, subprocess, sys, time, uuid, zlib
from pathlib import Path
```

**What:** Standard library imports.
**Why (notable ones):**
- `copy` — deep-copying optimizer state dicts for the warmup-reset trick (line 984).
- `zlib` — compressing the final int8 checkpoint. The competition scores on `code + model` size, so every byte matters.
- `uuid` — generating a unique run ID if none is provided. Prevents log collisions when running multiple experiments.

```python
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
```

**What:** External/PyTorch dependencies.
**Why:**
- `sentencepiece` is used for byte-pair-encoding (BPE) tokenization. The competition uses a custom 1024-token vocabulary (very small!) to keep embedding tables tiny.
- `DDP` enables multi-GPU training by synchronizing gradients across processes. The script is designed to work with `torchrun`.

---

## Lines 39-89: `Hyperparameters` Class

```python
class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    ...
```

**What:** A plain class (no `__init__`) where every attribute is a class variable with an environment-variable override.
**Why:** This is a zero-boilerplate config pattern. Every hyperparameter has a sane default, but you can override any of them via env vars (`VOCAB_SIZE=2048 python train_gpt.py`). No argparse, no YAML, no config files — just `os.environ.get`. This keeps the script self-contained and trivially reproducible: the environment fully specifies the run.

### Key hyperparameters explained:

| Line | Parameter | Default | Why |
|------|-----------|---------|-----|
| 41-44 | `data_path`, `train_files`, `val_files`, `tokenizer_path` | FineWeb 10B paths | Points to pre-sharded training data. The glob pattern `fineweb_train_*.bin` matches multiple shard files. |
| 45 | `run_id` | random UUID | Unique identifier for logs. Prevents overwriting if you run multiple experiments. |
| 46 | `seed` | 1337 | Reproducibility. Classic ML seed choice. |
| 49-51 | `val_batch_size`, `val_loss_every`, `train_log_every` | 524K, 1000, 200 | `val_loss_every=1000` means validate every 1000 steps. Validation is expensive, so not every step. |
| 54-59 | Training length params | 20K iters, 524K tokens/step, seq_len 1024, 600s cap | The **wallclock cap** (`max_wallclock_seconds=600`) is the competition constraint — you get 10 minutes. `train_batch_tokens=524288` = 512K tokens per step = `512 * 1024`, a standard power-of-2 batch. |
| 55-56 | `warmdown_iters`, `warmup_steps` | 1200, 20 | `warmup_steps=20` is for **torch.compile warmup**, not LR warmup — the model/optimizer states get reset after (line 1000). `warmdown_iters=1200` is a cosine-style LR decay at the end of training. |
| 60 | `qk_gain_init` | 1.5 | A learnable scalar multiplied into Q after QK-norm. Initializing at 1.5 (>1) effectively makes attention sharper at init. See line 582/595. |
| 63-73 | Model shape | vocab=1024, layers=9, dim=512, heads=8, kv_heads=4, mlp_mult=2 | Tiny model by design. `vocab_size=1024` is extraordinarily small — this is the "parameter golf" constraint. `num_kv_heads=4` with `num_heads=8` means **Grouped Query Attention** (GQA) — each KV head serves 2 Q heads, halving KV parameter count. `mlp_mult=2` means hidden dim = 1024, which is narrow (GPT-2 uses 4x). |
| 65-66 | `num_shared_blocks`, `shared_block_pattern` | `num_layers`, `"cycle"` | **Weight sharing**: by default, all 9 layers share the same block cores (weights). `"cycle"` means layer `i` uses core `i % num_shared_blocks`. With `num_shared_blocks = num_layers`, each layer gets its own core (no sharing). This is a parameter-saving knob. |
| 71 | `tie_embeddings` | `True` | The input embedding matrix IS the output projection. Saves `vocab_size * model_dim` parameters (512K params here). Standard trick from "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017). |
| 72 | `rope_base` | 10000.0 | Standard RoPE frequency base from the original paper. |
| 73 | `logit_softcap` | 30.0 | Caps logit magnitude via `tanh`. Borrowed from Gemma 2 — prevents logit explosion in small models. |
| 76-89 | Optimizer HPs | various | Separate learning rates for embeddings, head, matrices, and scalars. This is **muP-inspired** (maximal update parameterization) — different parameter shapes need different learning rates. `muon_momentum=0.95` and `backend_steps=5` control the Muon optimizer. `muon_momentum_warmup_*` linearly ramps momentum from 0.85 to 0.95 over 500 steps, which stabilizes early training. |

---

## Lines 98-170: Muon Optimizer

### `zeropower_via_newtonschulz5` (lines 98-111)

```python
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X
```

**What:** Computes the **matrix sign function** (or "zero-power" / orthogonalization) of a gradient matrix using Newton-Schulz iteration.

**Why:** Muon's core idea is: instead of using the raw gradient to update weight matrices, first **orthogonalize** it. This means the update direction has equal magnitude in all singular-value directions, preventing the optimizer from collapsing updates into the dominant gradient direction. This is the spectral version of "normalize your gradients" — but for matrices.

- `a, b, c = (3.4445, -4.7750, 2.0315)` — These are the 5th-order polynomial coefficients for the Newton-Schulz iteration that converges to the matrix polar factor. They're pre-computed to maximize convergence speed.
- `X /= X.norm() + eps` — Normalize first so the iteration starts in the convergence basin. This works because `||X||_spectral <= ||X||_Frobenius`, so dividing by the Frobenius norm guarantees the spectral norm is <= 1, which is inside the convergence basin for these polynomial coefficients. See the spectral norm explanation below.
- `transposed = G.size(0) > G.size(1)` — The iteration works better when the matrix is "wide" (more cols than rows). If tall, transpose, iterate, transpose back.

#### What is the spectral norm?

**The goal: find the spectral norm of a matrix A.**

The spectral norm asks a simple question: what's the most `A` can stretch any vector? Mathematically: `||A||_2 = max(||Ax||_2 / ||x||_2)`. Both norms here are just regular vector L2 norms.

**The problem: how do you find that maximum?**

You can't test every possible vector. So we need a smarter route. When you square the definition, you get `||A||_2^2 = max(x^T A^T A x / x^T x)`. The matrix `A^T A` naturally appears because squaring the L2 norm of `Ax` introduces `A^T`. This expression is the Rayleigh quotient of `A^T A`, and its maximum equals the largest eigenvalue of `A^T A`.

**So now we need eigenvalues of `A^T A`.**

We find them by solving `det(A^T A - lambda * I) = 0`. The `lambda * I` only subtracts from the diagonal because the identity matrix is zero everywhere else. Solving the resulting polynomial gives us the eigenvalues.

**How do these relate to singular values?**

SVD says `A = U @ Sigma @ V^T` (rotate, stretch, rotate). Plugging into `A^T A` gives `V @ Sigma^2 @ V^T`. The middle `U` cancels (`U^T U = I`) because it appears next to itself. `V @ Sigma^2 @ V^T` is an eigendecomposition — `V` provides the eigenvectors, `Sigma^2` provides the eigenvalues. So the eigenvalues of `A^T A` equal the squared singular values.

**Putting it all together:**

spectral norm -> max stretch -> need largest eigenvalue of `A^T A` -> that eigenvalue equals `sigma_max^2` -> spectral norm = `sqrt(sigma_max^2)` = **`sigma_max`**

The spectral norm of `A` is simply its largest singular value — its biggest stretching factor.

**Why this matters here:** The Newton-Schulz iteration only converges when the spectral norm (largest singular value) of the input is within a certain range (~1.7 for these coefficients). The `X /= X.norm()` line divides by the Frobenius norm, which is `sqrt(s1^2 + s2^2 + ... + sk^2)`. Since the spectral norm is just `s_max`, and `s_max <= sqrt(s1^2 + s2^2 + ... + sk^2)`, dividing by the Frobenius norm guarantees the spectral norm drops to <= 1 — safely inside the convergence basin.
- The inner loop computes `X_{k+1} = a*X_k + (b*A + c*A^2) @ X_k` where `A = X_k @ X_k^T`. After convergence, `X` is an orthogonal matrix that preserves the singular vectors of `G` but sets all singular values to 1.

Great question. "Orthogonal" here doesn't mean making things 90° in a geometric sense you'd visualize. It means something specific and powerful for optimization.

**What orthogonalization actually does in Muon:**

You have a gradient matrix G (the direction the optimizer wants to update weights). G has an SVD: G = UΣVᵀ. Orthogonalizing G means replacing it with **UVᵀ** — you keep the rotations but **set all singular values to 1**. You throw away Σ entirely.

Tying this back to our earlier discussion: Σ contains the singular values (the stretching factors). Some might be huge, some tiny. Orthogonalization says: "I don't care *how much* you want to stretch in each direction — I'll treat every direction equally."

**Why this helps:**

In normal SGD or Adam, if the gradient has one very large singular value and several small ones, the update is dominated by that one direction. The optimizer keeps hammering the same dominant direction while barely exploring others.Good question. Let me connect this to everything we've discussed.

**What "orthogonalize the gradient" means in Muon:**

Your gradient G is a matrix. It has an SVD: G = UΣVᵀ. The singular values in Σ might be wildly different — say σ₁ = 50, σ₂ = 0.3, σ₃ = 0.01. That means the gradient is screaming "move a LOT in direction 1" and whispering "barely move in direction 3."

Orthogonalization replaces G with **UVᵀ** — it sets all singular values to 1. Now every direction gets equal weight.

Here's a visual showing the difference:

/home/raven/Pictures/Screenshots/Screenshot From 2026-03-22 19-53-31.png

**Why this helps training:**

With a raw gradient, if σ₁ = 50 and σ₂ = 0.3, the optimizer takes a huge step in direction 1 and a tiny step in direction 2. Over many iterations, it keeps hammering direction 1 while barely exploring direction 2. This is wasteful — direction 2 might be exactly where the loss landscape has a useful descent path, like an escape route from a saddle point.

After orthogonalization, every direction gets an equal-sized step. The optimizer explores the full space uniformly. It's like the difference between a flashlight (one bright beam) and a lantern (equal light in all directions).

**Where Newton-Schulz fits:**

The "correct" way to orthogonalize is to compute the full SVD (G = UΣVᵀ), throw away Σ, and return UVᵀ. But SVD is expensive on GPUs. Newton-Schulz is an iterative method that approximates the same result using only matrix multiplications, which GPUs are extremely fast at, and can run stably in bfloat16.

In practice, just 5 iterations of Newton-Schulz get close enough — the result isn't exactly UVᵀ but something like US'Vᵀ where the singular values are noisy values around 1 instead of exactly 1. That turns out to be good enough for training.

So tying it all the way back: spectral norm measures maximum stretch → singular values are those stretch factors → orthogonalization sets all stretch factors to 1 → Newton-Schulz does this cheaply on GPUs → Muon uses this to make gradient updates treat all directions equally.

### `Muon` class (lines 114-170)

```python
class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True):
```

**What:** The Muon optimizer — applies orthogonalized updates with Nesterov momentum.

```python
    total_params = sum(int(p.numel()) for p in params)
    updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
```

**Why flatten into one buffer?** For distributed all-reduce. Instead of doing N small all-reduce calls (one per parameter), it packs all updates into one flat tensor and does a single all-reduce. This is dramatically faster because NCCL has high per-call overhead. See the detailed explanation below.

### How the Muon distributed all-reduce works

In **standard DDP**, PyTorch all-reduces **gradients** — and it already buckets them into large fused buffers automatically. So per-parameter all-reduce of gradients would be silly and DDP doesn't do that.

But **Muon doesn't all-reduce gradients**. It all-reduces the **post-Newton-Schulz orthogonalized updates**. Here's the flow:

1. Each rank has the full set of gradients (from DDP's gradient sync during backward)
2. Muon **shards the Newton-Schulz computation** across ranks — rank `r` only orthogonalizes parameters where `i % world_size == r`
3. Each rank writes its results into a flat buffer (zeros elsewhere)
4. They all-reduce (SUM) that buffer so every rank has the complete set of orthogonalized updates
5. Every rank applies the full update to its local copy of the model

So the all-reduce here is **not about syncing gradients** (DDP already did that). It's about **recombining the sharded Newton-Schulz outputs**. It's a work-distribution pattern:

```
         rank 0              rank 1
         ------              ------
grads:   [g0, g1, g2, g3]   [g0, g1, g2, g3]    <- identical (from DDP)
NS work: [NS(g0), 0, NS(g2), 0]   [0, NS(g1), 0, NS(g3)]   <- each rank does half
all-reduce SUM:
result:  [NS(g0), NS(g1), NS(g2), NS(g3)]  on both ranks
```

If there were only 1 GPU, there'd be no all-reduce at all — every rank would just orthogonalize every parameter itself. The sharding is purely a **compute parallelism** optimization, not a data-correctness requirement.

### Why one big all-reduce call beats many small ones

NCCL all-reduce has two cost components:

- **Latency (fixed overhead):** ~5-15us per call for kernel launch, synchronization, ring setup
- **Bandwidth (proportional to data size):** the actual bytes transferred over NVLink/PCIe

For a small matrix (say 512x512 = 1MB in bf16), the fixed overhead dominates. 45 calls x 10us = 450us of pure overhead. One fused call on the concatenated 45MB buffer pays the 10us overhead once and saturates the interconnect bandwidth much better.

So it's not that "many small all-reduces" is wrong in principle — it's that NCCL's per-call overhead makes it wasteful when you could batch them. DDP does the same thing internally (it groups gradients into ~25MB buckets before all-reducing). Muon just does it manually because it's operating outside of DDP's gradient machinery.

### Remaining Muon details

```python
    if i % world_size == rank and p.grad is not None:
```

**What:** Distributes the Newton-Schulz computation across ranks. Parameter `i` is processed by rank `i % world_size`.
**Why:** Newton-Schulz is expensive (multiple matrix multiplications per parameter). By sharding it across GPUs and then all-reducing, each GPU only orthogonalizes `1/world_size` of the parameters.

```python
    buf.mul_(momentum).add_(g)
    if nesterov:
        g = g.add(buf, alpha=momentum)
```

**What:** Standard Nesterov momentum: `buf = momentum * buf + grad`, then the actual update direction is `grad + momentum * buf`.

**Why the momentum appears twice (detailed trace):**

There **are** two multiplications by momentum, and that's the point. The two lines serve different roles:

1. **Line 1** updates the momentum buffer — this is the persistent state that accumulates across steps. Standard heavy-ball momentum.
2. **Line 2** computes the actual update direction using the **Nesterov look-ahead** — it "peeks" in the direction the momentum would carry you, then evaluates the gradient from that position.

Substituting `buf` into the second line:

```
g = grad + momentum * (momentum * buf_old + grad)
g = grad + momentum * grad + momentum^2 * buf_old
g = grad * (1 + momentum) + momentum^2 * buf_old
```

Compare to **classical momentum** (no Nesterov), which would just use `buf` directly as the update:

```
update = momentum * buf_old + grad                       # classical: just the buffer
```

vs **Nesterov**:

```
update = grad + momentum * (momentum * buf_old + grad)   # looks ahead
```

The intuition: classical momentum says "go where the accumulated gradient points." Nesterov says "first imagine you took the momentum step, then correct from there." That extra `momentum * grad` term is the correction — it weights the current gradient more heavily, making the optimizer more responsive to recent curvature changes.

In practice, Nesterov reduces oscillation when the momentum is overshooting. At `momentum=0.95`, the buffer has a lot of inertia. If the loss landscape curves away, classical momentum keeps barreling forward for several steps. Nesterov notices the curvature shift one step earlier because it evaluates the gradient at the look-ahead position.

The "doing it twice" is exactly the mechanism — the momentum appears squared (`momentum^2 * buf_old`) intentionally, because you're composing "where momentum carries you" with "the gradient at that future point."

```python
    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
    g *= max(1, g.size(0) / g.size(1)) ** 0.5
```

**What:** Orthogonalize, then apply a scale correction.
**Why the scale correction?** After orthogonalization, the Frobenius norm of the update is `sqrt(min(m,n))` for an `m x n` matrix. The `max(1, m/n)^0.5` factor normalizes this so that tall matrices (more rows than columns) get the same effective step size as square ones. Without this, layers with different aspect ratios would train at different effective rates.

```python
    if distributed:
        dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
```

**What:** Sum all updates across ranks. Since each rank only filled in its assigned parameters (and zeros elsewhere), the sum gives the complete update vector.

```python
    p.add_(g, alpha=-lr)
```

**What:** `p = p - lr * g` — apply the orthogonalized, momentum-corrected update.

---

## Lines 172-280: Tokenizer-Agnostic Evaluation

### `build_sentencepiece_luts` (lines 182-206)

```python
def build_sentencepiece_luts(sp, vocab_size, device):
```

**What:** Builds three lookup tables (LUTs) mapping each token ID to:
1. `base_bytes` — how many UTF-8 bytes the token represents
2. `has_leading_space` — whether the token starts with the SentencePiece word-boundary marker
3. `is_boundary_token` — whether it's a control/unknown/unused token

**Why:** The competition metric is **bits-per-byte (BPB)**, not loss-per-token. Different tokenizers compress text differently (a 1024-vocab tokenizer needs ~2 tokens where a 32K-vocab tokenizer needs ~1). BPB normalizes this by measuring compression in terms of raw bytes, making results comparable across vocabularies.

The leading-space handling (line 198-200) is subtle: SentencePiece encodes word boundaries as a leading special character. When calculating byte counts, the space byte should be counted **only if the previous token wasn't a boundary** (i.e., the space is inter-word, not sentence-initial). That logic lives in `eval_val` at line 268.

### `load_validation_tokens` (lines 209-218)

```python
def load_validation_tokens(pattern, seq_len):
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[: usable + 1]
```

**What:** Loads all validation shards into one contiguous tensor, truncated to a multiple of `seq_len` (plus one for the shifted target).
**Why:** The `+1` is because next-token prediction needs `(x[:-1], y[1:])` pairs — you always need one more token than you use. Truncating to a multiple ensures clean batching with no padding.

### `eval_val` (lines 221-280)

```python
def eval_val(args, model, rank, world_size, device, grad_accum_steps, val_tokens, ...):
```

**What:** Runs the full validation set and computes both `val_loss` (cross-entropy in nats) and `val_bpb` (bits-per-byte).

Key details:

```python
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
```

**Why:** Shards the validation set across ranks for parallel evaluation. Each rank processes its slice, then they all-reduce.

```python
    x = local[:-1].reshape(-1, args.train_seq_len)
    y = local[1:].reshape(-1, args.train_seq_len)
```

**What:** The classic causal LM setup — input is tokens `[0..n-1]`, target is tokens `[1..n]`.

```python
    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
```

**What:** Counts total bytes for the BPB calculation. If a target token has a leading space AND the previous token is not a boundary token, add 1 byte for the space.
**Why:** SentencePiece strips the leading space into its special marker but that space was a real byte in the original text. We only count it when there's a real preceding word (not at sentence start).

```python
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
```

**What:** `BPB = (loss_nats / ln(2)) * (tokens / bytes)`. Converting nats to bits, then scaling by the tokenizer's compression ratio.

---

## Lines 282-424: Post-Training Quantization

### Configuration constants (lines 290-310)

```python
CONTROL_TENSOR_NAME_PATTERNS = tuple(...)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_CLIP_PERCENTILE = 99.99984
```

**What:** Controls which tensors get quantized vs. kept in float.
**Why:**
- Small tensors (<=65K elements) are kept in float because quantizing them saves negligible space but can hurt quality — scales, norms, and biases fall here.
- `CONTROL_TENSOR_NAME_PATTERNS` lists names like `attn_scale`, `mlp_scale`, `resid_mix` — these are per-dimension scaling/mixing vectors that are very sensitive to quantization noise. Keeping them in fp32 preserves model quality.
- `INT8_CLIP_PERCENTILE = 99.99984` — Clips outliers before quantization. The 99.99984th percentile (~4.3 sigma for Gaussian) removes extreme outliers that would waste dynamic range. This is aggressive but works well for neural net weights.

### `quantize_float_tensor` (lines 323-342)

```python
def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    if t32.ndim == 2:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        ...
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE)
```

**What:** Per-row int8 quantization for matrices, per-tensor for vectors.
**Why per-row?** Different rows (output channels) of a weight matrix can have vastly different magnitudes. Per-row scales adapt to each row's range, giving much better quantization accuracy than a single per-tensor scale. The scale is stored in fp16 to save space (1 fp16 scale per row is negligible overhead).

The flow: clip outliers -> compute scale = `max_abs / 127` -> round to nearest int8 -> clamp to [-127, 127].

### `quantize_state_dict_int8` (lines 344-401)

**What:** Walks the full state dict and decides per-tensor: quantize to int8, keep as fp16 passthrough, or keep as exact passthrough.
**Why:** The competition scores on total file size. This function implements a tiered strategy:
1. Non-float tensors (rare) -> pass through as-is
2. Small float tensors (<=64K elements) -> downcast to fp16 (saves 50% vs fp32, no quant noise)
3. Large float tensors -> int8 + per-row scale (saves ~75% vs fp32)

The `passthrough_orig_dtypes` dict remembers whether each kept tensor was originally fp32 or bf16, so it can be perfectly restored later.

### `dequantize_state_dict_int8` (lines 403-424)

```python
    out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype)
```

**What:** Reverses the quantization: `float_value = int8_value * scale`, then cast back to the original dtype.
**Why the `view` trick?** The scale has shape `(rows,)` but the quantized tensor has shape `(rows, cols)`. The view adds trailing singleton dims so broadcasting works correctly: each row is scaled by its own scale factor.

---

## Lines 427-496: Data Loading

### `load_data_shard` (lines 431-445)

```python
def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype="<i4", count=256)
    if header[0] != 20240520 or header[1] != 1:
        raise ValueError(...)
```

**What:** Reads a binary shard file with a 256-int header followed by uint16 token data.
**Why:**
- `20240520` is a **magic number** (a date: 2024-05-20) that identifies valid shard files. If you accidentally point at a random file, this catches it immediately.
- `header[1] == 1` is a version check.
- `header[2]` stores the token count, which is validated against the actual file size.
- Tokens are stored as `uint16` (2 bytes each) — sufficient for vocab_size <= 65535 (our vocab is only 1024).

### `TokenStream` (lines 448-476)

```python
class TokenStream:
    def take(self, n: int) -> Tensor:
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
```

**What:** A sequential stream that reads tokens from shard files one after another, wrapping around to the first shard after the last.
**Why:** Simple and deterministic — no shuffling, no random access, no DataLoader workers. Each training step consumes exactly `n` tokens from the stream in order. This guarantees bit-for-bit reproducibility across runs. The `_advance_file` method handles shard boundaries seamlessly (a `take` call can span two shards).

### `DistributedTokenLoader` (lines 479-496)

```python
class DistributedTokenLoader:
    def next_batch(self, global_tokens, seq_len, grad_accum_steps):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span]
```

**What:** Wraps `TokenStream` for multi-GPU training. Each call consumes one global batch worth of tokens, then slices out this rank's portion.
**Why:** All ranks read from the **same** stream (same ordering), so their data is naturally disjoint — rank 0 gets the first span, rank 1 gets the second, etc. The `+1` per span is again for the `(input, target)` shift. `grad_accum_steps` divides the per-rank batch further so each micro-step gets its portion.

```python
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
```

**What:** Reshape into `(batch_size, seq_len)` pairs with the standard 1-position shift.

---

## Lines 498-605: Transformer Modules

### `RMSNorm` (lines 502-508)

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
```

**What:** Root-Mean-Square Layer Normalization — normalizes by `x / sqrt(mean(x^2))` without subtracting the mean and without a learnable affine (no gamma/beta parameters).
**Why:** RMSNorm is cheaper than LayerNorm (no mean computation) and works just as well in transformers. Having no learnable parameters here saves parameters — the scaling is handled by the separate `attn_scale` and `mlp_scale` in `BlockControls`.

### `CastedLinear` (lines 511-515)

```python
class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), ...)
```

**What:** A linear layer that **stores weights in fp32** but casts to the input's dtype (bf16) at compute time.
**Why:** Optimizer state quality. Adam accumulates first and second moments — doing this in bf16 causes drift and instability. By keeping weights in fp32, the optimizer sees full-precision parameters. The bf16 cast happens only during the forward/backward pass where the reduced precision is acceptable. This is a manual version of what AMP's `GradScaler` does, but more explicit and compatible with Muon (which isn't a standard PyTorch AMP-aware optimizer).

### `restore_low_dim_params_to_fp32` (lines 518-523)

```python
def restore_low_dim_params_to_fp32(module):
    for name, param in module.named_parameters():
        if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)):
            param.data = param.data.float()
```

**What:** After the model is cast to bf16 (line 871), this walks back through and forces scalars/vectors/control params to fp32.
**Why:** These small parameters (norm weights, scaling factors, mixing coefficients) are extremely sensitive to precision. A bf16 `attn_scale` parameter has only ~3 decimal digits of precision — not enough for the fine adjustments these parameters need to make during training.

### `Rotary` (lines 526-548)

```python
class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
```

**What:** Rotary Positional Embeddings (RoPE). Pre-computes inverse frequencies and caches cos/sin tables.
**Why:** RoPE encodes position by rotating pairs of dimensions in Q and K by angles proportional to position. Unlike absolute or learned positional embeddings, RoPE:
- Has zero learnable parameters (saves params in a "golf" competition)
- Decays attention naturally with distance (relative position bias emerges from the rotation)
- The frequency `inv_freq[i] = 1 / (base^(2i/dim))` creates a geometric series from high-frequency (nearby tokens) to low-frequency (distant tokens)

The caching (`_seq_len_cached`) avoids recomputing sin/cos every forward pass for the same sequence length.

### `apply_rotary_emb` (lines 551-554)

```python
def apply_rotary_emb(x, cos, sin):
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
```

**What:** Applies the rotation: splits the head dimension in half, applies a 2D rotation matrix `[[cos, sin], [-sin, cos]]` to each pair.
**Why:** This is the "complex number" rotation from the RoPE paper, but done in real arithmetic. Each consecutive pair of dimensions gets rotated by an angle that depends on position, effectively encoding position into the query/key representations.

### `CausalSelfAttention` (lines 557-605)

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)       # Q: full heads
        self.c_k = CastedLinear(dim, kv_dim, bias=False)    # K: fewer heads (GQA)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)    # V: fewer heads (GQA)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
```

**What:** Multi-head attention with GQA and QK-norm.
**Why (design choices):**

- **GQA** (`num_kv_heads=4` < `num_heads=8`): Each KV head is shared by 2 query heads. This saves 50% of KV parameters while losing minimal quality. Critical for parameter golf.
- **No bias**: Saves `4 * dim` parameters per attention layer. Standard in modern LLMs.
- `proj._zero_init = True`: The output projection is initialized to zeros. This means at init, attention contributes nothing — the residual stream passes through unchanged. This is a **residual stream initialization** trick that stabilizes deep networks at init (similar to fixup/zero-init residual from "Fixup Initialization").
- `q_gain`: A learnable per-head scalar applied to Q after QK-norm. Initialized at 1.5, which makes attention slightly sharper than the default QK-norm scale. This is a cheap (8 parameters) way to let each head learn its own "temperature".

**Forward pass (lines 585-605):**

```python
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
```

**QK-norm** — normalizes Q and K to unit RMS before computing attention. This prevents attention logits from exploding/vanishing as model depth or head dim changes. A key stability technique from "Scaling Vision Transformers to 22 Billion Parameters".

```python
    q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
```

Applies the per-head learnable temperature after normalization.

```python
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=(...))
```

PyTorch's fused SDPA — uses FlashAttention under the hood (forced on line 798). `is_causal=True` applies the causal mask without materializing it. `enable_gqa=True` tells SDPA to handle the KV head broadcasting automatically.

### `MLP` (lines 608-619)

```python
class MLP(nn.Module):
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.proj(x.square())
```

**What:** A **ReluSquared** MLP — applies ReLU, then squares the result.
**Why:** `relu^2` was proposed in "Primer: Searching for Efficient Transformers". It's sparser than GELU (ReLU zeros out negatives), and squaring amplifies large activations while suppressing small ones. This gives better gradient signal per parameter than standard ReLU. The MLP expansion factor is only 2x (vs. typical 4x), so every activation function choice matters more.

`self.proj._zero_init = True` — same zero-init trick as in attention. MLP contributes nothing at init.

### `BlockCore` and `BlockControls` (lines 622-652)

This is where the **shared weight architecture** is implemented, split into two parts:

**`BlockCore`** (lines 622-634): Contains the **heavy** parameters — attention Q/K/V/proj matrices and MLP fc/proj matrices. These are the parameters that can be **shared** across layers.

**`BlockControls`** (lines 637-652): Contains the **light** per-layer parameters — two RMSNorm layers, `attn_scale`, `mlp_scale`, and `resid_mix`. These are **never shared** — each layer gets its own.

```python
    self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
```

**What:** A 2xdim mixing matrix. `mix[0]` scales the current residual stream, `mix[1]` scales the original input `x0`.
**Why:** This is a **residual mixing** mechanism. Each layer can learn to blend the current activations with the original embeddings. This creates a "skip connection to the beginning" — useful when layers share weights, because the model needs a way to differentiate what each pass through the same weights does. At init, `mix = [1, 0]`, meaning pure residual (no mixing with x0).

```python
    def forward(self, x, x0, core):
        x = mix[0] * x + mix[1] * x0
        attn_out = core.attn(self.attn_norm(x))
        x = x + self.attn_scale * attn_out
        x = x + self.mlp_scale * core.mlp(self.mlp_norm(x))
```

**Why per-dimension scaling instead of a single scalar?** Each dimension of the residual stream carries different information. Per-dimension `attn_scale` and `mlp_scale` let the model learn to amplify or suppress attention/MLP contributions on a per-feature basis. This is more expressive than a scalar but far cheaper than a full matrix — only `dim` parameters each.

### `GPT` (lines 655-755)

```python
class GPT(nn.Module):
    def __init__(self, ...):
        self.num_encoder_layers = num_layers // 2    # 4
        self.num_decoder_layers = num_layers - ...    # 5
        self.num_skip_weights = min(...)              # 4
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim))
```

**What:** A **U-Net style** transformer with skip connections between the first and second halves.
**Why:** Borrowed from the U-Net architecture in diffusion models. The first half ("encoder") stores activations, and the second half ("decoder") receives skip connections from the corresponding encoder layer in reverse order. This helps gradient flow and lets the decoder layers access features from multiple levels of abstraction. The `skip_weights` are learnable per-dimension scalars (initialized to 1) that control how much of the skip signal to mix in.

```python
    self.block_cores = nn.ModuleList([BlockCore(...) for _ in range(num_shared_blocks)])
    self.block_controls = nn.ModuleList([BlockControls(model_dim) for _ in range(num_layers)])
    self.block_core_indices = self._build_block_core_indices()
```

**What:** Creates `num_shared_blocks` sets of heavy weights and `num_layers` sets of light control weights. The `block_core_indices` maps each layer to which core it uses.

**`_build_block_core_indices`** (lines 715-718):
- `"cycle"`: layer `i` uses core `i % num_shared_blocks` — e.g., with 3 cores and 9 layers: `[0,1,2,0,1,2,0,1,2]`
- `"chunk"`: distributes cores in contiguous chunks — e.g., `[0,0,0,1,1,1,2,2,2]`

**`_init_weights`** (lines 723-728):
```python
    if self.tie_embeddings:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
```

When embeddings are tied, they serve double duty (input + output), so the initialization std (0.005) is deliberately smaller than typical (0.02). This prevents the output logits from being too large at initialization, which would cause high initial loss and unstable gradients.

**Forward pass (lines 730-755):**

```python
    x = self.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
```

Embed tokens, normalize (prevents embedding magnitude from varying wildly), save `x0` for residual mixing in each block.

```python
    # Encoder half
    for i in range(self.num_encoder_layers):
        x = self.block_controls[i](x, x0, self.block_cores[self.block_core_indices[i]])
        skips.append(x)
    # Decoder half with U-Net skip connections
    for i in range(self.num_decoder_layers):
        if skips:
            x = x + self.skip_weights[i] * skips.pop()
```

`skips.pop()` reverses the order — the last encoder layer's output connects to the first decoder layer (like a U-Net).

```python
    logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
    return F.cross_entropy(logits.float(), targets, reduction="mean")
```

**Logit softcapping:** `30 * tanh(logits/30)` smoothly clips logits to `[-30, 30]`. For small logits (< ~10), this is approximately the identity. For large logits, it compresses them. This prevents confident-but-wrong predictions from producing huge gradients that destabilize training. Borrowed from Gemma 2.

The `.float()` cast before `cross_entropy` ensures the loss is computed in fp32, avoiding numerical issues from bf16 softmax.

---

## Lines 762-1006: Training Setup (`main()`)

### Lines 763-767: Compile setup

```python
    code = Path(__file__).read_text(encoding="utf-8")
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
```

`code` is saved to the logfile for reproducibility — the exact script that produced these results is embedded in the log.

`torch.compile` on the Newton-Schulz function generates fused CUDA kernels that are significantly faster than eager-mode PyTorch (avoiding kernel launch overhead for the iterative matrix multiplications).

### Lines 769-800: Distributed + CUDA setup

```python
    grad_accum_steps = 8 // world_size
```

**What:** With 1 GPU, accumulate 8 micro-batches. With 2 GPUs, accumulate 4 each. With 8 GPUs, no accumulation.
**Why:** Keeps the effective batch size constant at `8 * (train_batch_tokens / (world_size * grad_accum_steps)) * world_size = train_batch_tokens` regardless of GPU count. The constraint `8 % world_size == 0` ensures this divides evenly.

```python
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**What:** Enables TF32 precision for matmuls and convolutions.
**Why:** TF32 uses 10-bit mantissa (vs. fp32's 23-bit) in the tensor cores, giving ~8x throughput with minimal quality loss for training. This is the single biggest free speedup on Ampere+ GPUs.

```python
    enable_flash_sdp(True)
    enable_cudnn_sdp(False); enable_mem_efficient_sdp(False); enable_math_sdp(False)
```

**What:** Forces FlashAttention as the only SDP backend.
**Why:** FlashAttention is the fastest and most memory-efficient attention implementation. Disabling alternatives prevents PyTorch from falling back to slower implementations on edge cases.

### Lines 831-851: Seeding and tokenizer validation

All four RNG sources (Python, NumPy, PyTorch CPU, PyTorch CUDA) are seeded for full reproducibility.

The tokenizer vocab size is validated against `args.vocab_size` — a mismatch would silently produce garbage (embedding lookups into uninitialized rows or out-of-bounds indices).

### Lines 857-955: Model and optimizer construction

```python
    base_model = GPT(...).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
```

**What:** Create model -> cast everything to bf16 -> cast `CastedLinear` layers back to fp32 -> cast control params back to fp32.
**Why this dance?** The `.bfloat16()` call converts everything (including buffers, embeddings, etc.) to bf16 for fast compute. Then the two restore calls bring precision-sensitive parameters back. The result: activations flow through bf16 compute, but weights and small parameters stay in fp32 for optimizer quality.

```python
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
```

`dynamic=False` tells the compiler the input shapes won't change — enables aggressive optimizations. `fullgraph=True` requires the entire forward pass to be traceable in one graph (no graph breaks). This produces the best fused kernels but means you can't use Python control flow that depends on tensor values.

```python
    model = DDP(compiled_model, ..., broadcast_buffers=False) if distributed else compiled_model
```

`broadcast_buffers=False` because this model doesn't have any buffers that need syncing (RoPE caches are registered as non-persistent).

**Optimizer split (lines 879-931):**

Four separate optimizers for four parameter groups:

| Optimizer | Parameters | Algorithm | Learning Rate |
|-----------|-----------|-----------|--------------|
| `optimizer_tok` | Token embeddings | Adam | 0.05 (tied) or 0.6 (untied) |
| `optimizer_head` | LM head (if untied) | Adam | 0.008 |
| `optimizer_muon` | 2D matrix params in blocks | **Muon** | 0.04 |
| `optimizer_scalar` | 1D/scalar params + skip_weights | Adam | 0.04 |

**Why separate optimizers?** Different parameter shapes benefit from different optimization strategies:
- Matrices (Q, K, V, MLP weights) benefit from Muon's spectral normalization — it equalizes learning across all directions in weight space.
- Scalars/vectors don't have meaningful singular structure, so standard Adam is better.
- Embeddings have their own scaling needs — with tied embeddings the LR is much lower (0.05 vs 0.6) because the gradient includes both the embedding and LM head signals.

`fused=True` on all Adam instances — uses a single CUDA kernel per optimizer step instead of one per parameter.

### Lines 957-1006: Warmup + State Reset

```python
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for ...}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        # ... run warmup_steps forward/backward/optimizer steps ...
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states):
            opt.load_state_dict(state)
        train_loader = DistributedTokenLoader(...)  # reset data stream too
```

**What:** Run 20 training steps, then **throw away all the model/optimizer state changes** and reset the data loader.
**Why:** This is a **`torch.compile` warmup** trick. The first few steps through a compiled model trigger JIT compilation of CUDA kernels, which is slow and can include profiling/autotuning. By running throw-away steps first, the actual timed training starts with pre-compiled, warm kernels. This is critical when the competition is scored on wallclock time — you don't want 30+ seconds of compilation counted against your 10-minute budget.

The state is saved to CPU (`detach().cpu().clone()`) to avoid wasting GPU memory. The data loader is also reset so training sees the exact same data sequence as if warmup never happened.

---

## Lines 1008-1101: Main Training Loop

```python
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
```

**What:** The loop runs until either `iterations` is reached or the wallclock cap triggers an early stop.

### Validation (lines 1021-1042)

Validation happens at step 0 (baseline), every `val_loss_every` steps, and at the final step. The timer is paused during validation so it doesn't count against the wallclock budget.

### Forward/backward + gradient accumulation (lines 1052-1064)

```python
    for micro_step in range(grad_accum_steps):
        if distributed:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        x, y = train_loader.next_batch(...)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        train_loss += loss.detach()
        (loss * grad_scale).backward()
```

**What:** Accumulates gradients over `grad_accum_steps` micro-batches before stepping the optimizer.
**Why:**
- `require_backward_grad_sync = False` on all micro-steps except the last — this tells DDP to skip the all-reduce for intermediate micro-steps, reducing communication overhead. Gradients are only synchronized on the final micro-step.
- `loss * grad_scale` where `grad_scale = 1/grad_accum_steps` — divides the loss so that accumulated gradients equal the average over all micro-batches, not the sum.
- `torch.autocast` wraps the forward pass in bf16 mixed precision. The backward is automatically bf16 too.

### Muon momentum warmup (lines 1066-1069)

```python
    frac = min(step / args.muon_momentum_warmup_steps, 1.0)
    muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
```

**What:** Linearly ramps Muon's momentum from 0.85 to 0.95 over 500 steps.
**Why:** High momentum at the start of training can amplify noise from random initialization. Starting lower and ramping up lets the optimizer explore more broadly early on, then commit to directions more strongly later.

### LR scheduling (lines 1071-1073)

```python
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["base_lr"] * scale
```

**What:** Multiplies all learning rates by the warmdown factor `scale`.
**Why:** The `lr_mul` function (line 969) implements a **wallclock-aware warmdown**. Instead of decaying based on step count, it estimates remaining time from `elapsed_ms / step * warmdown_iters` and decays the LR linearly to 0 over the final `warmdown_iters` worth of estimated steps. This ensures the LR reaches ~0 right as the wallclock cap hits, regardless of how fast each step is. Clever for a time-limited competition.

### Wallclock cap (lines 1094-1100)

```python
    reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
    if distributed:
        dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
```

**What:** If any rank hits the time cap, all ranks stop together.
**Why:** In distributed training, different GPUs may have slightly different timings. `ReduceOp.MAX` ensures that if ANY rank is over time, everyone stops. Without this, one slow rank could hit the cap while others continue, causing a deadlock on the next collective operation.

`stop_after_step = step` doesn't stop immediately — it sets a flag so the loop runs one more iteration, which triggers the final validation. This ensures you always get a final BPB measurement.

---

## Lines 1107-1167: Serialization + Roundtrip Validation

```python
    torch.save(base_model.state_dict(), "final_model.pt")
```

Saves the raw fp32/bf16 state dict for debugging.

```python
    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_blob = zlib.compress(quant_raw, level=9)
```

**What:** Quantize -> serialize to bytes in memory -> zlib compress at maximum level.
**Why:** The competition scores on `code_bytes + model_bytes`. Int8 quantization cuts the model ~4x. zlib on top of that exploits redundancy in the quantized weights (many near-zero int8 values compress well). `level=9` is slowest but smallest — compression time doesn't matter, only the final file size.

```python
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    q_val_loss, q_val_bpb = eval_val(...)
```

**What:** Loads the quantized-then-dequantized weights back into the model and re-evaluates.
**Why:** This is the **roundtrip validation** — it verifies that the quantized model (which is what gets submitted) actually achieves the claimed BPB. If quantization degrades quality too much, you'd catch it here. The `strict=True` ensures every parameter was restored.

```python
    log0(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    log0(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
```

Both rounded and exact values are logged — the competition likely scores on full precision, so 8 decimal places matters.

```python
    if distributed:
        dist.destroy_process_group()
```

Clean shutdown of NCCL — prevents hanging processes.

---

## Overall Architecture Summary

This is a competition-optimized GPT training script that packs an impressive amount of modern techniques into ~1170 lines:

1. **Muon optimizer** for spectral-normalized matrix updates
2. **GQA** to halve KV parameters
3. **Weight sharing** (optional) to reuse block cores across layers
4. **U-Net skip connections** for better gradient flow
5. **QK-norm + learnable temperature** for attention stability
6. **ReluSquared** activation for better MLP efficiency
7. **Logit softcapping** from Gemma 2
8. **Residual mixing with x0** for shared-weight differentiation
9. **Zero-init residual branches** (attention + MLP output projections)
10. **Tied embeddings** with separate initialization
11. **Wallclock-aware LR warmdown** for time-limited training
12. **Int8 quantization + zlib** for minimum submission size
13. **torch.compile warmup** with state reset for fair timing

Every design choice serves the competition objective: minimize `BPB * (code_bytes + model_bytes)` within 10 minutes on a GPU.
---
Third, it is still an approximation. GPTQ is based on a local second-order Taylor picture. A later paper, First-Order Error Matters (2025/2026), argues that during progressive compensation the latent weights drift away from the original full-precision point, so the assumption that the first-order term stays negligible becomes flawed. In other words, even full-Hessian GPTQ is not “exact”; it is an elegant second-order approximation whose error model can break as quantization proceeds.

Fourth, ordering matters, and the standard ordering is only heuristic. The later geometric analysis of GPTQ shows that the usual “act-order” heuristic only uses the diagonal of the Hessian, not the full matrix, and that finding an optimal order is NP-hard. That means “full Hessian” does not magically solve the sequencing problem; there is still a heuristic layer on top.

Fifth, calibration data matters a lot, because the Hessian is estimated from activations. GPTQ’s original paper uses 128 random 2048-token C4 segments as calibration data, and later empirical work found that downstream performance can vary substantially with the choice of calibration data. So a full-Hessian estimate is only as good as the activation sample it is built from. That is especially relevant to the parameter-golf entry you linked, since its big novelty is replacing forbidden train/val calibration with self-generated text while keeping the stronger full-GPTQ quantizer.

---

 3. SpQR-style per-weight mixed precision — Complex but powerful

  Instead of "all MLP weights in layer 0-5 at int5," identify the specific 1-5% of
   weights in each layer that cause the most quantization error and keep them in
  fp16. The rest go to int5. This surgical approach wastes fewer bits on weights
  that quantize fine and preserves quality for the sensitive ones.

  The artifact size would be: 95% of weights at int5 + 5% at fp16. That's 0.95 ×
  0.625 + 0.05 × 2.0 = 0.694 bytes per param. Only 11% larger than pure int5
  (0.625), but much better quality.

---

  1. Weight pruning — zero out more weights. Zeros compress extremely well. The   
  current selective pruning only triggers when the artifact exceeds TARGET_MB. If
  we prune more aggressively (5-10% of weights instead of the current minimal     
  pruning), the LZMA ratio improves. The quality cost of pruning 5% of smallest
  weights is typically ~0.001 BPB.
  2. zstd instead of LZMA — different compressor might hit a different ratio. The
  current #1's earlier PRs used zstd-22. Worth trying.                            

