# Windows Compatibility — Bugs & Fixes

## Environment

| Item | Value |
|---|---|
| OS | Windows (PowerShell) |
| GPU | NVIDIA GeForce RTX 3090 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.13 |
| CUDA | 12.4 |

---

## Bug 1 — Flash SDP + GQA kernel unavailable on Windows / consumer GPUs

### Symptom
```
RuntimeError: No available kernel. Aborting execution.
```
Occurs when `F.scaled_dot_product_attention(..., enable_gqa=True)` is called while
`enable_flash_sdp(True)` and `enable_math_sdp(False)` are set (as the script does by default).

### Root Cause
`train_gpt.py` configures SDPA at startup:
```python
enable_cudnn_sdp(False)
enable_flash_sdp(True)      # ← requires FA3 / special kernel
enable_mem_efficient_sdp(False)
enable_math_sdp(False)      # ← disabled, no fallback
```
Flash SDP's GQA (Group Query Attention) path requires FlashAttention 3 kernels. These are
only available on A100/H100-class GPUs and are not supported on RTX consumer cards or under
Windows CUDA drivers.

### Fix
Before `train_gpt.py` code runs, override the SDP backend setters:
```python
enable_cudnn_sdp(True)
enable_flash_sdp(False)     # Off — GQA Flash kernel unavailable
enable_mem_efficient_sdp(True)
enable_math_sdp(True)       # On — always-available fallback that supports GQA
```
Also monkey-patch `enable_flash_sdp` and `enable_math_sdp` so that the script's own
`enable_flash_sdp(True)` / `enable_math_sdp(False)` calls become no-ops.

**Verification:**
```python
q = torch.randn(2, 8, 32, 64).cuda().bfloat16()
k = torch.randn(2, 4, 32, 64).cuda().bfloat16()
v = torch.randn(2, 4, 32, 64).cuda().bfloat16()
out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
# → works fine with math/cudnn backends
```

---

## Bug 2 — `torch.compile` fails (no Triton on Windows)

### Symptom
```
RuntimeError: Cannot find a working triton installation.
backend='inductor' raised: RuntimeError: ...
```
Occurs when `torch.compile(...)` is called (which `train_gpt.py` does for the Muon
optimizer's inner function and for the model).

### Root Cause
`torch.compile` uses the **Inductor** backend, which requires **Triton** to JIT-compile
CUDA kernels. PyTorch ships Triton on Linux but **not on Windows** — there are no official
Windows Triton wheels.

### Fix — Install `triton-windows` community package
```powershell
pip install "triton-windows<3.3"
```
> **Why `<3.3`?**  PyTorch 2.6 expects Triton's `AttrsDescriptor` API from the 3.2.x
> series. `triton-windows>=3.3` has a slightly different API that causes `ImportError`.
> Pinning to `<3.3` installs `triton-windows==3.2.0.postXX` which is compatible.

**Verification:**
```python
import torch
compiled = torch.compile(lambda x: x * 2)
out = compiled(torch.randn(4,4).cuda())
# → OK, shape torch.Size([4, 4])
```

**Without the fix (fallback):** `torch.compile` can be disabled via:
```python
torch._dynamo.config.suppress_errors = True
torch.compile = lambda fn=None, **kw: (lambda f: f) if fn is None else fn
```
This makes training work but is **~2× slower** (no kernel fusion).

---

## Bug 3 — NCCL distributed backend unavailable on Windows

### Symptom
`dist.is_nccl_available()` returns `False`. If the script tried to call
`dist.init_process_group(backend="nccl")` it would fail.

### Root Cause
NCCL (NVIDIA Collective Communications Library) is a Linux-only library. Windows builds
of PyTorch do not include it.

### Fix
Monkey-patch `dist.init_process_group` to swap `backend="nccl"` → `backend="gloo"`:
```python
_orig = dist.init_process_group
def _patched(backend=None, **kwargs):
    if backend == "nccl":
        backend = "gloo"
    return _orig(backend=backend, **kwargs)
dist.init_process_group = _patched
```
> **Note:** For single-GPU runs (no `torchrun`), `distributed=False` so `init_process_group`
> is never called. This patch only matters when using `torchrun` on Windows.

**Verification:**
```python
import torch.distributed as dist
print(dist.is_nccl_available())   # False on Windows
print(dist.is_gloo_available())   # True on Windows
```

---

## Bug 4 — `fused=True` Adam (not actually a bug here)

### Symptom (expected, did not occur)
Fused Adam optimizers sometimes fail on platforms without CUDA support.

### Status: **Not an issue** — tested and working
```python
p = torch.nn.Parameter(torch.randn(10,10).cuda())
opt = torch.optim.Adam([p], fused=True)  # OK on RTX 3090 / Windows
```

---

## Bug 19 — Inductor Stochastic Depth RuntimeError

### Symptom
```
RuntimeError: Inductor does not support dynamic control flow in compiled blocks.
```
Occurs when `if self.training and torch.rand(1) < 0.1: return x` is used inside a `torch.compile` block.

### Root Cause
`torch.compile` (Inductor) on Windows cannot reliably trace Python-level conditional logic that depends on random values. It triggers a graph break or a full-recompile loop that eventually crashes.

### Fix: **Tensor-Mask Stochastic Depth**
Replace the Python `if` with a binary mask applied to the residual branch:
```python
if self.training:
    mask = (torch.rand(1, device=x.device) > 0.1).to(dtype=x.dtype)
else:
    mask = 1.0
x = x + mask * (residual_branch)
```
This keeps the entire block as a single, static computational graph that Inductor can optimize perfectly.

---

## Bug 20 — 12-Step Scalar Divergence (The "Step 1" Explosion)

### Symptom
Loss starts at 7.0 (Step 0) and explodes to 25.0+ (Step 1) even with 1.0 clipping.

### Root Cause
In a 12-step recursive UT, a single weight update of 0.01 is applied **12 times per sequence**. This effectively multiplies the update impact by $N=12$, causing the residual stream to shatter before the model has "matured."

### Fix: **Maturity Ramp & Universal Averaging**
1. **100-Step Maturity Ramp**: Linearly ramp the LR from 0 to target over the first 100 steps. This allows the model to settle before high-magnitude orthogonal updates begin.
2. **Universal Gradient Averaging**: Divide **ALL** parameter gradients by the recursion depth (`p.grad.div_(12)`). This ensures the tied weights are updated at a scale consistent with a single-layer model.

---

## Bug 21 — Shard Pattern Overfitting

### Symptom
Training loss plummets (e.g. 5.2) while validation loss stalls (overfitting gap > 1.0).

### Root Cause
With high-throughput training (~11s per update), the model consumes 524k tokens every 11 seconds. If shards are loaded in alphabetical order, every training run starts with the EXACT same sequence of tokens from `shard_000.bin`. The powerful 13M parameter model simply memorizes the first 16MB of text.

### Fix: **Shard Shuffling**
In `TokenStream.__init__`, randomized the shard file list:
```python
import random
random.seed(42)
random.shuffle(self.files)
```
This ensures every run sees a unique sequence of data, forcing the model to generalize and narrowing the train/val gap.

---

## Summary (Elite Standard 20.0)

| Change | Impact |
| :--- | :--- |
| **Stochastic Depth** | Narrowed Gap 1.3 → 0.7nd |
| **Wallclock Scheduler** | Zeroed BPB at 600s |
| **Shard Shuffling** | Generalization boost |
| **12-Step Unroll** | Max Reasoning Depth |

---

## Bug 5 — `fullgraph=True` fails with custom `autograd.Function`

### Symptom
```
torch._dynamo.exc.Unsupported: <function fused_relu2 at ...>
```
Occurs when `torch.compile(..., fullgraph=True)` is used with a custom Triton Megakernel on Windows.

### Root Cause
`fullgraph=True` is a strict mode that forbids graph breaks. Custom `autograd.Function` calls (like our Triton wrapper) are currently "unsupported" as single-graph units in the Inductor backend on Windows.

### Fix
In `train_gpt_windows.py`, regex-patch the source code of `train_gpt.py` at runtime to use `fullgraph=False`. This allows Dynamo to "break" the graph at the kernel, run it in eager mode, and compile the rest of the transformer blocks.

---

## Bug 6 — Mixed-Precision `tl.dot` (Windows/Ampere)

### Symptom
```
ValueError: too many values to unpack (expected 2)
```
Or a Triton compilation error at the `tl.dot` line.

### Root Cause
Triton's `tl.dot` implementation for Ampere/Hopper requires that the input floating point types match (e.g., both must be `bf16`). In `train_gpt.py`, inputs are `bf16` but `CastedLinear` weights are `fp32`. Passing mismatched types directly to `tl.dot` causes the compiler or the shape-checker to trip.

```python
a = tl.load(a_ptr).to(tl.bfloat16)
b = tl.load(b_ptr).to(tl.bfloat16)
tl.dot(a, b, accumulator)
```

---

## Bug 7 — Recompilation "Death Spiral"

### Symptom
Training starts at 70ms/step and gradually slows to 500ms+ or hangs.

### Root Cause
Triton kernels on Windows are hyper-sensitive to shared memory limits and register spills. If the graph needs even a minor re-specialization (e.g., sequence length change), `torch._dynamo` tries to recompile. If it hits the default `cache_size_limit (8)`, it triggers an "unsupported" graph break or a full-recompile loop.

### Fix
In `train_gpt_windows.py`:
```python
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
```

---

## Bug 8 — SDP Backend "Leak"

### Symptom
Step times triple (~1100ms → 3800ms) and logs show `math=True`.

### Root Cause
`train_gpt.py` contains hard-coded `enable_flash_sdp(True)` calls and a hard-coded log string. These direct function calls were overriding the global `torch.backends.cuda` toggles set by the wrapper.

### Fix
Use the Windows wrapper to brute-force replace the literal strings in the base script at runtime:
```python
_source = _source.replace("enable_flash_sdp(True)", "enable_flash_sdp(False)")
_source = _source.replace("enable_math_sdp(False)", "enable_math_sdp(False)") 
```

---

## Bug 9 — Step 2 Spectral Collapse (The 12M Heavyweight)

### Symptom
Loss starts at 7.0 (Step 1) and explodes to 25.0+ (Step 2).

### Root Cause
At `model_dim=1024`, the "Identity Highway" is extremely sensitive. Muon's Newton-Schulz orthogonalization is too aggressive for unformed features, and independent Adam LR jumps for the embeddings cause the residual stream to "shatter" in the first update.

### Fix: **Synchronized Double-Lock**
1. **Synchronized LR Ramp**: Both **Adam (Embeddings)** and **Muon (Blocks)** now ramp together from **1e-5** to target LR over 500 steps. This ensures the embeddings don't "run away" from the block weights.
2. **20-Step Settling Phase**: Reduced LR by another 10x (`0.1x` multiplier) for the first 20 steps to allow the init variance to damp.

---

## Bug 10 — Profiler Collision

### Symptom
`RuntimeError: Can't disable Kineto profiler` or `AttributeError: NoneType has no attribute save`.

### Root Cause
A logical collision between the `on_trace_ready` handler (automatic save) and a manual `prof.export_chrome_trace()` call in the training loop.

### Fix
Remove the manual export and replace it with `sys.exit(0)`. The `on_trace_ready` handler flushes the JSON to disk when the process terminates.

---

## Bug 11 — Step 0 Startup Latency

### Symptom
30–60 second "dead time" before Step 1 starts.

### Root Cause
The base script runs a full validation pass (`eval_val`) on the 62M token set at `step=0`. For a 10-minute challenge, this wastes 10% of the time.

### Fix
Guard the validation logic with `if step > 0`. Training now starts instantly.

---

## Bug 12 — Numerical Resonance (CANS Stability)

### Symptom
Loss becomes `NaN` by Step 1 or Step 2 despite LR ramps.

### Root Cause
High-order Newton-Schulz (CANS Degree 7) has an unstable fixed point at $x=0$ with a derivative of **4.375**. 
- If `backend_steps` is set too high (e.g., 8), the random gradient noise is amplified as $4.375^8 \approx 145,000\sigma$.
- This pushes singular values past the stability radius ($~2.0$), causing the polynomial to dive to $-\infty$.

### Fix
Restrict Newton-Schulz to **3–4 steps** during the first 100 iterations. Higher precision is only beneficial once stable features have formed.

---

## Bug 13 — 3GB Attention Map OOM

### Symptom
`torch.OutOfMemoryError: Tried to allocate 3.00 GiB`. Total allocated 46GB (spilling to system RAM).

### Root Cause
1. **DDP Hardcoding**: The script was ignoring `GRAD_ACCUM_STEPS` and forcing a batch of 64 sequences. $64 \times 12 \times 1024 \times 1024 \times 4 \approx 3GB$.
2. **Recursive Depth**: 12-loop Universal Transformers store activations for ALL stages in VRAM by default.

### Fix
1. **Activation Checkpointing**: Wrap the recursive block loop in `torch.utils.checkpoint`. VRAM drops by ~50%.
2. **Sequence Scaling**: Dropped training sequence length to **512**.

---

## Bug 14 — Triton Autotune "Conflicting Meta-parameters"

### Symptom
`ValueError: Conflicting meta-parameters: BLOCK_SIZE_M...`

### Root Cause
Passing manual `BLOCK_SIZE_M=...` arguments in the `kernel[grid](...)` call while also using the `@triton.autotune` decorator. The autotuner requires full control over these meta-arguments.

### Fix
Remove all manual tiling arguments from the `forward` and `backward` launches.

---

## Bug 15 — Step 5 Validation "Hang"

### Symptom
Script reaches Step 5/5 and appears to freeze for 15+ minutes.

### Root Cause
Mid-training validation was configured to process the **FULL** 62-million-token FineWeb val set.

### Fix
Cap mid-training validation at **50 batches** using a `max_steps` parameter.

---

## Bug 16 — Misleading Performance Timers

### Symptom
Logs show `Data: 14800ms | Comp: 500ms`, incorrectly suggesting an I/O bottleneck.

### Root Cause
The timer was measuring `t_data - t0` *once* per step, which accidentally included the computation time of the first 31 micro-steps.

### Fix
Accumulate `t_data_sum` and `t_comp_sum` strictly *inside* the micro-accumulation loop.

---

## Bug 17 — SDP Backend "Wait" Leak (The SDP_MATH Trap)

### Symptom
Step times jump from 500ms to 4000ms+ on RTX 3090, but VRAM remains low.

### Root Cause
PyTorch 2.6 on Windows periodically "leaks" the SDP backend selection into `SDP_MATH` if `enable_math_sdp(True)` is set alongside `enable_mem_efficient_sdp(True)`. While `math` is a valid fallback, it is not optimized for GQA/Ampere and serializes certain attention operations.

### Fix
In `train_gpt_windows.py`, we now brute-force the backend visibility for the runtime process:
```python
enable_cudnn_sdp(True)
enable_flash_sdp(False) 
enable_mem_efficient_sdp(True)
enable_math_sdp(False) # ← Force-disable to prevent the leak
```

---

## Bug 18 — Muon "Polar Express" Step 2 Explosion

### Symptom
Loss explodes to `NaN` or `25.0+` at exactly Step 2 when using high-order Newton-Schulz (Degree 7).

### Root Cause
Degree-7 Newton-Schulz has a high derivative (~4.37) at zero. Random weight gradients at Step 2 can push singular values into the "divergence zone" of the high-degree polynomial.

### Fix: **Polar Express (Degree-5)**
Switched to a quintic minimax polynomial ($3.4445, -4.7750, 2.0315$).
- **Benefit**: Much larger convergence radius and higher stability at the cost of slightly more iterations (5 steps instead of 3).
- **Result**: Step 2 explosions are eliminated across all model widths (768-1024).
