# ROCm Support

## ROCm-focused changes

### 1. Backend detection and logging

- Added `is_rocm_build()` to detect ROCm by checking `torch.version.hip`.
- Added `accelerator_name()` so logs can report `rocm` or `cuda`.
- Startup logging now prints the selected accelerator backend.
- GPU diagnostics now use `amd-smi` on ROCm builds and `nvidia-smi` otherwise.

Why it matters:
- The script can now distinguish ROCm from CUDA at runtime and emit backend-appropriate diagnostics instead of assuming NVIDIA tooling.

### 2. Autocast path made ROCm-safe

- Introduced `autocast_kwargs()` and replaced direct calls like:
  - `torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)`
- Training and validation now call:
  - `torch.autocast(**autocast_kwargs())`
- The helper documents an important PyTorch ROCm detail: ROCm still uses the `"cuda"` device type in autocast APIs.

Why it matters:
- This centralizes mixed-precision configuration and makes the ROCm behavior explicit instead of scattering CUDA-only assumptions through the training loop.

### 3. Scaled dot-product attention backend selection is now conditional

- Added `maybe_enable_sdp_backends(log0)`.
- The old code hardcoded:
  - cuDNN SDP disabled
  - Flash SDP enabled
  - Memory-efficient SDP disabled
  - Math SDP disabled
- The new code:
  - disables cuDNN and memory-efficient SDP
  - enables Flash SDP only when `torch.backends.cuda.is_flash_attention_available()` reports support
  - falls back to math SDP otherwise
  - logs the final backend selection

Why it matters:
- ROCm environments may not support the same Flash Attention path as CUDA/NVIDIA. The new logic avoids forcing an unavailable backend and falls back cleanly.

### 4. `torch.compile` is now optional and fault-tolerant

- Added hyperparameters:
  - `ENABLE_COMPILE` (default `1`)
  - `COMPILE_MODE` (default `"default"`)
  - `COMPILE_FULLGRAPH` (default `0`)
- Added `maybe_compile_module(...)`:
  - skips compilation when disabled
  - attempts `torch.compile(...)`
  - logs a failure and falls back to eager mode on exception
- Replaced unconditional fullgraph compilation of the model with this helper.
- The separate compile of `zeropower_via_newtonschulz5` was commented out.

Why it matters:
- `torch.compile` support and stability can vary across ROCm stacks. This change removes the hard dependency on successful compilation and gives a safe fallback path.

### 5. Adam optimizer creation no longer assumes fused kernels

- Added `make_adam(...)` helper.
- The old code always passed `fused=True` to every Adam optimizer.
- The new code only enables fused Adam when:
  - the installed `torch.optim.Adam` signature exposes `fused`
  - `torch.cuda.is_available()` is true
- Token, scalar, and head Adam optimizers now all use the helper.

Why it matters:
- Fused Adam support is not guaranteed across ROCm/PyTorch combinations. This prevents failures caused by unconditionally requesting fused kernels.

### 6. Device setup wording no longer implies NVIDIA-only CUDA

- The accelerator setup section was renamed from `DISTRIBUTED + CUDA SETUP` to `DISTRIBUTED + ACCELERATOR SETUP`.
- The runtime error changed from `CUDA is required` to:
  - `A CUDA-compatible accelerator is required; ROCm builds also appear via torch.cuda`

Why it matters:
- PyTorch exposes ROCm devices through `torch.cuda`, which is easy to misread if the code and error messages only mention CUDA.

### 7. Attention tensor layout tightened up

- Added `.contiguous()` after transposing `q`, `k`, and `v` in `CausalSelfAttention.forward`.

Why it may matter for ROCm:
- This is not explicitly ROCm-only, but it can reduce layout-related issues and backend sensitivity in attention kernels.

## Overall impact

The main direction of the commit is to remove NVIDIA-specific assumptions from the training script while keeping the same `torch.cuda` execution path that PyTorch uses for ROCm. The changes make backend selection, diagnostics, mixed precision, SDP backend choice, optimizer configuration, and `torch.compile` behavior more tolerant of ROCm environments.
