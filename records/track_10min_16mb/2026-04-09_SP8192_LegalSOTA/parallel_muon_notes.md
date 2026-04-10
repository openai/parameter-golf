# Parallel Muon Implementation Plan

## What Changes

### 1. GPT Model — Parameter Banking
Replace per-layer CastedLinear weights with 4 contiguous 3D parameter banks:
- `qo_bank`: (2*N, dim, dim) — Q and Out projections  
- `kv_bank`: (2*N, kv_dim, dim) — K and V projections
- `mlp_up_bank`: (N, hidden, dim) — MLP up projections
- `mlp_down_bank`: (N, dim, hidden) — MLP down projections

Where N = num_layers (11).

Forward pass uses `F.linear(x, bank[layer_idx])` instead of `self.c_q(x)`.

### 2. Attention/MLP Classes
- Remove CastedLinear submodules (c_q, c_k, c_v, proj, fc, proj)
- Accept weight tensors as forward() arguments
- Block.forward takes weight slices from banks

### 3. Muon Optimizer — Parallel Communication
Replace DDP + all_reduce with:
1. After backward: launch async `reduce_scatter` for each bank (biggest first)
2. While RS in-flight: step Adam on small params (scalars, embeddings)
3. Wait for RS, run batched Newton-Schulz on local shard, launch `all_gather`
4. Pipeline: each AG overlaps with next bank's Newton-Schulz

### 4. Training Loop
- Remove DDP wrapper
- After backward: call `muon.launch_reduce_scatters()`
- Step Adam optimizers
- Call `muon.step()` (waits for RS, does PE, launches AG)

### 5. Compatibility with Depth Recurrence
Banks index by physical layer (0-10). Depth recurrence reuses layer indices
(encoder/decoder indices reference the same bank entries). This works naturally
since recurrence already reuses self.blocks[i].

### 6. Compatibility with TTT
TTT doesn't use the optimizer — it creates its own SGD optimizer.
Bank params are just regular parameters with .grad, so TTT SGD works fine.
Only need to make sure TTT's manual all_reduce still works on bank grads.

### 7. GPTQ
GPTQ quantizes by iterating named_parameters. Bank params appear as
qo_bank, kv_bank, etc. Need to handle differently — quantize each 2D slice
(bank[i]) separately, not the whole 3D tensor.

## Estimated Savings
- Newton-Schulz: 19.7ms → 1.3ms (15x faster)
- DDP overhead removed: ~2ms
- Total: ~3-5ms/step
- At 130ms/step (triple loop phase), this is ~3% improvement = ~70 extra steps

## Risk
- GPTQ integration needs careful handling of 3D bank → 2D slices
- EMA state dict needs to map bank params correctly
- torch.compile with banked forward may behave differently
