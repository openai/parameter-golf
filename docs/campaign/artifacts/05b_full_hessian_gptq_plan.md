# Plan: Full Hessian GPTQ (Session 05b)

## Context

The Session 03 anchor loses **0.00775 BPB** from pre-quant EMA (1.14472) to roundtrip (1.15247) due to naive per-row int6 quantization. Full Hessian GPTQ replaces this with Cholesky-based error compensation that weights quantization errors by their downstream impact. The referenced competitive PRs (#634, #1019, #1060, #1072) all implement variants of the GPTQ family; the exact decomposition/order will be verified against the PR diffs before coding. This is the single highest-impact isolated delta available — it changes only the post-training export path, zero risk to training convergence.

## Critical Files

| File | Role |
|------|------|
| `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py` | Source anchor (copy, don't mutate) |
| `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py` | Working copy (all edits here) |
| `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md` | This plan artifact |

## Key Architecture Facts for Implementation

**Model variable aliases** (anchor lines 1047-1071):
- `base_model` — raw unwrapped `GPT(...)` module. **This is what we use for hooks and calibration.**
- `compiled_model = torch.compile(base_model)` — compiled wrapper
- `model = DDP(compiled_model, ...)` — DDP + compiled (used for training/eval)
- After EMA application (line 1320): `base_model.load_state_dict(avg_state)` updates the raw model

**Current export rank pattern** (anchor lines 1349-1398):
- Lines 1349-1357: ALL ranks do `sd_cpu = ...`, `mixed_quantize_int6(...)`, zstd compress (redundant)
- Lines 1359-1371: Only `master_process` writes file + logs
- Line 1374-1375: `dist.barrier()`, then ALL ranks read `final_model.int6.ptz`
- Lines 1385-1456: ALL ranks do roundtrip eval

**For GPTQ, we restructure**: Hessian collection + GPTQ quantization + file write are rank-0 only. Non-master ranks skip to barrier, then all ranks read the file for roundtrip eval.

**Data loading** (anchor lines 443-487):
- `TokenStream(pattern)` — sequential reader, takes N tokens at a time from training shards
- `DistributedTokenLoader(pattern, rank, world_size, device)` — wraps TokenStream for DDP
- For rank-0-only calibration, use `TokenStream` directly (simpler, no rank distribution)

**Linear layers per block** (66 total across 11 blocks):
Weight shapes in PyTorch storage order `(out_features, in_features)`.
GPTQ operates on W as `(d_row, d_col)` where `d_col = in_features`.
H = X^TX has shape `(in_features, in_features)`.

| Layer | Constructor | Weight (out, in) | Input dim | Hessian H shape |
|-------|------------|-------------------|-----------|-----------------|
| `blocks.{i}.attn.c_q` | Linear(512, 512) | (512, 512) | 512 | (512, 512) = 1 MB |
| `blocks.{i}.attn.c_k` | Linear(512, 256) | (256, 512) | 512 | (512, 512) = 1 MB |
| `blocks.{i}.attn.c_v` | Linear(512, 256) | (256, 512) | 512 | (512, 512) = 1 MB |
| `blocks.{i}.attn.proj` | Linear(512, 512) | (512, 512) | 512 | (512, 512) = 1 MB |
| `blocks.{i}.mlp.fc` | Linear(512, 1536) | (1536, 512) | 512 | (512, 512) = 1 MB |
| `blocks.{i}.mlp.proj` | Linear(1536, 512) | (512, 1536) | 1536 | (1536, 1536) = 9 MB |

Note: c_q, c_k, c_v share the same input (output of attn_norm * scale), so they will have identical Hessians. No special sharing logic needed — just collect independently and accept redundancy. Total Hessian memory: ~154 MB CPU.

## Implementation Steps

### Step 0: Workspace setup
- `mkdir -p records/track_non_record_16mb/2026-03-29_full_hessian_gptq`
- `cp records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py`
- Create `README.md` and `submission.json` stub

### Step 1: Add `collect_hessians()` function (~45 lines)

**Location**: After `dequantize_mixed_int6` (line 420), before DATA LOADING section.

```python
def collect_hessians(model, data_files, device, num_samples=128, seq_len=2048, batch_size=4):
    """Collect H = X^T X for each CastedLinear layer using training data.

    Returns dict mapping module_name -> H (float32, on CPU).
    Only collects for CastedLinear modules (the int6 quantization targets).
    """
```

**Hook registration and name mapping** (critical — must be exact):
```python
hessians = {}
n_samples = {}
handles = []
# Build module_name -> CastedLinear mapping from base_model
for name, module in model.named_modules():
    if isinstance(module, CastedLinear):
        # Verify this module has a matching state_dict key
        weight_key = name + ".weight"
        assert weight_key in model.state_dict(), f"No state_dict key for {weight_key}"
        # Only collect for int6-targeted layers (mlp/attn)
        cat = _classify_param(weight_key)
        if cat not in ("mlp", "attn"):
            continue  # skip embeddings, other non-int6 layers
        handles.append(module.register_forward_pre_hook(_make_hessian_hook(hessians, name, n_samples)))
```

Log skipped layers (any CastedLinear that isn't mlp/attn):
```python
log0(f"gptq: collecting Hessians for {len(handles)} layers, skipped: {skipped_names}")
```

**Calibration forward call** — use `forward_logits`, NOT `forward`:
- `GPT.forward(input_ids, target_ids)` requires targets and computes loss — wrong for calibration
- `GPT.forward_logits(input_ids)` (anchor line 835) does the full forward pass through all blocks/layers without needing targets
- All `CastedLinear` hooks fire identically since the computation path through blocks is the same

**Calibration batching** (grounded in existing `TokenStream`):
```python
stream = TokenStream(data_files)  # rank-0 only, no DistributedTokenLoader
num_batches = num_samples // batch_size  # 128 // 4 = 32 batches
model.eval()
with torch.inference_mode():
    for _ in range(num_batches):
        tokens = stream.take(batch_size * seq_len).to(device=device, dtype=torch.int64)
        x = tokens.reshape(batch_size, seq_len)
        model.forward_logits(x)  # hooks fire on all CastedLinear layers
for h in handles:
    h.remove()
# Move Hessians to CPU
return {name: H.cpu() for name, H in hessians.items()}
```

**Calibration budget**: 128 sequences × 2048 tokens = 262,144 tokens. 32 forward passes at batch_size=4. Estimated ~3-5 seconds on 1xH100 in inference mode.

**Hook implementation**:
```python
def _make_hessian_hook(hessians, name, n_samples):
    def hook(module, input):
        x = input[0].detach().float()
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])  # (B*T, D)
        if name not in hessians:
            hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=x.device)
        hessians[name].addmm_(x.T, x)
        n_samples[name] = n_samples.get(name, 0) + x.shape[0]
    return hook
```

### Step 2: Add `gptq_quantize_layer()` function (~65 lines)

```python
def gptq_quantize_layer(W, H, block_size=128, percdamp=0.01, clip_range=31, actorder=True):
    """GPTQ-quantize W using Hessian H.
    Returns (q: int8 in [-32, clip_range], scale: fp16 per-row, degraded: bool).
    degraded=True if Cholesky failed and naive fallback was used.
    Format identical to quantize_int6_per_row output (first two elements).
    """
```

**Implementation source priority**: PR diffs from #634/#1019/#1060 are the primary reference for exact code structure. The Frantar et al. paper is the reference only for ambiguous math (e.g., block update formula).

**Algorithm** (implements the GPTQ family from Frantar et al.):

```
1. W = W.float().clone()           # (d_row, d_col)
   H = H.float().clone()           # (d_col, d_col)
   d_row, d_col = W.shape

2. # Dead columns: before actorder permutation
   dead = (H.diag() == 0)
   H[dead, dead] = 1.0
   W[:, dead] = 0.0

3. # Damping
   damp = percdamp * H.diag().mean().clamp_min(1e-6)
   H.diagonal().add_(damp)

4. # Actorder: sort columns by descending H diagonal (most important first)
   if actorder:
       perm = torch.argsort(H.diag(), descending=True)
       W = W[:, perm]
       H = H[perm][:, perm]

5. # Three-step Cholesky to get upper-triangular factor of H^{-1}
   # This Hinv_chol IS the working matrix for the inner loop.
   # Hinv_chol[j,j] is the per-column error scaling factor.
   # Hinv_chol[j, j+1:] is the error propagation vector within the block.
   try:
       L = torch.linalg.cholesky(H)                    # H = L @ L^T
       Hinv = torch.cholesky_inverse(L)                  # H^{-1} (full, intermediate)
       Hinv_chol = torch.linalg.cholesky(Hinv, upper=True) # upper-triangular factor
       del L, Hinv  # Hinv_chol is all we need
   except torch.linalg.LinAlgError:
       # DEGRADED: fall back to naive quantization for this layer
       cond_est = H.diag().max().item() / max(H.diag().min().item(), 1e-12)
       log0(f"gptq:WARNING Cholesky failed, cond~{cond_est:.1e}, falling back to naive")
       q, s = quantize_int6_per_row(W[:, torch.argsort(perm)] if actorder else W)
       return q, s, True  # degraded=True

6. # Per-row scales (from permuted W — row-max is column-permutation-invariant)
   row_max = W.abs().amax(dim=1)
   scale = (row_max / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)

7. # Block-wise GPTQ quantization
   Q = torch.zeros_like(W)
   for block_start in range(0, d_col, block_size):
       block_end = min(block_start + block_size, d_col)
       W_block = W[:, block_start:block_end].clone()
       Err = torch.zeros_like(W_block)
       Hinv_block = Hinv_chol[block_start:block_end, block_start:block_end]

       for j in range(block_end - block_start):
           w_col = W_block[:, j]
           d = Hinv_block[j, j]

           # Quantize this column using per-row scales
           q_col = torch.clamp(torch.round(w_col / scale.float()), -32, clip_range)
           Q[:, block_start + j] = q_col

           # Scaled error
           err = (w_col - q_col * scale.float()) / d
           Err[:, j] = err

           # Update remaining columns within block
           if j + 1 < block_end - block_start:
               W_block[:, j+1:] -= err.unsqueeze(1) * Hinv_block[j, j+1:].unsqueeze(0)

       # Lazy batch update: propagate block errors to all remaining columns
       if block_end < d_col:
           W[:, block_end:] -= Err @ Hinv_chol[block_start:block_end, block_end:]

8. # Un-permute back to original column order
   if actorder:
       invperm = torch.argsort(perm)
       Q = Q[:, invperm]

9. return Q.to(torch.int8), scale, False  # degraded=False
```

**Cholesky fallback**: If `LinAlgError` is raised, log the layer name and H condition number estimate (ratio of max/min diagonal). The per-run result is marked as "degraded GPTQ" — not a clean success.

### Step 3: Add `gptq_mixed_quantize_int6()` function (~35 lines)

```python
def gptq_mixed_quantize_int6(state_dict, int6_cats, hessians):
    """Like mixed_quantize_int6, but uses GPTQ for layers with Hessians.

    hessians: dict mapping module_name (e.g. "blocks.0.attn.c_q") -> H tensor.
    state_dict keys use param names (e.g. "blocks.0.attn.c_q.weight").
    Mapping: strip ".weight" suffix from param name to look up Hessian.
    """
```

Logic (identical to `mixed_quantize_int6` except for the int6 branch):
```python
    result, meta = {}, {}
    gptq_count, naive_count, fallback_count = 0, 0, 0
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        cat = _classify_param(name)
        # ... passthrough / control tensor logic identical to anchor ...
        if cat in int6_cats and t.ndim >= 1:
            # Strip .weight to get module name for Hessian lookup
            module_name = name.rsplit(".weight", 1)[0] if name.endswith(".weight") else name
            H = hessians.get(module_name)
            if H is not None and t.ndim == 2:
                q, s, degraded = gptq_quantize_layer(t, H)
                if degraded:
                    fallback_count += 1
                    log0(f"gptq: DEGRADED layer {module_name}")
                else:
                    gptq_count += 1
            else:
                q, s = quantize_int6_per_row(t)
                naive_count += 1
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int6"}
        else:
            # int8 path — identical to anchor
            q, s = quantize_float_tensor(t)
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = {"type": "int8"}
    log0(f"gptq: quantized {gptq_count} layers with GPTQ, {naive_count} with naive, {fallback_count} Cholesky fallbacks")
    return result, meta
```

**Output format**: Identical to `mixed_quantize_int6` — same dict structure, same int8 storage, same fp16 scales. `dequantize_mixed_int6` is completely unchanged.

### Step 4: Wire into main() — Restructured rank ownership

**After pre-quant EMA eval** (after line 1336), replace the entire serialization block (lines 1338-1371) with:

```python
    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION (GPTQ int6 + zstd)
    # -----------------------------

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

        # --- GPTQ: Hessian collection (rank-0 only) ---
        log0("gptq: collecting Hessians")
        t_hess = time.perf_counter()
        hessians = collect_hessians(
            base_model, args.train_files, device,
            num_samples=128, seq_len=args.train_seq_len, batch_size=4,
        )
        log0(f"gptq: {len(hessians)} Hessians in {1000*(time.perf_counter()-t_hess):.0f}ms")

        # --- GPTQ: quantization + compress + write (rank-0 only) ---
        sd_cpu_r0 = {k: v.detach().cpu().contiguous() for k, v in base_model.state_dict().items()}
        t_quant = time.perf_counter()
        quant_result, quant_meta = gptq_mixed_quantize_int6(sd_cpu_r0, {"mlp", "attn"}, hessians)
        log0(f"gptq: quantization in {1000*(time.perf_counter()-t_quant):.0f}ms")
        del hessians, sd_cpu_r0

        quant_buf = io.BytesIO()
        torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
        quant_raw = quant_buf.getvalue()
        if _COMPRESSOR == "zstd":
            quant_blob = zstandard.ZstdCompressor(level=22).compress(quant_raw)
        else:
            quant_blob = zlib.compress(quant_raw, 9)

        with open("final_model.int6.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = len(quant_blob)
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model int6+{_COMPRESSOR}: {quant_file_bytes} bytes")
        log0(f"bytes_code:{code_bytes}")
        log0(f"bytes_model_int6_{_COMPRESSOR}:{quant_file_bytes}")
        total_bytes = code_bytes + quant_file_bytes
        log0(f"bytes_total:{total_bytes}")
        log0(f"Total submission size int6+{_COMPRESSOR}: {total_bytes} bytes")
        if total_bytes > 16_000_000:
            log0(f"WARNING: total submission size {total_bytes} exceeds 16,000,000 byte cap!")

    # All ranks wait for rank 0 to finish quantization + file write
    if distributed:
        dist.barrier()

    # sd_cpu needed on ALL ranks for dequantize_mixed_int6(..., sd_cpu) template
    sd_cpu = {k: v.detach().cpu().contiguous() for k, v in base_model.state_dict().items()}

    # Roundtrip: decompress + dequantize into fresh eval model (all ranks)
    # ... existing roundtrip eval code unchanged from line 1376 onward ...
```

**Key changes vs anchor**:
- Hessian collection + GPTQ quantization + compress + file write are all inside `if master_process:`. Non-master ranks skip to the barrier.
- Rank 0 uses its own `sd_cpu_r0` for GPTQ (deleted after quantization).
- After barrier, ALL ranks rebuild `sd_cpu` from `base_model` — this is needed by `dequantize_mixed_int6(..., sd_cpu)` as the template for dtype/shape reconstruction. This is the same as the anchor's line 1383 usage.
- `hessians` is never defined on non-master ranks — no risk of undefined variable.

### Step 5: Smoke test (1xH100, ~120s)

```bash
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:15:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export PYTHONUNBUFFERED=1
    pip install --no-cache-dir sentencepiece zstandard &&
    python -u records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py
  '
```

Note: 1xH100 smoke test has `--ntasks=1` (no DDP). The code path goes through the `not distributed` branch, which means `master_process = True` and no barriers. `TokenStream` works without distributed setup since it's a simple sequential reader. Verify this explicitly.

**Gate criteria** (must all pass before 8xH100 run):
- [ ] Hessian collection completes without error
- [ ] GPTQ Cholesky succeeds for ALL layers (no fallbacks — if any layer falls back, investigate before proceeding)
- [ ] Quantized weights in [-32, 31] range
- [ ] Artifact size ≤ 16,000,000 bytes
- [ ] Roundtrip dequant loads successfully (strict=True)
- [ ] Eval produces finite BPB
- [ ] GPTQ wall-clock (calibration + quantization) < 30 seconds

### Step 6: Full 8xH100 run (600s)

```bash
srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard &&
    python -u records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py
  '
```

### Step 7: Documentation updates

Per CLAUDE.md conventions — update in order:
1. `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`
2. `docs/campaign/AGENT_SYNC.md` — new measured results, update next steps
3. `docs/codex-memory/decisions.md` — GPTQ decision and result classification
4. `docs/codex-memory/project-state.md`
5. `docs/codex-memory/next-session.md`

## Locked Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `block_size` | 128 | Standard in referenced PRs and GPTQ paper |
| `percdamp` | 0.01 | Standard in referenced PRs |
| `actorder` | True | Standard in referenced PRs |
| `clip_range` | 31 | Match anchor (scale = row_max/31, clamp to [-32, 31]) |
| `num_samples` | 128 | Match prompt budget (128 sequences target) |
| `seq_len` | 2048 | Match training seq_len |
| `batch_size` | 4 | 4 seqs × 2048 tokens per forward pass |
| `num_batches` | 32 | 128 sequences / 4 per batch |
| `calibration_data` | Training shards | NOT validation (leakage avoidance) |
| `clip_percentiles` | [1.0] only | Conservative start — GPTQ-lite clip search failed on artifact size |
| `compression` | zstd level 22 | Same as anchor |

## Artifact Size Risk Assessment

The anchor artifact is **15,751,324 bytes** with only **248,676 bytes headroom** to the 16MB cap.

**Why Full Hessian GPTQ should NOT bloat like GPTQ-lite did**:
- GPTQ-lite clip search changed *scales* (different percentile → different fp16 scale per row) → different value distributions → worse zstd
- Full Hessian GPTQ with `clip_percentiles=[1.0]` keeps **identical scales** (row-max / 31, same as anchor)
- GPTQ only changes *which int6 value each weight rounds to* — error compensation shifts some weights by ±1 from naive rounding
- The distribution of quantized values should be very similar (same range, same scales, just different rounding choices)

**Mitigation ladder if artifact exceeds cap**:
1. Verify `clip_percentiles=[1.0]` is active (no multi-percentile search)
2. Selective GPTQ: only apply to MLP layers (largest weights, most impact), keep naive for attention
3. Switch compression from zstd to lzma preset 6 (used by the #1 record)
4. Abort GPTQ, keep anchor quantization (nuclear fallback)

## Memory Budget

| Resource | Size | Fits? |
|----------|------|-------|
| Per-layer Hessian (dim=512) | 512×512×4 = 1.0 MB | Yes |
| Per-layer Hessian (dim=1536) | 1536×1536×4 = 9.0 MB | Yes |
| Total Hessians (66 layers on CPU) | ~154 MB | Easily |
| GPU working memory (1 layer at a time) | max 9 MB | Trivial |

## Wall-Clock Budget (post-training, rank-0 only)

| Operation | Time | Status |
|-----------|------|--------|
| Pre-quant EMA eval (all ranks) | ~5s | Existing |
| **Hessian collection (rank 0)** | **3-5s** | **New** |
| **GPTQ quantization (rank 0)** | **3-8s** | **New** |
| Compression + file write (rank 0) | ~3s | Existing |
| Barrier (non-master ranks wait) | — | **New** |
| Roundtrip eval (all ranks) | ~10s | Existing |
| Sliding eval (all ranks) | ~20s | Existing |

Total new time on rank 0: ~6-13 seconds. Well within 30s budget.

## Success Criteria

| Metric | Anchor | Target | Gate |
|--------|--------|--------|------|
| Pre-quant EMA val_bpb | 1.14472403 | ± 0.001 | Confirms training unchanged |
| Roundtrip val_bpb | 1.15247273 | Strictly < 1.15247273 | Any improvement counts |
| Roundtrip gap | 0.00775 | < 0.005 | Core GPTQ benefit |
| Sliding s64 val_bpb | 1.12904446 | < 1.12904446 | End metric improvement |
| Artifact size | 15,751,324 | ≤ 16,000,000 | CRITICAL hard cap |
| step_avg | 91.37 ms | 91.37 ± 1 ms | No training impact |
| GPTQ wall-clock | — | < 30s | Budget feasibility |
| Cholesky fallbacks | — | 0 | If any, result is degraded |

## What's NOT Changed

- Model architecture (11L U-Net, GQA, XSA, BigramHash)
- Training loop, optimizer (Muon+Adam), learning rate schedule, warmdown
- EMA collection and application
- Data loading during training
- Attention mechanism (SDPA with flash backend flags)
- `dequantize_mixed_int6` function (completely unchanged)
- Serialization format (int8 storage, fp16 scales, torch.save dict structure, zstd)
- Evaluation code (standard + sliding window)
- Container (NGC 26.03)

## Verification

After the full 8xH100 run:
1. Pre-quant EMA val_bpb within ±0.001 of 1.14472403 (confirms training unchanged)
2. Roundtrip val_bpb strictly better than 1.15247273
3. Artifact size ≤ 16,000,000
4. step_avg ~91.37ms (no training regression)
5. Zero Cholesky fallbacks (all 66 layers GPTQ'd cleanly)
6. If all pass: commit as `research(protocol): add Full Hessian GPTQ to Session 03 anchor`
7. After results: commit as `research(results): Full Hessian GPTQ measured on 8xH100`
8. If any Cholesky fallback occurred: record as degraded result in AGENT_SYNC.md with layer name and condition number
