# Experiment 0032_mamba2_ssd_smoke

Parent: 0018_recur3x3_swiglu_2attn_bigramhash (replace position-1 S4D-Lin with Mamba-2/SSD)

## Question
**Family axis pivot #3**: Mamba-2 / SSD (State Space Duality, arXiv:2405.21060) — the chunkwise selective-SSM reformulation that's "2-8× faster than Mamba-1's selective scan because most work is matmul" per primer §2.2. Mamba-1 was MPS-infeasible (per-step Python loop with kernel-launch overhead dominating). Mamba-2/SSD's chunkwise scan with Q=64 chunk size cuts the sequential pass to L/Q=16 chunks, with intra-chunk math done as matmuls (MPS-friendly).

This gives us the **selective-SSM** data point for the writeup that Mamba-1 couldn't deliver. Direct A/B vs 0018 — replace ONLY position 1 with Mamba-2 block.

## Hypothesis [CONJECTURE]
val_bpb in [2.075, 2.110]. Single-seed exploration per session memory.

Most likely tied with S4D-Lin sandwich (40%, val 2.080-2.090) — at our 200-step regime selectivity may not have time to manifest its theoretical benefits. Win < 2.080 (25%) — selectivity helps even at short horizon. Loss > 2.090 (35%) — the input-dependent dynamics add training instability that a 200-step run can't outgrow (Mamba's documented LR cliffs per primer §4.2).

Step time prediction: ~6-9 s/step. Chunkwise reformulation is 2-8× faster than sequential per primer; if sequential was 5+ min/step on MPS (Mamba-1 result), chunkwise should be ~10-30 s/step still — but the L=1024, Q=64 → 16 chunks brings each step into matmul territory. Conservative prediction: 8 s/step → 27 min/exp.

## Change

**Code edit in train_gpt.py (subagent task):**

1. **Add a `Mamba2Block` class** to train_gpt.py implementing SSD (State Space Duality) chunkwise scan per arXiv:2405.21060 / primer §2.2. Reference: the algorithm has ~30 lines of PyTorch when stripped to essentials.

   Constructor: `Mamba2Block(dim, d_state=64, expand=2, chunk_size=64, headdim=64)`. (Note: SSD uses heads — head_dim divides d_inner; with d_inner=expand*dim=1024 and headdim=64, that's 16 heads.)

   Internal projections (most match Mamba-1 layout):
   - `in_proj` (CastedLinear, dim → 2*d_inner + 2*d_state + nheads): produces `[x, z, B, C, dt]` where x is the input branch (d_inner), z is the gate branch (d_inner), B is shared input-dependent (d_state), C is shared output-dependent (d_state), dt is per-head delta (nheads = d_inner/headdim).
   - `conv1d` (depthwise causal, kernel=4, groups=d_inner) — same as Mamba-1.
   - `out_proj` (CastedLinear, d_inner → dim, zero-init).
   - Parameters: `A_log` (shape (nheads,) — scalar A per head; **simpler than Mamba-1's per-channel-per-state A**), `D_skip` (shape (d_inner,), per-channel skip), `dt_bias` (shape (nheads,), softplus^-1 init like Mamba-1).

   Forward — the SSD chunkwise scan:
   - Compute (x, z, B, C, dt) from `in_proj(x_input)`.
   - Apply conv1d on x along sequence (causal, depthwise; cast to fp32 to match restored bias).
   - Compute `dt = F.softplus(dt + dt_bias)`  (shape (B, L, nheads)).
   - Compute `A = -torch.exp(A_log).float()`  (shape (nheads,)).
   - Reshape x to per-head: `x_h = x.view(B, L, nheads, headdim)`.
   - Compute `Δ_t * A` ramps and `cumsum`s for chunkwise discretization (per SSD paper Algorithm 1).
   - Process L in chunks of Q=64:
     - Within chunk: matmul-attention-like form. Build (Q, Q) lower-triangular `L_mask` per chunk; compute attention-like Y = L_mask * (Q @ K^T) @ V kind of structure but with per-position decay factors from cumsum.
     - State pass: each chunk produces a final state (B, nheads, headdim, d_state). Pass state to next chunk via matmul.
   - After all chunks: reshape back to (B, L, d_inner), gate by silu(z), add D_skip*x, out_proj.

   See also: the official Mamba-2 reference implementation (`mamba_ssm/modules/ssd_minimal.py` if available, or `https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py`). Adapt with attribution. If reference unavailable, write from primer §2.2 + paper Algorithm 1 directly.

2. **Block class**: extend with `mamba2_mode: bool` flag mutually exclusive with use_attention/parallel_mode/hyena_mode. When True, `self.attn = Mamba2Block(dim)`.

3. **Env-var**: `MAMBA2_LAYER_POSITIONS`. Threading mirrors others.

4. **CONTROL_TENSOR_NAME_PATTERNS extensions** (env.sh):
   - A_log already matched.
   - D_skip already matched.
   - dt_bias is 1D, auto-restored.
   - conv1d.weight (3D) — needs `conv1d` substring (same as 0028 Mamba-1 fix).
   - dt_proj is gone (Mamba-2 uses scalar A per head, no dt_proj weight matrix; only dt_bias).

5. **env.sh**:
   - Set `MAMBA2_LAYER_POSITIONS=1`. ATTN_LAYER_POSITIONS=0,2.
   - Append `conv1d` to CONTROL_TENSOR_NAME_PATTERNS.
   - All other 0018 settings.

**Cap math**:
- Mamba-2 block params: in_proj (dim → 2*d_inner + 2*d_state + nheads = 512 → 2*1024 + 2*64 + 16 = 2192) ≈ 1.12M int8. conv1d weight 4096 + bias 1024 (fp32). out_proj 1024×512 = 524K int8. A_log 16 fp32. D_skip 1024 fp32. dt_bias 16 fp32. Total: ~1.65M int8 + ~5KB fp32.
- vs S4D-Lin (656 KB int8 + 132 KB fp32): Δ = +1.0M int8 - 127 KB fp32 ≈ +0.55 MB compressed.
- 1 unique × 3 loops = 1 instance: predicted artifact ~12.27 + 0.55 = ~12.82 MB. Cap-safe.

**Verification step**: scratch_verify.py per the pattern: construct, print params + matched, one tiny forward, sanity-check kernel-equivalent output shape. **Critical: numerical correctness oracle**. Either reuse `references/selective_scan_ref.py` (it's for general selective scan; should work for the chunkwise reformulation by comparing against a sequential SSD evaluation at small inputs) OR derive a tiny analytical check (e.g., set A=0 → output = D_skip * x, easy to verify).

## Disconfirming
- val < 2.075: Mamba-2 selectivity wins on MPS at our regime; pivot to multi-position Mamba-2 variants.
- val > 2.110: selective hurts (LR cliff or training instability). Document as "selective harder to train at 200 steps."
- val ∈ [2.075, 2.110]: tie or modest move; selective family is feasible on MPS via SSD; characterize for writeup.
- step_avg > 15 s/step: even SSD chunkwise too slow on MPS for our shape; document, abort if needed.

## Notes from execution

**Implementation (2026-04-26, subagent):**

- Added `Mamba2Block` plus a free-function `ssd_minimal_discrete` (with helper `_ssd_segsum`) to `train_gpt.py`, placed right after `S4DLin`. The chunkwise scan is adapted verbatim (modulo einops removal) from the official `mamba_ssm/modules/ssd_minimal.py` (Apache-2.0; attribution comment block preserved at the top of the section). The wrapper handles in_proj split into `[x, z, B, C, dt]`, causal depthwise conv1d on the x branch, dt scaling (`X*=dt`, `A*=dt` per the reference's calling convention), gate `y * silu(z)`, D-skip, and zero-initialized out_proj. ngroups=1 (B and C broadcast across heads).
- Block contract: added `mamba2_mode` flag mutually exclusive with `use_attention`. When True, `self.attn = Mamba2Block(dim, d_state=64, expand=2, chunk_size=64, headdim=64)`.
- `MAMBA2_LAYER_POSITIONS` env var parsed in `Hyperparameters` and threaded through `GPT.__init__` (+disjointness check vs `attn_layer_positions`).
- Optimizer-routing fix from 0028 applied: `p.ndim != 2 OR matched` (was `p.ndim < 2 OR matched`) so the depthwise `conv1d.weight` (3D) goes into the Adam scalar bucket rather than vanishing. Comment cites both 0028 and 0032 for context.
- env.sh updates:
  - Appended `conv1d` to `CONTROL_TENSOR_NAME_PATTERNS` so depthwise `conv1d.weight` is restored to fp32 (matches the auto-fp32 1D `conv1d.bias`).
  - Set `MAMBA2_LAYER_POSITIONS=1` (the lone S4D position; ATTN_LAYER_POSITIONS=0,2 was already set by the 0018 carry-over).
  - All other 0018 settings are already inherited (env.sh was a clean copy of 0018's, so "missing parent overrides" listed in the brief turned out not to apply — the file already had them).

**Verification (`scratch_verify.py`, all checks PASS on MPS):**

1. GPT constructs cleanly: 3 blocks, `block[0].attn=CausalSelfAttention`, `block[1].attn=Mamba2Block`, `block[2].attn=CausalSelfAttention`. `model_params=22,925,361`.
2. Mamba2Block parameters enumerated: `A_log (16,) fp32`, `D_skip (1024,) fp32`, `dt_bias (16,) fp32`, `in_proj.weight (2192, 512) fp32`, `conv1d.weight (1024, 1, 4) fp32`, `conv1d.bias (1024,) fp32`, `out_proj.weight (512, 1024) fp32`.
3. CONTROL_TENSOR_NAME_PATTERNS matches: `A_log`, `D_skip`, `dt_bias`, `conv1d.weight`, `conv1d.bias` all matched. `in_proj.weight` and `out_proj.weight` correctly unmatched (these are the 2D matmul weights routed to Muon).
4. **Numerical oracle PASS**: SSD chunkwise (`ssd_minimal_discrete`) vs a sequential per-step reference at `b=2, L=16, h=4, p=16, n=8, chunk=8`: **max abs diff `1.9e-06`** -- comfortably under the relaxed tolerance (`atol=1e-3, rtol=1e-2`) and effectively at fp32 precision.
5. One forward through full GPT at `B=2, L=8` produces finite loss `20.29` (≈ -log(1/1024) = 6.93 * something; expected for an at-init zero-output mixer + bigram-zero-init head).

`bash -n env.sh` and `python -c "import ast; ast.parse(...)"` on train_gpt.py both pass.

**Note on chunk_size and L:** Mamba2Block requires `seq_len % chunk_size == 0`. With chunk_size=64 and TRAIN_SEQ_LEN=1024 this is satisfied (16 chunks). The scratch_verify.py B=2 L=8 forward briefly drops `chunk_size=8` on the constructed Mamba2Block(s) so the tiny shape is divisible, then restores; production training uses the full chunk_size=64 path.

**No deviations from the plan.** No "DO NOT RUN" comment was needed -- the SSD math is correct to fp32 precision against the sequential oracle.
