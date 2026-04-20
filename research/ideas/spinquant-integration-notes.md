# SpinQuant integration notes for #1736

**Status:** 🟡 DESIGN — implementation-phase notes for spec 009 (SpinQuant hotstart on top of #1736). Findings from reading `train_gpt.py` at commit `154c9b8` (#1736 import). Captured 2026-04-20.

These notes supplement `research/specs/009-spinquant-hotstart.md` with the implementation-level tradeoffs that emerged from reading the code.

## Key finding: RMSNorm is gamma-free

Line 529:

```python
def forward(self, x):
    return F.rms_norm(x, (x.size(-1),), eps=self.eps)   # no weight arg
```

No learnable `gamma`. RMSNorm is already in the "gamma = 1" form that SpinQuant requires, so the usual RMSNorm-gamma-fold preprocessing step is **not needed** for this model. RMSNorm is rotation-equivariant directly:

`RMSNorm(R·x) = (R·x) / ||R·x||_RMS = (R·x) / ||x||_RMS = R · RMSNorm(x)` (orthogonal preserves L2).

## But there are five other per-channel multipliers on residual flow

#1736 has relocated what would traditionally be RMSNorm's gamma into separate per-block parameters that multiply residual-stream vectors at various points. All of these have to be dealt with before a residual-stream rotation R₀ can commute:

| Param | Shape | Where applied | Commutes with R₀? |
|---|---|---|---|
| `ln_scale_factor` | **scalar** per block (`1/√(layer_idx+1)`) | after RMSNorm, before linear | yes (scalar commutes) ✓ |
| `attn_scale` | `[d_model]` per block | after attn output, before residual add | **no — per-channel** |
| `mlp_scale` | `[d_model]` per block | after mlp output, before residual add | **no — per-channel** |
| `resid_mix` | `[2, d_model]` per block | **before** RMSNorm, mixes `lane0 + x0` in the parallel-residual formulation | **no — per-channel AND pre-norm** |
| `skip_weights` | `[num_skip_weights, d_model]` | multiplicative on encoder-saved residual before add to decoder residual | **no — per-channel** |
| `skip_gates` (if enabled) | `[num_skip_weights, d_model]` | sigmoid-gated `torch.lerp` on skip — `lerp(w·skip, lane0, sigmoid(g))` | **no — per-channel AND non-linear** |
| `parallel_post_lambdas` | scalar `[num_layers, 2, 2]` | residual-combination coefficients in two-lane parallel block | yes (scalar) ✓ |
| `parallel_resid_lambdas` | scalar `[num_layers, 2]` | residual-decay coefficients in parallel block | yes (scalar) ✓ |

So the meaningful fold targets are: `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`, `skip_gates`.

## Fold analysis per target

### `attn_scale` — foldable

Applied at line 870: `x_out = x_in + attn_scale[None, None, :] * attn_out`. Fold:

```python
# Before any rotation:
qo_bank[n + i] = qo_bank[n + i] * attn_scale[i][:, None]   # scale output rows
attn_scale[i] = 1.0
```

(Assumes `qo_bank[n + i]` is the attn output projection with shape `[d_model, d_model]` where axis 0 is output dim = residual; `attn_scale[i]` has shape `[d_model]` per residual channel.)

### `mlp_scale` — foldable

Identical pattern to `attn_scale` at lines 871-873:

```python
mlp_down_bank[i] = mlp_down_bank[i] * mlp_scale[i][:, None]   # scale output rows
mlp_scale[i] = 1.0
```

### `resid_mix` — not cleanly foldable

Line 862+: `mix = resid_mix[i]` applied before RMSNorm:

```
attn_read = mix[0] * lane0 + mix[1] * x0
x_post = attn_norm(attn_read) * ln_scale_factor
```

The pre-norm per-channel weighting changes RMSNorm's denominator non-uniformly:

`RMSNorm(diag(α)·x) = (diag(α)·x) / √mean((α·x)²)` — this is NOT equal to `diag(α) · RMSNorm(x)` or `RMSNorm(x) · α_scalar` in general.

So if we rotate the residual stream (`lane0 → R·lane0_orig`, `x0 → R·x0_orig`), the mix step produces:

`mix[0] * R·lane0_orig + mix[1] * R·x0_orig`

which does not equal `R · (mix[0]·lane0_orig + mix[1]·x0_orig)` unless `mix` is constant across channels (it isn't).

**Workarounds, all unsatisfying:**
1. Freeze `resid_mix` to its constant mean (lose the per-channel signal — changes float pass, invariance breaks).
2. Restructure the forward pass to apply `resid_mix` after RMSNorm (mathematically different; would need retraining).
3. Drop the rotation on the residual stream (Option B below).

### `skip_weights` — foldable but fiddly

Line 1102: `lane0 = lane0 + w * skip` where `w = skip_weights[skip_idx][None, None, :]` and `skip` is a *saved residual* from an earlier encoder layer. `skip` is not the output of a single matrix — it's the running residual at the point `skips.append(x)` happens. So you can't fold `w` into a single source-layer weight row cleanly.

**Workaround:** fold `skip_weights[k]` into the encoder's *final* residual-writing linear — whichever matrix last wrote into `skip` before it was saved. Typically that's the previous block's `mlp_down_bank[j]` or `qo_bank[n+j]`. Requires tracing which encoder layer corresponds to `skip_idx = k`.

Doable but adds 30–60 min of careful bookkeeping per skip-index mapping, and the mapping depends on `encoder_indices` / `decoder_indices` which in turn depend on `loop_start / loop_end` config. Fragile if those change.

### `skip_gates` — not foldable

Line 1099–1100: `lane0 = torch.lerp(w * skip, lane0, sigmoid(g))`. Sigmoid is non-linear; per-channel pre-sigmoid parameter can't be absorbed into any linear. If skip_gates is enabled, the skip path is permanently non-rotatable.

Good news: `skip_gates_enabled` is a Hyperparameter flag (line 962). If #1736 ships with `skip_gates_enabled = False`, we can ignore this case. Execution to verify.

## The three options

### Option A — Full SpinQuant (R₀ + R_a + R_m, with all folds)

**Requires:** fold `attn_scale`, `mlp_scale`, `skip_weights` into their source linears. Drop `resid_mix` entirely (by replacing with `mix[0] = mix[1] = 1.0` and accepting the float-pass change as a retraining delta — NOT a zero-cost rotation) OR restructure the forward pass (massive change).

**Expected Δ:** closest to #1695's claimed –0.005 bpb. But `resid_mix` compromise means we're not really doing float-invariant SpinQuant anymore — we'd be doing "SpinQuant + freeze resid_mix," which may well be worse than either alone.

**Integration time:** 1–2 days + significant debug risk.

### Option B — Internal-only (R_a + R_m, no R₀)

**Requires:** nothing on residual stream. Apply R_a per layer on V-output / O-input; apply R_m per layer on fc-output / proj-input. Purely inside attn and mlp blocks.

Banked-tensor ops per layer `i` (with R_a^ℓ shape `[d_head, d_head]` and R_m^ℓ shape `[d_ff, d_ff]`):

```python
# Attention internal rotation R_a[i]
kv_bank[n + i] = R_a[i] @ kv_bank[n + i]        # V output-side (rows)
qo_bank[n + i] = qo_bank[n + i] @ R_a[i].T       # O input-side (columns)

# MLP internal rotation R_m[i]
mlp_up_bank[i]   = R_m[i] @ mlp_up_bank[i]      # fc output-side (rows)
mlp_down_bank[i] = mlp_down_bank[i] @ R_m[i].T   # proj input-side (columns)
```

FP invariance check becomes trivial — each rotation is local to one block and none of the per-channel residual multipliers are touched.

**Expected Δ:** in standard SpinQuant ablations, internal rotations deliver ~60–70% of the total benefit. Naive estimate here: –0.003 bpb (vs –0.005 for full).

**Integration time:** 4–6 hours, low failure risk.

### Option C — Port #1695's exact approach

**Requires:** read `gh pr diff 1695` on `openai/parameter-golf`. Port whatever they did.

**Upside:** if they solved the `resid_mix` problem (e.g., their base didn't have it, or they retrained), we get a proven path.

**Downside:** their base was #1529-adjacent (older parallel-residual formulation), may not include the same per-channel multipliers as #1736.

**Integration time:** depends entirely on what they did.

## Recommendation

Check #1695 first (free, read-only). If clean → do what they did. If not → start with Option B. Reserve Option A as a stretch if B lands and we still have budget.

## Testing "all three"

Because the three options share ~70% of the code (rotation generator, banked-tensor ops, fp invariance check, eval loop), testing all three is cheaper than 3×:

- **Day 1:** read #1695, design common framework with toggles (`ROTATE_RESIDUAL`, `ROTATE_ATTN_INTERNAL`, `ROTATE_MLP_INTERNAL`, `FOLD_SCALES`).
- **Day 2:** implement B (internal-only), run spec 009-B (~$6, 10 min compute). Likely lands ~–0.003.
- **Day 3–4:** implement A's fold path (attn_scale, mlp_scale, skip_weights), decide what to do with resid_mix, run spec 009-A (~$6).
- **Day 5:** if #1695 is readable and differs, port and run 009-C (~$6).

Total compute cost across all three: ~$18 (same `final_model.pt` hotstarts all). Total research/execution time: ~5 days.

That leaves 5 days buffer (with 10 left from 2026-04-20) for spec 010 onward. Tight but feasible.

**Key risk:** if Option A's `resid_mix` compromise produces nonsense numbers, we sink a day on a dead end. Mitigation: always run B first so we have a fallback with a known good number before starting A.

## What the hotstart script needs to export

Regardless of option, `spinquant_hotstart.py` imports from `train_gpt.py`:

```python
from train_gpt import (
    Hyperparameters, GPT, ValidationData,
    serialize, deserialize, eval_val, eval_val_ttt_phased,
    BatchedTTTLoRA, timed_eval, restore_fp32_params,
)
```

And the pipeline is:

```
1. build GPT(h), load state_dict from final_model.pt  [hotstart from spec 008]
2. (if Option A) fold per-channel multipliers into weights
3. generate R_a per layer, R_m per layer, (optionally) R₀
4. apply rotations to banked state_dict, copy back into model
5. FP invariance check (forward pass on 1 batch pre/post)
6. call serialize(h, base_model, code)  [GPTQ runs internally on rotated weights]
7. call deserialize(h, device)
8. run the quantized-eval + TTT blocks from train_and_eval (lines 2969-3075)
9. write final.json
```

Step 8 is where a `run_ttt_eval(h, device, val_data)` helper in `train_gpt.py` (factored out of `train_and_eval`) would clean up the code. Option: inline copy-paste the 80 lines if refactoring feels risky.

## Open questions

1. Is `skip_gates_enabled` True or False in #1736's seed-42 config? (Determines whether Option A is even tractable.)
2. Is `resid_mix` trained meaningfully away from `[1, 1]` or does it stay near the init value? If near init, freezing it may be ~zero-cost — check via examining the loaded state_dict.
3. What did #1695 actually do about these multipliers on the #1529 base (which has a simpler parallel-residual formulation but probably still has `attn_scale`/`mlp_scale`)?
4. Is `torch.linalg.qr(torch.randn(d, d))` sufficient for R, or do we need structured Hadamard? For d=512 both are cheap; Hadamard is theoretically better for outlier spreading but random-orthogonal is also valid.

## Links

- Spec: `research/specs/009-spinquant-hotstart.md`
- Parent idea: `research/ideas/1736-improvement.md`
- Code under review: `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/train_gpt.py`
- Reference PRs: #1695 (SpinQuant V1 on #1529-adjacent base), original SpinQuant paper (Liu et al. 2024).

## Addendum (2026-04-20): how #1695 actually does it — different approach entirely

Read `gh pr diff 1695` after writing spec 009's 2-mode implementation. **#1695's approach is not static weight rotation with folds — it's online activation rotation with rotated-basis GPTQ.** This is a materially different design from what my earlier analysis assumed.

### #1695's design in 4 bullets

- **4 global Hadamard rotations**, generated deterministically from `SPINQUANT_SEED` and a per-site `tag`:
  - `R_attn_in` (d_model) — applied to residual x before Q/K/V projections: `x_qkv = x @ R_attn_in`
  - `R_attn_proj_in` (d_model) — applied to attn output y before O projection: `y = y @ R_attn_proj_in`
  - `R_mlp_in` (d_model) — applied to residual before fc: `x = x @ R_mlp_in`
  - `R_mlp_proj_in` (d_ff) — applied to MLP hidden (post LeakyReLU^2) before proj: `hidden = hidden @ R_mlp_proj_in`
- **Online at forward time**, not baked into weights. The rotations are `register_buffer`s on `CausalSelfAttention` and `MLP` modules, activated by a class-level flag `CastedLinear._sq_active`.
- **OFF during training**, ON after deserialize for eval + TTT. Dynamo constant-folds the branch away during training.
- **GPTQ Hessians are rotated accordingly** — `_spinquant_rotate_sd_and_H` (referenced in their diff) rotates the collected activation covariance matrices to match the rotated forward.

### Why this works (and sidesteps both our blockers)

1. **LeakyReLU problem vanishes.** `R_mlp_proj_in` is applied *after* `F.leaky_relu(...).square()`, not across it. No need to commute rotation through a nonlinearity.
2. **`resid_mix` problem vanishes.** Rotations are per-linear-input, never on the residual stream. The residual keeps its original basis, so all per-channel multipliers (`attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`) operate in their trained basis — unchanged.
3. **No float invariance needed.** The model *is* different from the un-rotated trained model. The bet: the rotation perturbation is small enough that post-quant val_bpb still improves, because GPTQ in the rotated basis sees more evenly distributed activation outliers.

### Cost

- Modifies `CausalSelfAttention.forward`, `MLP.forward`, and the TTT-path forward mirrors (two per block, parallel + sequential variants). ~100 LOC of forward-pass edits in `train_gpt.py`.
- Rotation of GPTQ's collected Hessian — 1 function, probably ~30 LOC.
- `install_spinquant_rotations(...)` to register buffers and flip the class flag — ~40 LOC (already visible in their diff).
- Total: ~200 LOC of actual code change, but it's invasive (touches hot training/eval paths).

### How this reframes our `full` / `port_1695` plan

- **Drop the "static weight rotation with folds" design.** `full` mode as previously scoped (residual-stream R_0 + attn_scale/mlp_scale/skip_weights folds + resid_mix freeze-to-mean) is the textbook SpinQuant V1 approach, but #1695 demonstrates that the online-activation variant delivers the same claimed −0.005 bpb without any of the fold complexity.
- **`port_1695` is the right follow-up, not `full`.** We should port their exact scheme: 4 rotation sites in the forward pass, rotated GPTQ Hessians, class-level activation flag. Much cleaner than trying to fold multipliers.
- **Our R_a-only `internal_only` mode is still useful** as a fp-invariant sanity check and an independent data point. If R_a alone delivers ~−0.002 and port_1695's full online stack delivers ~−0.005, that tells us attention-internal and MLP/residual rotations each contribute roughly half — matching SpinQuant ablations in the literature.

### Open sub-questions for the port

1. Does their `install_spinquant_rotations` get called at a specific point in `train_and_eval`? Presumably after `deserialize()` and before the quantized eval. Confirm insertion point.
2. Their `R_attn_proj_in` rotates `y` (attn output, shape `[B, T, num_heads * head_dim]`) entirely — not per-head. So this is a full d_model rotation on the concatenated attention output, not the per-head R_a I implemented. Related but different mathematical object.
3. How do they handle the post-attention gates (`attn_out_gate`, `gated_attn`) that exist in #1736 but not in their base #1529? These gates multiply `y` per-head before out_proj. A post-gate rotation on the full d_model dim may compose cleanly (the gate is applied before, rotation after), but worth double-checking.
4. GPTQ Hessian rotation — the exact expression for rotating `H = X^T X` when `X_new = X @ R`: `H_new = R^T @ H @ R`. Simple but needs to be applied to the right Hessians at the right time.
5. Their `_spinquant_rotate_sd_and_H` also touches state_dict — do they rotate any weights statically as well? Need to read that function.

This is worth a full follow-up spec (call it spec 011 — "port_1695: online Hadamard rotation with rotated GPTQ"). Not blocked on anything from spec 009; can be designed in parallel while 009 runs.
