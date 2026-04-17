# SKC/Engram Training Diagnosis — 2026-04-17

Scope: training convergence + step-time. Eval-time Legal TTT / Engram explicitly out of scope.

## 1. Executive summary

Two concrete bugs explain every observed regression; a third amplifies it:

1. **Recurrence schedule is gated on nominal `ITERATIONS` (200 000) not wall-clock/actual budget.**
   `train_gpt_verbose.py:4398`
   ```python
   _want_recur = args.recurrence_depth if step >= int(args.iterations * args.recurrence_start_fraction) else 0
   ```
   With `recurrence_start_fraction=0.35` and `iterations=200000`, recurrence first fires at step 70 000. The 10‑min budget on 2× RTX 3090 only reaches step ≈ 411, so **recurrence is dead in every run**. The same bug applies to `ema_start_fraction=0.4` (80 000) and the MoE/feedback schedules. The SKC value proposition depends on recurrence firing; it has never fired in any of these ablations.

2. **`competition_profile` unconditionally overwrites `args.recurrence_start_fraction`**
   `train_gpt_verbose.py:507`
   ```python
   args.recurrence_start_fraction = 0.35
   ```
   (Not guarded by `if _unset('RECURRENCE_START_FRACTION')`.)
   Run C was launched with `RECURRENCE_START_FRACTION=0.65` but the profile clobbered it back to 0.35. The config echo and the step-for-step-identical B/C trajectories confirm this:
   - B val@400 `5.9540 / 2.3327`, C val@400 `5.9513 / 2.3317`
   - B final `5.9352 / 2.3254`, C final `5.9353 / 2.3254`
   B and C are the **same run** with different seeds (or effectively the same seed). A/B/C was intended to isolate recurrence‑start, but A vs B/C only varies WD and A stopped at step 40 — we have no long-horizon recurrence-start data from this bundle.

3. **SKC/MLP scales and residual mixes sit in the WD‑carrying AdamW group.**
   `train_gpt_verbose.py:4069–4094`
   ```python
   for (name, p) in base_model.named_parameters():
       if 'engram.tables' in name: engram_params.append(p)
       elif 'tok_emb' or 'lm_head' or 'embed_proj' or 'per_layer_' in name: head_params.append(p)
       elif _is_skc_structural(name) or p.ndim < 2: adam_params.append(p)   # ← WD group
       else: muon_params.append(p)
   opt_adam = AdamW([{'params': adam_params, 'weight_decay': args.adam_wd}, ...])
   opt_head = AdamW(head_params, ..., weight_decay=args.adam_wd)             # ← per_layer_* also WD
   ```
   With `adam_wd=0.04` and `scalar_lr=0.001`, the effective decay on every scale/gate is `4e-5` per step. SKC init is `0.05`. Real gradient updates on these scalars are ≤1e-4 per step, so WD is the same order as the learning signal and actively fights it. The `skc_scale` drift of **3 parts in 10³** over 400 steps (diagnostics) matches this.

### Why engram looks weak but SKC looks dead-ish
- Engram actually **is** in the training forward path at full weight (`_engram_w=1.0` below `elapsed_fraction=0.6`, `train_gpt_verbose.py:2639–2644`). It's just that:
  - `engram.tables` update via AdamW (lr=0.015, wd=0). Only hash buckets that are *touched* update; with 32 768×192 buckets and 411 steps, norm drift <0.2 % is normal.
  - Non-table engram params (gates/adapters) fall back into `adam_params` with WD.
  - **Engram has no downstream amplifier.** Recurrence/feedback/capsule never fire, so Engram's retrieved context is read once and can only move the head — exactly what we see (`fastest_layer=vocab_bias` then `tok_emb` once it untiés).
- SKC receives gradients (`skc_gcov=1.00`) but the *effective* update is smothered by a combination of:
  1. WD on the scalar gate (`skc_scale`) that gates its entire contribution.
  2. Recurrence never activates — SKC's deep-block benefit requires >1 pass.
  3. Diagnostic buckets alias: `residual_scales` double-counts SKC's `skc_scale/mlp_scale/resid_mix`; `capsule_koopman` matches SKC's `koopman_mixer_*` (so its nonzero grads don't mean capsule is on). The laptop vs pod divergence may be partly diagnostic noise.

## 2. Evidence table

| run | policy | adam_wd | recur_start (cfg) | steps | val@50 | val@100 | val@200 | val@400 | final val | val_bpb | avg ms/step |
|-----|--------|---------|-------------------|-------|--------|---------|---------|---------|-----------|---------|-------------|
| A (first) | **strict*** | 0.04 | 0.35 | 40† | 8.905 | – | – | – | – | – | 1270 |
| A r2 | legacy | 0.04 | 0.35 | ≤40‡ | – | – | – | – | – | – | – |
| B | legacy | 0.00 | 0.35 | 411 | 8.9242 | 7.2128 | 6.3883 | 5.9540 | **5.9352** | 2.3254 | 1266 |
| C | legacy | 0.00 | 0.35§ | 411 | 8.9243 | 7.2132 | 6.3878 | 5.9513 | **5.9353** | 2.3254 | 1263 |

\* first A ran `runtime_path_policy=strict` despite the file name — the pre-fix launcher.
† wall-clock stopped A at only 40 steps per its log tail; no val checkpoints.
‡ `diagnostics_ablateA_legacy_wd004_r35_r2.jsonl` is 0 bytes; re-run did not populate diagnostics.
§ C was launched with env `RECURRENCE_START_FRACTION=0.65` but the competition_profile re-pinned it to 0.35 — confirmed by the CFG echo. **B and C are the same experiment.**

Every run: `feedback_enabled=0, capsule_enabled=0, compile_mode=none, engram_inject_layer=1, iterations=200000, max_wallclock=570`.

## 3. Component SKC/Engram analysis

From `diagnostics_ablateB_*.jsonl` (25 snapshots, 0..400):

- **SKC core** grad_coverage=1.00 throughout. `gw_ratio_mean` drops 0.269 → 2.4e-5 by step 20, recovers to ~3e-4 by step 400. `grad_norm_mean` grows ~50× (1.4e-5 → 7.5e-4); `weight_norm_mean` drifts 6.155 → 6.532 (+6.1 %). **Interpretation:** SKC is learning, but almost all of the measurable weight drift lives in the Ternary/Muon matrices, not the scale/gate. The `skc_scale` stays at ~0.05 → ~0.05015, so SKC's contribution coefficient is stuck at initialization despite stronger grads late.
- **Engram** grad_coverage=1.00. `weight_norm_mean` 623.2 → 623.1 (‑0.02 %). `gw_ratio_mean` 3.0e-5 → 2.9e-4. The table lives; it's sparsely updated. There is no sign that gradients are detached. But engram's downstream sink is only the encoder residual at layer `engram_inject_layer=1`, which is read by the head via 11 more blocks that ignore it unless SKC/feedback amplify it.
- **residual_scales** bucket double-counts SKC's per-block `skc_scale/mlp_scale/resid_mix`. It shows `gw_ratio_mean` ≈ 2–9e-4 (i.e. the same numbers as skc_core minus engram). Treat this bucket as redundant.
- **capsule_koopman** bucket shows grad_coverage=0.75 and weight_norm≈0.06 — those are SKC's `koopman_mixer_*` inner params, **not** the disabled capsule bank. The bucket pattern `('capsule','koopman')` is too broad. Rename/narrow.
- **fastest_layer** = `vocab_bias` for the first ~180 steps, then `tok_stem.tok_emb.weight` after untie. The model is head-dominated for the entire ablation window.

Answering the explicit questions:
- **Is SKC actually learning?** Partially — its matrices receive nontrivial gradients, but the learnable contribution gate (`skc_scale`) is frozen by WD+low `scalar_lr`, so the *model* barely "feels" SKC.
- **Did zero-WD help SKC?** We cannot tell from this bundle. A stopped at step 40; B/C have wd=0 but are identical. The skc_scale drift is the same order across A(40) and B(40): 0.050000→0.050015 vs 0.050000→0.050010. We need an A_full vs B_full comparison at matched steps.
- **Did `recurrence_start_fraction=0.35` vs 0.65 help?** Cannot test — both ran at 0.35 due to the profile bug, and neither reached 70 000 steps anyway.
- **Is Engram contributing to BPB?** Weakly. It's in the forward path, tables are updating, but with no recurrence/feedback the encoder-layer injection is the only exposure. That matches the small delta you saw vs baseline.
- **Why did laptop ablations show stronger Engram?** Highest-likelihood causes: (a) those runs used a non-strict/non-competition profile so recurrence/feedback/capsule were live, (b) Triton-engram kernels were on (here you had to disable with `TRITON_ENGRAM_ENABLED=0`), (c) different `engram_num_heads/orders/buckets` from the strict profile, (d) different `scalar_lr/wd` because `competition_profile` wasn't forcing them. Need to compare CFG dumps.
- **Head-dominant?** Yes, definitively — `vocab_bias` and then `tok_emb` are top every snapshot; `final_norm` grads are tiny; `skc_scale/mlp_scale/resid_mix` are effectively frozen.
- **Diagnostics trustworthy?** Mostly no at the bucket level. `residual_scales` aliases SKC scales; `capsule_koopman` aliases SKC `koopman_mixer`. Component pattern `'router'` in `_is_skc_structural` also spuriously matches any MoE router even when MoE is off.

## 4. Throughput bottleneck (1.27 s/step on 2× 3090, 10 128 tokens)

Per-step budget (coarse, order-of-magnitude):
- Eager (no compile) cost for ~20 M-param hybrid @ 10k tokens on 3090 bf16 ≈ 250–400 ms baseline.
- `DDP_FIND_UNUSED_PARAMETERS=1` + static_graph=0: +10–20 %.
- Dense `TRAIN_LOG_EVERY=1` + JSONL flush + per-param norm scan in `_poll_nn_diagnostics` (O(params) Python loop): +50–150 ms (Python-side on master rank only, but blocks step).
- Synchronous `grad_scale` / `vocab_bias.grad` reads on master every step: +5–10 ms GPU→CPU.
- Python graph breaks through `torch.compiler.is_compiling()` checks, Triton engram dispatch, and SKC scan checkpointing in eager: expected, absorbed.
- Data/dataloader: FineWeb sp8192 sharded local files on 3090 box — not obviously a stall; confirm with a tag that prints dataloader wait time.
- 3090 has ~1/5–1/6 the H100 throughput in bf16 Tensor Cores; the *absolute* 1.27 s on 3090 implies ~0.25–0.35 s/step on 8× H100 with the same ops, which is in the right ballpark.

Expected diagnostic overhead alone (TRAIN_LOG_EVERY=1 + DIAGNOSTICS_ENABLED=1) is 200–400 ms on this model; that is most of the gap from "expected eager 300 ms" to the observed 1266 ms. Compile would erase another ~50 %.

## 5. Ranked patch plan

### P0 — ML correctness / convergence

**P0-1. Make recurrence/EMA/feedback/Moe schedules wall-clock-aware, not step-index-aware.**
File: `train_gpt_verbose.py:4398` and every use of `int(args.iterations * args.<x>_start_fraction)`.
Minimal patch: replace step comparisons with `step_fraction(step)`, which is already defined and is the elapsed wall-clock fraction. Alternatively, pre-compute `recurrence_start_step = int(effective_steps * fraction)` where `effective_steps` is estimated from `MAX_WALLCLOCK_SECONDS / avg_step_ms` after warmup.
Acceptance: with `recurrence_start_fraction=0.35` and a 10-minute budget, recurrence activates around step 150 (~35 % of ~410 steps), not step 70 000. Verify with a diag log `recurrence_depth:3` appearing before step 200.

**P0-2. Respect env over competition_profile for schedule knobs.**
File: `train_gpt_verbose.py:507` (and nearby 508/511/515 for `matrix_lr`, `scalar_lr`, `ema_start_fraction`).
Patch (line 507):
```python
if _unset('RECURRENCE_START_FRACTION'):
    args.recurrence_start_fraction = 0.35
```
Apply the same `_unset` guard to `ema_start_fraction`, `scalar_lr`, `matrix_lr`, `muon_wd` at lines 510–515 so future sweeps are actually sweeps.
Acceptance: setting `RECURRENCE_START_FRACTION=0.15` in env produces a CFG echo of `recurrence_start_fraction=0.15`.

**P0-3. Move SKC/Engram/residual scales and gates to a no-WD group, with a dedicated LR.**
File: `train_gpt_verbose.py:4069–4095`.
Patch:
```python
skc_gate_patterns = ('skc_scale','mlp_scale','attn_scale','resid_mix',
                     'decay_rates','add_gate','mul_gate','recurrent_gate',
                     'vrl_alpha','mixer_scale','skip_weights','skip_weight',
                     'q_gain','gate_proj','router.','feedback_gate',
                     'content_scale','engram_gate')
adam_nodecay_scales_params = []
for (name, p) in base_model.named_parameters():
    if not p.requires_grad: continue
    if 'engram.tables' in name: engram_params.append(p); continue
    if 'tok_emb' in name or 'lm_head' in name or 'embed_proj' in name: head_params.append(p); continue
    if 'per_layer_' in name or any(k in name for k in skc_gate_patterns):
        adam_nodecay_scales_params.append(p); continue
    if _is_skc_structural(name) or p.ndim < 2:
        if id(p) in _ternary_param_ids: adam_nodecay_params.append(p)
        else: adam_params.append(p)
    else: muon_params.append(p)
...
opt_adam = AdamW(
    [{'params': adam_params, 'weight_decay': args.adam_wd},
     {'params': adam_nodecay_params, 'weight_decay': 0.0},
     {'params': adam_nodecay_scales_params, 'weight_decay': 0.0,
      'lr': args.scalar_lr * float(os.environ.get('SCALES_LR_MULT','3.0'))}],
    lr=args.scalar_lr, betas=(args.beta1, args.beta2), eps=args.adam_eps)
```
Acceptance: diagnostics show `skc_scale` drifting to ≥0.06 (or visibly ≥+20 %) by step 200 at `SCALES_LR_MULT=3` — a 1000× bigger signal than today.

**P0-4. Separate diagnostic buckets so `capsule_koopman` and `residual_scales` don't alias SKC internals.**
File: `train_gpt_verbose.py:2728–2738`.
Patch:
```python
component_rules = {
    'skc_scan':   ('decay_rates','mixer_conv','mixer_lowrank','mixer_diag','mixer_scale','spec_proj','gate_proj'),
    'skc_gates':  ('skc_scale','mlp_scale','attn_scale','resid_mix','per_layer_skc_scales','per_layer_mlp_scales','per_layer_attn_scales','per_layer_resid_mixes'),
    'skc_koopman':('koopman_mixer','koopman_state','koopman_conv','koopman_speculator'),
    'engram_tables': ('engram.tables','bigram_hash_table'),
    'engram_ctrl':   ('engram.router','engram.gate','engram.proj','engram.q_proj','engram.k_proj','engram.v_proj'),
    'feedback':      ('feedback',),
    'capsule':       ('capsule_bank','capsule_state','capsule_carry'),
    'head':          ('vocab_bias','lm_head','tok_emb','embed_proj'),
}
```
Buckets become mutually exclusive; require `not any(p matched earlier)` to avoid double-count.
Acceptance: `skc_koopman` is 0 when `capsule_enabled=0`; `engram_tables` and `engram_ctrl` are separately visible.

**P0-5. Add Engram-on vs Engram-off A/B under identical legacy/wd0/r=0.35 config with P0-1 in place.**
Acceptance: a single boolean flip (`BIGRAM_HASH_ENABLED=0`) moves final `val_bpb` by >0.01 one way or the other — if not, engram really is inert and we should either raise `ENGRAM_LR`, untaper the 0.6 gate, or inject it later (after recurrence is active).

### P1 — Throughput (only after ML path is healthy)

**P1-1. Keep diagnostics off for speed runs.** With `DIAGNOSTICS_ENABLED=0` and `TRAIN_LOG_EVERY>=10` the per-step Python scan and JSONL flush disappear. Expected savings 150–300 ms.
**P1-2. Re-enable block-specific `max-autotune` compile.** You confirmed ~7 s compile cost separately; expected step cost 0.4–0.6 s on 2× 3090.
**P1-3. Retire `DDP_FIND_UNUSED_PARAMETERS=1` once recurrence/feedback actually run every step.** It exists because path strict ran unused capsule/feedback params through DDP. After P0-1/P0-2, those paths are consistently used or consistently absent.
**P1-4. Skip `opt_matrix` active-grad logic on eager path:** the `active_grad_eps` gate is fine but Muon backend_steps=5 in eager is expensive; keep the existing warmdown ramp.
**P1-5. Profile one step with `torch.profiler` and publish to `scratch/prof_<runid>.json` to fingerprint any remaining stall (engram kernel, scan checkpoint, dataloader).**

### P2 — Speculative

- Move `engram_inject_layer` deeper (e.g. layer 6) so the head can actually read what Engram wrote.
- Replace the `TaperedGradients.apply(engram, _engram_w)` cliff at elapsed_fraction=0.6 with a smooth ramp that is wall-clock-aware (same fix as P0-1).
- Add an explicit "engram self-consistency" aux loss during training so engram tables receive a gradient that isn't just a by-product of the CE loss (similar to koopman_consistency_weight).

## 6. Next experiment matrix

After landing P0-1, P0-2, P0-3, P0-4 on a branch:

| exp | RUNTIME_PATH_POLICY | ADAM_WD | SCALES_LR_MULT | RECURRENCE_START_FRACTION | BIGRAM_HASH_ENABLED |
|-----|---------------------|---------|----------------|---------------------------|---------------------|
| E1 (control) | legacy | 0.00 | 1.0 | 0.35 | 1 |
| E2 (scales-unfrozen) | legacy | 0.00 | 3.0 | 0.35 | 1 |
| E3 (recur-early, wall-clock aware) | legacy | 0.00 | 3.0 | 0.20 | 1 |
| E4 (engram-off) | legacy | 0.00 | 3.0 | 0.20 | 0 |
| E5 (strict-corrected) | strict | 0.00 | 3.0 | 0.20 | 1 |

Shared env:
```
COMPILE_MODE=none DIAGNOSTICS_ENABLED=1 DDP_FIND_UNUSED_PARAMETERS=1
TRAIN_LOG_EVERY=1 TRAIN_LOG_EVERY_FRACTION=0
VAL_LOSS_EVERY=50 VAL_LOSS_EVERY_FRACTION=0
EXPORT_ALIGNED_TRAIN=0 EXPORT_PROXY_EVAL=0
ITERATIONS=200000 MAX_WALLCLOCK_SECONDS=570
TRAIN_BATCH_TOKENS=10128 TRITON_ENGRAM_ENABLED=0
```

Read-out per run: per-step train loss slope 0..50, 0..200; val@50/100/200/400; final val_loss / val_bpb; `skc_scale` drift; `engram.tables` weight_norm drift; `skc_gates` gw_ratio_mean trajectory; recurrence activation step.

Decision gate:
- If E2 beats E1 by >0.01 bpb and `skc_scale` drifts >20 %, P0-3 lands.
- If E3 beats E2 by >0.01 bpb, P0-1 lands and we tie all fractions to wall-clock.
- If E4 regresses <0.01 bpb vs E3, Engram is inert and we should escalate P2.

Separately, once compile is back on:
- Sweep `COMPILE_MODE ∈ {max-autotune-no-cudagraphs, max-autotune}` × `DDP_FIND_UNUSED_PARAMETERS ∈ {0,1}` at E3 settings; pick the fastest config that still converges identically.

## 7. Uncertainty

- We have no long-horizon run under `adam_wd=0.04` (A stopped at step 40, A_r2 left empty diagnostics). Claim "WD smothers SKC" is consistent with the math and the 40-step parity but is not proven at 400 steps.
- We have **no** run where recurrence fires. Until P0-1 lands, every claim about "SKC learning behavior" is a claim about the non-recurrent forward path only.
- Laptop vs pod divergence: the original strong-Engram laptop runs' exact CFG is not in this bundle; conclusions about "what changed" are inference from the strict/legacy gating, not direct comparison. Please capture the laptop CFG echo when you next repro it.
- `_poll_nn_diagnostics` reads `p.grad.norm()` post-backward but **before** `grad_clip_norm=1.0` and before optimizer step. Reported grad magnitudes reflect pre-clip grads; after clipping, effective updates are smaller. This does not change the qualitative conclusions but makes absolute `gw_ratio` numbers mild overestimates of applied update strength.
- Step-time attribution above is order-of-magnitude; a `torch.profiler` run is needed to finalize the breakdown.
