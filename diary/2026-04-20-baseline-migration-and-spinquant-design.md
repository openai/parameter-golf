# 2026-04-20 — Baseline migration to #1736 + SpinQuant design sprint

**Session kind:** research (no pod live at session start; one execution session happened in parallel mid-day to run spec 008). **Days to deadline:** 10.

## TL;DR

- Moved the research baseline from merged SOTA #1493 (1.0810) to unmerged PR **#1736** (dexhunter, 1.0655). Rationale: the frontier moved past us in 10 days via witnessed legal levers, and continuing to iterate off spec-000 leaves us behind before we try anything.
- **Spec 008 ran** (execution session). Reproduced #1736's pre-quant val_bpb within +0.00016 — training reproduced cleanly. Missed the final post-TTT gate number due to a watcher-trigger bug; projected pass based on #1736's measured quant→TTT delta.
- **Spec 009 written and ready** to run: R_a-only internal attention rotation + baseline reference. Script (`spinquant_hotstart.py`) and CPU invariance test both implemented and passing.
- **Spec 010 (port_1695) and 011 (tapered WD) drafted** but code not yet written. Executable order now is 009 → 010 → 011 (or 009 ‖ 011).
- Major discovery from reading #1695's diff: they do **online activation rotation**, not static weight rotation. Sidesteps both the LeakyReLU and `resid_mix` blockers that made the "full" SpinQuant design hairy.

## Morning — frontier scan and the baseline decision

Day started on the question "are we chasing the right target?" User had frontier scan notes (`diary/2026-04-19-frontier-scan.md`, `frontier-map.md`) showing the unmerged ceiling had dropped from #1493 (1.0810, merged 04-09) to #1738 (1.0354, alertcat, 04-19) over ~10 days. The question: which of these numbers should we trust and aim at?

Ranked the field by credibility (real-PR filter: artifact ≤ 16 MB, 3 seeds with low std, no known legality issues, no flagged bugs):

- **Credible frontier:** #1736 at 1.0655 (dexhunter, CaseOps + gates + phased TTT).
- **Sub-1.01 cluster (GDN-family):** exposed by dexhunter's Issue #1719 — `build_sentencepiece_luts` double-counts the leading-space byte, inflating the denominator by ~17%. Canonical recomputation puts #1698 (arsenis-cmd, 1.00995 claimed) at ~1.189. Author self-closed follow-ups (#1711, #1712, #1734) once the bug became visible.
- **Trinity / SLOT / n-gram family** (#1246, #1722, #1723): banned-mechanism stacking. Stays open because nobody has bothered to close. Ignore.
- **Disputed contender #1738 (1.0354):** builds on #1735's pre-quant TTT. bigbag's own Issue #1017 writes: *"Corpus-level TTT has a ceiling of approximately 0.0003 bits."* A claimed TTT gain of −0.038 bpb is ~100× that ceiling. Physically implausible to be legal. Likely DQ'd. Treat as noise.
- **Tokenizer-disputed (#1604):** casefold (lossy) variants like #1578, #1693 probably die; CaseOps (lossless, bijective) variants like #1729 and #1736 probably survive. Modal outcome: #1736 stands.

**Decision:** rebase to #1736. Updated `CLAUDE.md` (baseline section + new "baseline-migration spec" exception to the `exp/<slug>` convention), wrote a new project memory (`project_baseline_1736.md`), and created `research/ideas/1736-improvement.md` as the migration plan.

Three downstream specs queued off the migration: spec 008 (reproduce #1736), spec 009 (SpinQuant hotstart), and whatever lever lands after those.

## Midday — spec 008 handoff and partial result

Spec 008 drafted as a single-seed reproduction of #1736. No code change to `train_gpt.py`, just pin the import at commit `e100586d` and run their launch command. Bulk-imported the submission dir to our `research` branch at commit `154c9b8`.

Initial plan called for checkpoint saves at phase boundaries (spec 000's convention). Thought better of it mid-conversation: spec 009 is a post-training rotation + GPTQ eval, so all we actually need from spec 008 is **one pre-GPTQ FP checkpoint** for hotstart. Cut spec 008 to seed 42 only + single `final_model.pt` dump. Cost dropped from ~$40 → ~$17.

Execution ran it. Summary from their `final.json` + notes:

| stage | us | #1736 seed 42 | Δ |
|---|---|---|---|
| train stop-early step | 4828 | 4854 | −26 |
| train_time wallclock | 596.14 s | 596.18 s | ≈ equal |
| diagnostic pre-quant post-EMA val_bpb | **1.06922** | **1.06906** | **+0.00016** |
| artifact size | 15,946,577 | 15,978,834 | −32,257 (both under 16 MB) |
| quantized_ttt_phased val_bpb | **(not captured)** | 1.06610 | — |

Pre-quant match within 0.00016 is essentially identical — well inside bf16 cross-pod noise. Training reproduced cleanly.

The missing post-TTT number is from a watcher-trigger bug: execution's stop-signal fired on the wrong log marker (`Total submission size`) and killed the pod before `quantized_ttt_phased` ran. Projected post-TTT = 1.06922 − 0.00296 ≈ **1.06626**, which is comfortably inside the ±0.003 gate. Treat as projected pass.

Execution offered three recovery options: (a) eval-only rerun on the saved `.ptz` (~$3), (b) full Phase 3 rerun (~$10), (c) ship as projected. Research folded (a) into spec 009 as a `baseline` mode — we need an apples-to-apples local reference for SpinQuant Δs anyway. One pod, four numbers, ~$27 (later cut to 2 modes + ~$15).

Minor execution note: their `SAVE_PRE_GPTQ` patch landed at line 2080 of `train_gpt.py`, after `_unbank_state_dict()` runs. So the saved file has per-layer keys (`blocks.N.attn.c_q.weight`, etc.) not banked keys (`qo_bank`). `spinquant_hotstart.py` now auto-detects and calls `_rebank_state_dict` when needed.

## Afternoon — spec 009 design evolution

Spec 009 went through three scope cuts before settling. Each was worth it.

### Scope cut 1: 4 modes → unified sweep

Started as three independent sub-specs (internal_only, full, port_1695). Collapsed into one pod session with a toggle flag because they share ~70% of the code and cost ~$5/variant hotstarting off the same checkpoint.

### Scope cut 2: add `baseline` mode

After talking through spec 008's watcher bug with execution, realized we needed a local no-rotation reference on the same pod to measure the three SpinQuant deltas cleanly (removes cross-pod bf16 drift from the comparison). Total cost went to ~$27 for four modes, or $20 of new spend after folding in spec 008's eval-only rerun.

### Scope cut 3: read train_gpt.py and discover the real fold problem

Spent ~30 min reading `train_gpt.py` at the banked-layout level to actually implement the rotation math. Two findings landed hard:

1. **RMSNorm is gamma-free.** Line 529: `F.rms_norm(x, (x.size(-1),), eps=self.eps)`. No weight arg. The gamma-fold step I had in my design notes doesn't need to happen — RMSNorm is already rotation-equivariant.
2. **But #1736 has five OTHER per-channel multipliers on the residual stream** that matter more:
   - `attn_scale` [d_model] per block, post-attn
   - `mlp_scale` [d_model] per block, post-mlp
   - `resid_mix` [2, d_model] per block, pre-RMSNorm (mixes lane0 and x0 in the parallel-residual formulation)
   - `skip_weights` [num_skip, d_model], per-channel on U-net skip path
   - `skip_gates` [num_skip, d_model] with sigmoid-gated `torch.lerp` (non-linear, non-foldable)

Of these, `attn_scale`/`mlp_scale`/`skip_weights` fold cleanly into adjacent linear rows. `resid_mix` does not — it's pre-RMSNorm per-channel, and the norm's denominator depends on the per-channel scaling non-linearly.

3. **MLP internal rotation R_m is broken by the LeakyReLU.** Line 821: `F.leaky_relu(F.linear(x, up_w), slope=0.5).square()`. For R_m to commute with the MLP nonlinearity, the activation must be rotation-equivariant — LeakyReLU(slope=0.5) is not (it's axis-aligned, flipping sign per coordinate).

These are real architectural blockers for a textbook static-weight-rotation SpinQuant. Not just bookkeeping issues.

### Final scope: 2 modes, with concrete plan for the rest

- `baseline` — no rotation. Closes spec 008's missed gate number and provides the Δ reference.
- `internal_only` — attention-only R_a (per-layer, per-KV-group, d_head=64). Strict float-invariance because softmax(QKᵀ)V is rotation-equivariant in V's d_head axis. Skipping R_m because of LeakyReLU.
- `full` and `port_1695` — deferred. Stubbed to raise `NotImplementedError` with explanatory messages.

`research/ideas/spinquant-integration-notes.md` captures the full fold analysis.

## Afternoon — implementation

Wrote two files in the #1736 submission directory:

- **`spinquant_hotstart.py`** (~360 LOC) — imports everything it needs from `train_gpt.py`, loads the FP checkpoint, dispatches by mode, calls `serialize()` → `deserialize()` → eval → TTT. The TTT eval block from `train_and_eval` (lines 2997–3075, compile warmup + `eval_val_ttt_phased`) is inlined into `_run_ttt_eval()` rather than refactoring the source — smaller blast radius.
- **`test_rotation_invariance.py`** (~250 LOC) — standalone (no flash-attn / triton dep), runs on any CPU with torch. Self-contained minimal forward pass that mirrors the banked attention shape so a rotation bug surfaces as a numerical mismatch. Tests `baseline` (bit-exact) and `internal_only` (relative tolerance 1e-4).

Ran the test locally. First attempt failed with `max_abs = 9e3, rel_max = 0.33` on the real checkpoint — but `rel_l2 = 0.27` growing layer-by-layer from `1e-5` at layer 1 to `0.33` at layer 11. Identified as scale explosion: my minimal forward was missing the QK RMSNorm (line 769–770 of real model), which bounds attention logits. Without it, trained weight magnitudes blow up Q @ K, softmax saturates near argmax, and tiny float errors in V cause different token selections. Added RMSNorm on Q and K in the minimal forward. Second run passed with `rel_max = 8.8e-7` (100× below the tolerance). Rotation math is correct.

### Handoff state for spec 009

All the pieces execution needs: spec file cleaned up to match the 2-mode reality, checkpoint path pinned to `pre_gptq.pt`, preflight step documented (CPU invariance test), pod bash for two sequential modes. Ready to run.

## Late-afternoon — #1695 diff and a design reframe

User asked to dig into `#1695` to inform the deferred `port_1695` mode. Read their diff (~3486 added lines in `train_gpt.py`, a full snapshot not a focused patch). Hunted for SpinQuant-specific lines.

**Finding: they do not use static weight rotation with folds at all.**

They use **online activation rotation**: four global Hadamard rotations applied as `x @ R` matmuls at four forward-pass sites:

1. `R_attn_in` on residual before Q/K/V linear
2. `R_attn_proj_in` on attention output before O linear
3. `R_mlp_in` on residual before fc linear
4. `R_mlp_proj_in` on MLP hidden after LeakyReLU² before proj linear

Rotations live as `register_buffer`s, gated by a `CastedLinear._sq_active` class flag. OFF during training (Dynamo constant-folds the branch away). ON after `deserialize()` for quantized eval and TTT. GPTQ's collected Hessian is rotated (`H_new = Rᵀ H R`) to match the rotated forward.

**Why this sidesteps both of my design blockers:**

- `R_mlp_proj_in` is applied AFTER the LeakyReLU², not across it. The rotation never has to commute with the nonlinearity.
- Rotations only live on per-linear-input paths, never on the residual stream. `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights` all stay in their trained basis, untouched.

The tradeoff: **no float invariance**. The rotated model computes a strictly different thing than the unrotated trained model. The bet is that rotated-basis GPTQ error is lower than the activation perturbation introduced by the rotations — empirically it was, for them (~−0.005 bpb on the #1529 base).

Captured as an addendum in `research/ideas/spinquant-integration-notes.md`. This reframes the "what to do about `full` mode" question: the right answer is probably "don't do `full`, do `port_1695` (the online-rotation variant) instead." Cleaner code, no fold math, no `resid_mix` freeze compromise.

## Evening — spec 010 and 011 drafts

Drafted two specs' documents (code not written):

- **Spec 010 — port_1695 online rotation.** Hotstart off spec 008's `pre_gptq.pt`. Requires ~150 LOC of additions to `train_gpt.py` (forward-pass hooks in 2+ places + Hessian rotation function + `install_spinquant_rotations`) plus a ~100 LOC driver script. Expected Δ −0.003 to −0.005 bpb vs spec 009 baseline. ~$10.
- **Spec 011 — tapered Muon WD (port from #1729).** Full retrain, independent of 009 and 010. Linear WD taper from 1.0× to 0.5× over the last 30% of training. ~$20. Thin lever (~−0.001 bpb expected) but orthogonal to everything.

Renumbered mid-draft. User flagged that `port_1695` is the higher-impact lever and should be the next spec (spec 010), with tapered WD pushed to spec 011. Agreed — moved things around, commit `17c8521`.

## Where everything stands going into next session

| Spec | Doc | Code | Runnable now? |
|---|---|---|---|
| 008 — reproduce #1736 | ✅ | — (uses #1736's) | Done (partial — gate number projected, not measured) |
| 009 — SpinQuant baseline + R_a | ✅ | ✅ (`spinquant_hotstart.py` + `test_rotation_invariance.py`) | **Yes** |
| 010 — port #1695 online rotation | ✅ | ❌ (~280 LOC port work remaining) | No |
| 011 — tapered Muon WD | ✅ | ❌ (~30–50 LOC training-loop patch) | No |

Modal plan for the remaining days:

1. Execution runs spec 009 (~$15, two modes, 30 min GPU). Closes spec 008's projected-not-measured gate number and gets first SpinQuant data point.
2. Based on 009's number, either (a) write spec 010 code if the internal_only lever landed and we want the bigger online-rotation lever on top, or (b) pivot.
3. Parallel track: write spec 011's WD-taper patch and run independently.
4. Stack winners, run a 3-seed final confirmation on the best composition.

## Lessons this session

- **Reading 3000+ lines of someone else's training code for 30 minutes saves hours of spec-writing on wrong assumptions.** The RMSNorm-is-gamma-free discovery alone shaved a whole subsection out of the integration notes. The LeakyReLU and `resid_mix` findings dictated which spec modes are actually implementable vs which are traps.
- **If a design requires "freeze X to mean" or "accept small perturbation," that's a research question, not an implementation detail.** I almost committed to it for `resid_mix` in the static-rotation plan. Glad I kept it as "deferred" in the spec.
- **Writing a CPU invariance test first catches bugs before burning pod time.** The first test run failed, pointed me at the missing QK-norm in the minimal forward, and let me fix it without any pod cost. Same-day iteration cycle instead of a next-day-after-the-pod-run one.
- **A partial result from execution is often good enough to proceed.** Spec 008's missed post-TTT number could have triggered a full $10 rerun; folding the gap into spec 009's baseline mode costs an extra $5 and gets four measured numbers on the same pod.
- **When someone on the frontier solves a problem the "wrong" way, sometimes their way is actually better.** Spent a day designing how to fold five per-channel multipliers around a residual rotation. #1695 solved the same problem by not rotating the residual at all. Their approach drops float invariance, but avoids all the fold math. Read the reference implementation early next time.

## Open questions for the next research session

1. What does spec 009 actually return? If `internal_only` lands −0.002 or better, the online-rotation port in spec 010 is clearly worth the ~280 LOC.
2. Does the unbanked-checkpoint load in `spinquant_hotstart.py` actually work on the pod? Script uses `strict=False` + `_rebank_state_dict` fallback — should be fine, but worth checking the first log.
3. What's the right lever to bet on if spec 009's `internal_only` is null? Either (a) push on spec 010 anyway in case the 4-rotation variant works where single-rotation doesn't, or (b) pivot to a non-quant lever (spec 011 tapered WD, SwiGLU, layerwise LR decay).
4. Do we have budget and wall-clock headroom for a 3-seed confirmation at the end, or should we commit to single-seed all the way through?
