# Non-record Submission: Three-Layer Governance Probe + Inference-time Cheap Gate v2 (Signs-of-Life)

**Track:** `track_non_record_16mb` — submitted as a pre-compute-grant **signs-of-life** probe. Results below come from local probes on a minimal `real_cael` checkpoint, not from a full 8×H100 record run. This PR documents a three-layer governance supervision chain (Cael → Monday → Seryn) that has already produced a working **inference-time cheap gate** on a small probe set, and the real-machine experiment plan that the grant would fund.

**Author:** SyntagmaNull
**Date:** 2026-04-19
**Status:** signs-of-life / pre-grant — not competing for the leaderboard at this time

---

## TL;DR

In the tiny-model / short-budget regime of this challenge, lightweight *content-level* interventions (persona corpora, governance-style tags, late-onset injections, neutral→mixed→question schedules) did **not** reliably move target relational-quality signals. What *did* first show up as a readable mechanism signal was **governance sensitivity** — not *what* the model says, but *when* something should be intervened on, and *whether* the intervention actually helped.

We built a three-layer supervision chain around that observation:

- **Cael** — the base model generating continuations on world-like samples.
- **Monday** — a policy layer deciding whether and how to intervene on Cael's trajectory.
- **Seryn** — an audit layer checking whether the intervention actually pulled the trajectory back, overshot, or was misapplied.

On a small probe the chain separates two usefully-distinct regimes:

- **Persistent-error zone** (`error_persist_steps ≥ 2`, or clear drift) — redirection pays off.
- **Pre-error / light-anomaly zone** (hesitation, `top_gap_band` low) — intervening *hurts*; mark-only is the correct policy.

The submitted artifact is an **inference-time cheap gate** (`cheap_gate_v2_candidate`) that encodes exactly that split: redirect on persistent-error, mark-only on pre-error, no model re-training required. On our probe set this lowers trajectory mismatch without introducing new overgovernance artifacts. Training-side auxiliary supervision (zone-classification, decision-shape) only yielded modest baseline stabilization; the inference-time gate is currently the stronger signal.

---

## Why this is worth a non-record PR

This is not a leaderboard-BPB submission. The claim is narrower and structural:

1. The three-layer supervision chain **is not a concept** — it is wired up, it ingests real model logits from a minimal Cael checkpoint, and it emits checkable per-step records (see `probe_results/usefulness_probe_v0/rows.jsonl`).
2. The chain produces a **readable decision geometry** before the model produces readable content. The `error_persist_steps` / `top_gap_band` / `uncertainty_band` / `intervention_effect` fields already separate two action regimes cleanly.
3. That decision geometry is **immediately convertible to an inference-time policy** (`cheap_gate_v2_candidate`) that earns local signs-of-life utility — without changing the model or the tokenizer.

This matches exactly the non-record submission spirit: *"Breakthrough ideas are rarely immediately state-of-the-art."* We are filing the PR now, with signs-of-life evidence, so that the compute grant has a concrete pre-registered object to verify.

---

## The three-layer chain, concretely

Implemented in `code/build_three_layer_prototype_data.py` (the builder), with two backends:

- `mock_v0` — rule-driven, used for pipeline shakeout.
- `real_cael` — the chain consumes continuations and logits from a minimal `real_cael` checkpoint, exported via `code/export_minimal_real_cael_checkpoint.py`.

Per world-sample the chain emits:

1. **Cael continuation** — the generated trajectory.
2. **Cael trace** — `error_mode`, `error_persist_steps`, `uncertainty_band`, `top_gap_band`, `local_site`.
3. **Monday `target_move`** — the policy's decision: redirect / mark / no-op.
4. **Post-Monday effect** — the measured `intervention_effect` after the policy acts.
5. **Seryn `target_audit`** — whether the intervention was helpful, unnecessary, or overshot.

This is the structure on which every downstream probe sits.

---

## What has been tried and judged

This is the important context — it narrows what the grant run should actually test.

### Content-level injection lines — judged insufficient in this regime

- Persona corpora (Monday-style, Seryn-style).
- Governance / reflective / late-onset injection at varying schedules.
- neutral → mixed → question short schedules.
- Lightweight interruption / branch-content designs.

None of these produced stable relational-quality movement in the tiny-model / short-budget regime. We are **not** asking for grant compute to retry these.

### Three-layer governance line — first readable fork

Governance-layer signals (when/whether to intervene, whether intervention helped) separated into readable families before content-level signals did. This is the pivot point that led to the current submission.

### Usefulness probe v0 (`probe_results/usefulness_probe_v0/`)

Small-sample probe — `rows.jsonl` carries per-step records, `inspection.md` summarizes the manual read. Key observation: applying heavy intervention on `pre_error` hesitation samples (e.g. `world_006`-style) **hurts**, while applying it on `persistent_error` samples helps. This is the geometry `cheap_gate_v2_candidate` encodes.

### Auxiliary supervision probe v0 (`probe_results/aux_supervision_probe_v0/`)

Two training-side targets tried, both with a minimal aux head on the base model:

- `zone_classification_v0` — predict zone label at chunk level.
- `decision_shape_v1` — predict `target_move` shape directly.

See `aux_supervision_probe_results.csv`, `decision_shape_coverage_audit.md`, `*_usefulness_rows.jsonl`, `*_train_curve.jsonl`.

Headline: small baseline-stability improvement, but the inference-time cheap gate is still the larger signal. The current bottleneck reads as `class support / supervision geometry` rather than step count — which is exactly what the coverage audit shows.

---

## The cheap gate (what the grant run verifies)

`cheap_gate_v2_candidate` is a rule expressed on the chain's trace fields, not a learned head:

- If `error_mode == "persistent"` and `error_persist_steps ≥ 2` → **redirect**.
- Else if `uncertainty_band == "high"` or `top_gap_band == "narrow"` → **mark-only** (no action).
- Else → **no-op**.

On the small probe this (a) preserves the win in the persistent-error zone, (b) removes the overgovernance cost in the pre-error zone, (c) keeps audit distributions healthy, (d) introduces no new visible side effect.

The real-machine experiment the grant would fund is a focused three-test sequence:

1. **Inference-time verification.** Re-run the cheap gate on a more realistic checkpoint / larger rollout — does the persistent-error redirect win survive, and does the pre-error mark-only policy remain the safer one?
2. **Coverage expansion.** Does the two-family decision geometry (persistent-error vs pre-error) stay stable under larger sample coverage, or do new important families appear?
3. **Training-side fair retest.** Only after 1 and 2 stand: one clean retest of `decision_shape_v1` with class support geometry corrected first, to judge whether training-side governance gain can be pulled up to match the inference-time gate.

No new mechanism families are being introduced. The grant is asking for real resources to **verify** an already-isolated small-regime signal, not to fish for a new one.

---

## Files in this submission

- `README.md` — this file.
- `submission.json` — metadata.
- `code/build_three_layer_prototype_data.py` — the three-layer builder (`mock_v0` + `real_cael` backends).
- `code/export_minimal_real_cael_checkpoint.py` — exporter for the minimal Cael checkpoint the chain consumes.
- `code/run_usefulness_probe_v0.py` — inference-time cheap-gate evaluator.
- `code/run_aux_supervision_probe_v0.py` — training-side `zone_classification_v0` / `decision_shape_v1` probe harness.
- `code/run_three_layer_timing_probe*.py` / `code/run_three_layer_conflict_probe.py` / `code/analyze_three_layer_*.py` — supporting probes / analyzers referenced in the writeup.
- `code/three_layer_nested_training_prototype_spec.md` — full spec of the three-layer prototype.
- `code/three_layer_prototype_data_builder_spec.md` — full spec of the builder contract.
- `probe_results/usefulness_probe_v0/` — inference-time probe outputs (`rows.jsonl`, `inspection.md`, `manifest.json`).
- `probe_results/aux_supervision_probe_v0/` — training-side probe outputs (`aux_supervision_probe_results.csv`, `decision_shape_coverage_audit.md`, `inspection.md`, per-run `*_train_curve.jsonl` and `*_usefulness_rows.jsonl`, `train_examples.jsonl`). **Model checkpoints (`*.pt`, ~2 MB each) are excluded** — rerun `run_aux_supervision_probe_v0.py` to regenerate.

## How to reproduce

```bash
# 1. Export a minimal Cael checkpoint (or drop in your own)
python3 code/export_minimal_real_cael_checkpoint.py

# 2. Rebuild the three-layer prototype data
python3 code/build_three_layer_prototype_data.py --backend real_cael

# 3. Run the inference-time cheap-gate probe
python3 code/run_usefulness_probe_v0.py

# 4. (Optional) Run the training-side aux supervision probe
python3 code/run_aux_supervision_probe_v0.py
```

## Lineage and credits

- No overlap with any leaderboard record — non-record.
- Companion submission: **Online Data Scheduler — `unique_only + lookahead=16` (Signs-of-Life)** (separate PR). The two PRs share a project-level thesis: in the 16 MB / 10-min regime, the first readable governance signal is *structural* — on the data-feeding side in one PR, on the intervention-policy side in the other — not content-level.

<\!-- ================================================================== -->
<\!-- GPT 填充区（线 B three-layer GPT）：你可以往下面补充 -->
<\!-- - usefulness probe 的精确 mismatch 差数字 -->
<\!-- - aux probe v0 vs v1 vs baseline 的 curve 对比 -->
<\!-- - cheap_gate_v2_candidate 的决策表格 / rule table -->
<\!-- - decision_shape_coverage_audit 的具体发现（哪些 class support 不够） -->
<\!-- - 真机阶段三步测试各自的 accept/reject 判据 -->
<\!-- - 不要加 records/... 之外的新 top-level 文件 -->
<\!-- ================================================================== -->


---

## Update (2026-04-20)

Since the previous Line B README snapshot, the main progress is not "more numbers" but a clearer boundary on what currently works, what partially internalizes, and what still must remain external.

### 1. Training-side governance is no longer just a weak probe

After fixing class support for the decision-shape target, `decision_shape_v2` became the first auxiliary target that improved both:

- baseline continuation stability
- and the downstream `cheap_gate_v2` usefulness panel

This indicates that training-side governance gain is real in the current tiny regime, but it required the target geometry to actually open up in data.

### 2. Training-side aux and inference-time gate are not redundant

Interaction probes now support the following read:

- training-side aux improves the model body / baseline trajectory
- inference-time `cheap_gate_v2` still performs the last-step correction
- the two are partially overlapping but clearly complementary

So the current picture is not "the model learned the gate away," but rather: some governance boundary can be internalized, while final correction still benefits from an external scaffold.

### 3. Final-step correction still must remain external, and cannot yet be too soft

A weakened gate collapsed consistently:

- error-zone samples that were previously `stable_redirect` fell back to `still_drifting`
- degradation reappeared immediately

This currently sets a hard boundary: the model can learn to fall in less often, but final rescue still requires an external scaffold, and that scaffold cannot yet be weakened too much.

### 4. Shared objective is now attached as a design principle, not a new loss

The repo now includes `shared_objective_v0`, defined conservatively as minimizing:

- useless drift
- mistimed governance
- no-gain intervention

This is intentionally not promoted to a training target yet. At the current stage it functions as a design summary that aligns the three layers:

- **Cael:** drift / off-track continuation
- **Monday:** mistimed or excessive intervention
- **Seryn:** auditing whether these bad states are actually reduced

### Current boundary

The current best read of Line B is:

- some governance boundaries can be learned into the model
- inference-time governance remains useful and non-redundant
- final-step correction still has to live in an external scaffold
- the shared objective is now clearer, but is not yet mature enough to be optimized directly

This is a better-scoped understanding of the problem than the repo had two weeks ago, and is the main reason this branch is now worth pausing rather than continuing to dig deeper under the current tiny-budget regime.
