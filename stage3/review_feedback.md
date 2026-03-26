# Stage 3 Review Feedback

This note is a review only. It does not propose in-place edits to the stage3 code.

## Addendum (2026-03-24)

After the latest upstream scan and fork sync, I would tighten the stage3 review in three ways:

- merged `#549` is now best read as a strong bridge record, not the current frontier template
- the biggest missing modern lanes are now `XSA-all` and curriculum / shard ordering
- the exciting TTT-specific work is `K-LoRA + Min-NLL`, but that is a separate eval-time protocol branch and should not be used to justify the no-TTT shortlist

Open vs merged status matters here:

- `#549`: merged
- `#616`: merged, README-only
- `#614`: closed, not merged
- `#615`: open
- `#638`: open
- `#639`: open
- `#650`: open

So stage3 should not talk as though those later PRs are merged truths. They are high-signal evidence, not settled baseline facts.

## Addendum (2026-03-23)

I revisited stage3 after:

- rebasing `parameter-golf` to newer upstream changes
- updating `stage2_1` around the newer no-TTT frontier
- scanning newer upstream PRs directly with `gh`

The important shift is that stage3's older March-20 hypothesis set is now partly outdated. The newer upstream no-TTT frontier is centered much more strongly on:

- Full GPTQ
- LeakyReLU(0.5)^2
- XSA
- EMA
- Partial RoPE + LN Scale
- then VRL / VE128 / stronger exact n-gram priors as second-wave extensions

That changes how I would triage the stage3 hypothesis set.

### Immediate New Finding

Stage3 is now weaker than I previously reported on one practical point: the dry run is no longer clean against the current merged root script.

Commands run:

```bash
python3 -m py_compile pgolf/parameter-golf/stage3/orchestrate_stage3.py pgolf/parameter-golf/stage3/patches.py
python3 pgolf/parameter-golf/stage3/orchestrate_stage3.py --phase sanity --dry-run
```

Result:

- `py_compile` still passes.
- `--phase sanity --dry-run` now fails because the `label_smoothing` patch no longer matches the current root `train_gpt.py`.

Concretely:

- `patch_label_smoothing()` in [`stage3/patches.py`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/patches.py) searches for an older `F.cross_entropy(...)` call shape that no longer exists.
- So `H6` is not just weak as a hypothesis. It is currently broken as an executable slot.

That means stage3 is now both:

- experimentally behind the frontier
- partially drifted out of sync with the current root script

### Hypothesis Triage

This triage compares:

- the stage3 hypothesis set as represented by `experiment_plan.md`, `run_configs.json`, and `attack_surfaces.md`
- the newer stage2_1 hypothesis set
- the newer upstream PR evidence

#### Abandon As First-Wave Stage3 Hypotheses

These are not "forbidden forever." They should be removed from the first-wave stage3 screen if the goal is to chase the current frontier efficiently.

- `H6` label smoothing
  - Reason: weak prior, broken patch, and the newer upstream frontier does not support it as a top explanation of the remaining gap.

- `H7` compile autotune as a score hypothesis
  - Reason: this is an infra/throughput probe, not a primary `val_bpb` mechanism. It belongs in a systems check, not in the same shortlist as frontier architecture/export moves.

- NorMuon as a lead hypothesis
  - Reason: stage3 still treats `NorMuon + MuonWD` as a top Pack-1 story, but the newer upstream evidence makes MuonWD the live helper and NorMuon the demoted part.

- solo SmearGate as a first-wave lane
  - Reason: newer upstream no-TTT winners do not depend heavily on it. It has become a mid-tier helper, not a frontier anchor.

- OrthoInit as a first-wave lane
  - Reason: still plausible as a helper, but it no longer looks like the best next explanation of the current gap.

#### Good / Still Defensible

These ideas remain sound and should still inform later work.

- the stage3 runtime patching architecture
- repeated controls and noise-floor estimation
- clean separation of eval/export/training lanes as a planning principle
- MuonWD as a helper mechanism
- Bigram memory as a category, though not necessarily plain hashed BigramHash
- the attack-surfaces document's later frontier section, which correctly elevates GPTQ, LeakyReLU^2, EMA, XSA, Partial RoPE, LN Scale, VRL, and VE128

#### Good But Should Be Extended

These are the hypotheses I would extend rather than discard.

- `BigramHash` -> extend toward `CountInitBigram`
  - Reason: the newer upstream evidence makes exact or count-initialized n-gram priors more interesting than plain learned hash memory alone.

- `XSA4` -> extend to `XSA-all` when compression/export headroom allows
  - Reason: `#587` suggests the question is now about how much XSA the artifact budget can support, not whether XSA works at all.

- `QAT` -> extend into "verify-active late QAT / export-aligned QAT"
  - Reason: the mechanism is still relevant, but stage3's framing is too old. After the newer upstream evidence, QAT is not the lead story; it is an alignment/helper story that must be verified to actually be active.

- `BigramHash / SmearGate` architecture lane -> extend with `VE128` and `VRL`
  - Reason: the newer no-TTT frontier is getting more leverage from value-path and context-path additions than from old bigram-only priors.

### Comparison: Stage3 vs Stage2_1

My current view is:

- `stage2_1` is directionally closer to the true frontier
- stage3 still has useful exploratory pieces, but its Pack-1 shortlist is too anchored to the older community story

What stage2_1 gets more right now:

- strong frontier-aligned base as the control
- Full GPTQ as the main export hypothesis
- LeakyReLU^2, EMA, XSA, Partial RoPE, LN Scale as first-wave no-TTT mechanisms
- VRL and VE128 as second-wave structural lanes
- explicit separation between real frontier targets and temporary runnable proxies

What stage3 still contributes:

- a clean review of older attack surfaces
- a useful decomposition of causal lanes
- runtime patch composability

What I would now consider outdated in stage3's hypothesis framing:

- the idea that the best Pack-1 screen should be centered on `NorMuon / SmearGate / QAT / OrthoInit / LabelSmoothing / compile`
- the assumption that "community repeated techniques" are still the best frontier explanation

### Recommendation

If stage3 were to be refreshed conceptually, I would:

- keep the orchestration and patching model
- keep the lane decomposition
- drop `label_smoothing` and `compile_autotune` from the main hypothesis shortlist
- demote `NorMuon`, solo `SmearGate`, and solo `OrthoInit`
- replace the first-wave shortlist with:
  - Full GPTQ
  - LeakyReLU(0.5)^2
  - EMA
  - XSA
  - Partial RoPE + LN Scale
- then use a second-wave extension set:
  - VRL
  - VE128
  - CountInitBigram / stronger exact n-gram priors
  - verified-active late QAT

## What I Checked

Commands run:

```bash
python3 -m py_compile pgolf/parameter-golf/stage3/orchestrate_stage3.py pgolf/parameter-golf/stage3/patches.py
python3 pgolf/parameter-golf/stage3/orchestrate_stage3.py --phase sanity --dry-run
```

Additional validation:

- Each runtime patch in `stage3/patches.py` was applied against the live root `pgolf/parameter-golf/train_gpt.py` and `py_compile` passed for all of them.
- The label-smoothing patch is the current exception at orchestration time: it compiles as a standalone patch file, but its target string no longer matches the merged root script, so the end-to-end dry run now fails before job materialization completes.

## High-Confidence Findings

### 1. Attribution is better than before, but not fully clean

`run_configs.json` describes stage3 as a "mechanism-level screen", but the actual candidates are not consistently mechanism-level.

- `H1` still bundles `NorMuon + MuonWD` in one slot.
- `H2` and `H3` are now correctly split into `SmearGate` and `BigramHash`.
- `H7` is now a pure `compile_autotune` probe, which is cleaner than the older `compile + warmup` bundle.

So the config is improved relative to the older stage3 draft, but it is still not fully mechanism-pure because `H1` remains a package rather than a single mechanism.

My view:

- `NorMuon + MuonWD` is defensible as one package if the explicit claim is "community-paired optimizer discipline package", not "mechanism-level optimizer screen".
- Splitting `SmearGate` and `BigramHash` was the right correction and should be kept if stage3 continues to exist.
- `compile_autotune` is now properly isolated, but it still should not be treated as a frontier BPB hypothesis.

### 2. The plan and the actual implementation disagree in important places

There are several design mismatches between `experiment_plan.md`, `attack_surfaces.md`, `run_configs.json`, and `patches.py`.

- `experiment_plan.md` still reflects the older slot map, where `H2` is `SmearGate + BigramHash`, `H3` is `QAT`, `H4` is `OrthoInit + muP`, `H5` is `label smoothing`, and `H6` is `FA3 + compile autotune + warmup=5`.
- The actual `run_configs.json` has moved to `H2=SmearGate`, `H3=BigramHash`, `H4=QAT`, `H5=OrthoInit`, `H6=label_smoothing`, and `H7=compile_autotune`.
- `attack_surfaces.md` has advanced much further than either of those and now correctly highlights frontier mechanisms such as GPTQ, LeakyReLU^2, EMA, XSA, Partial RoPE, LN Scale, VRL, and VE128.

This is more than naming drift. It affects interpretation. If `H4` wins, that is evidence for orthogonal init, not for `OrthoInit + muP`. If `H6` wins, that is evidence for the compile/warmup bundle, not for the broader systems story written in the plan.

Right now the deepest document in stage3 is `attack_surfaces.md`, the executable truth is `run_configs.json`, and `experiment_plan.md` is the stalest artifact.

### 3. The main design problem is now stale planning, not just slot bundling

`experiment_plan.md` says not to mix causal lanes in the same screen pack and to keep candidates mechanism-level where possible. That is the right principle. The actual config partially violates it:

- `H1` still mixes two optimizer mechanisms.
- The stage description still markets the screen as mechanism-level even though it contains one explicit package test.
- More importantly, the plan document is lagging the config by an entire slot remap.

So the problem is no longer simply "bad bundling in the config." It is that the three stage3 artifacts disagree about what stage3 actually is.

### 4. `--phase all` does not actually mean the full declared pipeline

The CLI exposes:

- `sanity`
- `screen`
- `composite`
- `decision`
- `final_single`
- `champion_8x`
- `all`

But `all` only runs:

- `sanity`
- `screen`
- `final_single`
- optional `champion_8x`

It does not run `composite`, and it does not run `decision`, even though both phases are declared in the config and exposed in the CLI.

This is risky because an operator can reasonably assume `all` means "the whole stage3 pipeline". It does not. Right now it means "the short screen plus a direct promotion path".

### 5. Promotion logic is too narrow for the stated goals

`write_phase_summary()` and `recommend_promotions()` are built around `delta_post_quant_bpb`. That is acceptable for a simple ranking loop, but it does not match the stage3 writeup, which explicitly separates:

- training dynamics
- export quality
- eval policy
- systems throughput

Examples:

- QAT is supposed to tolerate worse pre-quant quality in exchange for smaller quant gap.
- SYS is supposed to win on ms/step and total steps, not necessarily on short-horizon post-quant BPB.
- INIT is supposed to show up early in the learning curve, not necessarily only in the final 180s point estimate.

If all promotions collapse to post-quant BPB ranking, then stage3 is not actually honoring its own lane-specific kill rules.

## Experimental Design Feedback

### What is strong

- Runtime patching per slot is the right execution model.
- Control duplication with `R0A` and `R0B` is the right instinct.
- The causal-lane decomposition in `experiment_plan.md` is good. It is the strongest part of the stage3 package.
- The attack-surface work is broad enough to avoid local idea collapse.

### What is weak

- Attribution and packaging are muddled.
- The written mechanism stories are ahead of what is actually implemented.
- The runner summary logic is simpler than the stated decision logic.
- The phase graph is ambiguous because declared phases and automated phases diverge.

### What I would trust from stage3

- The patching architecture.
- The idea of separate training, export, and eval lanes.
- The need for repeated controls in noisy short screens.
- The usefulness of stage3 as a source of candidate ideas.

### What I would not trust from stage3 without revision

- Fine-grained conclusions about which mechanism won.
- Any claim that `H4` tested muP.
- Any claim that `H6` tested FA3.
- Any operator assumption that `--phase all` runs the full designed process.

## Machine Utilization Check

Short answer: no, not at every phase.

What the current scheduler actually does:

- `run_parallel_phase()` assigns one visible GPU per slot and launches one process group per slot. If you run 8 slots on 8 GPUs, you get 1 GPU per slot.
- If you run fewer slots than GPUs in `run_parallel_phase()`, the unused GPUs stay unused. There is no automatic repartitioning there.
- `run_partitioned_phase()` does repartition all visible GPUs across the promoted slots. So with 2 tasks on 8 GPUs, each task gets 4 GPUs.
- `run_serial_phase()` is used for the single champion path and can hand the full requested GPU set to one slot.

So the answer to the concrete question is:

- 8 tasks on 8 GPUs -> yes, 1 GPU per task.
- 2 tasks on 8 GPUs -> yes only if those 2 tasks are being run through the partitioned phase.
- 2 tasks on a parallel screen phase -> no, they will still use 1 GPU each and 6 GPUs will sit idle.

That means stage3 currently maximizes machine usage in the later partitioned phases, but not in the early parallel phases.

## Bottom Line

Stage3 still has a sound runtime-patching architecture, but it is no longer cleanly dry-runnable against the merged root. The main problems are now:

- frontier drift
- doc/config divergence
- at least one real orchestration break (`label_smoothing`)

If this is used as inspiration for later work, the pieces to keep are:

- per-run patched working copies
- explicit slot metadata
- lane-aware experiment planning

The pieces to treat carefully are:

- bundled candidates presented as mechanism-level evidence
- stale plan/code mismatches around `muP`, `FA3`, and the slot map itself
- summary logic that ranks on one metric even when the plan says to use lane-specific criteria
