# Stage 3 Review Feedback

This note is a review only. It does not propose in-place edits to the stage3 code.

## What I Checked

Commands run:

```bash
python3 -m py_compile pgolf/parameter-golf/stage3/orchestrate_stage3.py pgolf/parameter-golf/stage3/patches.py
python3 pgolf/parameter-golf/stage3/orchestrate_stage3.py --phase sanity --dry-run
```

Additional validation:

- Each runtime patch in `stage3/patches.py` was applied against the live root `pgolf/parameter-golf/train_gpt.py` and `py_compile` passed for all of them.
- The dry run produced slot-local patched `train_gpt.py` working copies under `stage3/runs/...`, which is the right high-level pattern and matches the tested Enigma style better than pre-editing the shared root.

## High-Confidence Findings

### 1. The claimed unit of attribution is too coarse

`run_configs.json` describes stage3 as a "mechanism-level screen", but the actual candidates are not consistently mechanism-level.

- `H1` bundles `NorMuon + MuonWD` in one slot.
- `H2` bundles `SmearGate + BigramHash` in one slot.
- `H6` bundles `compile_autotune + warmup reduction`.

That means the screen is not identifying which mechanism works. It is identifying whether each bundled package works. Those are different conclusions. This matters because Pack 2 is supposed to stack survivors, and you only want to stack survivors when you understand what survived.

My view:

- `NorMuon + MuonWD` is defensible as one package if the explicit claim is "community-paired optimizer discipline package", not "mechanism-level optimizer screen".
- `SmearGate + BigramHash` is less defensible as one package because they are not the same mechanism. One is a recurrence-like token blending prior; the other is a parameterized hashed bigram memory.
- `compile_autotune + warmup=5` is the weakest bundle because it mixes throughput and stability effects.

### 2. The plan and the actual implementation disagree in important places

There are several design mismatches between `experiment_plan.md`, `attack_surfaces.md`, `run_configs.json`, and `patches.py`.

- The docs repeatedly describe `H4` as `OrthoInit + muP`, but the actual slot and patch are just `orthoinit`.
- The docs repeatedly describe `H6` as `FA3 + compile autotune + warmup=5`, but the actual slot only changes `COMPILE_MODE` and `WARMUP_STEPS`; there is no FA3 patch in `patches.py` and no FA3 env gate in `run_configs.json`.
- The attack surface doc treats `muP` and `FA3` as real attack surfaces, but the runner is not screening them.

This is more than naming drift. It affects interpretation. If `H4` wins, that is evidence for orthogonal init, not for `OrthoInit + muP`. If `H6` wins, that is evidence for the compile/warmup bundle, not for the broader systems story written in the plan.

### 3. The config violates one of the plan's best rules

`experiment_plan.md` says not to mix causal lanes in the same screen pack and to keep candidates mechanism-level where possible. That is the right principle. The actual config partially violates it:

- `H6` mixes a throughput knob with a schedule/stability knob.
- The stage description markets the screen as mechanism-level while using packaged candidates.

So the written plan is stronger than the implemented experiment design. I would trust the plan more than the current slot design.

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

Stage3 is runnable enough to dry-run and its runtime patching approach is sound. The main problems are experimental-design and interpretation problems, not immediate syntax or orchestration breakage.

If this is used as inspiration for later work, the pieces to keep are:

- per-run patched working copies
- explicit slot metadata
- lane-aware experiment planning

The pieces to treat carefully are:

- bundled candidates presented as mechanism-level evidence
- doc/code mismatches around `muP` and `FA3`
- summary logic that ranks on one metric even when the plan says to use lane-specific criteria
