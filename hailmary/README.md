# Hailmary

Moonshot search space for `parameter-golf`.

As of 2026-04-09, `hailmary` should be read as:

- the wide moonshot layer on the current executable strong local base
- with an explicit execution contract for what is active now
- and separate deferred frontier-rebase ideas

This folder is not a patch pack. It is a first-principles map of the full high-upside solution space under the real constraints:

- `600s` wallclock
- `16MB` artifact cap
- final score is deployed `val_bpb`, not just pre-quant loss

Files:

- [attack_surfaces.md]( nanoevolve/pgolf/parameter-golf/hailmary/attack_surfaces.md)
  - line-by-line attack surface map of the current merged [`train_gpt.py`]( nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - which surfaces can move `val_bpb` a little versus drastically
- [hypotheses.md]( nanoevolve/pgolf/parameter-golf/hailmary/hypotheses.md)
  - mechanism families, math intuition, bottleneck classes, negative knowledge, and moonshot hypotheses
- [rebase_hypotheses.md]( nanoevolve/pgolf/parameter-golf/hailmary/rebase_hypotheses.md)
  - deferred moonshot families after the April frontier shift to SP4096/SP8192 + recurrence + full GPTQ + SDClip + embedding GPTQ + MuonEq-R
- [execution_contract.md]( /Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/execution_contract.md)
  - active executable base, active primary/support packs, and what remains legacy or deferred
- [curation.md]( nanoevolve/pgolf/parameter-golf/hailmary/curation.md)
  - subagent-steelmanned narrowing of the moonshot set into distinct mechanism families
- [patches.py]( nanoevolve/pgolf/parameter-golf/hailmary/patches.py)
  - runtime patch library for hailmary experiments
- [run_configs.json]( nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json)
  - packs, slots, controls, and promotion rules
- [orchestrate_hailmary.py]( nanoevolve/pgolf/parameter-golf/hailmary/orchestrate_hailmary.py)
  - no-idle-GPU runner that materializes patched `train_gpt.py` copies per slot
- [run_strategy.py]( nanoevolve/pgolf/parameter-golf/hailmary/run_strategy.py)
  - small wrapper entrypoint for the hailmary runner

Working rule:

- `stage2_1` is the frontier-aligned exploitation lane
- `hailmary` is the wide high-upside exploration lane

## 2026-03-25 Principle Check

`hailmary` now has three explicit layers in code:

- a runnable lead pack: `phase_split`
- runnable support packs: export/eval/context/geometry/throughput
- deferred rebuild packs for the larger broken-invariant families

That distinction matters.

The older runnable packs are still useful, but they were biased toward mechanisms that were already easy to patch:

- export
- eval
- explicit priors
- geometry
- throughput

That is wider than `stage2_1`, but it still under-samples the stronger mechanism classes that likely matter next:

- phase-split training
- checkpoint/export selection
- parameter-family splits
- two-stage curriculum
- context-budget splits
- alternating objective cycles

So `hailmary` should now be read as:

- current runnable moonshot infrastructure
- a real lead process-split pack in [`run_configs.json`]( nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json)
- plus a rebuild target described in [rebuild.md]( nanoevolve/pgolf/parameter-golf/hailmary/rebuild.md)

Concretely:

- `ttt_family` is runnable now and is the new default pack
- `phase_split` is runnable as a lead process/control pack
- `checkpoint_selection` and `staged_curriculum` are now also runnable lead packs
- `alternating_objective` is the active support pack
- `G1/G2` remain active export anchors in the finalist wave
- `moonshot_core`, `moonshot_second_wave`, `moonshot_geometry`, and `moonshot_throughput` are retained as legacy packs, not as the active tournament thesis
- `parameter_family_split` and `context_stage` remain explicit rebuild lanes and are still marked `needs_patch`

Important caveat:

- `Full GPTQ` is now implemented as a runtime patch in [`patches.py`]( nanoevolve/pgolf/parameter-golf/hailmary/patches.py)
- the deferred export pack in [`run_configs.json`]( nanoevolve/pgolf/parameter-golf/hailmary/run_configs.json) is now runnable
- the port is intentionally adapted to the current non-banked root script, not copied wholesale from the banked record submissions

## 2026-04-09 Active Scope

The April PR analysis changed what counts as a real moonshot.

The missing mass is now clearly:

- `SP4096/SP8192`
- `full GPTQ + SDClip + GPTQ embeddings`
- `depth recurrence`
- `MuonEq-R`
- then `pre-quant TTT / dTTT / ETLB / parallel residuals`

So `hailmary` now separates:

- active executable moonshot packs on the current strong local 8192-tokenizer base
- deferred frontier-rebase moonshots that still need dedicated code

See [execution_contract.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/hailmary/execution_contract.md) for the active runnable claim and [rebase_hypotheses.md]( nanoevolve/pgolf/parameter-golf/hailmary/rebase_hypotheses.md) for the deferred frontier-rebase moonshot families.

The most important runnable change after that rebase is:

- `pre_quant_ttt` is now a real runtime patch
- `ttt_family` is now a primary pack that screens:
  - raw-state narrow TTT
  - EMA-state narrow TTT
  - longer-horizon TTT
  - broader dTTT-style tails
  - checkpoint-selected TTT
  - checkpoint-selected TTT plus late QAT
