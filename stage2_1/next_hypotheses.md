# Stage 2_1 Next Hypotheses

This is the next-wave slate after the `stage2_1` postmortem.

The design rule is simple:

- stop searching for one more local helper on top of the current SOTA-aligned default
- search for mechanisms that break a false invariant in the current process
- only keep hypotheses with a plausible path to beating the deployed compressed score at `600s`

The current false invariants are:

- one training regime for the entire run
- one data order for the entire run
- one objective: fit first, compression later
- one checkpoint: the last one
- one update law for all tensor families
- one context budget for all phases of training

## Selection Bar

A next-wave hypothesis is only worth running if it satisfies all of:

- distinct causal story from `stage2_1`
- natural observable signal at `600s`
- plausible path to lower deployed `post_quant_bpb`, not just lower early train loss
- easy enough to port as a runtime patch without rewriting the whole program

## H401: Late Compression Alignment

- Mechanism:
  Train the current SOTA-aligned default normally for most of the run, then switch the final `15-25%` into a compression-aware regime.
- Why this is distinct:
  It attacks the train-to-deploy mismatch directly instead of hoping raw fit transfers through quantization.
- Expected signal:
  Neutral or slightly worse early raw loss, smaller quant gap, better final `post_quant_bpb`.
- Cheapest observable:
  Compare pre-quant vs post-quant gap on the same `600s` horizon.
- Likely failure mode:
  If the late alignment starts too early, it damages representation quality before the model has stabilized.
- Patch surfaces:
  - phase gate in the main training loop of [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - export/quantization block in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - late-only patch gates in [patches.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patches.py)
- Concrete patch idea:
  Late-only QAT or late-only quantization-noise injection.

## H402: Deployed Checkpoint Selection

- Mechanism:
  Stop assuming the last checkpoint is the best submission. Keep a small set of late checkpoints and choose by deployed roundtrip score.
- Why this is distinct:
  It breaks the "last step wins" assumption without changing the training law at all.
- Expected signal:
  Little change in raw training traces, real gain in final deployed score.
- Cheapest observable:
  Compare final `post_quant_bpb` for the last `K` late checkpoints on the same run.
- Likely failure mode:
  If the deployed optimum is flat across late checkpoints, the extra evaluation buys little.
- Patch surfaces:
  - late training loop checkpoint hooks in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - export/eval block in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - runner summary parsing in [orchestrate_stage2_1.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/orchestrate_stage2_1.py)
- Concrete patch idea:
  Best-of-last-`4` selection across raw and EMA checkpoints.

## H403: Two-Stage Curriculum

- Mechanism:
  Use one shard order early and a different shard order late.
- Why this is distinct:
  `T4` showed static curriculum is nearly relevant. The natural next move is to stage it, not to keep retuning one fixed order.
- Expected signal:
  Better early learning than control and better late robustness than static curriculum.
- Cheapest observable:
  `600s` deployed score versus control and versus static `size_desc`.
- Likely failure mode:
  If the two orders are too similar, the stage split is fake and the result stays within noise.
- Patch surfaces:
  - dataset file ordering in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - shard-order env handling in [patches.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patches.py)
- Concrete patch idea:
  `size_desc` before warmdown, then either natural order or hard-shard order during warmdown.

## H404: Three-Phase Optimizer Law

- Mechanism:
  Split warmup, bulk training, and late consolidation into separate optimizer regimes.
- Why this is distinct:
  It breaks the current "one optimizer schedule for the whole run" assumption.
- Expected signal:
  Slower or safer warmup, equal or better bulk fit, smaller late quant gap.
- Cheapest observable:
  Compare final `post_quant_bpb` and the raw/deployed gap, not only early train loss.
- Likely failure mode:
  If the phase boundaries are wrong, this becomes a noisy retune of the current schedule.
- Patch surfaces:
  - momentum and lr schedule logic in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - late-only env switches in [patches.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patches.py)
- Concrete patch idea:
  slower matrix warmup, unchanged bulk, then late decay increase plus perturbation off.

## H405: Parameter-Family Late Freeze

- Mechanism:
  Stop moving all parameter families the same way late in training.
- Why this is distinct:
  It breaks the "all trainable tensors should keep adapting until the end" assumption.
- Expected signal:
  Similar raw fit, lower deployed damage, potentially smaller artifact drift.
- Cheapest observable:
  Compare final deployed score and submission size versus the same parent run without freezing.
- Likely failure mode:
  Freeze the wrong family and the model underfits the final phase.
- Patch surfaces:
  - optimizer group construction in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - main loop phase gate in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
- Concrete patch idea:
  freeze embeddings and head late, keep only matrix trunk adapting.

## H406: Alternating Objective Microcycles

- Mechanism:
  Do not blend all objectives every step. Run normal steps most of the time and periodic consolidation steps.
- Why this is distinct:
  It is a process split, not a coefficient tweak.
- Expected signal:
  Better deployed score than always-on regularization, with less damage to representation quality.
- Cheapest observable:
  Same-parent comparison against an always-on late regularizer.
- Likely failure mode:
  If the consolidation step is too weak, it does nothing; if too strong, it becomes another destabilizing penalty.
- Patch surfaces:
  - micro-step logic in the main loop of [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - patch scheduling in [patches.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patches.py)
- Concrete patch idea:
  every `N` steps, run a late alignment step with quant-noise or consolidation-only updates.

## H407: Two-Stage Context Budget

- Mechanism:
  Use a cheaper training context early and restore full context late.
- Why this is distinct:
  It breaks the "the whole run must spend the same context budget" assumption.
- Expected signal:
  More updates early, then better long-context consolidation late.
- Cheapest observable:
  more steps in the same wallclock without losing late deployed score
- Likely failure mode:
  If early short context damages the learned representation too much, the late switch cannot recover it.
- Patch surfaces:
  - batch/sequence arguments and loader calls in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - schedule env parsing in [patches.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patches.py)
- Concrete patch idea:
  `1024` or `1536` early, `2048` during the final phase.

## H408: Dual-Track Snapshot Export

- Mechanism:
  Maintain more than one submission candidate inside the same run and export both.
- Why this is distinct:
  It breaks the "one weight state, one export path" assumption.
- Expected signal:
  better final deployed score from either raw or EMA or compression-aligned track, even when the train curve looks the same
- Cheapest observable:
  compare raw-final, EMA-final, and late-aligned-final on the same run
- Likely failure mode:
  if all tracks collapse to the same deployed optimum, the extra bookkeeping is wasted
- Patch surfaces:
  - EMA state logic in [patches.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/patches.py)
  - export block in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py)
  - runner summary code in [orchestrate_stage2_1.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage2_1/orchestrate_stage2_1.py)
- Concrete patch idea:
  export raw, EMA, and late-aligned variants and keep the best deployed artifact.

## What Not To Regenerate

Do not spend the next stage on:

- another small training-side helper in the `B` family
- throughput-only hypotheses as if they are score mechanisms
- static curriculum variants without a phase split
- early-only perturbation as a lead story
- mechanisms whose only signal is better `90s` or `180s` screens

## Recommended Search Order

1. `H402` deployed checkpoint selection
2. `H401` late compression alignment
3. `H403` two-stage curriculum
4. `H404` three-phase optimizer law
5. `H405` parameter-family late freeze
6. `H407` two-stage context budget
7. `H406` alternating objective microcycles
8. `H408` dual-track snapshot export

The key design rule is:

- do not ask "which helper beats the current default?"
- ask "which broken invariant explains why the current default still dominates at `600s`?"
