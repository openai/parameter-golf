# Stage 3.2 Rebase Hypotheses

Date: 2026-04-09

These are the **next** controller hypotheses for `stage3_2` after the April frontier shift.

They assume the base stack is no longer the old SP1024 trunk.
The intended base is:

- `SP4096` or `SP8192`
- `11L`
- `full GPTQ`
- `SDClip`
- `GPTQ embeddings`
- `depth recurrence`
- `MuonEq-R`

This stage is now a **support/control layer** on top of that base.

## Rule

Every surviving `stage3_2` controller must change a first-order process decision on the new stack:

- when recurrence activates or strengthens
- when deploy pressure begins
- when checkpoint/export state should be captured or selected
- when TTT-prep behavior should begin
- when expensive mechanisms should be throttled or intensified

## C201 Recurrence Activation Controller

- Mechanism:
  - control `loop_count` or loop activation boundary by training state
  - start with lower virtual depth, then move to higher recurrence when the trunk is ready
- Why:
  - [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L33) says earlier loop activation helps, but recurrence is also the mechanism most likely to overpay if activated too crudely
- Dominant lane:
  - training quality + throughput
- Expected impact:
  - medium to large
  - `0.003 - 0.010 BPB`
- Earliest signal:
  - `180s`
- Failure mode:
  - trigger arrives too early and recurrence steals too much step budget

## C202 Recurrence-Quant Coupling Controller

- Mechanism:
  - coordinate late deploy controls with recurrent blocks only
  - stronger QAT / clip / checkpoint capture on looped layers than on non-looped layers
- Why:
  - recurrence only became frontier-viable once quantization damage was neutralized
- Dominant lane:
  - export / quant damage
- Expected impact:
  - medium
  - `0.003 - 0.009 BPB`
- Earliest signal:
  - `600s`
- Failure mode:
  - extra complexity adds no gain beyond a globally strong GPTQ pipeline

## C203 Best Deployed State Under Pre-Quant TTT

- Mechanism:
  - adapt snapshot cadence and checkpoint selection specifically for the state that will feed pre-quant TTT
  - do not assume the best TTT starting point is the last EMA state
- Why:
  - the new frontier uses pre-quant TTT as a major downstream stage, so checkpoint choice now matters even more
- Dominant lane:
  - deploy / state selection
- Expected impact:
  - medium
  - `0.003 - 0.010 BPB`
- Earliest signal:
  - only long run
- Failure mode:
  - TTT washes away the differences between late checkpoints

## C204 TTT Freeze-Depth Controller

- Mechanism:
  - adapt how many early blocks are frozen during pre-quant TTT based on late-run state
  - example: freeze first `2`, `6`, or `9` blocks depending on stability and quant gap
- Why:
  - the latest PRs show multiple successful freeze-depth choices, which implies this is not a fixed constant
- Dominant lane:
  - eval / TTT
- Expected impact:
  - medium
  - `0.002 - 0.008 BPB`
- Earliest signal:
  - only long run with TTT
- Failure mode:
  - too much policy freedom; no stable best choice across seeds

## C205 ETLB Activation Controller

- Mechanism:
  - enable and scale eval-time logit bias only when checkpoint/export state is stable enough
  - treat ETLB as a controlled late eval mechanism, not an always-on trick
- Why:
  - [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L83) and [pr_analysis.md](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/stage3/pr_analysis.md#L94) show ETLB is real but small
- Dominant lane:
  - eval policy
- Expected impact:
  - small
  - `0.001 - 0.003 BPB`
- Earliest signal:
  - same-checkpoint eval bakeoff
- Failure mode:
  - ETLB stays purely additive and does not justify controller complexity

## C206 Systems-Aware Recurrence Controller

- Mechanism:
  - intensify recurrence or parallel-residual compute only when measured `step_avg_ms` remains inside a safe band
  - otherwise fall back to cheaper virtual-depth settings
- Why:
  - the new frontier makes throughput support meaningful only when attached to expensive recurrence and kernel choices
- Dominant lane:
  - throughput / more steps
- Expected impact:
  - small to medium
  - `0.001 - 0.005 BPB`
- Earliest signal:
  - `90s`
- Failure mode:
  - controller mostly turns mechanisms off and learns nothing useful

## C207 Recurrence-to-TTT Transition Controller

- Mechanism:
  - learn the handoff point between training-time recurrence exploitation and pre-quant TTT preparation
  - alter EMA, snapshot cadence, and deploy pressure near the handoff
- Why:
  - the frontier now has two strong stages: recurrence during train, TTT before export
- Dominant lane:
  - process split
- Expected impact:
  - medium
  - `0.003 - 0.009 BPB`
- Earliest signal:
  - `600s`
- Failure mode:
  - handoff logic is too noisy; a static split is already good enough

## C208 Composite Era-6 Controller

- Mechanism:
  - combine:
    - recurrence activation control
    - deploy-state selection
    - TTT freeze-depth selection
    - recurrence-aware late quant control
- Why:
  - the new frontier is explicitly multi-stage; the controller should optimize the stage transitions, not just one local knob
- Dominant lane:
  - process / deploy / TTT
- Expected impact:
  - large
  - `0.005 - 0.015 BPB`
- Earliest signal:
  - only long run
- Failure mode:
  - interaction complexity exceeds what the bounded controller can optimize cleanly

## Keep / Demote Summary

- `keep`:
  - deployed-state selection
  - alternating late objective pressure
  - family-specific late control
- `demote`:
  - generic curriculum-by-state
  - context-budget control on the old trunk
- `add`:
  - recurrence activation
  - recurrence-quant coupling
  - TTT freeze-depth control
  - recurrence-to-TTT transition control
