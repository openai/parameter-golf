# hm2 Hypotheses

## Thesis

The stage-level idea is not "bigram is good" or "TTT is good".

It is:

- `bootstrap mechanisms` and `late receivers` are different jobs
- the same mechanism should not necessarily stay active for all 600 seconds
- the right unit of search is the handoff policy

## Families

### H601: Static Bootstrap Prior

- Mechanism: keep the count-initialized bigram prior active for the whole run.
- Why: establishes whether the observed early gain is real and front-loaded.
- Expected signature: strong `early_loss_delta`, weaker `late_loss_delta`, `front_loaded` pattern.
- Slots: `B2`, `R1`

### H602: Fixed Fade

- Mechanism: linearly reduce prior scale over a fixed late window.
- Why: preserve early optimization help, then give the trunk the late basin.
- Expected impact: `0.002-0.008` BPB if late flattening is real.
- Slots: `B3`, `B6`, `B7`

### H603: Fixed Freeze

- Mechanism: stop updating the prior after a boundary, but keep it present.
- Why: tests whether late harm comes from continued drift instead of continued presence.
- Expected impact: `0.001-0.006` if drift is the problem.
- Slots: `B4`

### H604: Plateau-Triggered Handoff

- Mechanism: trigger fade or freeze only when recent training improvement stalls.
- Why: a fixed handoff time is probably wrong across runs.
- Expected impact: `0.003-0.010` if the right handoff is state-dependent.
- Slots: `B5`, `R2`, `R3`, `R4`, `R7`

### H605: Snapshot Receiver

- Mechanism: after handoff, pick the best deployed late snapshot instead of trusting the last state.
- Why: front-loaded mechanisms may create better intermediate deployed states than final states.
- Expected impact: `0.002-0.008`.
- Slots: `B6`, `R2`, `R5`, `R7`

### H606: TTT Receiver

- Mechanism: after handoff, let a narrow pre-quant TTT receiver finish the late phase.
- Why: the trunk may need a small adaptation receiver once the early scaffold has done its job.
- Expected impact: `0.003-0.012`.
- Slots: `B7`, `R3`, `R6`, `R7`

### H607: Raw-vs-Deployed Snapshot Receiver

- Mechanism: after handoff, compare choosing the best raw late checkpoint against choosing the best deployed late checkpoint.
- Why: front-loaded mechanisms may shift where the best raw and best deployed states occur.
- Expected impact: `0.001-0.005`.
- Slots: `R4`

## Success Criteria

An `hm2` candidate is real only if all three hold:

- it beats the base control on `post_quant_bpb`
- its diagnostics show the intended early/late shape
- its receiver or handoff event explains the gain better than "it got lucky"
