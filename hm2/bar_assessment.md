# hm2 Bar Assessment

`hm2` is meant to clear the higher idea bar.

## Why It Clears

### 1. It breaks a real invariant

Older stages mostly assumed:

- one mechanism for the whole 600s

`hm2` breaks that directly:

- early bootstrap
- explicit late handoff
- distinct late receiver

### 2. It is code-mandatory

The stage is not just prose. The active mechanism surface is implemented in:

- `bootstrap_handoff`
- `phase_diagnostics`
- `countinit_bigram`
- `checkpoint_selection`
- `pre_quant_ttt`
- `late_qat_active`

### 3. It leaves evidence behind

`hm2` does not only rank by final score.

Each run records:

- early / mid / late loss deltas
- handoff events
- late validation evidence

That means a failed run can still teach the next stage something concrete.

### 4. It swings at a first-order process question

The question is not "does a helper patch help?"

It is:

- can a front-loaded mechanism be made useful overall by controlling when it stops owning the run?

That is a large enough process question to matter.

## Expected Lift

This is not a tokenizer/recurrence-sized jump.

My expected range is:

- likely: `0.003-0.010` BPB
- optimistic: `0.012-0.015`

Why not larger:

- the trunk is still fixed
- the search is about phase ownership, not base architecture

Why still worth running:

- it tests a reusable mechanism class
- it creates better future hypotheses even if the direct win is modest
