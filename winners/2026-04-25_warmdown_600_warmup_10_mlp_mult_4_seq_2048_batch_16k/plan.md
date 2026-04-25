# Experiment 0012_seq_len_2048_on_winner

Parent: canonical (env-var sibling of `winners/2026-04-25_warmdown_600_warmup_10_mlp_mult_4`)

## Question
The canonical `TRAIN_SEQ_LEN=1024` doubles to 2048 with no change in
parameter count. Records contain seq2048 (2026-03-18 LongContextSeq2048
at 1.2060) and seq4096 (2026-03-19 TrainingOptSeq4096 at 1.2014); both
showed positive direction at H100 scale. Does the longer effective
context help on the 200-step MPS smoke?

The relevant constants:
- TRAIN_BATCH_TOKENS=8192 (fixed): with seq_len=1024, that's 8 sequences
  per batch; with seq_len=2048, 4 sequences per batch. Same total tokens
  per step, so per-step compute is similar (within attention-quadratic
  effects).
- VAL_BATCH_SIZE=8192 (fixed): same reasoning.
- VAL_BPB is averaged per byte, not per sequence — fair comparison.

The win mechanism: longer context lets the model see more preceding
tokens for next-token prediction, reducing entropy on tokens that
benefit from longer-range structure (paragraph-level, document-level).
Cross-sequence boundaries are not crossed (causal mask), so 2048 means
the last token of each sequence has up to 2047 preceding tokens
available instead of 1023.

Architectural note: attention is O(seq²) so per-step time will roughly
double in the attention layer. RoPE base is 10000 — handles 2048
seq_len without issue (well within typical extrapolation range).

## Hypothesis [LIKELY]
Δ vs 0008 winner ≈ +0.005 to +0.020. Records show positive direction;
on the smoke, the larger receptive field per token should provide
fewer-bits per byte on the entropy-rich tokens. Predicting modest
gain because the MLP/attention blocks are still small (d=512, 8 heads).

## Change
`env.sh`:
- `LR_WARMUP_STEPS=10`
- `WARMDOWN_ITERS=600`
- `MLP_MULT=4`
- `TRAIN_SEQ_LEN=2048`

No code edits. Artifact unchanged (no new params, RoPE buffers are
on-device only and not saved).

## Disconfirming
- Δ ≤ +0.005: longer context doesn't pay enough at sp1024/d=512/200 steps.
  Could be capacity-limited (model too small to benefit from extra context)
  or step-budget-limited (200 steps not enough to learn long-range
  patterns).
- Δ < 0: unstable interaction with the schedule — doubling effective
  per-token compute costs may slow convergence at fixed step budget.
- Significant step_avg slowdown (>2× baseline): attention quadratic is
  hitting a regime that destroys throughput. Might force a smaller
  TRAIN_BATCH_TOKENS to keep compute fair.
- NaN: longer sequences can cause attention scores to drift differently
  numerically; bf16 may not handle as cleanly.

## Notes from execution
**First attempt crashed** at the first batch fetch:
`RuntimeError: shape '[-1, 2048]' is invalid for input of size 1024` —
`local_tokens = train_batch_tokens / (world_size * grad_accum_steps)
= 8192 / 8 = 1024` doesn't fit `seq_len=2048`. `eval_val` has an explicit
assertion for the same constraint. Hardcoded `grad_accum_steps = 8 // world_size`
forces this when world_size=1 (MPS).

**Fix**: bump `TRAIN_BATCH_TOKENS` and `VAL_BATCH_SIZE` to 16384 (so each
micro-step gets exactly one 2048-length sequence). This **doubles the
optimizer-step batch size**, confounding the experiment: a positive Δ
will be ambiguous between "longer context helps" and "bigger batch helps."
Accept the confounding for first-pass directional info. Pure-context test
would require modifying the loader to accept smaller seq_len * grad_accum
combinations (out of scope for env-var-only).

Also doubling effective per-step tokens means the lr_mul schedule is now
"applied to bigger updates" — could affect stability.
