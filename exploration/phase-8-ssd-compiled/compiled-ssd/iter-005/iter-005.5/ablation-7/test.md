# Ablation 7: Remove RoPE on B/C matrices

## Hypothesis
RoPE on B/C matrices (complex SSM via rotary position embeddings) is causing the
regression from iter-003.5 (val_bpb=1.600) to iter-005.5 (val_bpb=1.98).

## What was removed
1. `_apply_rope()` helper function
2. `self.theta_proj` QATLinear layer (dim -> ngroups * d_state//2)
3. Forward pass: theta computation, cumsum of angles, cos/sin, and _apply_rope calls on B and C
4. `theta_proj.weight` from QAT_WEIGHT_SUFFIXES
5. `d_state % 2 == 0` assertion (only needed for RoPE's chunk(2) split)

## Rationale
- **Extra parameters**: theta_proj is a QATLinear(1024, 32) = 32K params. Small, but it
  also adds a full matmul in the forward pass per depth-recurrence iteration.
- **Extra compute per iteration**: theta_proj matmul, cumsum along sequence dim,
  cos/sin pair, and the rope rotation (chunk + cat) on both B and C. This runs
  N_ITERS=8 times per forward pass = 16 rope applications + 8 theta_proj matmuls +
  8 cumsums per training step.
- **Cumsum is a graph-break risk**: torch.cumsum along the sequence dimension is
  sequential and can prevent efficient compilation/fusion.
- **Questionable value at L=1024**: RoPE provides position-dependent phase rotation.
  At short sequence lengths the SSM's own decay (A_log with log-uniform timescale
  init spanning -4.5 to 0.5) already encodes rich positional/temporal structure.
  The B_bias/C_bias further shape the state interaction. Adding learned rotary
  frequencies on top may over-parameterize the positional encoding, leading to
  training instability or slower convergence.
- **Mamba-3 context**: The RoPE-on-B/C trick comes from Mamba-3 which targets much
  longer contexts (4K-8K+). At L=1024 with chunk_size=64 (16 chunks), the benefit
  is marginal while the compute cost is fixed.

## Expected outcome
- Fewer FLOPs per step -> higher tokens/sec -> more tokens seen in 10 min
- Simpler optimization landscape -> faster convergence
- If RoPE was actively hurting: direct BPB improvement from removing a bad inductive bias

## Metrics to compare
- val_bpb (primary)
- tokens_per_sec (should increase)
- train_loss curve shape (convergence speed)
