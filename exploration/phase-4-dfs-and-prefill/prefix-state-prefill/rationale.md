## Prefix-State Prefill Hypothesis

These two branches test the same idea on two different SSDGolf bases:

- `train_gpt_champion_prefill.py`: clean champion-style shared-block SSD base
- `train_gpt_batch262k_prefill.py`: current heavier Griffin/DFS-style branch

### Change

Each training microbatch is split into:

- a no-grad prefix of `PREFIX_STATE_PREFILL_TOKENS`
- a supervised suffix of `TRAIN_SEQ_LEN`

The prefix is run through `forward_stateful()` only to build the compressed SSD state.
That state is detached, then reused as the initial state for the suffix loss computation.

### Why this is worth testing

- It gives each supervised token longer horizontal context without backpropagating through the whole prefix.
- It uses the SSM strength directly: compressed state carry across long sequence chunks.
- It keeps the backward graph short, unlike training-time full long-context unrolling.

### Supporting choices

- The champion copy now has true per-iteration SSD state carry instead of a placeholder state path.
- The batch262k copy defaults to `EVAL_STRIDE=64` and disables training/eval vertical carry unless explicitly re-enabled.
- Both copies log:
  - `prefix_state_prefill_tokens`
  - `processed_batch_tokens`
  - `tok/s` based on all processed tokens
  - `loss_tok/s` based on supervised suffix tokens only

### Intended first runs

- Champion branch: test whether detached horizontal memory improves BPB without wrecking throughput.
- Batch262k branch: test whether horizontal-only memory helps more than the current vertical-carry-heavy training regime.

## Transformer D2S Hypothesis

The third branch in this directory, `train_gpt_transformer_d2s.py`, is a different bet:

- start from the stronger 10-layer leaderboard-style GPT, not the raw March 17 naive baseline
- keep the fast dense attention path intact
- late-swap only the heavy MLP matrices to a 2D Monarch factorization

### Why this is worth testing

- The transformer lineage already proved it can recover well under the 10-minute wallclock.
- Its MLPs are large enough that a late compression swap may be survivable without wrecking the attention stack.
- Keeping the Monarch factors as 2D parameters avoids the Conv-shaped Muon/export problems that hurt the SSD D2S branches.

### Intended first run

- Force the MLP-only D2S swap late enough that the dense model has already learned most circuits, but early enough to leave recovery budget.

## Transformer Griffinization Hypothesis

The dense SwiGLU transformer now looks stable enough to treat as a real architecture track.
The next change is to make the block more Griffin-like without changing the outer model:

- keep the same 10-layer transformer shell
- keep bigram hash, smear gate, and encoder/decoder skip-weight structure
- keep the SwiGLU MLP
- change only the block schedule so attention and MLP branches read the same mixed residual state and are merged in parallel

### Why this is worth testing

- It is a clean A/B against the current dense smoke: same parameters, same features, different branch scheduling.
- Parallel branch compute is closer to the recurrent Griffin intuition than a serial Transformer block.
- If it helps BPB or throughput in dense form, we have a better base before re-introducing any late compression.
