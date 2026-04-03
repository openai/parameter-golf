# Spectral Flood Walk LM — V0 Training Spec

## Summary

This document defines the `v0` training path that matches the retrieval-only eval spec in [2026-03-30-spectral-flood-walk-v0.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/plans/2026-03-30-spectral-flood-walk-v0.md).

The central principle is simple:

> `v0` should train with retrieval in the loop from the start, using the same memory interface that evaluation will use.

The point of `v0` is not to max out eval-time memory. It is to test whether a fixed, training-derived retrieval pool can improve prediction on unseen validation text when:

- the controller is trained to emit useful queries
- the seed codes are trained to expand into useful runtime states
- the reader is trained to interpret retrieved states correctly

## Scope

`v0` training includes:

- recurrent experts
- gating
- query projection
- compact seed-code generation
- learned expansion basis
- retrieval-conditioned read decoder
- full-vocab normalized decoding

`v0` training does not include:

- online eval-time append-only memory
- eval-time TTT
- approximate search
- giant HBM-filling runtime pools

Those belong to `v1` and later.

## Training Objective

The training question is:

> Can we learn a controller + retrieval interface such that a compact training-derived codebook expands into runtime states that are useful enough to help on validation?

That means training has to optimize three linked pieces, not just one:

```python
controller_state -> query
seed_target      -> compact_code -> expanded_runtime_state
controller + retrieved_runtime_states -> full_vocab_logits
```

If training only optimizes NLL without explicitly respecting this interface, the eval path will be brittle.

## Training Unit

Training should use the same chunk unit as eval:

- chunk size: `8` tokens

For each chunk `c_t`:

1. Consume the prefix before the chunk with the recurrent experts.
2. Produce a fused controller state `h_t`.
3. Project `h_t` into a quantized retrieval query `q_t`.
4. Retrieve neighbors from the training bank using exact search.
5. Expand retrieved codes into runtime states.
6. Use the read decoder to combine `h_t` with the retrieved states.
7. Decode to full-vocab logits and score the chunk.

Minimal shape:

```python
h_t = controller(prefix_tokens_before_chunk)         # [512]
q_t = quantize_int8(norm(W_q @ h_t))                 # [128]

nbrs = bank.topk(q_t, k=64)
r_t = expand_codes(nbrs.codes)                       # [64, runtime_dim]

ctx_t = read_decoder(h_t, r_t, nbrs.meta)
logits_t = chunk_decoder(ctx_t, chunk_prefix)
loss_nll = cross_entropy(logits_t, chunk_targets)
```

The quantized query path needs one explicit clarification: the retrieval query is quantized at train time with a fake-quant / straight-through estimator, not only at eval time.

In other words:

```python
q_float = norm(W_q @ h_t)
q_t = q_float + stopgrad(quantize_int8(q_float) - q_float)
```

This keeps the forward path aligned with eval while still allowing gradients to flow back into `W_q` and the controller.

The retrieval top-`k` selection itself remains non-differentiable, which is fine. Gradients flow through:

- `L_nll` via the reader and decoder
- `L_ret` via the contrastive query/key path

not through the discrete neighbor identity chosen by `topk`.

## What Gets Learned

`v0` learns all of these jointly:

- `8` recurrent experts
- expert gate
- expert projection heads
- query projection
- retrieval key projection
- seed-code projection
- expansion basis
- read decoder
- LM head

I also recommend a dedicated target encoder used only during training and export.

This target encoder is allowed to be richer than the online controller because it does not need to exist in the eval hot path in the same way.

High-level target-code path:

```python
r_tgt = target_encoder(chunk_tokens, next_chunk_tokens)
code_t = code_proj(r_tgt)                    # compact artifact code
state_t = expand(code_t)                     # runtime retrieval state
```

This is the key `v0` compromise:

- the artifact stores compact codes
- eval retrieves expanded runtime states
- training must learn to make those expanded states actually useful

## Expert Recurrence

The recurrent experts should use a token-conditioned but state-linear affine recurrence, not a GRU.

Default expert step:

```python
a_t = sigmoid(W_a @ x_t + b_a)                 # [128]
g_t = W_g @ x_t + b_g                          # [r]
u_t = W_u @ x_t + b_u                          # [128]

s_{t+1} = a_t * s_t + u_t + U @ (g_t * (V.T @ s_t))
```

Recommended default:

- expert hidden size: `128`
- low-rank update rank: `8`

Reasoning:

- affine in the previous state
- cheap enough for `v0`
- much closer to the intended Spectral Flood Walk flavor than a generic GRU
- more amenable to scan-style implementations than a heavily state-nonlinear recurrence

## Seed Code Semantics

The seed-code design is the most important architectural choice in `v0`.

There are three natural options:

1. Compress the controller hidden state directly.
2. Compress a reader-space state optimized for retrieval.
3. Compress a continuation-aware target state.

My default recommendation for `v0` is option `3`.

Why:

- the seed pool is derived from training data only
- the seed pool is fixed at eval time
- `v0` should give this fixed pool the best possible chance to transfer

So the training/export pipeline should let the seed encoder see:

- the current chunk
- optionally the next chunk or short continuation

That gives the code a stronger continuation signal than a plain prefix-only state.

This is slightly less aligned with `v1`, where online writes will be prefix-only. That is acceptable because `v0` is a transfer test for the fixed training-derived pool.

The query/key path should therefore be explicitly asymmetric:

- query comes from the prefix-only controller state
- key comes from the continuation-aware target encoder
- `W_q` and `W_k` are separate learned projections
- both sides get their own normalization and temperature scaling

We should not force them into a symmetric shared embedding space.

## Training Bank

The training bank should directly simulate the eval pool instead of using a surrogate.

Recommended `v0` training-bank properties:

- same key width as eval: `128` bytes `int8`
- same code format as eval
- same metadata layout as eval
- same `topk=64`
- same exact retrieval semantics as eval

Because the `v0` pool is only `64K-128K` entries, direct simulation is practical.

### Distributed Training Bank

The simplest distributed version is:

- full model is replicated data-parallel across `8` GPUs
- each rank keeps a replicated compressed training bank
- new candidate entries are all-gathered every `N` steps, for example every `16-64` steps
- retrieval stays local and exact on each rank

This keeps the retrieval path faithful while avoiding constant communication on every chunk.

## Losses

Primary loss is still NLL, but `v0` needs additional structure-aware losses so the seed-code path becomes useful.

Recommended starting objective:

```python
L = L_nll + 0.10 * L_ret + 0.05 * L_recon + 0.01 * L_gate
```

### `L_nll`

Standard chunk-level language-model loss with full-vocab normalization:

```python
probs = softmax(logits_t, dim=-1)
L_nll = -log(probs[target_tokens]).mean()
```

### `L_ret`

Asymmetric continuation retrieval loss:

```python
k_pos = key_proj(stopgrad(r_tgt))
L_ret = info_nce(q_t, k_pos, negative_keys)
```

This teaches the controller query to retrieve chunks with useful future continuations, not just superficially similar local text.

That is the right retrieval target for `v0`.

The working assumption is that InfoNCE is sufficient to bridge the train/eval asymmetry **if** we make the asymmetry architectural rather than pretending the spaces are identical. That means:

- separate `W_q` and `W_k`
- separate normalization
- same retrieval dimensionality for dot-product search
- same fake-quant path on both sides before bank insertion/search

### `L_recon`

Expansion-basis consistency loss:

```python
L_recon = mse(expand(code_t), stopgrad(r_tgt))
```

This exists because the seed entries are not raw states in the artifact. They are compact codes that are decoded at eval startup.

So the basis is not just a packing trick. It is part of the model and needs its own training signal.

Possible alternatives:

- reconstruct controller hidden state directly
- reconstruct only the read-decoder input space
- reconstruct whatever state best improves downstream NLL

`v0` should commit to one default here:

- reconstruct the **read-decoder input space**

That is the state actually consumed at retrieval time, so it is the most task-aligned reconstruction target for the compressed seed codes.

The exact latent used for that read-decoder input space can still evolve, but the default optimization target should not.

### `L_gate`

Small load-balancing regularizer:

```python
L_gate = load_balance_loss(gate_probs)
```

Its only job is to prevent total expert collapse early in training.

## Training Schedule

Use a simple three-phase schedule.

### Phase 1: warmup

Roughly `0-5%` of training.

Goals:

- stabilize controller and decoder
- prevent noisy retrieval from dominating too early
- let the query and code paths begin aligning

Recommended behavior:

- retrieval disabled or heavily dropped out
- target encoder and expansion basis already active

### Phase 2: retrieval-in-the-loop

Roughly `5-90%` of training.

Goals:

- learn query formation
- learn useful seed codes
- learn expansion basis
- learn read decoder

This is the main `v0` training regime.

### Phase 3: export stabilization

Roughly `90-100%` of training.

Goals:

- reduce retrieval dropout
- stabilize candidate seed-bank quality
- finalize export entries

This phase should make the exported artifact look as much like eval usage as possible.

## Retrieval Dropout

The controller should not become helpless without the bank.

So `v0` should use retrieval dropout during training:

- higher early in training
- lower late in training

This encourages:

- controller competence on its own
- graceful dependence on retrieval instead of total collapse into the bank

Exact schedule is open, but the need for some dropout is real.

## Candidate Entry Format

Each candidate training-bank entry should use the same logical format that export will use:

```python
entry = {
    "key": q_t_int8,
    "code": quantize(code_t),
    "expert_id": argmax(gate_t),
    "position": chunk_position_bucket,
}
```

This keeps train and eval tightly aligned.

## Export Path

Do not build the seed pool in a separate expensive post-pass.

Instead:

1. Maintain a fixed-size candidate bank during training.
2. Continuously score candidate usefulness.
3. Export the final bank directly into the artifact.

This keeps everything inside the same training run and avoids the exact kind of extra artifact-construction ambiguity that has invalidated other approaches.

## Candidate Retention

The candidate bank should not just be a FIFO buffer.

The candidate bank should use a cheap proxy for usefulness rather than a true counterfactual "with vs without entry" score.

Recommended retention score:

```python
score = novelty * read_mass * chunk_surprise
```

Where:

- `novelty`: discourages near-duplicate entries
- `read_mass`: cheap proxy for usefulness, for example average reader attention mass or attention-mass-weighted value norm
- `chunk_surprise`: prefers entries from harder regions

This avoids an expensive counterfactual ablation pass while still preferring entries the reader actually uses.

If we want a slightly richer proxy later, the next step would be:

- average attention mass times projected-value norm
- or a first-order attribution proxy from the read decoder

but `v0` should start simple.

## Throughput Estimate

The full `v0` pipeline is heavier than the current baseline, so we should at least have a rough budget before implementation.

Using the current baseline global batch size:

- `524,288` train tokens/step
- `8`-token chunks
- `65,536` chunks/step globally
- `8,192` chunks/step/GPU

For a `64K` bank with `128B` keys, each query scans:

```text
64K * 128B = 8MB
```

So each GPU reads about:

```text
8,192 * 8MB ≈ 64GB
```

of key memory per training step.

At H100 bandwidth, that is plausible. In practice it suggests:

- the query scan is a real cost but not absurd
- the target encoder is the first thing to simplify if the step time grows too much

Reasonable initial expectation:

- a step-time regime on the order of the better baseline runs or somewhat slower
- retrieval scan plus the extra target encoder pass are the first suspects if throughput disappoints

So the first throughput test should explicitly compare:

1. controller + reader only
2. controller + reader + retrieval
3. controller + reader + retrieval + target encoder

That will tell us quickly whether the target encoder is affordable in the 600-second budget.

## Default V0 Configuration

If we froze `v0` today, I would start with:

- `8` learned recurrent experts
- chunk size `8`
- exact retrieval over a `64K` training bank
- continuation-aware target encoder for seed-code generation
- `64B` seed codes
- learned expansion basis to `FP8` runtime states
- cross-attention read decoder
- `L_nll + L_ret + L_recon + small gate regularizer`
- short warmup, then retrieval active for most of training

This configuration directly matches the intended `v0` eval interface and maximizes the chance that the fixed seed pool is genuinely useful.

## Success Criteria

`v0` training is successful if it produces an artifact such that:

1. The eval path works end-to-end with exact retrieval and full normalized distributions.
2. The seed-code expansion basis produces runtime states that the reader can actually use.
3. Retrieval helps or at least plausibly helps compared with controller-only decoding.
4. The exported fixed pool transfers to validation text well enough to justify `v1`.

If retrieval adds only noise, then `v0` still did its job scientifically by showing that training-derived fixed memory alone is not sufficient.

## Open Questions

### 1. Should the target encoder see only the current chunk, or the current chunk plus a short continuation?

My recommendation is current chunk plus continuation, but this remains an explicit experiment.

### 2. What exactly should `L_recon` reconstruct?

Options:

- controller state
- retrieval state
- reader input state
- some task-aligned latent better than any of the above

The default is now reader input state. The remaining open question is whether that default is enough, not what the default should be.

### 3. How much retrieval dropout is enough?

Too little and the controller may collapse into bank dependence.

Too much and the retrieval interface never becomes useful.

### 4. Is a replicated training bank fast enough?

At `v0` scale it should be, but this should be verified with a real distributed smoke run rather than assumed forever.

### 5. How large does the candidate bank really need to be?

`64K` is a good default, but if transfer is weak we may need to test whether the issue is code quality or simple coverage.

### 6. How important is expert provenance?

We are storing `expert_id` in metadata, but `v0` still needs to prove that the reader actually benefits from it.

### 7. How much train/eval asymmetry remains after InfoNCE?

The current plan is:

- prefix-only controller queries
- continuation-aware target keys
- asymmetric projections with matched retrieval dimensionality

This is likely good enough for `v0`, but it should be treated as an explicit empirical question rather than an article of faith.

## Expected Follow-On

If `v0` works, the natural `v1` changes are:

- append-only online eval memory
- direct runtime raw-state writes
- larger runtime pool growth through validation
- eventually variable payload width and richer retention logic

At that point the architecture stops being a fixed training-derived retrieval test and starts becoming the full eval-time memory machine it is meant to be.
