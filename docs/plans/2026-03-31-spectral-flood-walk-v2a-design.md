# Spectral Flood Walk LM — V2a Residual Table Design

## Summary

`V2a` is the first Spectral Flood Walk design that stores something directly predictive.

The core idea is:

> keep a strong small transformer as the base model, give it the best exact context we can afford at eval time, then add explicit multi-order residual corrections from a large GPU-resident context table.

This is the proposed replacement for both:

- `v1a` semantic-memory-as-MLP-replacement
- `v1b` episodic pooled-hidden retrieval

Those earlier versions failed for the same underlying reason:

> the stored object was not useful enough for next-token prediction.

`V2a` fixes that by storing **low-rank logit corrections** instead of hidden states.

## What V2a Is Trying To Answer

`V2a` asks one narrow question:

> does a large explicit residual table over recent token context improve `val_bpb` when added on top of a strong long-context transformer base?

More specifically:

1. How much does stronger exact eval context buy by itself?
2. Does a multi-order residual table improve on that stronger base?
3. Does online residual updating on validation text help more than a fixed residual table?

## Core Thesis

The best way to spend extra eval-time hardware is:

1. **first on exact context**
2. **then on explicit residual corrections**
3. **only later on heavier ideas like small ensembles**

This is guided by three observations already in the repo:

- sliding-window eval gives large gains with no architecture change
- depth recurrence repeatedly lost under fixed wallclock
- storing vague latent state and asking a reader to recover signal did not work for us

Relevant repo references:

- [2026-03-19_SlidingWindowEval/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md)
- [2026-03-21_DepthRecurrence_MixedPrecisionQuant/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-03-21_DepthRecurrence_MixedPrecisionQuant/README.md)
- [2026-03-31-spectral-flood-walk-v1b-design.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/docs/plans/2026-03-31-spectral-flood-walk-v1b-design.md)

## High-Level Architecture

`V2a` has two learned objects and one large runtime object.

### 1. Base Model

A strong small causal transformer compressed into the `16MB` artifact budget.

Its job is still standard language modeling:

```python
base_logits = model(prefix_tokens)
```

The difference is that `V2a` assumes the base model is evaluated with the strongest legal exact-context strategy we can afford:

- sliding-window eval
- high `EVAL_BATCH_SEQS`
- near-maximal usable prior context for scored tokens

This is the part that productively uses the KV cache and exact token history.

### Base Spine Requirement

`V2a` should **not** start from the underpowered `V1a` exploratory controller.

Before any residual-table experiment is taken seriously, the base model should be brought much closer to the competition's proven transformer stack. The relevant reference point in this repo is the family of `11`-layer `512d` models in the low-`1.12x` to low-`1.15x` range under sliding-window eval:

- [2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md)
- [2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)
- [2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md)

So the default `V2a` base spine should look like:

- `11` transformer layers
- `d_model = 512`
- `8` attention heads, `4` KV heads
- `MLP_MULT = 3`
- vocab `1024`
- tied embeddings / LM head
- sequence length `2048`
- sliding-window eval with stride `64`
- Muon + AdamW with decoupled `0.04` weight decay
- EMA
- late QAT / GPTQ-lite style post-quant path if needed for the final artifact

This does **not** mean every exploratory `1xH100` run has to reach `1.12x bpb`. It does mean the architecture and training stack should be as close as possible to a real competitive base, so residual tables are being judged on top of something honest instead of on top of a weak toy controller.

### How Many Base Spines?

`V2a` should test **two** base spines, not one and not many.

That is enough to tell us whether the residual-table mechanism is:

- broadly useful
- tightly coupled to one particular base flavor
- or simply not helping

Testing only one spine would make a negative result ambiguous.
Testing a large zoo of spines would turn `V2a` into a base-model sweep instead of a residual-memory test.

So the scope should be:

#### Spine A — Plain Strong Transformer

The closest clean implementation of the proven 11L competition stack:

- `11` layers
- `512d`
- `8` heads / `4` KV heads
- `MLP_MULT = 3`
- tied embeddings
- strong sliding-window eval

This is the control spine.

#### Spine B — Strong Transformer + One Proven Enhancement

The same backbone with one small, already-proven enhancement layered on top.

Good candidates:

- late-layer XSA
- the strongest already-proven eval/context configuration
- a clean EMA/GPTQ-lite export path if it changes the effective final model enough to matter

The point is not to invent a new family.
The point is to test whether residual tables still help on top of a slightly more competitive flavor of the same strong base.

### Explicit Non-Goal

`V2a` should **not** test:

- recurrence
- MoE variants
- large architecture family sweeps
- more than two spines

until the residual-table mechanism itself shows a positive signal.

### 2. Shared Residual Basis

A small learned matrix:

```python
U: [V, r]
```

where:

- `V` is vocab size
- `r` is a small correction rank, for example `8`, `16`, or `32`

`U` is stored in the artifact and shared across all residual nodes.

### 3. Large Residual Tables

At eval time, we maintain multi-order hashed residual tables:

```python
T_1, T_2, T_3, T_4, ...
```

Each table maps a recent token context hash to a small coefficient vector:

```python
c_h in R^r
```

The final correction is:

```python
delta_logits = U @ c_total
```

and the full prediction remains clean and legal:

```python
logits = base_logits + delta_logits
probs = softmax(logits)
```

That means every prediction is still a complete normalized distribution over the whole vocabulary.

## Conceptual Trie, Physical Hashed Tables

The right mental model is still a trie over token sequences:

```text
"the" -> correction
"the cat" -> correction
"the cat sat" -> correction
```

But the right implementation is **not** a literal trie.

On GPU, `V2a` should use:

- rolling hashes over recent tokens
- flat per-order residual tables
- simple table probes, not pointer chasing

So the physical implementation is:

```python
c_total = 0
for n in orders:
    h_n = rolling_hash(last_n_tokens)
    c_total = c_total + alpha[n] * table[n].lookup(h_n)
logits = base_logits + U @ c_total
```

This gives trie semantics with GPU-friendly storage and lookup.

## Why This Is Different From What Failed

### v0 / v1b failure mode

`v0` and `v1b` stored hidden states and hoped retrieval would somehow recover predictive value from them.

That failed because the stored object was too indirect.

### V2a value object

`V2a` stores a direct answer to:

> for this context pattern, how should the logits move?

That is a much better memory object.

The stored value is no longer:

- pooled hidden state
- summary hidden state
- product-key code that must be interpreted

It is:

```python
c_h  # low-rank logit correction coefficients
```

which becomes logits through the shared basis `U`.

## Online Update Rule

The online update is the strongest part of `V2a`.

After scoring a token, we already have:

```python
probs = softmax(logits)
target
```

The gradient of cross-entropy with respect to logits is:

```python
g_logits = probs - one_hot(target)
```

Since:

```python
delta_logits = U @ c
```

the natural projected residual update is:

```python
g_c = U.T @ g_logits
```

So each traversed table node can be updated after scoring with:

```python
c <- (1 - beta) * c - eta * g_c
```

or a slightly more stable EMA-style version:

```python
grad_ema[h] <- rho * grad_ema[h] + (1 - rho) * g_c
c[h] <- c[h] - eta * grad_ema[h]
```

This is much cleaner than training a reader to extract signal from a latent bank.

## Orders And Backoff

`V2a` should start with a small number of orders:

- unigram
- bigram
- trigram
- 4-gram

The correction can combine them additively:

```python
c_total = c_1 + c_2 + c_3 + c_4
```

or with learned scalar gates:

```python
c_total = a_1 * c_1 + a_2 * c_2 + a_3 * c_3 + a_4 * c_4
```

The additive version is the right first implementation.

## Artifact Budget

`V2a` should keep the artifact simple.

Initial target split:

- `6-10MB`: base transformer
- `0.05-0.5MB`: shared residual basis `U`
- `1-4MB`: compressed seed residual tables
- `0.2MB`: code / metadata

The attractive property here is that `U` is tiny.

For example, with:

- `V = 1024`
- `r = 16`
- `int8` basis values

we get:

```text
1024 * 16 = 16,384 bytes ≈ 16KB
```

Even with scales and metadata, it is tiny.

So most of the artifact budget should still go to the base transformer.

In other words:

- `V2a` is **not** "small transformer plus giant table"
- `V2a` is "competitive compressed transformer plus residual table"

The residual path only deserves credit for whatever it adds beyond a serious base.

## Runtime Memory

Unlike the failed hidden-state memory paths, the runtime table here is cheap per node.

If a node stores:

- `r = 16` int8 coefficients = `16B`
- hash / stats / EMA / metadata = roughly `16-48B`

then one node is on the order of:

```text
32B to 64B
```

That means `V2a` is likely to be **semantically strong** before it is **VRAM-filling**.

That is okay.

The base model's exact context and KV cache are still the main sink for productive VRAM.
The residual tables are there to hold useful corrective structure, not just to consume memory for its own sake.

## Seed Table Construction

There are two versions to test.

### Version A — Training-derived static seeds

During training:

1. run the base transformer
2. compute residual target:

```python
g_c = U.T @ (probs - one_hot(target))
```

3. aggregate these projected gradients into hashed context tables
4. export the compact table state

This gives a fixed artifact-derived residual memory.

### Version B — Eval-time online growth

At eval:

1. start from a small seed table
2. after each scored token or chunk:
   - compute `g_c`
   - update the relevant residual nodes
3. future predictions benefit from those updates

This is the more interesting version, but `V2a` should test the static and online variants separately.

## Coprocessor Status

`V2a` does **not** need a learned coprocessor.

That is deliberate.

The fixed projected-gradient update is already a principled adaptive rule. We should not add a learned memory refiner until we know that:

1. the stored object is correct
2. the fixed update already helps

So:

- `V2a`: no learned coprocessor
- later `V2b`: maybe add one if the residual tables work

## Evaluation Modes

One `V2a` run should compare at least three modes on the same trained base model.

### `context`

Base transformer only, with the best exact eval context strategy.

This is the control.

### `context+static_residual`

Base transformer plus fixed residual tables loaded from artifact.

This answers:

> does explicit context-conditioned residual memory help at all?

### `context+online_residual`

Base transformer plus residual tables that update after scoring.

This answers:

> does same-stream adaptation help once the memory object is directly predictive?

## What Success Looks Like

Good outcomes:

- `context+static_residual` beats `context`
- `context+online_residual` beats both

Especially strong:

- the gain survives longer runs, not only short smokes
- the gain increases with stronger exact context rather than disappearing

Bad outcomes:

- residual tables are flat or negative
- online updates overfit instantly
- the base transformer with better context already absorbs all the gain

## Experiment Ladder

### 0. Base Spine Upgrade

Before residual tables, we should first port `V2a` onto a leaderboard-like transformer spine.

This stage is successful if:

- the model architecture matches the proven 11L/512d/MLP3x family closely enough to be credible
- the eval protocol matches strong sliding-window practice
- the quantization/export path is realistic enough that the final artifact target is not fantasy

This stage is not about proving a new idea. It is about refusing to test the new idea on top of a weak base.

### 1. Stronger Context Baseline

Measure the best context-only baseline first.

Questions:

- how much eval-time gain is available from context alone?
- how much eval wallclock is left afterward?

Run this for both Spine A and Spine B.

### 2. Static Residual Tables, Small Rank

Try:

- orders `1..3`
- `r = 8`
- simple additive backoff

Questions:

- does direct residual memory help at all?
- is the artifact math still comfortable?

Run this for both Spine A and Spine B.

### 3. Static Residual Tables, Larger Rank / More Orders

Try:

- `r = 16` or `32`
- orders `1..4`

Questions:

- are gains rank-limited?
- do higher orders help or mostly sparsify the table?

Only do this for the better of Spine A / Spine B from Step 2.

### 4. Online Residual Updates

Turn on post-score updates with:

- fixed step size
- optional EMA smoothing

Questions:

- does the table adapt beneficially to the validation stream?
- is the gain stable or does it drift into overfit?

Again, only do this on the better-performing spine unless the Step 2 results are too close to separate.

### 5. Only Then Consider Small Ensemble

If `V2a` is already positive, then a later `V2b` can test:

- `2` small base models
- average logits
- residual correction on top

This is explicitly later, not part of `V2a`.

## Two-Spine Experiment Matrix

The intended initial matrix is:

| Spine | Mode | Purpose |
|---|---|---|
| A | context | clean strong-base control |
| A | context + static residual | test whether residuals help on the plain spine |
| A | context + online residual | test same-stream adaptation on the plain spine |
| B | context | slightly stronger flavored control |
| B | context + static residual | test whether residuals survive on the enhanced spine |
| B | context + online residual | test same-stream adaptation on the enhanced spine |

This is enough to answer the most important `V2a` question without exploding the search space.

## Phase 2 Queue

If Phase 1 shows a positive residual-table signal, `V2a` should immediately move to one queued orthogonal comparison instead of reopening architecture brainstorming during pod time.

The default queued Phase 2 axis should be:

### Phase 2A — Tokenizer / Lexical Granularity Shift

Keep the strong transformer family broadly the same, but change the lexical granularity.

Default candidate:

- `1024 BPE` strong spine from Phase 1
- versus an `8192 BPE` or similarly larger-tokenizer spine if artifact math still fits

Why this is the first queued orthogonal comparison:

- residual tables are fundamentally keyed by recent token context
- tokenization changes table sparsity, collision structure, and n-gram usefulness directly
- this is a meaningfully different base without turning `V2a` into a completely different architecture family

Questions Phase 2A answers:

- do residual tables work better when tokens are more word-like?
- does a larger tokenizer reduce table depth requirements?
- does the base model get so much better that the residual path becomes unnecessary?

### Phase 2B — Architecture Flavor Shift (Backup Only)

Only if Phase 2A is inconclusive or blocked by artifact math, queue one more orthogonal but still leading-flavored base.

Good candidates:

- a U-Net / skip-heavy transformer variant already represented in the repo's strong line
- a late-layer-enhanced transformer that changes representation flow but remains in the same general 11L competitive family

This is deliberately the backup, not the first orthogonal test.

### Explicit Phase 2 Non-Goals

Even in Phase 2, still avoid:

- recurrence
- MoE
- binary / ternary / radically different quantization families
- more than one orthogonal axis at a time

The point of queueing Phase 2 is to remove uncertainty, not to pre-approve a sprawling branch explosion.

## Current Status After The First 1×H100 Round

The corrected exploratory result is now known.

The first pod run surfaced an evaluation bug in the lightweight `V2a` runner:

- overlapping sliding windows were double-counting scored tokens

After fixing that, the best under-cap exploratory configuration is:

- `spine_variant = xsa`
- `model_dim = 448`
- `num_layers = 9`
- `residual_table_size = 49152`

Three corrected `1xH100` seeds on that config produced:

- mean `context val_bpb = 2.43654`
- mean `static residual val_bpb = 2.39399`
- mean `online residual val_bpb = 2.33665`
- mean `delta_static = -0.04255`
- mean `delta_online = -0.09989`
- std(`delta_online`) = `0.00359`
- max artifact = `15,874,794` bytes

So the current evidence is:

1. The residual-memory mechanism survives bug correction.
2. Online residual updating is consistently better than static residual tables.
3. The mechanism can be kept under the real decimal `16,000,000` byte cap.
4. The exploratory runner's absolute BPB is still far from competitive.

That means `V2a` is now a **validated mechanism**, not a final submission stack.

## V2a.1 Graft Plan

The right next step is **not** to keep tuning this lightweight runner in place.

The right next step is to graft the validated residual mechanism onto a much stronger competition-grade base model.

### Host Base

Use the strongest already-proven transformer training stack we can borrow from this repo's leading line, ideally one of:

- [2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md)
- [2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)

The important thing is to inherit:

- the stronger optimizer stack
- the stronger training schedule
- the stronger quantization/export path
- the stronger eval-context setup

### Minimal Integration Surface

The graft should change as little as possible about the host runner.

Add only:

1. `ResidualBasis`
   A shared low-rank correction basis `U`.

2. `ResidualRouter`
   Multi-order hashed context IDs from the already-scored token suffix.

3. `ResidualTables`
   Static and online tables stored outside the base model weights.

4. One post-logit hook:

```python
logits = base_logits + delta_logits
```

where:

```python
delta_logits = U @ c_total
```

Everything else should remain host-runner-native.

### Evaluation Contract

The corrected `V2a` lesson is now non-negotiable:

- when using overlapping eval windows, only score newly introduced tokens
- only apply `online_residual` updates to those scored positions

This should be treated as part of the mechanism, not an implementation detail.

### First Graft Matrix

The first graft matrix should stay small:

1. host base only
2. host base + static residual
3. host base + online residual

Only after that is positive should we re-open:

- tokenizer granularity
- larger rank
- additional orders
- any ensemble idea

### Why This Is The Right Next Move

The exploratory runner already answered the architectural question we needed answered:

> storing directly predictive low-rank logit corrections and updating them online can help.

What it has **not** answered is the submission question:

> does that mechanism still help once the base model is good enough to matter on the real leaderboard?

`V2a.1` should answer that question directly.

## Open Questions

1. How closely can we match the strong 11L/512d leaderboard stack before introducing any residual-table complexity?
2. Should the base transformer remain at vocab `1024`, or does a larger tokenizer improve the residual-table tradeoff?
3. What is the best rank `r` before diminishing returns dominate?
4. Should the node update be pure projected SGD or EMA-smoothed projected SGD?
5. Is token-level updating better than chunk-level updating for stability?
6. Do we need separate bases `U_n` for different n-gram orders, or is one shared basis enough?
7. How much of the total gain comes from stronger exact context alone versus the residual table?

## Bottom Line

`V2a` is the first Spectral Flood Walk design that uses memory in the most direct possible way:

> store corrections, not latent states.

That does not guarantee it wins. But it does mean the next experiments are asking the right question.
