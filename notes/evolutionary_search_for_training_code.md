# Evolutionary Search for Training Code

This note captures the three-level framing from the ChatGPT draft, then adds the repo-specific corrections that matter for `pg_enigma` and `search_harness.py`.

## Core idea

There are at least three levels of evolutionary search here.

### 1. Parameter search

This is the shallowest level.

Typical mutations:

- learning rate
- betas
- weight decay
- warmup
- batch size
- grad clip
- loss weights
- data mixture ratios

This is useful, but it is mostly black-box HPO. It helps once a real mechanism is already in the neighborhood, but it rarely discovers a new training law by itself.

### 2. Policy search over training decisions

This is where the search stops being "just a config search" and becomes search over a conditional training procedure.

Typical mutations:

- when to anneal
- when to switch optimizers
- when to upweight hard data
- when to increase sequence length
- when to freeze or unfreeze components
- when to change regularization
- when to change exploration vs exploitation in data sampling

At this level, the individual is not just a parameter vector. It is a controller.

Examples:

- if validation plateaus and grad noise drops, lower LR less aggressively
- if rare-token loss lags, shift the data mixture
- if instability rises, raise EMA or clip more

This is closer to evolving a training controller than tuning a config.

### 3. Algorithm search

This is the deepest level.

Typical mutations:

- optimizer update rule
- moment estimation logic
- per-parameter scaling
- trust-region logic
- layerwise adaptation
- curriculum algorithm
- sample selection mechanism
- loss decomposition logic

At this level, the search is not only tuning training. It is mutating parts of the learning algorithm itself.

That is the point where evolution becomes search over methods.

## Why this framing is good

The draft is directionally right for a few reasons.

### Search over deltas from a trusted baseline

This is correct.

Do not evolve arbitrary code from scratch. Start from a strong baseline and represent each candidate as a patch over that baseline.

Good units of variation are things like:

- modify a scheduler
- insert a condition
- replace an optimizer block
- alter a sampling policy
- add or remove one mechanism

That keeps the search grounded and makes attribution legible.

### Hierarchical genome

This is also correct.

Not all mutations should live at the same scale. Scalar knobs, controller rules, algorithmic blocks, and systems tricks should not all mutate at the same rate.

Most generations should do local edits.
A smaller fraction should do structural jumps.

That is how you avoid random thrashing.

### Multi-fidelity evaluation

This is one of the biggest practical truths.

Do not fully evaluate everything. Use a ladder:

1. compile / unit-test / smoke-train
2. tiny run for stability
3. short run for learning speed
4. medium run for a generalization proxy
5. full run only for elites

The real selection target is not "best final model every time." It is "best evidence of promise per unit compute."

### Select on trajectory shape, not only endpoint

Also right.

For training code, a single scalar is usually too dumb. The search often cares about:

- speed of improvement
- stability
- variance across seeds
- memory / throughput
- downstream metrics
- robustness to mild data perturbation

That makes staged or multi-objective selection more appropriate than a single-winner regime.

### Keep an archive, not only a champion

This is critical in deceptive search spaces.

The archive should preserve lineages that are:

- fastest learners
- most stable
- cheapest
- best final quality
- weird but promising

One global winner collapses diversity too early.

### Semantic mutation beats syntactic mutation

This is one of the strongest points in the note.

Do not mutate raw tokens blindly.
Mutate units of meaning, such as:

- replace cosine decay with plateau-aware decay
- add layerwise LR decay
- switch AdamW to a hybrid update after step K
- reweight examples by uncertainty
- introduce a curriculum switch at a loss threshold
- add norm-based clipping for selected modules

Blind edits mostly create broken code or fake novelty.

### Learn the mutation policy

This is where LLM guidance becomes useful.

The model should not be treated as the optimizer itself. It should be used to improve the proposal distribution:

- which operator classes are worth trying now
- which kinds of changes helped in similar regimes
- which classes of mutations keep failing

That is much stronger than asking for random patches.

## What I think is missing

The note is strong, but it is missing a level above all three search levels.

### Missing Level 0: measurement-contract search

Before parameter search, policy search, or algorithm search, there is a more basic question:

**What exactly are we selecting on in this cycle?**

Examples:

- prequant `val_bpb`
- final exact post-quant `val_bpb`
- size / latency constrained quality
- progress per FLOP
- robustness under small dataset shifts

If that measurement contract is wrong, all downstream search levels get distorted.

This is one of the main ways search goes bad:

- a good idea gets killed because the proxy was weak
- a bad idea wins because the cheap metric was easy to hack
- the system optimizes a stage-local score that does not survive deployment

So for this repo, there are really at least four levels:

1. measurement-contract search
2. parameter search
3. policy/controller search
4. algorithm search

And in practice there is also a separate base-contract layer, where tokenizer, vocab, architecture, or global optimizer stack can change.

### Base search should be separate from family search

Tokenizer search, vocab search, architecture search, and global optimizer-stack search are not the same thing as policy mutations inside a frozen base.

Those should be treated as separate frontier moves.

If you mix base moves and within-base family mutations in one tournament, ranking becomes hard to trust.

### Family before tuning

Another missing point is that local tuning should come only after a mechanism family has survived.

Bad pattern:

- invent a weak family
- immediately sweep thresholds inside it
- spend compute polishing a premise that never proved itself

Better pattern:

1. test the family against controls
2. prove the mechanism is real
3. only then run local mutation inside the family

This is one of the biggest corrections to make when the search feels busy but unproductive.

### Deep algorithm search needs executable proxies

The note is right to call for algorithm search, but in practice this level is easy to romanticize and hard to execute.

The safe version is:

- define one claimed mechanism
- compile several concrete realizations of that mechanism
- test those realizations against a strong control

Do not ask for one giant optimizer rewrite and assume it means anything.

## What this means for this repo

For `train_gpt.py` and the Enigma flow, the right split is roughly:

### Parameter search

Use this after a family or base has already earned deeper attention.

This belongs mostly in downstream tournament execution, not in the main creative search loop.

### Policy/controller search

This should be the main frontier for `pg_enigma`.

It is where the search can still produce consequential changes while remaining attributable and testable.

Examples:

- state-triggered schedule changes
- optimizer handoff policies
- phase-conditioned compute allocation
- curriculum switches
- selector logic

### Algorithm search

This should be rare, sparse, and strongly verified.

Algorithmic mutations should probably consume a small share of proposal budget until the harness is proving that it can distinguish real wins from noisy artifacts.

### Base search

This should remain a first-class separate lane:

- tokenizer / vocab
- architecture
- optimizer stack
- dataset contract

These are not "cheating." They are legitimate base changes. But they need clean controls.

### Deployment / selector search

This also deserves its own lane whenever the metric that matters at the end does not match the metric that is easiest to measure during training.

For example:

- checkpoint/export choice
- quantization family
- post-train selection logic

## Practical budget split

The draft suggests something like:

- 80% small mutations
- 15% medium policy changes
- 5% deep algorithmic changes

I think that is directionally right, but I would interpret it carefully.

For this repo, the real split should be closer to:

- most budget on proving or refining consequential families
- some budget on local mutation inside already-proven families
- a small budget on deep structural jumps
- occasional base-frontier rounds when all within-base work stalls

So the key distinction is not only mutation size. It is also **what level the search should currently be operating at**.

If the search is flat, the right answer is often not "try more local edits." It is "move up one level."

## Where people go wrong

The draft list is good. I would restate the main failure modes like this.

### 1. Searching too wide too early

If everything is mutable, almost all compute is wasted.

### 2. Using bad proxies

If the cheap metric does not correlate with the real objective, the search reward-hacks the proxy.

### 3. Collapsing diversity too early

One winner too soon means local-optimum lock-in.

### 4. Ignoring variance

A single-seed win is often noise.

### 5. Failing to tax complexity

Search loves ugly hacks unless you penalize them.

### 6. Confusing family search with local tuning

This is a major one for this project.

The system keeps mutating details of an unproven family instead of asking whether the family premise itself should survive.

### 7. Not freezing enough inside a tournament

Across the whole project, the search space can be wide.
Inside one comparison cycle, the contract must be narrow enough for attribution.

## What Enigma should operationalize

If this note is taken seriously, the harness should enforce a few things.

### 1. Every candidate should declare its search level

At minimum:

- metric / lane
- base contract
- family mechanism
- local realization

That prevents hidden compounds.

### 2. The verifier should test consequentiality explicitly

The verifier should reject:

- scalar retunes disguised as new ideas
- children of unproven families
- mixed-level compounds with no clean attribution
- big algorithm stories with no executable proxy

### 3. The compiler should emit multiple realizations of one mechanism

The compiler should not collapse a family into one patch.

It should produce multiple executable realizations of the same claimed mechanism so the mechanism, not one brittle implementation, gets tested.

### 4. Downstream packs should stay family-homogeneous

One pack should answer one causal question, plus controls.

That is how later promotion remains meaningful.

### 5. Postmortem should learn which mutation classes pay off

The analyst stage should accumulate evidence like:

- schedule/controller families worked in unstable regimes
- tokenizer/base changes helped more than policy changes
- selector mutations mattered only late
- algorithmic jumps mostly failed under current fidelity

That is how the system starts learning the mutation policy itself.

### 6. Advance-round should be level-aware

The next round should not just "continue search."
It should decide whether to:

- refine within a proven family
- promote a base challenger
- compose proven survivors
- reset and move the search up a level

## Bottom line

This is a strong note.

The main additions I would make are:

1. put **measurement-contract search** above everything else
2. keep **base search** separate from within-base family search
3. force **family proof before local tuning**
4. treat **algorithm search as sparse, proxy-based, and heavily verified**

The deepest idea I agree with is this:

evolution is not just searching for a better model; it is searching for a better path through model space under finite compute.

In this repo, that means the search system should optimize not only code variants, but also the level at which it is searching, the evidence contract it trusts, and the kinds of mechanisms it is willing to preserve.
