# Enigma First-Pass Strategy For Parameter Golf

## Goal

Apply the Enigma premise generically to Parameter Golf:

1. enumerate the real solution space,
2. narrow it without collapsing onto one local favorite,
3. cover each major mutation surface with at least one deliberate first-pass probe,
4. retire bad regions quickly,
5. preserve enough diversity that AlphaEvolve can still discover non-obvious combinations.

This is not an optimizer-only plan. The search space here includes architecture, optimizer/training dynamics, tokenizer/data, systems throughput, evaluation behavior, and artifact compression.

## Ground Truth Constraints

- Objective: minimize final round-tripped `val_bpb`.
- Hard runtime cap: training must finish under `10 minutes` on `8xH100`.
- Hard artifact cap: code bytes plus compressed model bytes must stay under `16,000,000` bytes.
- The baseline already sits close to the size ceiling:
  `15,863,489` total bytes, leaving only about `136 KB` of slack.
- The baseline also hits the wallclock cap before exhausting steps:
  `13,780 / 20,000` steps in `600s`.

Implication:

- Anything that hurts step time can lose even if it improves per-step learning.
- Anything that adds much code can lose even if it improves loss.
- First-pass search should prefer compact, local, high-leverage mutations.

## Early-Experiment Note From The Original Repo

The upstream repo is explicit about how to start early experiments:

- use local MLX smoke runs first when iterating locally,
- use reduced data prefixes such as `--train-shards 1` for quick bring-up,
- test on cheaper hardware and even `1xH100` before spending on `8xH100`,
- keep the full fixed validation split for anything that is meant to be comparable,
- the core `train_gpt.py` already enforces the approximate 10-minute wallclock budget.

Enigma interpretation:

- early experiments are for falsification and ranking broad families,
- they are not for claiming victory,
- they should preserve alignment with the real track whenever possible,
- expensive full-track runs should only be spent on survivors from clearly distinct neighborhoods.

## Surface Area Map

### Primary Search Surface: `train_gpt.py`

This is the main battleground. The file contains almost every material lever:

- `Hyperparameters`: model shape, batch shape, optimizer constants, wallclock behavior.
- `Muon`: matrix optimizer behavior for 2D transformer weights.
- token-agnostic evaluation: the actual scoring path.
- quantization and compression: directly affects artifact viability and final score.
- token loading and batch construction: affects both stability and speed.
- transformer modules: architecture, recurrence/skip behavior, attention/MLP structure.
- training loop: LR schedule, warmup/warmdown, grad clipping, validation cadence, stopping behavior.

### Secondary Search Surface: `data/`

- tokenizer choice and tokenizer rebuild path,
- dataset export path from the fixed published docs cache,
- shard-count control for cheap early experiments,
- potential retokenization path if we decide the tokenizer itself is a frontier surface.

### Reference Surface: `records/`

- the 10-minute baseline is the real comparator,
- the 4-hour non-record run shows what improves with more optimization time,
- both provide empirical anchors for throughput, code size, model size, and validation movement.

## Search Taxonomy

We should treat the solution space as six top-level families.

### Family A: Parameter Allocation And Architecture

Question:
Given the same artifact cap, how should parameters be spent?

Subspaces:

- depth vs width,
- attention head count vs KV head count,
- MLP expansion ratio,
- tied vs untied embeddings,
- skip/recurrence/residual mixing structure,
- long-context or test-time-compute ideas only if they survive the code-size budget.

Why it matters:

- Parameter Golf is fundamentally an `L(N)` problem, not just an optimizer problem.
- Architecture can dominate optimizer tweaks if the current allocation is poor.

### Family B: Training Dynamics And Optimizer Design

Subspaces:

- Muon behavior,
- Adam/tied-embedding behavior,
- schedule shape,
- batch size and sequence length,
- warmup/warmdown design,
- gradient clipping or norm control,
- per-group learning-rate allocation.

Why it matters:

- The leaderboard metric is reached under a strict short horizon, so bootstrap behavior matters heavily.

### Family C: Tokenizer And Data Representation

Subspaces:

- vocabulary size,
- tokenizer family,
- retokenization of the same published docs,
- training shard prefix choice during early experiments,
- sequence packing assumptions.

Why it matters:

- the challenge is tokenizer-agnostic and scored in BPB,
- embeddings are a large fraction of parameters in tiny models,
- tokenizer choice changes both parameter allocation and compression behavior.

### Family D: Throughput And Systems

Subspaces:

- step time,
- communication pattern,
- precision choices,
- fused vs unfused behavior,
- validation cadence,
- compilation warmup cost.

Why it matters:

- more steps inside the same 600 seconds is itself a first-class improvement axis.

### Family E: Artifact And Compression

Subspaces:

- quantization scheme,
- passthrough rules,
- code-size footprint,
- control-parameter storage choices,
- architecture choices that produce more compressible weights.

Why it matters:

- final score is measured after round-trip quantization,
- a better pre-quant model can still lose if the artifact expands or quantizes badly.

### Family F: Evaluation-Aware Ideas

Subspaces:

- sequence length used for eval,
- model behaviors that help compression specifically,
- representation choices that improve token-byte efficiency.

Why it matters:

- the objective is BPB on the fixed validation set, not generic train loss.

## How To Narrow The Search Space Without Missing It

Enigma should not search everything at once. The first pass should narrow in stages.

### Stage 1: Build A Morphology, Not A Grab Bag

Create an explicit grid with one row per family and columns for:

- mechanism,
- expected gain channel,
- expected cost channel,
- cheapest faithful test,
- observability in subscale runs,
- overlap neighborhood,
- retirement condition.

The point is to avoid ten variants of the same idea while leaving entire families untouched.

### Stage 2: Pick One Probe Per Distinct Neighborhood

For the first serious wave, choose only hypotheses that are:

- high leverage,
- cheap in code bytes,
- cheap in step-time risk,
- orthogonal to one another.

If two ideas hit the same mechanism, only one survives the first pass.

### Stage 3: Use Cheap Experiments For Elimination, Not Promotion

Cheap experiments should answer:

- does this family move the metric at all,
- does it slow the loop,
- does it threaten the artifact budget,
- does it destabilize training.

They should not be used to declare final winners.

### Stage 4: Promote By Family Diversity

When escalating to real 8xH100 10-minute runs, the portfolio should include survivors from different families:

- one architecture/allocation idea,
- one optimizer/bootstrap idea,
- one artifact/compression idea,
- optionally one tokenizer/data idea if the implementation cost is justified.

This gives coverage of the real solution space instead of overfitting one local optimizer hill.

### Stage 5: Use Leave-One-Out And Counterfactuals Early

If a composite wins, immediately ask:

- which part actually carries the gain,
- which part is only neutral noise,
- which part is secretly harmful but masked by another improvement.

This is the direct Stage 5 Enigma lesson from NanoChat and should be part of the generic framework.

## First-Pass Portfolio

This is the initial coverage plan, not the final answer.

### Bucket 1: Architecture / Allocation

Probe ideas:

- rebalance depth vs width around the same rough compressed-size budget,
- adjust `MLP_MULT`,
- adjust `NUM_KV_HEADS`,
- test whether tied embeddings remain optimal once tokenizer or vocab changes.

Why first-pass:

- these are compact edits with potentially large effect size.

### Bucket 2: Optimizer Bootstrap

Probe ideas:

- Muon schedule variants,
- embedding-specific Adam variants,
- LR ratio changes between embeddings, matrices, and scalars,
- warmup and warmdown schedule reshaping.

Why first-pass:

- short-horizon training is dominated by early optimizer behavior.

### Bucket 3: Artifact / Compression

Probe ideas:

- int8 quantization rule changes,
- passthrough threshold changes,
- reducing the count or precision of small preserved tensors,
- architecture choices chosen partly for compressibility.

Why first-pass:

- the baseline has almost no size slack.

### Bucket 4: Tokenizer / Vocabulary

Probe ideas:

- vocab-size changes near the current `1024`,
- alternative tokenizer families trained on the same published docs cache,
- evaluating whether a different tokenizer reduces embedding burden enough to pay for itself.

Why not first-first:

- this is powerful but higher complexity and higher validation risk.
- it should enter once the simpler trainer-local surfaces are characterized.

### Bucket 5: Throughput

Probe ideas:

- reduce validation overhead during exploratory runs,
- examine sequence-length and batch-shape tradeoffs,
- look for low-code simplifications that improve effective steps.

Why first-pass:

- a step-time win can be as valuable as a learning-rule win under this challenge.

## Retirement Rules

Retire a family quickly if it:

- increases step time materially without an offsetting validation improvement,
- expands the artifact budget too far,
- only improves train loss but not round-tripped `val_bpb`,
- overlaps a stronger surviving idea in the same neighborhood,
- depends on too much added code to be leaderboard-safe.

## Promotion Rules

Promote a family if it:

- improves final `val_bpb` on aligned runs,
- keeps or improves effective steps under the wallclock cap,
- preserves artifact safety margin,
- remains understandable enough that AlphaEvolve can continue mutating around it,
- occupies a distinct mechanism neighborhood from the other survivors.

## Practical First Pass

1. Map each editable surface in `train_gpt.py` into one of the six families above.
2. Pick one compact first-pass probe per family.
3. Use cheap aligned runs to retire obvious losers.
4. Run a small diverse survivor slate on real 10-minute conditions.
5. Perform leave-one-out on any winning composite.
6. Only then let AlphaEvolve search deeper inside the surviving neighborhoods.

## Bottom Line

The first pass should not ask, "what is the best optimizer mutation?"

It should ask, "which distinct mechanism families are even alive under the 10-minute and 16MB constraints?"

That is the generic Enigma move:

- narrow by ontology,
- cover by neighborhood,
- promote by evidence,
- retire aggressively,
- never confuse local optimizer tweaking with coverage of the actual solution space.
