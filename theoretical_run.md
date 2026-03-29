# Parameter Golf Theoretical Run

This is a thought experiment for the current baseline and the first two experiment directions.

It is not a claim about measured results.

The goal is to reason about the full optimization chain:

- training time
- evaluation time
- capability before export
- quality loss after export
- final artifact size
- final judged `val_bpb`

## 1. The Core Equation

The final judged score is not just "how good the model is."

A better mental model is:

`final_score = base_capability - export_survival + eval_gain`

Where:

- `base_capability` means the model's pre-export quality
- `export_survival` means how much quality is lost when we compress and reload the artifact
- `eval_gain` means any legal score improvement at evaluation time

Subject to:

- `artifact_bytes < 16,000,000`
- `train_time < 600s`
- `eval_time < 600s`

So the optimization target is really:

`best final_score per unit of artifact byte and wallclock`

## 2. Variables That Matter

### Capability Variables

- `L`: effective depth
- `D`: model dimension
- `M`: MLP expansion
- `V`: vocabulary size
- `E`: embedding dimension or factorized embedding bottleneck
- `C_train`: total useful training compute inside the wallclock

These mostly affect:

- pre-export `val_bpb`

### Compressibility Variables

- `B_w`: bitwidth allocation by tensor family
- `Q_clip`: clipping / scaling policy
- `Q_train`: whether training is export-aware
- `S_meta`: metadata overhead for scales, codebooks, exceptions
- `S_zip`: how well the serialized payload compresses

These mostly affect:

- post-roundtrip `val_bpb`
- artifact bytes

### Evaluation Variables

- `K_ctx`: effective context at eval
- `U_eval`: update frequency during legal adaptation
- `A_eval`: adaptation parameter count

These mostly affect:

- judged `val_bpb`
- eval wallclock

### Runtime Variables

- `t_step`: average training step time
- `t_eval`: average eval throughput
- `t_compile`: warmup / compile overhead

These mostly affect:

- how much actual optimization fits in the budget

## 3. Baseline Theoretical Chain

Baseline structure:

- dense GPT-like model
- tied embeddings
- 9 layers
- `512d`
- baseline int8+zlib export

Theoretical profile:

### Training

Strengths:
- simple dense path
- H100-friendly
- compile-friendly
- easy to optimize

Weaknesses:
- spends bytes fairly uniformly at export
- not especially clever about capability-per-byte

Expected training behavior:
- strong stability
- decent throughput
- limited architectural efficiency

### Export

Strengths:
- simple and robust
- low implementation risk

Weaknesses:
- likely not optimal byte allocation
- probably leaves some artifact budget unused or inefficiently used

### Eval

Strengths:
- simple
- low legality risk

Weaknesses:
- no extra judged-score lift

### Best Theoretical Outcome

The baseline is not likely to be the frontier winner, but it is a strong anchor because:

- it survives export
- it trains reliably
- it provides a clean reference for deltas

## 4. Experiment 01: Mixed Export

This experiment keeps training fixed and changes only the artifact policy.

## Chain

### Training Time

Expected effect:
- almost none during training

Why:
- training graph is unchanged

Risk:
- none at train time

### Eval Time

Expected effect:
- small or negligible

Why:
- the artifact is dequantized once on load, then evaluation runs dense

Risk:
- slightly more decode overhead at load time
- not a major issue unless serializer complexity grows

### Artifact Size

Expected effect:
- should decrease if int4 MLP packing works as intended

Why:
- MLP matrices are usually a large chunk of bytes
- packing them harder should save real payload bytes

Main uncertainty:
- metadata and scale storage
- whether zlib likes the packed format

### Output Quality

Expected effect:
- pre-export quality unchanged
- post-export quality may get worse

This experiment wins only if:

- byte savings are meaningful
- roundtrip degradation is small enough to justify later reinvestment

### Best Theoretical Outcome

Best-case path:

1. mixed export saves enough bytes to create real headroom
2. roundtrip loss stays nearly flat
3. later we spend the saved bytes on a stronger model

Expected near-term best outcome:

- not necessarily a better score immediately
- but a better size-quality frontier

This is a compression-first experiment.

## 5. Experiment 02: Factored Embeddings

This experiment keeps export mostly fixed and changes the embedding budget.

## Chain

### Training Time

Expected effect:
- slight overhead from the extra projection

Why:
- one more linear on input
- tied path also projects back before logits

Risk:
- slower convergence if the bottleneck is too small

### Artifact Size

Expected effect:
- modest parameter savings

At the current baseline shape:

- baseline tied embedding parameters: about `1024 x 512 = 524,288`
- factorized path with `E=128`: `1024 x 128 + 128 x 512 = 196,608`
- rough saving: `327,680` parameters

That is useful, but not a giant change by itself.

### Output Quality

Expected effect:
- could stay flat or get slightly worse

Why:
- the embedding bottleneck may lose expressivity

Why it may still be worth it:
- if semantic structure is low-rank enough
- if the saved parameters can later be reinvested better elsewhere

### Best Theoretical Outcome

Best-case path:

1. factorized embeddings barely hurt capability
2. saved parameter/byte budget is reused in depth, width, or MLP
3. final score improves via better capability-per-byte

Expected near-term best outcome:

- small score change either way
- but useful information about embedding inefficiency

This is a capability-per-byte experiment.

## 6. Combined Path: What Happens If Both Work

If both directions work, the combined strategy is:

1. use a more byte-efficient embedding path
2. use mixed export so MLPs compress harder than fragile tensors
3. spend reclaimed budget on better hidden compute

That gives a more realistic frontier path than either experiment alone.

### Combined Best-Case Story

- factorized embeddings save static parameter budget
- mixed export saves artifact bytes
- the recovered budget funds more useful hidden computation
- post-export degradation stays controlled

This is the first serious path toward:

`same wallclock, better capability-per-byte, same artifact cap`

## 7. Why Runtime Still Dominates

Even with perfect compression, a slow model can still lose.

Reason:

- if `t_step` rises too much, total useful optimization falls
- if `t_eval` rises too much, legal eval improvements become unusable

So the best theoretical design is not:

- strongest model
- or smallest artifact

It is:

- strongest compressible model that still fits the wallclock

## 8. Theoretical Priority Ordering

### Highest Probability Of Immediate Value

1. mixed export
2. factorized embeddings
3. warmup / warmdown / schedule cleanup

Why:
- low isolation risk
- easy to compare
- directly tied to the challenge objective

### Medium-Term Value

4. mild shared depth / recurrence
5. larger vocab with careful accounting
6. stronger export-aware QAT

Why:
- higher upside
- more coupled to runtime and correctness risk

### Late / Moonshot Value

7. native BitNet path
8. learned fast weights
9. TurboQuant-style online quantized state

Why:
- high upside
- much harder to land cleanly under the full challenge constraints

## 9. Expected Best Outcome From The Current Plan

If the current plan works well, the best realistic near-term outcome is:

- a model that is only modestly better pre-export
- but much better post-export per artifact byte
- and therefore easier to scale inside the `16MB` cap

That is the right shape of progress.

The first major win is not necessarily:

- "we immediately beat SOTA"

It is more likely:

- "we build a more efficient optimization surface than the baseline"

Meaning:

- better byte allocation
- better post-roundtrip robustness
- better capability-per-byte

Once that is true, larger architecture bets become worth taking.

## 10. Short Version

The challenge pipeline is:

1. train a strong model
2. make it survive compression
3. legally improve judged eval score
4. keep the whole system under wallclock

The current expected best path is:

- use factorization to reduce wasted static budget
- use mixed export to reduce wasted artifact bytes
- keep the runtime path simple and fast
- only add small eval-time gains later

So the first target is not just "better loss."

It is:

`better final roundtrip score per byte per second`
