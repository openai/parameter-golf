# Consequence Self-Check Rubric

Use this rubric both when **proposing** and when **verifying** hypotheses.

The point is not to reward elegant prose.

The point is to decide whether a hypothesis is a **real search move**.

## 1. Consequence

Ask:

- does this change the search at the metric, base, family, composition, or reset level?
- which consequential axis changes?

Valid consequential axes:

- `representation_contract`
- `compute_allocation`
- `regime_transition`
- `artifact_selection`
- `deployment_path`
- `model_contract`
- `search_strategy_reset`

### Score

- **5** - clearly changes a first-order mechanism or search level
- **4** - clearly consequential, though narrower than the best ideas
- **3** - maybe real, but still suspiciously close to local tuning
- **2** - mostly retuning or reframing language around a local change
- **1** - clearly not consequential

Keep only ideas that can plausibly score **4 or 5**.

## 2. Novelty Against History

Ask:

- does this break a different false invariant from prior failed lines?
- or is it a reworded version of the same line?

### Score

- **5** - clearly opens a new mechanism class
- **4** - meaningfully distinct from prior failures
- **3** - related but maybe not distinct enough
- **2** - near-duplicate family
- **1** - obvious repetition

## 3. Falsifiability

Ask:

- is there a smallest decisive probe?
- is there an observable that should move early?
- is there a cheap falsifier?
- is the measurement plan explicit enough that the probe can actually be judged?

### Score

- **5** - crisp probe, signal, and kill rule
- **4** - solid probe with minor ambiguity
- **3** - plausible but still too broad
- **2** - hard to falsify cheaply
- **1** - basically unfalsifiable

Keep only ideas that can plausibly score **3 or better**.

## 4. Lane Integrity

Ask:

- is the idea honest about which lane it attacks?
- does the evaluation lane match the mechanism?

### Score

- **5** - lane is explicit and measurement is aligned
- **4** - mostly aligned with minor ambiguity
- **3** - lane story is muddy
- **2** - likely lane mixing
- **1** - lane is incoherent

## 5. Implementation Honesty

Ask:

- can this run now with the current executable catalog?
- if not, is it explicitly marked as needing a new primitive or a new base cycle?
- does the hypothesis state why that implementation claim is honest?

### Score

- **5** - implementation claim is exact and honest
- **4** - mostly honest
- **3** - some ambiguity about what exists
- **2** - likely hiding missing machinery
- **1** - clearly dishonest

Keep only ideas that can plausibly score **3 or better**.

## Rewrite rule

If an idea is interesting but scores low on consequence, do **not** keep it as-is.

Instead rewrite it upward:

- from local tuning -> program family
- from program family -> base contract
- from base contract -> search reset

The verifier should prefer a **strong rewrite** over a weak keep.
