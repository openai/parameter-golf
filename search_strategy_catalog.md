# Search Strategy Catalog for Optimizing `train_gpt.py`

## Goal

Optimize an executable training program and its surrounding contract to the **lowest credible `val_bpb`** on a given dataset.

For this repo, that means treating `train_gpt.py` as the object to optimize, not any one stage folder. The script already:

- supports a bring-your-own SentencePiece tokenizer via `TOKENIZER_PATH` and `VOCAB_SIZE`
- computes tokenizer-aware `val_bpb`
- logs quantized end-state metrics such as `final_int8_zlib_roundtrip_exact val_bpb`

So the legitimate search space includes:

- training dynamics
- checkpoint/export selection
- quantization/deployment path
- tokenizer/vocab choice
- architecture or global stack changes

The mistake is to let one search surface dominate forever just because an earlier stage emphasized it.

---

## First correction: there is no permanent fixed cycle

There is **not** one permanent stage-3-like frozen cycle for the whole project.

Instead, there are **temporary search cycles**.

- Across the whole project, you may search tokenizer, vocab size, prequant path, post-quant path, architecture, and training stack.
- Inside any one tournament, you freeze only the dimensions needed for a fair comparison.

That distinction matters:

- **project-level search** can move the base
- **cycle-level search** should only vary one class of causal question at a time

If you forget this, the search gets trapped in the worst of both worlds:

- too local to discover a new frontier
- too mixed to trust the rankings

---

## What a search strategy actually is

A search strategy is not "some mutations."

A search strategy is a choice of:

1. **unit of variation** - what is allowed to change
2. **unit of comparison** - what counts as a fair control
3. **dominant metric** - what decides survival
4. **evidence contract** - what data must be collected
5. **reset rule** - when to stop mutating and change strategy level

If any of those are vague, you are not searching. You are just generating edits.

---

## Search levels

| Level | Main question | Typical things that change | Things that should stay frozen |
| --- | --- | --- | --- |
| **L0. Metric/lane** | What outcome are we actually optimizing right now? | prequant vs post-quant vs final exact vs size/latency tie-breaks | everything else |
| **L1. Base contract** | Which executable base deserves deeper search? | tokenizer, vocab size, architecture, optimizer stack, dataset variant | within-base family logic |
| **L2. Program family** | Which consequential mechanism class is real? | controller, schedule family, optimizer handoff, selector, branch policy, quant path family | chosen base contract |
| **L3. Within-family mutation** | What is the best local version of a proven family? | thresholds, magnitudes, small wiring edits, counts | base and family identity |
| **L4. Composition** | Which proven families actually combine? | pairings or stacks of proven survivors | base and solo-family evidence |
| **L5. Reset/reframe** | Are we searching at the wrong level entirely? | the search strategy itself | the already-observed evidence |

Most "bad to worse" search behavior happens because the work should have moved **up** a level, but instead kept mutating **down** the tree.

---

## Consequential mutation test

A mutation is **consequential** if it changes at least one of these:

1. **representation contract**  
   tokenizer, vocab size, data encoding
2. **compute allocation**  
   batch tokens, sequence-length curriculum, which phase gets compute
3. **regime transitions**  
   optimizer handoff, warmdown gating, state-triggered policy changes
4. **artifact selection**  
   which checkpoint/export state wins
5. **deployment path**  
   quantization or export method family
6. **model contract**  
   depth/width/head structure or another true base change

A mutation is **not consequential** if it only:

- nudges a scalar
- adjusts a threshold inside an unproven family
- adds a helper without changing the execution program
- sweeps local parameters before the causal premise has survived

### Examples

| Consequential | Not consequential |
| --- | --- |
| switch from fixed warmdown to velocity-gated warmdown | change warmdown threshold from 0.20 to 0.23 before the family has survived |
| switch tokenizer/vocab from SP1024 to SP8192 | tweak one tokenizer-related constant inside the same tokenizer regime |
| replace "last checkpoint wins" with "best of last K" selector | tune K from 4 to 5 before selector search has survived |
| change quantization family | retune one calibration batch count inside an unproven quant family |

---

## Strategy catalog

## S0. Metric-lane decomposition

**Purpose:** decide what "better" means before running any tournament.

Use when:

- prequant and post-quant are telling different stories
- a candidate improves training but loses after quantization
- throughput or size is dominating decisions without being explicit

Outputs:

- dominant metric for this cycle
- secondary metrics
- rejection conditions
- whether training and deployment should be searched together or separately

Typical result:

- **training cycle:** optimize prequant `val_bpb`
- **deployment cycle:** optimize final exact post-quant `val_bpb`
- **tokenizer cycle:** optimize `val_bpb` under a changed tokenizer contract with a simple training base

---

## S1. Base frontier scout

**Unit of search:** executable base contracts.

Vary things like:

- tokenizer/vocab regime
- architecture family
- global optimizer stack
- dataset slice or data contract
- prequant vs quantization-first training stack

Do **not** vary:

- local child mutations inside each base
- branch portfolios
- controller thresholds

Use when:

- no trustworthy frontier base exists
- recent tournaments were flat or catastrophic
- all interesting mutations seem to be built on a suspicious base

Pack shape:

- 2 controls on the current best-known base
- 2-4 challenger bases
- each challenger is as simple and canonical as possible

Promotion rule:

- promote only bases that beat control beyond control spread
- if control spread is large, fix infrastructure first

This is the right strategy when the real question is "are we standing on the wrong hill?"

---

## S2. Tokenizer/vocab scout

**Unit of search:** representation contracts.

This is a specialized form of base search and deserves its own strategy because `train_gpt.py` explicitly allows tokenizer replacement while scoring by `val_bpb`.

Vary:

- tokenizer family
- vocab size
- tokenizer/data alignment

Freeze:

- training program
- selector logic
- quantization family

Use when:

- compression metric suggests the tokenizer may be a ceiling
- improvements are tiny despite many training-side mutations
- different tokenizer settings dominate more than training changes

Warning:

- tokenizer changes are real search, not cheating
- but they must be compared as **whole-base variants**, never as per-slot overrides in a family tournament

---

## S3. Training-program family search

**Unit of search:** consequential execution programs on a frozen base.

Examples:

- state-conditioned controllers
- optimizer handoff families
- compute-allocation schedules
- checkpoint/export selection logic
- late branch policies

Use when:

- one base looks credible
- you need to discover which mechanism class matters
- you want bigger moves than scalar tuning

Freeze:

- tokenizer/vocab
- architecture
- dataset contract
- base optimizer stack, unless the family itself is about changing it

Good first-order families:

- schedule/regime transitions
- artifact selectors
- late-branch portfolios
- quant-path families

Bad first-order families:

- tiny local retunes
- child variants of an unproven parent
- compounds made from two unproven families

---

## S4. Artifact/selector search

**Unit of search:** what state or artifact is chosen as the winner.

Examples:

- last checkpoint vs best of last K
- EMA vs raw
- best export state vs final state
- branch winner selection

Use when:

- training looks decent but final deployed score is inconsistent
- prequant curves are promising but end artifacts are bad
- multiple late states are clearly behaving differently

The key is that selector search should compare **the same training regime** under different selection rules, not totally different training programs.

---

## S5. Quantization/deployment search

**Unit of search:** post-training export and deployment family.

Examples:

- calibration family
- quantization family
- export artifact format
- deployment-time evaluation path

Use when:

- prequant `val_bpb` is good but post-quant exact is weak
- training-side families are not the bottleneck anymore
- deployment cost or submission size is the real frontier

Freeze:

- prequant training artifact
- tokenizer contract
- training-side family

If you mix training-side and quantization-side search in one short screen, the ranking usually reflects time-budget mismatch instead of the true winner.

---

## S6. Local exploitation search

**Unit of search:** children of a proven family.

Allowed:

- thresholds
- magnitudes
- counts
- small structural variants within the same family identity

Use only when:

- a family already survived a parent-level comparison
- the early signal appeared
- the causal story is credible

Do **not** use when:

- nothing has survived yet
- you are emotionally attached to a family
- results are flat and you are hoping micro-tuning rescues a wrong premise

This is where most wasted time accumulates.

---

## S7. Composition search

**Unit of search:** interactions between proven survivors.

Only allow composition when:

- both component families have solo evidence
- they act on different causal surfaces
- you can name the expected interaction

Examples:

- a schedule family plus a selector family
- a training family plus a quant family

Bad composition:

- "take all the winners and stack them"

Good composition:

- "this schedule creates different late states, and this selector should exploit exactly that"

---

## S8. Reset/reframe search

**Unit of search:** the search strategy itself.

Use when:

- multiple recent packs made no progress
- winners do not replicate
- all improvements are below control spread
- local mutations keep failing
- stage labels are constraining thinking

Reset/reframe asks:

1. Are we on the wrong search level?
2. Is the dominant metric wrong?
3. Is the base contract wrong?
4. Is the lane mixed?
5. Are we missing a primitive or just searching the wrong family?

This is the strategy that prevents "stage 3 after stage 3 after stage 3" from turning into ritual.

---

## How to choose strategy from data

| Observed pattern | Likely cause | Next strategy |
| --- | --- | --- |
| controls unstable | infrastructure or orchestration error | stop search and fix instrumentation first |
| all candidates catastrophic and slower | wrong family class or broken base | S1 base frontier scout or S8 reset/reframe |
| all candidates flat near control | search is too local | move up from S6 to S3 or from S3 to S1 |
| prequant improves but final exact degrades | deployment is the bottleneck | S5 quantization/deployment search |
| final results depend heavily on which checkpoint is chosen | selector is bottleneck | S4 artifact/selector search |
| tokenizer-related runs dominate metric differences | representation contract is bottleneck | S2 tokenizer/vocab scout |
| repeated child mutations never help | parent premise is false | retire family and move up a level |
| compile/materialize failures dominate | primitive gap or tooling issue | build primitive/instrumentation backlog first |

---

## Recommended controller for a model or agent

Use this loop:

1. **Choose the dominant metric and lane.**
2. **Choose the highest unresolved search level.**
3. **Freeze everything below that level.**
4. **Run the smallest decisive pack.**
5. **Retire aggressively.**
6. **Only deepen after a real survivor exists.**
7. **If there is no survivor, change strategy, not just mutations.**

That one rule matters most:

> if no family has survived, the next action is usually a new strategy, not better child mutations

---

## Minimum evidence contract

Every run should leave enough evidence for the next strategy decision.

At minimum record:

- executed program identity
- dataset path
- tokenizer path
- vocab size
- dominant metric
- prequant `val_bpb`
- post-quant / final exact `val_bpb`
- submission size
- throughput or step-time
- selected artifact/checkpoint
- control spread
- promotion or retirement reason
- whether the declared early signal appeared

Without that, the model cannot decide whether the next move is:

- a new family
- a new base
- a new tokenizer
- a new quant path
- or an infrastructure fix

---

## What to stop doing

1. Do not let stage names define the search objective.
2. Do not mix tokenizer/base search with family admission inside one tournament.
3. Do not keep mutating a family that has never shown its declared signal.
4. Do not compose unproven families.
5. Do not let export-heavy ideas compete under the same budget as pure training ideas unless the lane is intentionally shared.
6. Do not confuse "more code changed" with "more consequential."

---

## Short version

The right question is not:

> what mutation should I try next?

The right question is:

> what **search level** is currently wrong?

Once that is clear, choose a strategy from this catalog, freeze everything below that level, and run the smallest decisive tournament that can answer the question.
