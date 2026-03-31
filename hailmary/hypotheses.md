# Hailmary Hypotheses

This is the moonshot hypothesis set for `parameter-golf`.

It is deliberately wider than `stage2_1`. The point is not to pick the safest next patch. The point is to cover the full high-upside solution space without collapsing into minor variants of the same idea.

## Abstract Problem

Maximize deployed next-token predictive quality under:

- fixed training wallclock
- fixed artifact size
- fixed code budget
- fixed dataset legality

Equivalent abstract form:

- limited online optimization budget
- lossy model transmission channel
- context-limited evaluator
- tiny decision budget for expensive confirmations

Hidden trade:

- `capacity vs compressibility`
- `useful updates vs per-step complexity`
- `raw checkpoint quality vs exported artifact quality`
- `train-time learning vs eval-time adaptation`

## Bottleneck Classes

The score can be bottlenecked by all of these at once:

- representation bottleneck
- optimization bottleneck
- compression bottleneck
- evaluation bottleneck
- throughput bottleneck
- search bottleneck

The main mistake is to assume only one of them matters.

## Invariants

Likely true:

- the winning artifact must survive quantization/compression
- byte budget is a first-class design variable, not a reporting constraint
- early training dynamics matter disproportionately under `600s`
- eval context and export policy can move score without changing the raw checkpoint
- data order can matter materially under a hard time cap even when the data itself is unchanged

Likely false invariants worth breaking:

- one precision for all weights
- one attention rule for all layers
- one optimizer law for all matrix params
- one context policy for all eval tokens
- one checkpoint selection rule at the end

## 2026-03-25 Principle Check

The hypothesis map itself is still broad enough. The problem is that the runnable slate did not fully follow it.

The main miss was scale:

- too many runnable ideas were still static mechanisms
- too few runnable ideas broke the larger process invariants

So the next `hailmary` generation standard is:

- keep export and eval lanes
- but force lead ideas from phase split, checkpoint selection, parameter-family split, and budget split

If a moonshot only changes a local helper, it is no longer a lead hypothesis.

## Negative Knowledge

Do not over-regenerate these as lead stories:

- plain NorMuon as the main frontier explanation
- generic label smoothing
- standalone compile mode as a primary score mechanism
- standalone SmearGate as a moonshot
- standalone OrthoInit as a moonshot
- FA3-first narratives without checking compressibility and actual step gain

## Operator-Sampled Families

Each family is written as:

- mechanism
- why it could move the score a lot
- math intuition
- dominant lane
- cheapest observable
- likely failure mode

## H901: Export Internalization

- Operator: `internalize`
- Mechanism: train and checkpoint-select for the deployed artifact, not the raw model
- Why high-upside: the largest current gap is often in `L_export`, not `L_train`
- Math intuition:
  - if export creates perturbation `e`, then `Delta L ~= grad^T e + 1/2 e^T H e`
  - Full GPTQ wins because it optimizes loss-weighted quantization error, not naive norm error
- Dominant lane: export
- Cheapest observable: same-checkpoint export bakeoff
- Failure mode: the export method is too expensive or fragile to operationalize

Concrete moonshots:

- Full GPTQ on banked weights
- checkpoint selection by deployed score
- export-aware late QAT

## H902: Capacity Reallocation Through Compression

- Operator: `reallocate`
- Mechanism: save bytes in export, then spend them on more predictive parameters
- Why high-upside: bytes saved by quantization/compression can be turned directly into lower `L_repr`
- Math intuition:
  - score gain can come from either reducing quantization distortion or buying more parameters
  - the best move is whichever gives larger `dL / d(byte)`
- Dominant lane: export + architecture
- Cheapest observable: artifact headroom plus same-wallclock ablation
- Failure mode: extra parameters are slow or low-value

Concrete moonshots:

- int5 MLP + int6 attention + fp16 fragile tensors
- spend freed bytes on `11L-12L`, VE128, or explicit n-gram priors

## H903: Explicit Low-Order Language Priors

- Operator: `externalize`
- Mechanism: stop forcing the transformer to relearn all short-range statistics from scratch in 10 minutes
- Why high-upside: low-order transitions are highly compressible and highly predictive
- Math intuition:
  - if `p(x_t | x_{t-1}, x_{t-2})` carries large mutual information, then a cheap explicit prior can remove entropy the main trunk no longer has to model
- Dominant lane: architecture
- Cheapest observable: early-loss drop and deployed score on short screens
- Failure mode: prior overfits low-order structure and steals capacity from higher-order modeling

Concrete moonshots:

- CountInitBigram head
- trigram sidecar
- hybrid learned-plus-count prior

## H904: Value-Path Specialization

- Operator: `specialize`
- Mechanism: give the value path its own memory machinery instead of making attention and MLP absorb everything
- Why high-upside: several strong recent results suggest value transport is under-modeled
- Math intuition:
  - attention picks information, but value channels carry content
  - if long-range predictive information is bottlenecked in values, then explicit value transport lowers `L_repr`
- Dominant lane: architecture
- Cheapest observable: no-TTT training screens with post-quant evaluation
- Failure mode: added structure is redundant with skips or XSA

Concrete moonshots:

- VRL
- VE128
- value-only sidecars

## H905: Context Expansion Without Full Cost Explosion

- Operator: `stage`
- Mechanism: concentrate stronger context handling where it matters most
- Why high-upside: score is often limited by context truncation, not raw parameter count
- Math intuition:
  - later layers can spend compute on disambiguation after early layers build local features
  - so context mechanisms may have higher marginal value in late depth
- Dominant lane: architecture + eval
- Cheapest observable: same-checkpoint eval and deeper-layer ablations
- Failure mode: context mechanism is too expensive or too noisy for the budget

Concrete moonshots:

- XSA-all
- late-layer-only XSA with stronger export
- context mode chosen by layer group

## H906: Eval Maximization

- Operator: `externalize`
- Mechanism: give each scored token the best legal context and the best legal adaptation
- Why high-upside: this directly attacks `L_eval`
- Math intuition:
  - if each target token is counted once, then more legal left context monotonically helps unless it destabilizes predictions
- Dominant lane: eval
- Cheapest observable: same-checkpoint eval bakeoff
- Failure mode: eval policy is too slow or adaptation overfits

Concrete moonshots:

- exact overlap-aware sliding
- doc-aware state resets
- score-first or backward-looking TTT
- K-LoRA plus best-epoch-per-document selection

## H907: Training-Deployment Phase Split

- Operator: `stage`
- Mechanism: use one objective early for fast learning and another late for export alignment
- Why high-upside: the final artifact is judged after compression, not at raw bf16 weights
- Math intuition:
  - early on, maximize learning progress
  - late on, minimize deployed loss under the export channel
- Dominant lane: training + export
- Cheapest observable: late-phase ablation on a strong base
- Failure mode: phase switch destabilizes or arrives too early

Concrete moonshots:

- EMA then GPTQ on EMA
- tight SWA only in the final low-LR region
- late active QAT aligned to the export quantizer

## H908: Depth and Family Asymmetry

- Operator: `specialize`
- Mechanism: stop using homogeneous rules across layers and tensor families
- Why high-upside: not all layers contribute equally to score or artifact bytes
- Math intuition:
  - marginal utility of one more bit, one more dimension, or one more context mechanism varies by depth and tensor role
- Dominant lane: architecture + export
- Cheapest observable: family-specific ablations
- Failure mode: too many discrete choices, weak observability

Concrete moonshots:

- XSA only where it matters most, then XSA-all if funded
- mixed precision by tensor family
- separate optimizer/export policy for explicit memory modules

## H909: Compression-Aware Geometry

- Operator: `tighten`
- Mechanism: shape weights to be easier to compress without losing predictive power
- Why high-upside: if two models have similar raw loss, the more compressible one can either score better after export or fund more parameters
- Math intuition:
  - a rotation or pruning rule can change code-length and quantization error together
  - the target is minimum deployed loss at fixed bytes, not minimum raw distortion
- Dominant lane: export
- Cheapest observable: artifact-size and deployed-score bakeoff
- Failure mode: compression gain is real but predictive loss outweighs it

Concrete moonshots:

- Hadamard / QuIP-lite style rotation
- mild pruning for entropy coding
- tensor-family-specific compressors

## H910: Deployed Checkpoint Selection

- Operator: `externalize`
- Mechanism: export and score multiple late checkpoints, then choose the best deployed artifact rather than the last state
- Why high-upside: the deployed optimum may not coincide with the raw training optimum
- Dominant lane: export
- Cheapest observable: best-of-last-`K` late checkpoint bakeoff
- Likely failure mode: late deployed scores are flat, so the extra selection step adds little

## H911: Two-Stage Curriculum

- Operator: `stage`
- Mechanism: one shard order early, another late
- Why high-upside: fixed ordering is likely too blunt; stage-specific data emphasis may matter more than one global order
- Dominant lane: training
- Cheapest observable: `600s` comparison against static curriculum
- Likely failure mode: both phases are too similar, so the split is fake

## H912: Parameter-Family Late Freeze

- Operator: `factorize`
- Mechanism: embeddings, head, controls, and trunk matrices stop moving at different times
- Why high-upside: not all tensor families should keep adapting equally late in training
- Dominant lane: training + export
- Cheapest observable: deployed score and artifact comparison on the same parent stack
- Likely failure mode: freeze the wrong family and lose too much fit

## H913: Two-Stage Context Budget

- Operator: `reallocate`
- Mechanism: cheaper context early, full context late
- Why high-upside: early training may benefit more from extra updates than from maximal context length
- Dominant lane: training + throughput
- Cheapest observable: steps gained at equal wallclock and final deployed score
- Likely failure mode: early short context harms representation too much to recover

## H914: Alternating Objective Microcycles

- Operator: `stage`
- Mechanism: alternate normal optimization with periodic deployment-alignment steps
- Why high-upside: avoids poisoning the whole training trajectory with a secondary objective
- Dominant lane: training + export
- Cheapest observable: same-parent comparison versus always-on late regularization
- Likely failure mode: alignment steps are either too weak to matter or too strong and destabilize training

## H910: Throughput Reinvestment

- Operator: `reallocate`
- Mechanism: buy more useful updates, not just lower ms/step
- Why high-upside: if throughput gain converts into hundreds of extra steps, it can reduce `L_opt` materially
- Math intuition:
  - under a hard wallclock cap, the relevant derivative is `dL / d(step)` times extra steps bought
- Dominant lane: systems + training
- Cheapest observable: step-count-at-cap plus matched loss
- Failure mode: faster kernels hurt compile stability, compressibility, or numerical behavior

Concrete moonshots:

- eliminate warmup waste
- backend choice on score-per-wallclock, not kernel speed alone
- parallel optimizer variants

## H911: Curriculum Under a Time Cap

- Operator: `borrow`
- Source domain: curriculum learning / control
- Abstract mechanism: spend early budget on the most leverage-per-step information
- Translated mutation: curriculum over shard order, context length, or transition difficulty
- Why mapping is valid: this benchmark is budget-limited optimization, not convergence-limited optimization
- Dominant lane: data + training
- Cheapest observable: early loss and final deployed score on equal wallclock
- Failure mode: curriculum biases the model toward the wrong distribution slice

Concrete moonshots:

- short-to-long context curriculum
- count-first transition curriculum
- entropy-ranked shard scheduling
- perplexity-ranked shard scheduling

## H912: Anti-Dominant Simpler-Core Branch

- Operator: `invert`
- Mechanism: reduce trunk complexity, push more work into explicit priors, export quality, and eval
- Why high-upside: the dominant story may be overinvesting in the transformer trunk
- Math intuition:
  - if a portion of predictive entropy is cheaply handled by priors or eval-time adaptation, the best byte allocation may be a simpler trunk plus stronger side systems
- Dominant lane: cross-lane
- Cheapest observable: full-stack comparison against the frontier template
- Failure mode: trunk becomes too weak and priors cannot rescue it

Concrete moonshots:

- smaller trunk + CountInitBigram + VE128 + GPTQ + strong eval
- simpler trunk with more exact side memory

## Coverage Matrix

| Family | Train | Export | Eval | Architecture | Data | Systems |
| --- | --- | --- | --- | --- | --- | --- |
| H901 Export Internalization | secondary | primary | none | none | none | none |
| H902 Capacity Reallocation | secondary | primary | none | primary | none | none |
| H903 Explicit Low-Order Priors | secondary | secondary | none | primary | primary | none |
| H904 Value-Path Specialization | secondary | secondary | none | primary | none | none |
| H905 Context Expansion | secondary | none | primary | primary | none | none |
| H906 Eval Maximization | none | none | primary | none | none | none |
| H907 Training-Deployment Phase Split | primary | primary | none | none | none | none |
| H908 Depth and Family Asymmetry | primary | primary | none | primary | none | none |
| H909 Compression-Aware Geometry | none | primary | none | secondary | none | secondary |
| H910 Throughput Reinvestment | primary | none | none | none | none | primary |
| H911 Curriculum Under Time Cap | primary | none | none | none | primary | none |
| H912 Simpler-Core Branch | primary | primary | primary | primary | none | secondary |

## Highest-Upside Hailmary Set

If the question is strictly "what could change `val_bpb` drastically under the constraints?", the strongest moonshot set is:

1. Full GPTQ plus checkpoint-by-deployed-score selection
2. CountInitBigram or stronger exact n-gram prior
3. XSA-all funded by better export
4. VRL plus VE128 value-path stack
5. eval maximization with exact overlap-aware scoring and legal adaptation
6. shard-order curriculum under the fixed 600-second budget

Those are not the safest ideas. They are the ideas most likely to move the score by a visibly large amount if they work.

## Mutation Directions

For an evolutionary or staged search, the mutation operators should be:

- export precision map mutation
- explicit prior family mutation
- context mechanism depth mutation
- value-path augmentation mutation
- phase-boundary mutation
- checkpoint-selection mutation
- eval-policy mutation

Guardrails:

- compare export mutations on the same checkpoint first
- compare eval mutations on the same checkpoint first
- do not mix throughput probes with score hypotheses in one early pack
- only stack side memories after export headroom is real

## Bottom Line

The moonshot space is not "more tricks around the baseline."

It is:

- redesign the exported artifact
- add cheap explicit structure the trunk cannot learn in time
- move more legal information into eval context or adaptation
- spend every saved byte on higher-value predictive structure
