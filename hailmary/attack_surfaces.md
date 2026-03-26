# Hailmary Attack Surfaces

This document is grounded in the current merged [`train_gpt.py`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py).

It is not a list of community tricks. It is a first-principles map of where the score can move a lot.

## Objective Math

For a fixed tokenizer and a correct evaluation policy that counts each validation byte exactly once,

`val_bpb = val_loss / ln(2) * tokens_per_byte`

and `tokens_per_byte` is effectively constant on the fixed validation split.

So the real problem is:

`min_theta E[-log p_theta(x_t | context_t)]`

under four extra penalties:

- wallclock cap: only a limited number of updates fit in `600s`
- artifact cap: only a limited number of parameters and bits fit in `16MB`
- deployment penalty: exported weights are not the trained weights
- eval-context penalty: the scoring context may be shorter or weaker than the model could use

Useful decomposition:

`L_final = L_repr + L_opt + L_export + L_eval`

where:

- `L_repr`: model class cannot represent the right conditional distribution
- `L_opt`: model could represent it, but training did not reach it in time
- `L_export`: trained model is good, exported artifact is worse
- `L_eval`: model/artifact is good, but the scoring procedure leaves context or adaptation on the table

A drastic move is one that attacks one of those four terms directly.

## Hard Constraints

- code should still stay reasonably implementable in this repo
- no illegal use of validation data
- no hidden external artifact payloads
- improvements must survive the real compressed roundtrip, not just raw checkpoints

## Surface Map

## 1. Hyperparameters and Global Budget Allocation

Reference:

- [`train_gpt.py:39`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L39)
- [`train_gpt.py:87`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L87)

Current assumptions:

- 9 layers by default
- seq `1024`
- MLP `2x`
- weak frontier schedule defaults
- warmup steps are spent only to prime compile

Why this matters:

- these lines decide the capacity/step/context trade before any patch is written
- a weak base causes patch ranking to chase baseline repairs instead of frontier gaps

Drastic levers:

- fund a larger model by export savings rather than by training tricks
- asymmetric context: train short enough to get more updates, evaluate long enough to exploit context
- base schedule as a first-class mechanism, not an afterthought

High-upside mutations:

- `11L-12L + seq2048 + stronger warmdown` style base
- `33M+` param branches funded by int5/mixed-bit/GPTQ compression
- train/eval context asymmetry instead of one shared `TRAIN_SEQ_LEN`

## 2. Muon Geometry and Optimizer Law

Reference:

- [`train_gpt.py:96`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L96)
- [`train_gpt.py:168`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L168)

Current assumptions:

- plain Muon
- one matrix optimizer law for all transformer matrices
- no weight decay
- no second-moment normalization
- no depth-specific or family-specific optimizer specialization

Why this matters:

- optimizer error enters `L_opt`
- optimizer-induced weight scale enters `L_export`

First-principles angle:

- Muon orthogonalizes update direction, but it does not directly optimize deployability
- the competition score cares about the exported model, so an optimizer that makes weights more compressible can beat one that only minimizes raw training loss

Drastic levers:

- training-deployment co-design in the optimizer itself
- specialize by tensor family: attention, MLP, embedding, sidecar memory, value path
- phase-conditioned optimizer law, not static law

Moonshot candidates:

- Muon with explicit deployment regularization
- depth-varying Muon momentum / decay
- value-path optimizer separate from attention/MLP optimizer
- Parallel Muon or reduced communication optimizer to buy more steps

## 3. Validation Policy and What Is Actually Counted

Reference:

- [`train_gpt.py:219`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L219)
- [`train_gpt.py:278`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L278)

Current assumptions:

- disjoint windows
- every prediction only sees the context inside one non-overlapping chunk
- no document reset logic
- no test-time adaptation

Why this matters:

- this surface directly attacks `L_eval`
- unlike training changes, it can move score on the same checkpoint

First-principles angle:

If the model benefits from left context and the metric counts tokens once, then the correct scoring policy is to give each token maximal legal left context while counting it exactly once.

Drastic levers:

- exact overlap-aware sliding eval
- doc-isolated state resets when boundary leakage is harmful
- legal TTT or score-first adaptation

Moonshot candidates:

- overlap-aware exact tail scoring
- score-first or backward-looking TTT
- eval policy selected per artifact family, not one global policy

## 4. Export, Quantization, and Compression

Reference:

- [`train_gpt.py:288`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L288)
- [`train_gpt.py:422`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L422)

Current assumptions:

- int8 export
- percentile clipping
- zlib
- no Hessian-aware quantization
- no mixed precision by tensor family
- no rotation, pruning, or entropy coding specialization

Why this matters:

- this is pure `L_export`
- for the current frontier, this is the single highest-leverage lane

First-principles angle:

If exported weights are `w + e`, then for small perturbations:

`Delta L ~= grad^T e + 1/2 e^T H e`

which is exactly why Hessian-aware GPTQ matters. The objective is not to make quantization error small in norm. It is to make it small in loss.

Drastic levers:

- Full GPTQ
- mixed-bit allocation by layer or tensor family
- fragile-tensor passthrough
- compression-aware rotations or pruning that buy parameter headroom

Moonshot candidates:

- Hessian GPTQ on banked weights
- int5 MLP / int6 attention / fp16 fragile tensors
- post-EMA quantization, not raw-final quantization
- artifact-level policy search: compression backend, precision map, passthrough map

## 5. Data Stream and Token Order

Reference:

- [`train_gpt.py:429`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L429)
- [`train_gpt.py:494`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L494)

Current assumptions:

- deterministic sequential stream
- no curriculum
- no shard prioritization
- no domain-specific ordering

Why this matters:

- under a short wallclock, early data order changes what the model learns before time runs out

First-principles angle:

Training under a hard time cap is not the same as training to convergence. Early gradient budget should be spent on the highest-value statistics first.

Drastic levers:

- curriculum over context difficulty
- front-load low-order statistics or high-frequency transitions
- schedule shard order to reduce early optimizer waste

Moonshot candidates:

- count-first / transition-first curriculum
- short-to-long context curriculum
- high-entropy document prioritization

This lane is risky because it is easier to make wrong than to make strong.

## 6. Parameterization of the Core Model

Reference:

- [`train_gpt.py:500`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L500)
- [`train_gpt.py:724`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L724)

Current assumptions:

- RMSNorm without affine
- full RoPE on all head dims
- relu^2 MLP
- plain causal attention
- no value sidecar
- no explicit n-gram prior
- U-Net skips and resid_mix already exist

Why this matters:

- this surface is most of `L_repr`
- if the model class misses the right inductive bias, no optimizer patch fixes it

### 6a. Attention Path

Reference:

- [`train_gpt.py:555`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L555)
- [`train_gpt.py:603`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L603)

Drastic levers:

- XSA on some or all layers
- value residual transport
- attention asymmetry by depth
- selective context mechanisms that help long-range prediction without full KV explosion

Moonshot candidates:

- XSA-all when export headroom exists
- VRL from early value states into later blocks
- per-layer attention mode specialization

### 6b. MLP and Activation

Reference:

- [`train_gpt.py:606`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L606)
- [`train_gpt.py:617`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L617)

Drastic levers:

- LeakyReLU(0.5)^2
- wider MLP funded by compression
- activation that better preserves gradient and quantization robustness

Moonshot candidates:

- LeakyReLU^2 as the default activation
- MLP width pushed above the current frontier template when compression allows
- activation/width specialization by depth

### 6c. Embedding and Explicit Priors

Reference:

- [`train_gpt.py:669`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L669)
- [`train_gpt.py:724`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L724)

Drastic levers:

- explicit bigram or trigram priors
- count-initialized transition heads
- value embeddings or sidecar memories

First-principles angle:

In a 10-minute run, the model may never fully internalize low-order transition structure from scratch. Hard-coding or strongly initializing that structure can free capacity for higher-order patterns.

Moonshot candidates:

- CountInitBigram head
- trigram sidecar with tiny parameter budget
- VE128 funded by better export

## 7. Initialization

Reference:

- [`train_gpt.py:693`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L693)
- [`train_gpt.py:698`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L698)

Current assumptions:

- tied embedding normal init
- zero-init projections
- otherwise mostly default linear init

Why this matters:

- initialization mostly attacks early `L_opt`
- under short horizon, early curvature and signal scale matter much more than in long training

Moonshot candidates:

- orthogonal or spectrum-shaped init
- projection init chosen for quantization friendliness, not just stability
- count- or corpus-informed init for explicit prior modules

This lane looks more helper-grade than lead-grade in the newer frontier.

## 8. Compile, Kernel Choice, and Effective Step Budget

Reference:

- [`train_gpt.py:736`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L736)
- [`train_gpt.py:844`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L844)

Current assumptions:

- compile everything
- flash SDPA only
- warmup by doing real gradient steps and then rewinding

Why this matters:

- this attacks `L_opt` indirectly by increasing updates
- but throughput only matters if it converts into useful updates without hurting compression or code path stability

Drastic levers:

- remove wasted warmup
- backend choice based on score, not only ms/step
- separate compile strategy for different module families

Moonshot candidates:

- compile/warmup redesign that buys real steps
- backend search that includes artifact compressibility
- selective decompilation if guard pressure is harming speed

This is a real lane, but it is not the dominant lane unless it buys a lot of steps.

## 9. Optimizer Split Across Tensor Families

Reference:

- [`train_gpt.py:846`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L846)
- [`train_gpt.py:894`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L894)

Current assumptions:

- one Adam law for embeddings
- one Muon law for all matrix parameters in blocks
- one Adam law for all scalar/control params

Why this matters:

- the model is heterogeneous, but the optimizer split is still coarse

Moonshot candidates:

- embeddings on a different schedule from matrix params
- sidecar memories or value-path modules on their own optimizer
- export-fragile tensors trained with explicitly smaller norm growth

## 10. Schedule and Phase Behavior

Reference:

- [`train_gpt.py:924`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L924)
- [`train_gpt.py:961`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L961)

Current assumptions:

- wallclock-aware linear warmdown
- compile warmup consumes real training compute
- no LR warmup
- no explicit phase transition in objective or export alignment

Why this matters:

- late training should optimize the deployed artifact, not just the raw checkpoint

Moonshot candidates:

- late export-aware phase
- switch objective or optimizer law after a boundary
- checkpoint collection only in the final low-LR region

## 11. Main Loop and Training Objective

Reference:

- [`train_gpt.py:972`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L972)
- [`train_gpt.py:1056`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1056)

Current assumptions:

- pure next-token cross-entropy
- no EMA
- no SWA
- no QAT
- no export-aware loss shaping

Why this matters:

- pure training loss is not the competition objective
- the winning model is the exported model under the final eval policy

Drastic levers:

- EMA / tight SWA
- late QAT or export-aligned fake quant
- loss that internalizes deployment distortion

Moonshot candidates:

- EMA as the exported checkpoint
- late active QAT aligned to the final quantizer
- choose the export checkpoint, not just the last checkpoint

## 12. Serialization and Model Selection

Reference:

- [`train_gpt.py:1068`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1068)
- [`train_gpt.py:1119`](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py#L1119)

Current assumptions:

- save the final weights
- quantize once
- evaluate once

Why this matters:

- the best raw checkpoint, the best EMA checkpoint, and the best quantized checkpoint need not be the same object

Moonshot candidates:

- checkpoint selection by deployed score
- export policy search on one trained checkpoint
- compression-aware pruning after training

## Which Surfaces Could Move `val_bpb` Drastically?

`Drastic` here means likely to matter at the `0.005+` scale, not just a tiny polish gain.

Highest-upside surfaces:

- export/quantization: Full GPTQ, mixed bits, precision map, checkpoint-specific export
- context/eval: exact overlap-aware eval, TTT, stronger context mechanisms
- architecture/value path: XSA-all, VRL, VE128, explicit n-gram priors
- capacity reallocation: spend bytes saved in export on more useful parameters

Medium-upside surfaces:

- optimizer/schedule co-design
- data curriculum
- throughput changes that buy real steps

Mostly helper-grade unless composed:

- standalone init tricks
- standalone label regularization
- standalone compile tweaks

## Bottom Line

If the goal is a moonshot, the path is not "slightly better baseline training."

It is one of:

1. reduce `L_export` much harder than the current script
2. change the model class so it captures more predictable structure per byte
3. use more legal context or adaptation at evaluation
4. convert saved bytes into much more predictive parameters
