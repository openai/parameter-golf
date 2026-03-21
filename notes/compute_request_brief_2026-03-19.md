# Parameter Golf Compute Request Brief

## Objective

Train a better `<=16,000,000` byte artifact for the Parameter Golf challenge under the real `10` minute `8xH100` training cap.

The central lesson from local work is that this is a fixed-time efficiency problem, not just a parameter-count problem. The best path is likely:

1. more useful tokens per second,
2. more model quality per byte,
3. only then more elaborate architecture.

## What We Already Learned Locally

Negative map:
- full-depth AttnRes loses under matched wallclock
- selective AttnRes is less bad, but still loses to baseline locally
- naive shared-depth recurrence loses
- naive latent memory loses
- naive recurrent memory loses
- naive widening loses
- naive PTQ `int4` is too lossy as a direct drop-in

Positive map:
- quantization is still a real byte lever
- selective late routing is structurally more plausible than global routing
- wallclock-aware comparisons matter more than equal-step comparisons

Conclusion:
- stop spending time on broad naive architecture sweeps
- prioritize throughput, curriculum, and compression-aware model design

## Proposed Research Tracks

### Track 1: Fixed-Time Training Efficiency

Primary idea:
- variable sequence length curriculum / dataset decomposition

Rationale:
- likely the cleanest way to buy more useful training in a hard time budget
- lower sequence lengths early should improve throughput and optimization
- full sequence length is still used later and at evaluation

Secondary ideas:
- optimizer/schedule tuning on the fast baseline backbone
- H100-native mixed precision / FP8 if feasible

### Track 2: Parameter-Efficient Backbone

Primary idea:
- shared-backbone recurrence `v2`

Not:
- naive full block sharing

Instead:
- share heavy matrices
- keep tiny per-depth controls unique
- add stronger step conditioning

Secondary idea:
- Transformer-XL / compressive segment recurrence style memory

### Track 3: Compression Co-Design

Primary idea:
- quantization-friendly backbone + export

Not:
- plain post-training `int4` on an unchanged model

Instead:
- mixed `4/8-bit`
- smoother / more quantization-friendly parameter distributions
- byte savings spent on small targeted capacity increases

## Immediate H100 Plan

Phase 1:
- reproduce and profile baseline on CUDA
- measure throughput bottlenecks and actual training volume in the real stack

Phase 2:
- run sequence-length curriculum experiments first
- compare under matched wallclock

Phase 3:
- run small optimizer/schedule sweep on the best fast backbone

Phase 4:
- only then revisit backbone changes:
  - shared-trunk recurrence `v2`
  - segment recurrence
  - quantization-aware variants

## Why Compute Credits Are Needed

The local Mac work is good for rejecting ideas cheaply, but the actual competition objective depends on:
- CUDA/H100 throughput,
- wallclock-limited training dynamics,
- post-quantized artifact behavior in the real training stack.

Several of the most promising directions are inherently GPU-bound:
- sequence-length curriculum under large token throughput
- FP8 / systems efficiency work
- compression-aware retraining
- richer shared-backbone variants

## Deliverables We Plan To Produce

- a remote branch with reproducible experiment harness updates
- short-run CUDA pilot logs
- clear matched-wallclock comparisons
- one or two serious candidate branches for full `8xH100` runs

## Current Recommendation

The best next engineering move is:

1. implement variable sequence length curriculum,
2. validate it locally,
3. move it to the CUDA path,
4. request compute for the broader three-track program above.
