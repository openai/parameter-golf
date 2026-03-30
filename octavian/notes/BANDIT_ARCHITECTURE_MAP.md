# Bandit Architecture Map

## What Bandit is

Bandit combines two systems:

1. **Crawler base model**
   - Source: `experiments/Bandit/train_gpt.py`
   - Configured by `experiments/Bandit/run.sh`
   - Current intended setup:
     - `USE_CRAWLER=1`
     - `NUM_FLAT_LAYERS=4`
     - `NUM_CRAWLER_LAYERS=1`
     - `CRAWLER_LOOPS=4`
     - `INST_DIM=32`
     - `DELTA_NET_HEADS=0`
     - `CRAWLER_QUANT_INT8=1`

2. **Shared n-gram oracle stack**
   - N-gram eval order 9
   - Shared score-first tables across ranks
   - Cubric 3D warm-start / adaptive alpha logic
   - Complementary training via bigram predictability downweighting

## Base model structure

`CrawlerGPT` is the key architecture.

### Flat section

- A flat encoder/decoder section with skip connections.
- `num_flat_layers=4` means the model first processes tokens through unique blocks.
- Skip connections preserve a U-Net-like path and stabilize reconstruction.

### Crawler section

- `crawler_blocks` are **shared** blocks.
- They are reused for `crawler_loops` passes.
- In Bandit, there is 1 crawler block looped 4 times.
- This is the Frugendorff core: reuse parameters to free budget for width / compression.

### Instruction / FLOW mechanism

- Each crawler pass can receive a loop-specific perturbation.
- Current mechanism is **FLOW**:
  - project current hidden state to a small bottleneck (`loop_inst_proj`, dim=`INST_DIM`)
  - expand back with loop-specific `loop_inst_up[k]`
  - add this to the current state before the shared block fires
- This is better than static preplanned loop offsets because it conditions each loop on the output of the previous loop.

### DeltaNet path

- Optional associative memory module after the crawler blocks.
- Two implementations exist:
  - `DeltaNetMemory`: Python token loop
  - `CanonicalDeltaNet`: FLA chunk delta rule CUDA path
- Important current causality fix:
  - state is **not carried across loops** in `_run_crawler`
  - comments explicitly note cross-loop carry leaks future information
- Current Bandit run script has `DELTA_NET_HEADS=0`, so DeltaNet is disabled in the baseline.

## Compression / post-processing stack

Bandit does not end at training loss.

### Quantization

There is a substantial quant/export subsystem in `train_gpt.py`:

- mixed int6/int8 export
- per-row quantization
- GPTQ quantization
- **loop-aware GPTQ calibration** for crawler models
- special handling for crawler blocks:
  - crawler weights can stay int8 for wider dynamic range
  - motivation: shared weights serve multiple loop contexts and can unravel under narrower quantization

### SWA / EMA / late-stage stabilization

The run config uses:
- `EMA_START_STEP=4400`
- `EMA_DECAY=0.99`
- `SWA_EVERY=50`

This matters because existing research notes strongly suggest the crawler’s small advantage may be damaged or erased during post-processing.

## Oracle / eval stack

### Shared score-first n-gram eval

`eval_val_sliding_hashed_ngram(...)` is not a trivial add-on. It is a major subsystem.

Key properties:
- all ranks share identical table state
- scoring is done before chunk updates
- buckets track context and full token counts by order
- adaptive alpha depends on model entropy
- Cubric 3D adjusts alpha multipliers per:
  - order
  - entropy bin
  - count bin

### Mixer path

There is also a learned mixer head path:
- neural model probability
- n-gram expert probabilities
- learned per-token blending with a neural floor

This means Bandit has at least **three interacting surfaces**:
1. crawler base architecture
2. quantization / export stability
3. oracle blending and eval dynamics

## Current evidence from existing report

From `experiments/RESEARCH_REPORT_crawler_signal_analysis.md`:

- most of the crawler advantage appears to be **width**, not recursion
- recursion signal was reported as near-zero in per-step C/N analyses
- more looping may create early gain that decays
- post-processing appears hostile to shared weights
- there may still be a smaller residual sharing/regularization benefit worth saving

## Initial read of the real optimization problem

Bandit is probably **not** a "make crawler deeper and loop harder" problem.
It looks more like a:

1. preserve the tiny real benefit of sharing
2. prevent post-processing from destroying it
3. couple the preserved signal to the oracle more effectively
4. accelerate the hot path enough to search the space aggressively

## Likely hot spots worth instrumenting

1. `_run_crawler(...)`
2. loop instruction path (`loop_inst_proj`, `loop_inst_up`)
3. DeltaNet path when enabled
4. loop-aware GPTQ calibration and crawler int8 policy
5. shared n-gram scoring / Cubric updates
6. learned mixer path
