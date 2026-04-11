# Hailmary Rebase Hypotheses

Date: 2026-04-09

These are the new moonshot families after the frontier moved to:

- `SP4096/SP8192`
- `full GPTQ`
- `SDClip`
- `GPTQ embeddings`
- `depth recurrence`
- `MuonEq-R`

This file is not about polishing the old SP1024 stack.
It is about where high-upside mass still exists **after** adopting the new base.

## What Counts As A Moonshot Now

A new `hailmary` idea should satisfy at least one of:

- changes the tokenizer/representation/export trade fundamentally
- changes the recurrence/depth story fundamentally
- changes the TTT paradigm materially
- changes the deploy/eval object materially
- creates a byte-neutral way to buy much more predictive structure

## HM-61: Recurrence Program Search

- Mechanism:
  - do not just choose “recurrence on/off”
  - search over:
    - which blocks loop
    - how many loops
    - when loops activate
    - whether parallel residuals apply only in deeper virtual layers
- Why:
  - the new frontier treats recurrence as a first-order architecture knob, not a side detail
- Dominant lane:
  - architecture + throughput
- Expected impact:
  - large
  - `0.006 - 0.018 BPB`
- Failure mode:
  - quantization and throughput debt erase the virtual-depth gain

## HM-62: Tokenizer–Architecture Co-Design

- Mechanism:
  - compare `SP4096` vs `SP8192` jointly with:
    - recurrence depth
    - bigram sidecars or no sidecars
    - embedding quant format
- Why:
  - the new frontier shows vocabulary size is not just preprocessing; it changes what explicit local structure is still needed
- Dominant lane:
  - representation + export
- Expected impact:
  - large
  - `0.008 - 0.020 BPB`
- Failure mode:
  - code and export budget become the bottleneck instead of modeling quality

## HM-63: GPTQ Pipeline Redesign

- Mechanism:
  - move beyond “full GPTQ exists”
  - search:
    - embedding GPTQ strategy
    - SDClip variants
    - AR self-generated calibration variants
    - layer-sensitive damping / ordering
- Why:
  - the frontier is now quantization-pipeline dominated again, but in a more specialized way
- Dominant lane:
  - export / quant damage
- Expected impact:
  - large
  - `0.005 - 0.015 BPB`
- Failure mode:
  - added quant sophistication yields only marginal gains over already-good full GPTQ

## HM-64: Product-Key / Learned Local Priors

- Mechanism:
  - replace plain BigramHash with:
    - product-key bigram
    - count-initialized product bigram
    - SP8192-with-light-local-prior hybrid
- Why:
  - the latest frontier suggests larger vocab partly substitutes for explicit bigram structure, but not necessarily completely
- Dominant lane:
  - architecture
- Expected impact:
  - medium
  - `0.003 - 0.010 BPB`
- Failure mode:
  - larger vocab already absorbs the gain and the sidecar becomes redundant

## HM-65: Pre-Quant TTT Family Search

- Mechanism:
  - treat pre-quant TTT as a full design family:
    - freeze depth
    - epoch count
    - optimizer type
    - per-block LR groups
    - discriminative TTT variants
- Why:
  - TTT is no longer speculative; it is one of the largest remaining gains
- Dominant lane:
  - deploy / eval
- Expected impact:
  - large
  - `0.010 - 0.035 BPB`
- Failure mode:
  - TTT overfits short-horizon evaluation or becomes too expensive for the allowed budget

## HM-66: Eval-Time Bias / Adapter Layer

- Mechanism:
  - push beyond plain ETLB into:
    - token-bias only
    - bias + low-rank last-layer adapter
    - chunk-wise latent mask on top of a frozen base
- Why:
  - ETLB and LatentMask both suggest there is still cheap eval-time headroom after the core stack is strong
- Dominant lane:
  - eval policy
- Expected impact:
  - medium
  - `0.002 - 0.008 BPB`
- Failure mode:
  - extra eval-time adaptation is too weak compared with full pre-quant TTT or too expensive to justify

## HM-67: Compression-Aware Recurrence

- Mechanism:
  - explicitly design recurrent stacks for compressibility:
    - loop selection chosen with quant damage in mind
    - looped layers get different export treatment
    - noisy/late QAT aligned to recurrent blocks
- Why:
  - the merged recurrence research says the main failure mode is quantization compounding through repeats
- Dominant lane:
  - architecture + export
- Expected impact:
  - medium to large
  - `0.004 - 0.012 BPB`
- Failure mode:
  - extra complexity replicates what strong full GPTQ already solves

## HM-68: SLOT-Lite / Frozen-Base Test-Time Paradigm

- Mechanism:
  - instead of full SLOT, search lighter frozen-base per-sample adaptation:
    - bias-only
    - low-rank latent mask
    - hidden-state delta without full paradigm jump
- Why:
  - the SLOT result is too large to ignore, but it may be overkill as a first port
- Dominant lane:
  - eval / TTT
- Expected impact:
  - large if real
  - `0.010 - 0.050 BPB`
- Failure mode:
  - paradigm mismatch; lightweight variants do not inherit enough of SLOT’s benefit

## Moonshot Priority

The highest-value new hailmary families are:

1. `HM-62 Tokenizer–Architecture Co-Design`
2. `HM-63 GPTQ Pipeline Redesign`
3. `HM-65 Pre-Quant TTT Family Search`
4. `HM-61 Recurrence Program Search`

Those are where the new frontier signal is strongest.
