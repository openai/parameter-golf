# Experiment Feedback
# The main agent writes results here. The Researcher reads them.
# Format: one entry per experiment.

# Updated: 2026-04-06

## Current Best: 1.2490 BPB (exp13, VE128 on layers 9,10)
## Quick Baseline: 4.2889 BPB (107 steps, 60s)

## Recent Results (apr6 session)

### exp15: SWA (scale<1.0, every=50) → DISCARD 1.3150 BPB
- SWA averaging over entire warmdown dilutes good weights with earlier worse weights
- Need scale<0.2 trigger like SOTA to only capture final weights

### exp16: GPTQ-lite clip search (5 percentiles) → DISCARD 1.2576 BPB  
- Likely seed variance (different commit = different seed via eval.sh)
- The technique is validated by SOTA, but our result inconclusive

### exp17: warmdown 500→620 → DISCARD 1.2579 BPB
- 500 is already optimal for ~1066 steps on 1xH100

### exp18: Full Hessian GPTQ + AR self-gen → RUNNING
- AR generation (16 seqs x 512 tokens) takes ~13s
- Hessian collection takes ~17s  
- GPTQ quantization takes additional time per weight matrix
- Full eval in progress

## Notes
- 1xH100, ~570ms/step, ~1068 steps in 600s
- Current artifact: 13.2 MB (2.7 MB headroom)
- 3 consecutive discards → go bolder: trying Full GPTQ (significant structural change)
- Key insight: seed variance makes ±0.01 BPB changes hard to detect

