# DEPRECATED — Do Not Use As Base For New Experiments

Scripts here use the **old n-gram eval stack**:
- `eval_val_sliding_hashed_ngram` + `TrainNgramOracle`
- Incompatible with competition eval harness
- BPB numbers are NOT comparable to leaderboard scores

## What to use instead

Always start new experiments from:
```
experiments/X_wing_cubric_lite/xwing_green_1/train_gpt.py
```

That script uses:
- `BackoffNgramMixer` + `eval_val_sliding_ttt`
- GPU-offloaded n-gram system
- Score-first TTT — matches competition eval protocol

## What's preserved here

| Experiment | What was learned |
|------------|-----------------|
| FX_Wing | Content-derived loop instructions (perturbation); quant gap +2.93 BPB |
| FX_Wing_Delta | Flow instructions (recompute inst from current x); quant gap +0.006 BPB — H0 confirmed |
| FX_Wing_Delta_DN | DeltaNet + gradient checkpointing fix (tbptt_chunk=64); not evaluated on correct stack |
| FX_Wing_Sigma | Entropy-gated instructions plan; not implemented |

## The architecture insight is valid

The **flow instruction** result (FX_Wing_Delta) is real:
- Quant gap: +2.93 → +0.006 BPB
- The fix: recompute `inst_k = up_k(proj(x))` from current x at each loop, vs static pre-planned instructions

This needs to be ported onto the xwing_green_1 base to get competition-comparable numbers.
