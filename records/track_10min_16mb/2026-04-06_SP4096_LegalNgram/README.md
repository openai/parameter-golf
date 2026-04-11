# SP4096 + Depth Recurrence + Parallel Residuals + Legal N-Gram

**val_bpb = 1.08457715** | **seed 42** | **2-GPU 40-minute probe** | **15,967,527 bytes**

This folder starts from the original `#1333` SP4096 / recurrence / parallel-residual stack and evaluates a separate prefix-only legal n-gram path with `EVAL_MODE=ngram` and `SLOT_ENABLED=0`.

## Result

| Metric | Value |
|------|------:|
| Pre-quantization post-EMA BPB | 1.09390451 |
| Int6 roundtrip BPB | 1.10589735 |
| Sliding-window BPB | 1.08719574 |
| **Legal n-gram BPB** | **1.08457715** |
| **N-gram gain vs sliding** | **-0.00283638** |
| Total submission size | 15,967,527 |

## Main Idea

The training stack stays with the original SP4096 recurrent base. The only new scoring path is a legal n-gram overlay:

1. Build prefix-only token / within-word / word-start experts from already-seen tokens.
2. Run the frozen language model normally to obtain full-vocab logits.
3. Apply a one-token bias from the chosen expert.
4. Renormalize over the full vocabulary.
5. Score the current token exactly once in a single left-to-right pass.

## Legal N-Gram Details

- Prefix-only state updates in `online_ngram_state.c`
- Token n-gram expert
- Within-word continuation expert
- Word-start expert
- One-token logit tilt plus full-vocab renormalization
- No target-conditioned gating
- No two-pass rescoring
- No weight updates during evaluation

```

## Notes

- This result keeps the original `#1333` training base and swaps the final evaluator to legal n-gram.
- The measured gain is real, but modest: roughly `0.0028 BPB`.
- The current implementation is still CPU-heavy in the n-gram blending path.

## Credits

PR #1218 @clarkkev, PR #1285 @dexhunter, PR #1204 @msisovic, PR #1289 @MatoTeziTanka, PR #1260 @dexhunter, PR #1217 @bigbag, PR #1333 @aryanbhosale, PR #1145 @g-w1
