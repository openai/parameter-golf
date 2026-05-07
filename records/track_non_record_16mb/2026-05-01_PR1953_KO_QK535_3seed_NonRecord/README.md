# Non-Record Submission: PR1953 K+O-only TTT + QK_GAIN_INIT=5.35

This is a **non-record** submission documenting a clean follow-up to the PR #1953 stack. The best single seed was promising (`s42 = 1.05767136` val_bpb), but the 3-seed mean regressed to `1.05857803`, so this is submitted as a reproducibility / negative-variance finding rather than a record claim.

## Summary

Starting from the PR #1953-style stack, this run keeps score-first phased TTT but restricts the TTT LoRA adapters to **K and O only** while increasing `QK_GAIN_INIT` to `5.35`.

Key env changes relative to the PR #1953 family:

```bash
EVAL_SEQ_LEN=2688
TTT_EVAL_SEQ_LEN=2688
TTT_MASK=no_qv
TTT_K_LORA=1
TTT_MLP_LORA=0
TTT_O_LORA=1
TTT_LOCAL_LR_MULT=0.75
QK_GAIN_INIT=5.35
```

## Results

All runs used 8xH100 SXM, 600s train budget, 600s eval budget, CaseOps data, and phased score-first TTT.

| Seed | Train step | Pre-quant BPB | Quantized BPB | TTT BPB | val_loss | TTT eval time | Artifact bytes |
|------|-----------:|--------------:|--------------:|--------:|---------:|--------------:|---------------:|
| 42 | 4927 | 1.06087844 | 1.06891249 | **1.05767136** | 2.31457858 | 434.877s | 15,978,954 |
| 314 | 4927 | 1.06164848 | 1.06975107 | **1.05854316** | 2.31648639 | 390.025s | 15,983,413 |
| 7 | 4900 | 1.06277681 | 1.07073062 | **1.05951958** | 2.31862318 | 391.064s | 15,978,698 |
| **Mean** | - | **1.06176791** | **1.06979806** | **1.05857803** | **2.31656272** | **405.322s** | **15,980,355** |
| **Std** | - | **0.00095480** | **0.00090998** | **0.00092460** | **0.00202338** | **25.601s** | **2,651** |

Maximum artifact size was `15,983,413` bytes, leaving `16,587` bytes of headroom under the 16,000,000-byte cap. Maximum final TTT eval time was `434.877s`; the strict verifier's summed diagnostic+TTT eval-time fields max at `455.926s`.

## Interpretation

The K+O-only mask looked high-EV from the seed-42 result, but the seed-314 and seed-7 reruns show that the effect is not stable enough to support a record submission. The useful finding is that removing MLP LoRA from TTT does not catastrophically break the PR #1953 stack and may reduce variance on some seeds, but the 3-seed average does not beat the strongest open clean PR frontier.

## Rule Compliance Notes

- Artifact cap: all three seeds are below `16,000,000` bytes.
- Train budget: all three seeds stop via `stopping_early: wallclock_cap` at about `596s`.
- Eval budget: all three `quantized_ttt_phased` evals are below `600s`.
- Causal scoring: inherited PR #1953-style score-first phased TTT; each token is scored before any local TTT update can use it.
- BPB accounting: canonical byte-level BPB path from the included `train_gpt.py` snapshot; full validation shards are scored once.

## Packaging Checks

- `train_gpt.py` size is `159,688` bytes and matches the code size reported by all three seed logs.
- Hyperparameter blocks differ only by `seed`, `run_id`, and `logfile`.
- No absolute local paths or placeholders are present in the submitted logs or metadata.
- The CaseOps prep script was smoke-tested from a clean `/tmp` copy on a 10-doc sample; it produced a validation shard with 10 BOS markers.

## Included Files

- `train_gpt.py` - exact 159,688-byte code snapshot used by all three runs.
- `run.py` - minimal local runner wrapper.
- `train_seed42.log`, `train_seed314.log`, `train_seed7.log` - full logs with source snapshot and final metrics.
- `prepare_caseops_data.py`, `lossless_caps.py`, `tokenizers/` - data preparation support files.
- `submission.json` - non-record metadata.
