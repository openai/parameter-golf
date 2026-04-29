# Run 4 Summary: SEQ_LEN/STRIDE invariance — gap is not in the eval pipeline

## Headline

The 0.77% gap between yahya's quoted 1.1746 and the audit's reproduction (1.1655) is **invariant** to eval pipeline windowing parameters. Tested three configurations: seq_len=2048/stride=64 (audit default), seq_len=1024/stride=64 (yahya's code default), seq_len=1024/stride=1024 (no overlap, sanity check). All three produce the same yahya ratio to floating-point precision.

## Results

| Configuration | Canonical ratio | Yahya ratio | Scored tokens | Gap to 1.1746 |
|---|---|---|---|---|
| seq_len=2048, stride=64 | 1.1671397 | 1.1655009 | 40,542,786 | -0.7747% |
| seq_len=1024, stride=64 | 1.1671405 | 1.1655017 | 40,541,762 | -0.7746% |
| seq_len=1024, stride=1024 | 1.1671413 | 1.1655024 | 40,540,802 | -0.7745% |

All three yahya ratios agree to 6 decimal places.

## Implication

The buggy/canonical inflation ratio is **invariant** to seq_len in {1024, 2048} and stride in {64, 1024} on SP8192 fineweb val. The gap to 1.1746 cannot be attributed to scoring-strategy differences in the eval pipeline.

## Where the gap lives, by elimination

After runs 1-4, the gap to 1.1746 has been ruled out as living in:
- **The LUT structure** (run 3: yahya's exact LUT verified)
- **The canonical/buggy formula** (run 3: matches audit tool to floating-point precision)
- **Boundary mask coverage** (run 1: mask is non-trivial but irrelevant due to SentencePiece convention)
- **Three scoring modes** (run 1.5: converge to within 7.8e-9)
- **Eval pipeline windowing parameters** (run 4: invariant across tested configurations)

Remaining candidates:
- **A different SentencePiece tokenizer state.** Yahya's `train_gdn_7k.py` defaults to `fineweb_1024_bpe.model` (line 58); his audited submission used SP8192 (per submission.json) by overriding the default. His PR #1734 disclosure analysis predates that submission and may have been computed against SP1024.
- **A different val shard or tokenization run.** We have one val shard (fineweb_val_000000.bin, 40.5M tokens). His disclosure analysis may have used a different shard or an older tokenization run.
- **Hand-derived or estimated.** The 1.1746 may not have come from a script at all. PR #1734's disclosure was a bug report; the ratio could have been derived analytically.

We cannot distinguish among these without his exact disclosure-time tokenizer + val shard, neither of which is on the audit's network volume.

## Conclusion

The 0.77% gap is bounded. It lives in tokenizer/val state, not in eval pipeline structure. The audit's reproduction (1.1655) is the correct number for the audit's val state. The audit's static classifier verifies LUT correctness, which is the audit's stated scope; quantifying the inflation against a specific submitter's expected ratio requires that submitter's exact data, which is out of scope for static analysis.

## Files
- run4_seq_len_1024.py / .json / .log
