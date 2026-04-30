# SP8192 CaseOps + 2560 No-Q/V Adaptive Hedge Token N-Gram

Status: seed42 record-track proof complete. Seed0 and seed1234 follow-up runs were launched in parallel and will be appended if they finish in time.

## Result

Current optimized seed42 proof:

| Metric | Value |
| --- | ---: |
| BPB | 1.06082922 |
| Prior same-method exploratory BPB | 1.06083091 |
| Scored tokens | 47,851,520 |
| Scored bytes | 151,074,499 |
| Docs | 50,000 |
| Artifact bytes | 15,929,395 |
| Cap margin | 70,605 |

Doc-order hash:

```text
33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41
```

## Runtime

The optimized seed42 package proof clears the 10-minute evaluation cutoff on 8xH100 SXM.

| Variant | BPB | Inner eval | Total eval | Wrapper wallclock | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| batch 32 before log-prob reuse | 1.06083116 | 588.5s | not retained | 631s | score/package good, wrapper over target |
| batch 48 | 1.06083288 | 636.9s | not retained | 679s | slower from memory pressure/load imbalance |
| optimized batch 32 with hint log-prob reuse | 1.06082922 | 544.1s | 566.3s | 585s | selected proof |

## Method

This candidate starts from the PR #1915 seed42 quantized model and uses an eval-only follow-up stack:

- lower eval-time per-document TTT LR: `0.000075`
- eval/TTT context length: `2560`
- Q/V LoRA disabled for eval-time TTT
- K/MLP/O/lm_head LoRA active
- normalized token-level causal n-gram Adaptive Hedge scoring overlay

The n-gram overlay scores a normalized distribution over the official SP8192 CaseOps token alphabet. It uses strict-prefix hints only and updates state after scoring the current token.

## Size Accounting

| Component | Bytes |
| --- | ---: |
| compressed model | 15,872,234 |
| counted `train_gpt.py` wrapper | 57,161 |
| total | 15,929,395 |
| cap | 16,000,000 |
| margin | 70,605 |

The custom n-gram Python/C logic is embedded into the counted `train_gpt.py` wrapper. No uncounted helper files are required.

## Evaluation Data Safety

The final package path uses a validation-only data view:

- train shards visible during final eval: `0`
- validation token shards: `5`
- validation byte shards: `5`
- train-shard glob/listing skipped in final eval mode

## Legality Notes

- Official SP8192 CaseOps token alphabet.
- Full normalized token distribution after the n-gram tilt.
- Per-document score-first TTT.
- `TTT_WARM_START_A=0`; LoRA resets per document.
- `PHASED_TTT_PREFIX_DOCS=0`; no validation prefix adaptation.
- No global validation SGD or cross-document adaptive state.
- No byte PPM, custom tokenizer, target-conditioned lookup, or external network dependency.

## Included Files

- `train_gpt.py` - counted self-contained wrapper.
- `train_seed42.log` - optimized batch-32 seed42 proof log.
- `submission.json` - seed42 metadata.
- `package_size.json` - package accounting from the selected proof.
- `eval_data_manifest.json` - final eval data-view proof.
- `ENGINEERING_LOG.md` - engineering log for reviewers.

## Relationship To PR #1915

PR #1915 remains the conservative submitted anchor at 1.06504520 BPB. This submission is a separate follow-up and intentionally does not modify the PR #1915 record folder.
