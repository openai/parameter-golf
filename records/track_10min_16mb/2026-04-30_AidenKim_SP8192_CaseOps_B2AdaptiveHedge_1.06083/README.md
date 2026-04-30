# Draft: SP8192 CaseOps + B2 Adaptive Hedge Token N-Gram

Status: draft pending final runtime proof. This folder is staged as a ready-to-edit PR candidate, not a final record claim unless the optimized proof clears the official runtime requirement.

## Result

Current best packaged seed42 proof:

| Metric | Value |
| --- | ---: |
| BPB | 1.06083116 |
| Prior same-method exploratory BPB | 1.06083091 |
| Scored tokens | 47,851,520 |
| Scored bytes | 151,074,499 |
| Docs | 50,000 |
| Artifact bytes | 15,929,336 |
| Cap margin | 70,664 |

Doc-order hash:

```text
33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41
```

## Runtime Status

The package is under the 16 MB cap and reproduces the score, but runtime is still being tightened.

| Variant | BPB | Inner eval | Wrapper wallclock | Note |
| --- | ---: | ---: | ---: | --- |
| batch 16 | not kept | 633.8s | 674s | too slow |
| batch 24 | not kept | 621.5s | 663s | too slow |
| batch 32 | 1.06083116 | 588.5s | 631s | best current runtime point; wrapper over target |
| batch 48 | 1.06083288 | 636.9s | 679s | slower from memory pressure/load imbalance |

An optimized batch-32 proof is being tested with mathematically equivalent hint-log-prob reuse. If that proof clears runtime and reproduces the score/count/hash, this draft can be converted into the final PR. If it does not, this folder should be treated as a non-record / runtime-caveated follow-up.

## Method

This candidate starts from the PR #1915 Path A seed42 quantized model and uses an eval-only Path A+ stack:

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
| counted `train_gpt.py` wrapper | 57,102 |
| total | 15,929,336 |
| cap | 16,000,000 |
| margin | 70,664 |

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
- `train_seed42.log` - current batch-32 proof log.
- `submission.json` - draft metadata; update runtime fields after optimized proof.
- `package_size.json` - package accounting from the current proof.
- `eval_data_manifest.json` - final eval data-view proof.
- `ENGINEERING_LOG.md` - concise engineering log for reviewers.

## Relationship To PR #1915

PR #1915 remains the conservative submitted anchor at 1.06504520 BPB. This draft is a separate Path A+ follow-up and intentionally does not modify the PR #1915 record folder.
