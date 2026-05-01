# SP8192 CaseOps + 2560 No-Q/V Adaptive Hedge Token N-Gram

Status: record-track seed42 proof complete, with three runtime-compliant seed proofs.

## Result

Headline seed42 proof:

| Metric | Value |
| --- | ---: |
| Seed42 BPB | 1.06082922 |
| 3-seed mean BPB | 1.06157781 |
| Scored tokens | 47,851,520 |
| Scored bytes | 151,074,499 |
| Docs | 50,000 |
| Max artifact bytes | 15,932,067 |
| Cap margin | 67,933 |

Doc-order hash:

```text
33236cc6bd19fa6b89e06d441d3fcd8eb37dc8540f6a4f2b627b20af10894a41
```

## Per-Seed Proofs

| Seed | BPB | Inner TTT eval | Total eval wallclock | Wrapper wallclock | Runtime status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | 1.06082922 | 544.1s | 566.3s | 585s | under 600s |
| 0 | 1.06158291 | 546.7s | 568.5s | 587s | under 600s |
| 1234 | 1.06232130 | 545.6s | 568.1s | 586s | under 600s |

The headline claim is the seed42 record-track proof plus supporting reproducibility evidence. The 3-seed mean is reported for transparency and is not claimed to beat the displayed leaderboard mean.

## Method

This candidate starts from the PR #1915 quantized artifacts and uses an eval-only follow-up stack:

- lower eval-time per-document TTT LR: `0.000075`
- eval/TTT context length: `2560`
- Q/V LoRA disabled for eval-time TTT
- K/MLP/O/lm_head LoRA active
- normalized token-level causal n-gram Adaptive Hedge scoring overlay

The n-gram overlay scores a normalized distribution over the official SP8192 CaseOps token alphabet. It uses strict-prefix hints only and updates state after scoring the current token.

## Size Accounting

| Component | Bytes |
| --- | ---: |
| max compressed model | 15,874,515 |
| counted `train_gpt.py` wrapper | 57,552 |
| max total | 15,932,067 |
| cap | 16,000,000 |
| margin | 67,933 |

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
- `train_seed42.log` - optimized seed42 proof log.
- `train_seed0.log` - optimized seed0 proof log.
- `train_seed1234.log` - optimized seed1234 selected-scorer proof log.
- `submission.json` - metadata and per-seed results.
- `package_size.json` - package accounting from the selected proof source.
- `eval_data_manifest.json` - final eval data-view proof.
- `ENGINEERING_LOG.md` - engineering log for reviewers.

## Relationship To PR #1915

PR #1915 remains the conservative submitted anchor at 1.06504520 BPB. This submission is a separate follow-up and intentionally does not modify the PR #1915 record folder.
