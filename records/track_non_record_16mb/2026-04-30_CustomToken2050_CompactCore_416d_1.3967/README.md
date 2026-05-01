# Non-Record: CustomToken2050 Compact Core 416d (val_bpb: 1.3967)

This is a non-record, in-progress submission exploring a compact custom-token modeling path. The tokenizer implementation and private training-loss helpers are proprietary and intentionally excluded from this public artifact; this run uses pre-tokenized token IDs and a compact dense model trained against those IDs.

The goal of this entry is not to claim SOTA. It documents an early two-day experiment showing that a small dense model can reach a useful BPB range with a specialized token representation while preserving significant runtime memory headroom for future retrieval, reference-cache, or adaptive-context systems.

## Result

| Metric | Value |
|---|---:|
| validation BPB | `1.396683` |
| validation loss | `1.9260` |
| shape | `416/4/9/1kv`, MLP3 |
| vocabulary | `2050` custom token IDs |
| batch tokens | `1,048,576` |
| final step | `6000` |
| continuation runtime | `1551.3s` on 1x H100 SXM |
| throughput | `~1.35M tokens/sec` on 1x H100 |
| parameter count | `14,108,260` |
| packed model estimate | `~13.8 MB` with current int8+zlib path |

## Compute Context

This was a low-budget two-day sprint. Most early iteration and curve selection started locally on an `AMD Radeon RX 7900 XTX`; the H100 portion was used only after the direction looked worth testing at higher batch size.

As of this temporary submission, the cloud spend for the H100 SXM work was about `$7.50` at approximately `$1.505/hour`, with about `$12.50` remaining from an initial `$20` credit load.

## Run History

The best run was a continuation from a 4000-step checkpoint:

- step 4000: `1.401375`
- step 4300: `1.400610`
- step 4800: `1.399954`
- step 5000: `1.399820`
- step 5400: `1.399401`
- step 5700: `1.399090`
- step 5800: `1.397688`
- step 5900: `1.397380`
- step 6000: `1.396683`

## Notes

This entry intentionally avoids publishing the tokenizer implementation or private loss helpers. The public result should be read as evidence for the compact-token modeling direction, not as a fully open tokenizer release or a leaderboard-reproducible record. The dense model itself is small, and the observed runtime memory headroom is part of the research motivation.

This was started and run as a two-day sprint, so the result should be read as an early proof of direction rather than a mature optimized submission.

## Reproducibility Boundary

The included `train_gpt.py` is a public-safe executable disclosure stub. It is included because the submission instructions request a `train_gpt.py` file in each record folder, but the full trainer depends on private tokenization and loss code that is not part of this public submission.

Future work for this path includes:

- EMA/SWA checkpoint averaging
- depth recurrence / shared-block recurrence
- improved quantization and packing
- teacher or advisor transfer into the compact-token student
- retrieval/reference-cache systems that use the saved runtime memory

## Included Files

- `README.md` - this summary
- `submission.json` - metadata
- `train.log` - sanitized training log with proprietary tokenizer identifiers removed
- `train_gpt.py` - executable public-safe disclosure stub
