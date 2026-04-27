# Stack Integration + Legal TTT + Parallel Muon

**Reviewer-ready submission folder:** `records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/`

This folder packages the promoted 2026-03-28 submission artifact as a self-contained review surface. The executable `train_gpt.py` here is byte-identical to the already-proven 2026-03-23 record script, so the three copied seed logs below are inherited evidence for this promoted folder rather than newly rerun 2026-03-28 training jobs.

## Audited submission summary

- **Canonical metric:** `legal_ttt`
- **Three-seed mean val_bpb:** 1.119367
- **Three-seed std:** 0.000464
- **Conservative max total submission size:** 15,990,006 bytes
- **Artifact-size contract:** under `16,000,000` bytes
- **Provenance status:** byte-identical promoted/proven `train_gpt.py` scripts (`sha256 df7aa00cc6a0c959fbc95f2665ec4ef6b7f869f05d20f4523510a1ad16f1e674`)

## Inherited 3-seed evidence

The three logs in this folder were copied from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` after `experiments/audit_submission_package.py` confirmed that:

1. each log resolves to `chosen_metric: legal_ttt`
2. the promoted 2026-03-28 `train_gpt.py` matches the proven 2026-03-23 `train_gpt.py` byte-for-byte
3. every audited artifact stays within the `16,000,000` byte envelope

| Seed | Chosen metric | `val_bpb` | `Total submission size int6+lzma` | Evidence file in this folder | Canonical alias? |
|------|---------------|-----------|-----------------------------------|------------------------------|------------------|
| 1337 | `legal_ttt` | 1.1192 | 15,977,386 | `train_seed1337.log` | No |
| 42 | `legal_ttt` | 1.1200 | 15,876,510 | `train_seed42.log` | No |
| 2025 | `legal_ttt` | 1.1189 | 15,990,006 | `train_seed2025.log` | Yes (`train.log` alias) |
| **Mean / std** | `legal_ttt` | **1.119367 +/- 0.000464** | **max 15,990,006** | copied 3-seed evidence | -- |

## Log inventory and the `train.log` alias

This folder intentionally carries the copied per-seed evidence plus one reviewer-friendly canonical alias:

- `train_seed1337.log` - copied audited seed log
- `train_seed42.log` - copied audited seed log
- `train_seed2025.log` - copied audited seed log
- `train.log` - reviewer-friendly alias for `train_seed2025.log`; these two files are byte-identical and share sha256 `408f9895815ad8f2317aa42a14b4b1953df9828a480d1bd572b630d487c8f3ff`

A reviewer who reads only `train.log` still sees a real accepted proof log. A reviewer who reads all three `train_seed*.log` files gets the full inherited evidence behind the mean/std summary.

## Exact run contract

Run from the repository root so the default dataset and tokenizer paths resolve correctly.

```bash
MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=9000 \
EVAL_STRIDE=64 \
TTT_ENABLED=1 \
python records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_gpt.py \
  > records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log 2>&1

python experiments/verify_run.py \
  records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train.log
```

Expected verifier behavior:

- prints `chosen_metric: legal_ttt`
- prints `val_bpb: <value>` from the accepted metric path
- rejects logs that only expose fallback metrics

The promoted script also supports the non-TTT fallback path `TTT_ENABLED=0`, in which case `experiments/verify_run.py` falls back to `final_int6_sliding_window_s64`. That fallback is part of the runtime contract, but the submission numbers in this folder come from the audited `legal_ttt` runs above.

## Submission metadata source

`submission.json` in this folder was generated from the audit payload rather than hand-entered numbers. It records:

- the exact audited mean `val_bpb`
- the audited `val_bpb` standard deviation
- the conservative max `bytes_total` across the three copied seed logs
- the current `train_gpt.py` code size for the promoted artifact

To regenerate the package summary mechanically, rerun:

```bash
python experiments/audit_submission_package.py \
  records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_seed1337.log \
  records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_seed42.log \
  records/track_10min_16mb/2026-03-28_StackIntegration_LegalTTT_ParallelMuon/train_seed2025.log
```

## Relationship to S04 random-map adapters

`records/track_non_record_16mb/2026-03-28_RandomMapAdapters_Stack/` is a separate non-record package. It documents an S04 random-map adapter experiment with its own non-TTT comparison contract and placeholder/local-runtime evidence. This S04 package is not part of the submission evidence. It does not supply the numbers, logs, or provenance for this promoted S05 submission folder.

## Key artifact paths

- `train_gpt.py` - promoted submission script
- `README.md` - reviewer-facing provenance and run contract
- `submission.json` - audit-derived submission metadata
- `train.log` - canonical reviewer-facing alias
- `train_seed1337.log`, `train_seed42.log`, `train_seed2025.log` - copied inherited evidence
