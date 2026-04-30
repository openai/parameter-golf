# PR 1271 — Scylla Byte Accounting Audit (post-mortem of PR #1184)

**Author:** (see git log, same author as PR 1272 — negative results series)
**Claimed BPB:** PR #1184's Scylla stack actually lands at 1.1289 with corrected meta, not claimed 0.9491.
**Seeds:** 1337 used for audit
**Hardware:** 8xH100 (same as #1184)

## Files retrieved
- `records__track_10min_16mb__2026-04-02_Scylla_Byte_Accounting_Audit__README.md`
- `records__track_10min_16mb__2026-04-02_Scylla_Byte_Accounting_Audit__retokenize_proper.py`

Note: `correct_meta.npz` is binary and was not extracted to the archive.

## Claimed changes (from README, verbatim)

> An audit of PR #1184's Scylla tokenizer byte accounting. I ran their exact code with corrected candidate.meta.npz and proper val data. The result: 1.1289 BPB, not 0.9485. The sub-1.0 claim was a measurement error.

The bug: PR #1184's candidate.meta.npz has 27 byte-fallback tokens (IDs 75-101) with base_bytes=3 instead of 1. Overcounts the byte denominator by ~4%.

Originally flagged by @dexhunter on PR #1143 (closed for exactly this reason); PR #1184 reuses the same buggy npz.

Results table:
| | PR #1184 (buggy) | Corrected |
|---|---|---|
| Val tokens | 62,363,648 | 62,609,408 |
| Val NLL | 1.928 | 1.916 |
| Sliding BPB | 0.9491 | 1.1289 |
| Train shards | 194 | 207 |

Gap decomposition (+0.180 BPB total):
- Model quality (NLL diff): +0.010
- Byte accounting: +0.133
- Val text/token boundary diffs: +0.037
- **93% of gap is byte accounting, not model quality.**

Conclusion: With corrected accounting, Scylla stack lands at ~1.13 BPB, essentially same as SP1024 stack ~1.11-1.12. Requested review from @0hq @valerio-oai.
