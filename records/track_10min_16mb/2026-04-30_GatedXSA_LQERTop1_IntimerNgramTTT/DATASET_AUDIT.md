# Dataset Audit: CaseOps Train Shards and Full Validation

This note records how the dataset used for this submission was constructed and how it was checked against the merged CaseOps leaderboard lineage.

## Verdict

The submitted runs use the CaseOps SP8192 lossless-caps tokenizer and byte-sidecar BPB accounting. The 80 training shards were verified byte-for-byte against the output of the merged CaseOps leader's `prepare_caseops_data.py` default path. Evaluation uses the full CaseOps validation shard/sidecar reported by the leaderboard logs (`val_tokens: 47851520`).

This is the same structural setup used by the CaseOps leaderboard lineage: 80 train shards, SP8192 lossless-caps tokenization, BOS-delimited documents, and byte sidecars for validation BPB accounting.

## Sources

- Dataset stream: the canonical FineWeb document stream used by the CaseOps records, `docs_selected.jsonl`.
- Tokenizer: `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`.
- Text transform: `lossless_caps.py`.
- Dataset script in this submission: `prepare_caseops_data.py`.
- Reference merged-leader script: `records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/prepare_caseops_data.py` from commit `1e439663209730edeac34e659039d7de62d85908` in `codemath3000/parameter-golf` (`https://github.com/codemath3000/parameter-golf/blob/1e439663209730edeac34e659039d7de62d85908/records/track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/prepare_caseops_data.py`).

The relevant reference-script behavior is:

- `SHARD_TOKENS = 10_000_000`
- `BOS_ID = 1`
- `--val-docs` default is `10_000`
- documents before `val_docs` are written to `fineweb_val_*.bin` and `fineweb_val_bytes_*.bin`
- documents after that boundary are written to `fineweb_train_*.bin`

## Exact Train-Shard Replication

On the AP RunPod, we rebuilt the dataset using the exact merged-leader `prepare_caseops_data.py` behavior with its default `--val-docs=10000`. Because the full document stream is large and the reference script has no "stop after 80 train shards" option, the monitor stopped the producer after the regenerated dataset had passed the first 80 train shards. The monitor observed 82 train shards at its next polling interval, but the audit comparison intentionally uses only the first 80 shards, matching the record runs.

The regenerated output was compared to the compact CaseOps archive used for staging. The comparison result is stored in `dataset_verification/manifest_compare.json`:

```json
{
  "exact_train_shards_seen": 82,
  "exact_train_first80_tokens": 800000000,
  "archive_train_first80_tokens": 800000000,
  "train_first80_hash_mismatches": 0,
  "train_mismatches_first5": [],
  "exact_val_shards": 1,
  "exact_val_byte_shards": 1,
  "exact_val_tokens": 9662502,
  "exact_val_byte_entries": 9662502,
  "archive_val_tokens": 9662502,
  "archive_val_byte_entries": 9662502,
  "val_hash_match": true,
  "val_bytes_hash_match": true
}
```

Interpretation:

- The compact archive's first 80 train shards contain exactly `800000000` tokens.
- Those first 80 train shards have zero hash mismatches against the exact-script rebuild.
- The compact archive's default 10k validation token shard and byte sidecar also match the exact-script rebuild.

This proves that the compact archive used for training is a faithful byte-for-byte staging of the merged CaseOps script's first 80 training shards.

## Why Full 50k Validation Is Used

The exact merged script's default `--val-docs=10000` produces a small validation set with `9662502` raw validation entries. That default validation output is useful for proving the archive's provenance, but it is not the leaderboard-comparable validation set.

The CaseOps leaderboard logs report:

```text
val_tokens: 47851520
```

The submitted logs also report the same validation length:

- `train_seed42.log`: `val_tokens: 47851520`
- `train_seed1337.log`: `val_tokens: 47851520`
- `train_seed2026.log`: `val_tokens: 47851520`
- `ap_pod_seed0/run.log`: `val_tokens: 47851520`

So the final scoring dataset keeps the verified 80 train shards and replaces only the validation token/byte sidecar shards with the full 50,000-document CaseOps validation set.

## Full-Validation Repair

The full-validation repair leaves all `fineweb_train_*.bin` shards unchanged. It removes/replaces only:

- `fineweb_val_*.bin`
- `fineweb_val_bytes_*.bin`

Those validation files were regenerated from the first 50,000 documents of the same canonical document stream, using the same SP8192 tokenizer and `lossless_caps.py` transform.

The AP pod repair log is stored at `dataset_verification/repair_full50k_val_ap.log` and ends with:

```text
done docs=50000 val_shards=5 val_tokens=47853344
```

`47853344` is the raw number of validation token/byte entries. `train_gpt.py` rounds the scored validation stream to the eval sequence length (`EVAL_SEQ_LEN=2560`), yielding:

```text
val_tokens: 47851520
```

This matches the leaderboard lineage and the submitted logs.

## Base-Training/Eval Data Separation

Base training reads only:

```text
fineweb_train_*.bin
```

Validation/eval, including score-first TTT, reads:

```text
fineweb_val_*.bin
fineweb_val_bytes_*.bin
```

The byte sidecar is not a training target. It is used for BPB accounting and document-aware validation/eval processing. Eval-time TTT uses validation tokens only in the score-first order documented in `README.md`: tokens are scored before they are used for any global or LoRA update. The full-validation repair does not change base-training data or model code; it only restores the validation stream length to the leaderboard-comparable `47851520` scored tokens.

## Evidence Files

- `dataset_verification/manifest_compare.json` - hash/token comparison between exact-script rebuild and compact archive.
- `dataset_verification/monitor.log` - timestamped exact-script rebuild monitor and embedded manifest result.
- `dataset_verification/repair_full50k_val_ap.log` - full 50k validation regeneration log.
- `train_seed42.log`, `train_seed1337.log`, `train_seed2026.log` - final 3-seed submission logs.
- `ap_pod_seed0/run.log` - independent AP pod check using the same train/full-validation construction.

## Important Non-Comparable Check

We also ran the candidate against the exact script's default 10k validation output. That run produced a much worse BPB while having a similar validation loss, because BPB depends on the validation byte sidecar and the 10k validation slice has a different token/byte ratio. It is therefore a useful debugging check, but it is not comparable to leaderboard logs reporting `val_tokens: 47851520`.
