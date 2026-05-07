# Mockingbird

10k-vocab CaseOps body on the SOTA architecture, derived from PR1855.

This is a **non-record** submission. It does not beat the current leader. It is filed as evidence of the SP10240 CaseOps lane on the same compression / phased-TTT machinery as PR1855, for comparison with the SP8192 lane.

## Results

| Seed | val_bpb (quantized_ttt_phased) | Steps | Total submission size |
|------|--------------------------------|-------|------------------------|
| 42   | 1.06204667                     | 5,264 | 15,816,988 B           |
| 0    | 1.06226648                     | 5,231 | 15,818,783 B           |
| 1    | 1.06299064                     | 5,221 | 15,810,544 B           |
| **mean** | **1.06243460**             |       | **15,818,783 B (max)** |

Hardware: 8×H100 SXM · 600s wallclock · `bytes_code`: 163,036 (uncompressed) / 41,220 (compressed)

## Architecture

11L · dim 512 · `mlp_mult=3.75` · loop_start=3, loop_end=5, `enable_looping_at=0.45`

- **Vocab/data**: SP10240 CaseOps lossless-caps tokenizer (10,240 tokens), FineWeb 10B sidecar with byte-level loss accounting
- **Quantization**: per-group, embed int7, matrix int6, LQER asymmetric rank-4
- **Eval**: PR1855 phased LoRA TTT — `prefix_docs=2500`, `phases=3`, `chunk=48`
- **Compression**: pergroup
- **Train budget**: 600 s wallclock, hard 16 MB artifact cap

## Lineage

This is the SP10240 sister of PR #1855 (`510d03e0fc355406c9fd06f92d23b8c5aedea7fb`), which used the same CaseOps + LQER + phased-TTT machinery on SP8192 and reported a 3-seed mean of 1.06107587 post-phased-TTT.

The architecture is held fixed; only the tokenizer / vocab dimension changes (8192 → 10240). The 10k vocab consumes more bytes in the embedding table, so the body is shrunk to MLP3.75 (vs the SP8192 record's wider body) to stay under the 16 MB cap. `enable_looping_at=0.45` matches the same family.

## Seeds

The three runs used identical code and hyperparameters; only the random seed changed. The committed `train_gpt.py` is the seed-42 run (the strongest of the three). Seeds 0 and 1 differ only in `Hyperparameters.seed = N` (line 479 in this file) and the bookkeeping fields `TEST_ID` / `TEST_DATE` / `RUN_KIND` / blurb (lines 433–446). The training body is byte-identical.

Seed choice (`42`, `0`, `1`) reflects the seed-repeat batch we ran on this lane; this submission does not use the protocol's `444 / 300` convention because these specific runs were not re-executed at those seeds.

## Reproduce

```bash
# From repo root, with flash-attention/hopper on PYTHONPATH
SKIP_GPTQ=1 SEED=42 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-05-01_Mockingbird_8xH100/train_gpt.py
```

For seeds 0 and 1, change line 479 (`Hyperparameters.seed = 42`) to `0` or `1` respectively. The default `SEED` env var is overridden inside the file.

## Artifacts

Per-seed compressed artifacts (`final_model.int6.ptz`) and SHA256 hashes are recorded in `submission.json`. Each artifact is well under the 16 MB cap (max 15.82 MB).
