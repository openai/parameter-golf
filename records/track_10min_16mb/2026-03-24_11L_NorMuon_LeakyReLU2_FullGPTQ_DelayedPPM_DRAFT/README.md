# Draft: 11L NorMuon + LeakyReLU^2 + Full GPTQ + Delayed PPM

This folder is a draft snapshot of a record attempt built on top of
`2026-03-23_NorMuon_11L_EMA_GPTQ-lite`.

It is intentionally minimal for early review:

- `train_gpt.py`
- `trie_bench.c`
- this `README.md`

No results, logs, or `submission.json` are included yet.

## Status

Work in progress. This draft PR is meant to document the method and the
current code path before the full validation sweep is finished.

The eventual record claim, if any, will only be made after:

1. 3-seed validation is complete
2. the `0.005` nat improvement bar is satisfied at `p < 0.01`
3. final logs and metadata are added

## Method

This stack combines four ideas on the current 11-layer NorMuon base:

1. **NorMuon**
   Polar Express Newton-Schulz coefficients plus variance reduction on Muon
   updates, as introduced in
   `records/track_10min_16mb/2026-03-23_NorMuon_11L_EMA_GPTQ-lite`.

2. **LeakyReLU^2**
   Replace `ReLU(x)^2` with `LeakyReLU(x, 0.5)^2` in the MLP.

3. **Full GPTQ**
   Replace the previous GPTQ-lite export with a Hessian-aware GPTQ pass for
   int6 post-training quantization.

4. **Delayed PPM**
   Add a delayed outside-context-only PPM bank at final sliding-window eval.
   The bank only contains targets from positions at least `2048` tokens behind
   the current prediction point, so it cannot reuse anything already visible
   inside the transformer's local context window.

## Delayed PPM Details

The delayed PPM path is adapted from the delayed PPM record idea and keeps the
same overall structure:

- `k_values = [16, 12, 8, 6]`
- `min_confs = [1.0, 1.0, 1.0, 0.95]`
- `min_counts = [1, 1, 1, 1]`
- `boost_k = 15`
- `delay = 2048`
- `bos_id = 1`

The implementation uses a small C helper (`trie_bench.c`) compiled to
`libtrie.so` during eval to keep the hit-building pass fast enough for real
record-track ablations.

## Compression

The current draft prefers **Kanzi** when the jar is available locally, falling
back to `zstd` and then `zlib` otherwise. The intention is to keep the best
artifact-size path available while the modeling stack is being validated.

## Notes

- This draft is **not** claiming a record yet.
- The code is included now so the method can be reviewed before the validation
  package is finalized.
- If the stack validates cleanly, this folder can be updated with logs,
  `submission.json`, and final README results in a follow-up commit.
