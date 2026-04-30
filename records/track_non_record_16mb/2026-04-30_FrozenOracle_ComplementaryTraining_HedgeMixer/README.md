# Frozen N-gram Oracle + HedgeMixer + SGD TTT (non-record submission)

**Author:** Dhruv Puri ([@dhruvpuri](https://github.com/dhruvpuri)), 2026-04-30

A hybrid n-gram + neural language model for OpenAI's Parameter Golf 2026 (16 MB artifact, 10 minutes training on 8×H100, scored by bits-per-byte on FineWeb val).

This is a methodology submission, not a record claim. I didn't have 8×H100 access, so the full-scale numbers (11L/512d) aren't here. The pipeline is validated end-to-end on Kaggle T4×2 with NCCL DDP. See [JOURNEY.md](./JOURNEY.md) for the research arc, agent-assisted review loop, and reproducing instructions.

## What's in the patch

Three pieces that share an artifact, plus three smaller fixes.

**1. A frozen n-gram oracle.** [`build_ngram_oracle.py`](./build_ngram_oracle.py), 250 lines, NumPy only. Scans FineWeb training tokens once offline. Builds orders 1 through 8 as int8 log-probabilities with Laplace smoothing. Orders 1 and 2 are exact. Orders 3 through 8 use FNV-1a hashed contexts with bucket counts going from 4096 down to 256. zstd-22 compressed. A NumPy/Torch FNV-1a equivalence test runs before every build. If the offline NumPy hash and the online Torch hash disagree on any sample, the build aborts.

**2. HedgeMixer with oracle experts.** The base stack already had a 5-expert online ensemble (neural, online uni/bi/tri, decay cache). I added one expert per loaded oracle order, taking the count to 13. Mixing in log-space, multiplicative-weights update on per-token NLL. Warm prior `log_w[0] = 2.0` so short eval streams aren't dominated by Hedge convergence noise. With no oracle loaded the mixer reduces to the original 5-expert form.

**3. A magic-prefixed versioned artifact format.** 16-byte header (4-byte magic `0x50474152`, 1 version byte, 3 reserved, neural and oracle blob lengths), then the neural blob (int6 per-row + zstd-22) and the oracle blob. One file under the 16 MB cap. The version byte means future schema changes fail loudly instead of silently mis-slicing. Reload uses an in-memory `FrozenNgramOracle.from_bytes` classmethod, no per-rank temp files.

Plus, in `train_gpt.py`:

- A `TTT_OPTIMIZER=sgd` switch (lr=0.002, momentum=0.9), matching [PR #967](https://github.com/openai/parameter-golf/pull/967)'s reported -0.041 BPB.
- `LEAKY_RELU_SLOPE` configurable. Setting 0.75 matches [PR #977](https://github.com/openai/parameter-golf/pull/977)'s -0.008 BPB.
- Bucketed `dist.all_reduce` in TTT, replacing about 100 per-parameter NCCL launches per micro-step with one.

About 280 new lines, 9 changed lines, all gated by environment variables. With `NGRAM_ORACLE_PATH=""` and `TTT_OPTIMIZER=adamw`, runtime behavior matches the base.

## What I actually ran (Kaggle T4×2 NCCL DDP)

| Stage | Result |
|---|---|
| Build | Oracle 3.42 MB / 10M tokens / 2.5 s |
| Train | 8L/384d, 13.4M params, 172 steps in 180s, `world_size:2 grad_accum_steps:4` |
| Quantize + bundle | int6 + zstd-22, artifact 6.85 MB / 16 MB (neural 3.43 MB + oracle 3.42 MB + 16 B header) |
| Reload | `oracle:loaded from artifact orders=[1, 2, 3, 4, 5, 6, 7, 8]`, both ranks via `from_bytes` |
| TTT | SGD, 3,786 chunks, oracle in HedgeMixer experts, 7,078 s wall on T4×2 |
| Magic prefix check | `Header + blobs total: 7,181,197 == 7,181,197: True` |
| Exit code | 0 |

The 2.54 BPB from this run is a sanity check, not a competition number. 8L/384d trained for 180 seconds isn't going to land near 1.05 to 1.10. What it does prove is that the whole pipeline runs cleanly under DDP: HedgeMixer table updates, in-memory oracle reload, and the bucketed all-reduce path all work on more than one rank, which is what single-GPU testing can't show.

## H100×8 extrapolation

| Stage | Kaggle T4×2 (measured) | H100×8 (estimated, 11L/512d) |
|---|---|---|
| Training step | 1.05 s | 80 to 100 ms |
| TTT chunk | 1.87 s | 80 to 100 ms |
| Total wall | ~2 h | 13 to 17 min |

T4 to H100 single-card bf16 is ~15x, DDP 2 to 8 is ~3.3x in practice. Net per-step gain is ~25x after accounting for the 2x larger competition model. Fits inside the 10-minute train + 10-minute eval budget with room to spare.

## Negative results

| What I tried | Result | Why it didn't ship |
|---|---|---|
| Byte-level CTW (`ctw_prototype.py`) | Eval BPB 6.33 vs target <1.2; 21.3 MB compressed vs target <5 MB; 16.7K bytes/sec vs target >100K | 256-symbol alphabet at depth 8 has too many states. Killed in 2 days, redirected to FNV-hashed token-level oracle. |
| Inline complementary loss (`loss * weight.mean()`) | Mathematically not equivalent to per-token reweighting | Removed. Standalone `complementary_training_loss` is kept for reference; needs inside-graph integration to be correct. |
| `bi_counts[prev, targets] += 1.0` in HedgeMixer | Non-deterministic on duplicate indices, silent correctness bug | Replaced with `index_put_(..., accumulate=True)`. |

## Compliance

The frozen-oracle pattern was rejected once already in this cohort ([PR #924 ruling](https://github.com/openai/parameter-golf/issues/1017)). I designed this to hold against [Issue #1017](https://github.com/openai/parameter-golf/issues/1017): training tokens only (`build_ngram_oracle.py` never reads `fineweb_val_*.bin`), deterministic build (fixed FNV-1a, fixed Laplace constant, no RNG), no eval-time data dependence (oracle is read-only during train/TTT/eval), and bundled inside the 16 MB cap at 3.42 MB.

## Limitations

- No 8×H100 validation, no 3-seed mean, no competition-scale BPB number.
- Oracle build verified on a 10M-token slice (Kaggle) and a 100M-token shard (local). Full 80-shard build is extrapolated, not measured.
- `complementary_training_loss` is implemented but not wired into training. Per-token reweighting needs logits access from inside the compiled graph; the function is kept for that future integration.
- HedgeMixer's `bi_counts` is dense `vocab × vocab`, asserted for `vocab_size <= 2048`. SP4096 vocab would need a hashed bigram table.
