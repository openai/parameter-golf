# Data & tokenizer

Reference: [`research/AXES_ANALYSIS.md#axis-6-data--tokenizer`](../../research/AXES_ANALYSIS.md)

*Tokenizer choice (SP1024 / SP4096 / SP8192 / BPE variants), shard ordering, data filtering, FineWeb-Edu substitution, Rho-1-style selective LM, curriculum.*

## Hypothesis

The challenge train stream appears to be a frozen shuffled snapshot rather than a deliberately education-filtered corpus, so a cleaner train-only substitution may help under the 600s token budget even if it introduces some validation-distribution mismatch.

## Experiments

| ID | Date | Branch | Config | val_bpb | Base | Notes |
|----|------|--------|--------|---------|------|-------|
| `fwedu100-sp8192-pr1493` | 2026-04-14 | `main` | Official SP8192 tokenizer + official val + FineWeb-Edu-only train shards | pending | PR1493 | First directional probe. Pure Edu is expected to test the distribution-shift ceiling before mixed-train followups. |

## Findings

- The published challenge docs are a frozen shuffled export, not obviously a FineWeb-Edu-style cleaned subset.
- With 10-minute training, the model only sees an early slice of the train stream, so train data order and mixing strategy matter.

## Next

- Build the `100% FineWeb-Edu` train-only variant with the original SP8192 tokenizer and unchanged val split.
- Run the merged PR1493 command against the alternate `DATA_DIR`.
- If pure Edu loses, try an interleaved original/Edu mix before touching the tokenizer.

Reference runbook: [fineweb_edu_sp8192.md](fineweb_edu_sp8192.md)
