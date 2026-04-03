# Scylla: Corrected Byte-Exact Tokenizer Path

This PR packages the corrected, official revision of **Scylla**, our TokenMonster-derived tokenizer line for Parameter Golf.

We were pleased to see Scylla open what appears to be the competition's first substantial custom-tokenizer line. We were even more pleased, in the end, that people read it closely enough to break it. The critique from `@NoesisGenesis`, `@dexhunter`, and later `@andrewbaggio1` on byte accounting and exactness was correct and genuinely helpful. It forced a deeper audit than we had originally performed, and the result is better for it.

We were also delighted to see other competitors start building with Scylla in PRs like `#1184`, `#1274`, and `#1289`. But once the byte-accounting issue had been correctly surfaced, it was clear that the responsible thing to do was not to defend the old path harder, but to rebuild it properly.

What we present here is **Scylla, revised**: a robust, byte-exact tokenizer path for the fixed FineWeb validation text, together with the metadata and audit artifacts needed to review it.

This is **not** a leaderboard claim. It is a tokenizer contribution and a corrected reference path for future Scylla-based work.

For clarity: in this folder, **Scylla** means the corrected official revision. The original `998`-token path from PR `#1143` is superseded by the artifact set here.

## What Was Wrong Before

The original `998`-token Scylla path from PR `#1143` had two separate correctness problems:

1. Its byte-accounting metadata treated TokenMonster tokens as if their decoded byte lengths were context-free.
2. Its retokenized validation stream was not byte-identical to the fixed FineWeb validation text.

Those are distinct failures, and both matter for a tokenizer-agnostic `val_bpb` benchmark.

The repair path was not obvious at first. In the first byte-native audit lane, a converted Scylla-family vocabulary round-tripped `187/200` sampled validation documents exactly, while `13` remained stubbornly wrong. Those failures clustered almost entirely in non-ASCII / UTF-8 cases. The first clue was incomplete high-byte fallback coverage; fixing that collapsed the failure surface dramatically. The remaining holdouts included Turkish dotted `İ`, which exposed a deeper capcode interaction. That was the moment the shape of the real fix became clear: not another local patch, but a genuinely byte-native tokenizer regime.

## What Changed In Corrected Scylla

Corrected Scylla uses a byte-native TokenMonster regime:

- `capcode = 0`
- `charset = none`
- `normalization = none`
- explicit `0x00..0xFF` byte fallback coverage

The bundle/export path also needed two additional corrections:

- `charset:none` TokenMonster decoded strings must be interpreted as raw bytes via `latin-1`, not `utf-8`
- a synthetic zero-byte BOS token must be inserted at dataset/export time so the flat shard format preserves document boundaries exactly

The resulting tokenizer metadata and dataset bundle now admit exact, reviewable byte accounting.

## Full-Validation Exactness

We ran a strict full-validation audit against the fixed SP1024 FineWeb validation source. The corrected Scylla bundle yields:

- `source_val_docs = 50000`
- `bundle_val_docs = 50000`
- `source_bytes = 151080891`
- `meta_bytes = 151080891`
- `decoded_bytes = 151080891`
- `bad_docs = 0`
- `meta_overcount_frac = 0.0`
- `decoded_drift_frac = 0.0`

That is the whole point of this revision. The source text, the decoded tokenizer stream, and the metadata-derived denominator now agree exactly on the full validation shard.

## Included Artifacts

- `scylla.yaml`
  The corrected Scylla tokenizer artifact.
- `scylla.meta.npz`
  The corrected byte-accounting metadata.
- `manifest.json`
  Bundle manifest for the corrected full-data export.
- `BUILD_NOTES.md`
  Construction notes, invariants, and the exact audit path for future Scylla-based work.
- `FULL_VAL_AUDIT.json`
  Full-validation exactness audit results.

## Why We Are Publishing This

We think novel tokenizer work belongs in this competition. It changes the shape of the problem in an interesting way, and it deserves to be explored in public rather than in a private thicket of half-verified local hacks.

So this PR is meant as a community contribution:

- a corrected Scylla reference path
- an explicit accounting story
- a cleaner base for future tokenizer experimentation

We hope others extend it, stress it, improve it, and, ideally, beat it.

## Thanks

We are indebted to `@NoesisGenesis`, `@dexhunter`, and `@andrewbaggio1` for pressing on the exactness and byte-accounting questions. Their scrutiny materially improved this work.
