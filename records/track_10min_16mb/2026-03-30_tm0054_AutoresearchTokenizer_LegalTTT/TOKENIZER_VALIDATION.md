# Tokenizer Validation

Our submission replaces the published `sp1024` tokenization with a promoted TokenMonster-derived tokenizer ("Scylla"). Here is a summary of the changes and a simple note to help reviewers audit the changes.

## What changed

This submission changes both:

- the tokenizer
- the dataset tokenization

Specifically, it replaces the published SentencePiece `sp1024` tokenization with a promoted TokenMonster-derived tokenizer:

- tokenizer name: `tm0054_candidate` (nicknamed "Scylla", "0054" was the autoresearch experiment iteration that produced it)
- vocab size: `998`

## Runtime scoring method

`val_bpb` is still computed using explicit per-token metadata LUTs:

- `base_bytes`
- `has_leading_space`
- `is_boundary_token`

Those LUTs are loaded from `candidate.meta.npz` at runtime. The downstream `val_bpb` byte-counting logic is unchanged from the standard metadata-driven path.

## Full-data bundle provenance

The competition bundle was built by retokenizing the published `sp1024` FineWeb export in shard order.

Included bundle metadata:

- source tokenizer family: published `sp1024`
- target tokenizer family: `tm0054_candidate`
- train shards: `79`
- val shards: `1`

The manifest in this folder records the resulting shard and token counts.

## Validation checks run

The corrected full-data bundle passed repository preflight with:

- expected train shards: `79`
- expected val shards: `1`
- tokenizer metadata path present

The corrected run logs also show:

- `tokenizer_kind=tokenmonster`
- `TOKENIZER_META_PATH` loaded
- full-data train loader using `79` train shards

## BOS/EOS handling

The exported tm0054 competition bundle stores the retokenized shard stream used by the run, and the runtime scorer operates only on the stored token ids plus the explicit metadata arrays in `candidate.meta.npz`.

The important property for review is that the scoring path is metadata-driven and deterministic; it does not depend on runtime SentencePiece inspection or any target-conditioned tokenizer logic.

## Complete audit package
Everything needed for tokenizer review is present in this record folder:

- promoted `train_gpt.py`
- promoted tokenizer vocab
- tokenizer metadata
- bundle manifest
- full-data train logs for all three seeds
