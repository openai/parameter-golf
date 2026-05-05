# Scylla (novel tokenizer) + Legal Score-First TTT (val_bpb: 1.08056553)

## Results

| Seed | step_avg | steps | roundtrip | sliding | legal_ttt_exact | bytes_total |
|------|----------|-------|-----------|---------|-----------------|-------------|
| 42 | 84.63ms | 7091 | 1.10466967 | 1.08295388 | **1.08008661** | 15,866,740 |
| 1337 | 84.71ms | 7084 | 1.10565088 | 1.08398224 | **1.08102737** | 15,850,756 |
| 2026 | 84.65ms | 7089 | 1.10490932 | 1.08315990 | **1.08058261** | 15,849,792 |
| Mean | 84.66ms | 7088 | 1.10507662 | 1.08336534 | **1.08056553** | 15,855,763 |

Against the currently accepted leader [#549](https://github.com/openai/parameter-golf/pull/549) at `1.1194`, this is an improvement of `0.03883447` BPB, or about `3.47%`.

## Summary

This submission combines three ideas:

1. A backward-looking, score-first TTT evaluation path following the accepted PR `#461` framework.
2. A custom TokenMonster-derived tokenizer (`Scylla`) selected through iterative [autoresearch](https://github.com/karpathy/autoresearch) and proxy validation rather than manual guesswork.
3. A full-data retokenized FineWeb competition bundle using that tokenizer, with runtime `val_bpb` accounting driven by explicit per-token metadata rather than SentencePiece runtime inspection.

Our strategy is a stack change that starts at the tokenizer and runs all the way through evaluation:

- tokenizer family search
- budget-aware tokenizer screening
- proxy promotion and rejection of dead ends
- exact runtime byte accounting
- full-data retokenization into the promoted tokenizer
- legal score-first adaptive evaluation

To the best of our knowledge, this is also among the first leaderboard-caliber submissions in the competition to change the tokenizer itself rather than inherit the published `sp1024` tokenization. If reviewers spot an earlier example we missed, we would be happy to correct that framing; either way, we think tokenizer search is a genuinely promising avenue here and welcome scrutiny and follow-up work.

## Tokenizer Journey

The tokenizer work went through several iterative stages. The short version is that we tried the obvious thing first, watched it flatten out, and then had the good sense to stop being sentimental about it.

### 1. SentencePiece autoresearch

We first built an [autoresearch](https://github.com/karpathy/autoresearch) loop around SentencePiece. That loop optimized tokenizer candidates against a FineWeb-aligned screening metric and later against budget-aware heuristics.

This turned out to be useful exploration, but not the winning path:

- locally, (i.e., on my MacBook Pro) SentencePiece candidates improved the cheap tokenizer-screen metric
- in proxy model runs with beefier hardware, those gains mostly failed to transfer
- the search quickly saturated in a narrow neighborhood

That negative result mattered. It told us that “better tokenizer statistics” were not enough by themselves, and that larger vocabularies were often buying slim marginal gains with too much artifact budget. It also gave us permission to leave SentencePiece alone instead of continuing to hammer on a local maximum.

### 2. TokenMonster sidecar and proxy calibration

We then evaluated [TokenMonster](https://github.com/alasdairforsythe/tokenmonster) as a challenger family. Early cheap-screen results suggested that small TokenMonster vocabularies, especially around the `1024` regime, were more promising than either larger TokenMonster vocabularies or the best SentencePiece variants.

Proxy validation sharpened that impression:

- large TokenMonster variants did not hold up, but small TokenMonster variants did
- the best direction was not “bigger tokenizer”, it was “simpler tokenizer, slightly pruned, same strong byte efficiency”

### 3. TokenMonster-only autoresearch

We then narrowed the search into a TokenMonster-only lane. After broadening the proposal policy away from tiny local resize-only edits, the best line became a lightly pruned derivative of `english-1024-clean-v1`.

That candidate, tracked internally as `tm0054` and nicknamed **[Scylla](https://grokipedia.com/page/Scylla)**, kept the good byte efficiency of the parent vocabulary while reducing waste in the active vocabulary.

This was then promoted through:

- tokenizer screening
- proxy validation
- matched local training comparison
- legal-TTT ladder testing
- full-data bundle export

The important negative result was that larger-vocab and SentencePiece-side improvements looked better on cheap screening than they did in proxy or full runs. The winning lesson was not “make the tokenizer bigger.” It was “make the tokenizer better aligned to the artifact budget and to the tiny-model learning dynamics.”

If this submission does end up being among the first tokenizer-changing entries seriously pushed to the top of the leaderboard, we would be delighted to see other people push on the same door. This competition has been especially exciting for cultivating unusual and interesting ideas, and we think tokenizer search deserves a place in that mix.

## Full-Data Bundle

For the corrected competition path, we built a full-data `Scylla` bundle from the published `sp1024` FineWeb export by retokenizing in shard order.

The corrected bundle uses:

- `79` train shards
- `1` val shard
- preserved shard ordering
- preserved validation ordering

Runtime tokenizer assets:

- `candidate.vocab`
- `candidate.meta.npz`

The metadata artifact supplies:

- per-token byte lengths
- leading-space flags
- boundary-token flags

so the runtime path does not need SentencePiece to inspect tokenizer internals during evaluation.

A compact audit note is included in `TOKENIZER_VALIDATION.md`.

## Legality

This record path is intended to stay within the currently accepted legality standard:

- no target-conditioned mixing
- score-first TTT only
- full-data retokenized bundle with explicit metadata-driven byte accounting

Backward-looking, score-first TTT following PR `#461`'s framework:

- score a chunk first
- only then adapt on that already-scored chunk
- never use future tokens to change the distribution assigned to already-scored tokens

Score-first protocol: the model scores each validation chunk before adapting on it. No token is ever re-scored after adaptation. This follows the causal score-before-update TTT pattern that organizers have treated as legal in the adaptive track discussion and accepted submissions.

## Implementation Notes

The main script in this folder is the promoted legal TTT stack adapted for tokenizer bundles:

- `TOKENIZER_PATH` points to the promoted tokenizer vocab
- `TOKENIZER_META_PATH` points to the exported metadata LUTs
- `TTT_ENABLED=1`

The strongest path found so far combines:

- the promoted `Scylla` tokenizer
- legal score-first TTT
- the current tuned 11-layer legal stack

## Included Files

- `train_gpt.py`
- `candidate.vocab`
- `candidate.meta.npz`
- `manifest.json`
- `train_seed42.log`
- `train_seed1337.log`
- `train_seed2026.log`
- `TOKENIZER_VALIDATION.md`

## Acknowledgements

Thanks to **@0hq** and **@valerio-oai** for organizing, maintaining, and moderating an unusually fun and technically demanding competition.

The tokenizer lane also benefited from reading and learning from other competitors’ public work, especially the broader discussion around legal evaluation methods and tokenizer tradeoffs.
