# Non-Record Submission: Autoresearch Low-Rank-Q + Final-Tail-Q + Tied-Embedding Float (1xRTX 5090)

This is a non-record submission documenting the best accepted line from my search repo, [jadechip/autoresearch-parameter-golf](https://github.com/jadechip/autoresearch-parameter-golf).

The repo is built around a Codex/autoresearch loop on a single RTX 5090: run one fixed-budget experiment, commit the code change, keep meaningful wins, revert losers, and use git history plus a structured results table as the search memory. This submission snapshots the best accepted line from that search rather than the later March 22 exploratory tip runs.

## Best Accepted Result

- Run: `ar5090-20260321-231130`
- Commit: `905cc4d`
- Post-quant `val_bpb`: **1.529478563**
- Artifact size: **9,190,936 bytes**
- Total wallclock: **427.855572 s**
- Steps: **1088**
- Tokens processed: **66.846720M**
- Params: **17.500192M**
- GPU: **1x RTX 5090**

Baseline comparison from the same 5090 sweep:

| Run | val_bpb | Artifact bytes | Delta vs baseline |
|-----|---------|----------------|-------------------|
| `baseline_5090_5min_rerun` | 1.570137665 | 7,174,403 | — |
| `ar5090-20260321-231130` | **1.529478563** | 9,190,936 | **-0.040659102** |

## What The Search Actually Found

This repo did not win by making the recurrent baseline deeper or globally wider. The winning path was:

1. Start from a recurrent/shared-block compact transformer.
2. Reduce repeated/shared recurrence and spend more capacity on unique late blocks.
3. Use low-rank `q_proj` as a compute/byte reallocation tool.
4. Move to a compact carrier plus stronger unique tail.
5. Shrink the global update size to fit many more optimizer steps in the same wallclock.
6. Delay MLP fake quant until the `640 -> 768` curriculum reaches full context.
7. Restore full-rank `q_proj` only on the final tail block.
8. Keep the tied embedding in fp16 export.

The best accepted line is roughly:

- compact carrier: `stem=0`, `shared_layers=1`, `recurrence_loops=1`, `tail_layers=3`
- low-rank `q_proj` baseline on most blocks (`q_low_rank=128`)
- full-rank `q_proj` only on the final tail block
- selective attention fake quant disabled during training
- delayed MLP fake quant at the full-context boundary
- smaller update shape: `2 x 30720`
- fp16 tied-embedding export

## Why This Is A Non-Record Submission

This is a search-tier result from a single RTX 5090 repo, not a leaderboard attempt packaged around the official 8xH100 record process. The contribution here is the search method and the resulting model family:

- git-native autoresearch loop
- recurrent/shared-block model that converged toward a stronger compact carrier
- low-rank-Q reallocation plus selective precision as the main late-stage win
- a full sweep table showing both the wins and the negative results

## Later Follow-Ups

The later March 22 runs in `results.tsv` pushed closer to the hard cap with coarser float/precision bundles, but they did not beat the accepted best. Representative follow-ups:

| Run | val_bpb | Artifact bytes | Note |
|-----|---------|----------------|------|
| `ar5090-20260322-061953` | 1.535039374 | 15,261,334 | shared-Q reallocation + final-tail MLP float regressed |
| `ar5090-20260322-090834` | 1.537538912 | 15,267,914 | shared full-rank Q + final-tail float regressed |
| `ar5090-20260322-094022` | 1.535119154 | 12,007,446 | likely current tip experiment; still worse than accepted best |

Those follow-ups are included because the negative results are part of the method: broad shared-carrier float bundles and near-cap precision stacks were usually worse than smaller, more targeted precision spends.

## Files Included

- `train_gpt.py` — snapshot of the accepted-best training script from commit `905cc4d`, adjusted only so it resolves the upstream repo's `data/` paths and counts `train_gpt.py` in artifact bytes
- `results.tsv` — full later 5090 sweep table used as the structured experiment log for this repo
- `submission.json` — metadata for the best accepted line

The full development history and current codebase live in the source repo:

- <https://github.com/jadechip/autoresearch-parameter-golf>
