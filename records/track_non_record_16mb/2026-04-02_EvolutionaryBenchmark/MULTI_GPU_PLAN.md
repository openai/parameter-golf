# Multi-GPU Committee Plan

The single-H100 committee results point toward a consistent rule:

- within one GPU, depth beats early breadth
- late breadth can help, but it should not steal too much early branch training
- the best use of additional H100s is likely breadth across devices, not shallow breadth inside each device

## Current Single-GPU Signal

At equal `480` branch-seconds on seed `1337`, the current ordering is:

- `4x120`: `val_bpb ~= 2.114`
- `4x90 -> 8x15`: `val_bpb ~= 2.130`
- `4x60 -> 8x30`: `val_bpb ~= 2.167`
- `4x60 -> 16x15`: `val_bpb ~= 2.226`
- `4x30 -> 8x45`: `val_bpb ~= 2.233`

That is strong evidence that each GPU should first produce a small set of competent branches before we spend additional hardware on wider committees.

## Recommendation

### Search mode on `8x H100`

Use the hardware to parallelize committee frontier runs, not to make each run wider by default.

- GPU 0: `4x120`
- GPU 1: `6x80`
- GPU 2: `8x60`
- GPU 3: `4x90 -> 8x15`
- GPU 4: `4x60 -> 8x30`
- GPU 5: `4x60 -> 8x15 -> 16x7.5`
- GPU 6-7: extra seeds or stronger-base variants (`XSA`, `QK-gain 4.0`)

This is the purpose of [runpod_queue_committee_frontier.sh](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-02_EvolutionaryBenchmark/runpod_queue_committee_frontier.sh).

### Inference / submission mode on `8x H100`

Treat each GPU as an independent committee trainer for a small deep branch set, then ensemble across GPUs.

Example:

- each GPU trains `4` branches for `120s`
- each GPU holds a local top-`4` committee
- the cluster ensembles `32` strong branches total

That avoids the main failure mode we saw on one GPU: going wide too early and weakening each branch.

## Cluster Ensembling Sketch

Each rank owns a local committee shard and contributes only a probability sum:

```python
prob_sum_local = torch.zeros(batch, seq, vocab, device=device, dtype=torch.float32)

for states_chunk in local_branch_chunks:
    logits = vmap_eval(states_chunk, inputs)          # [local_branches, B, T, V]
    prob_sum_local += torch.softmax(logits.float(), dim=-1).sum(dim=0)

dist.all_reduce(prob_sum_local, op=dist.ReduceOp.SUM)
probs = prob_sum_local / global_branch_count
```

The communication payload is only `[batch, seq, vocab]`, not all model weights or all per-branch logits.

## Hypothesis To Test Next

If the current staged results hold across the second seed, the likely best next large-hardware bet is:

1. strengthen each local branch set with `XSA` / `QK-gain`
2. keep local committees small and deep
3. use cross-GPU breadth for the final ensemble

That is the hardware-native version of the committee idea.
