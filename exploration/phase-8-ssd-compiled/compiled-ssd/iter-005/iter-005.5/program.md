# iter-005.5 — Final Configuration Sweep

## Root Cause Analysis

iter-003.5 achieved 1.600 BPB (10 min, 1×H100, 960 steps, 209K tok/s).
iter-005.5 sf_a achieved 1.895 BPB (5 min, 1×H100, 983 steps, 428K tok/s).

Despite 2× throughput, 005.5 was WORSE. Investigation found:
1. **embed_lr=0.6** — untied embeddings used the baseline's 0.6 LR instead of 0.03
2. **warmdown=2400** — with ~1960 steps in 10 min, warmdown consumed the entire run
3. These bugs masked any throughput gains from the 005 codebase

## Hypothesis

The 005.5 codebase IS faster (428K vs 209K tok/s). If we fix the LR and warmdown
bugs and match iter-003.5's config, we should get ~1900 steps in 10 min (vs 960)
and beat 1.600 BPB significantly.

## Three Configs (5 min each, 1×H100)

### Config A: iter-003.5 reproduction on 005.5 codebase
```
N=4, seq=1024, LR=0.03, warmdown=1200, tied_emb=yes, stateful=no
```
**Hypothesis:** If 005.5 throughput is real (~428K tok/s), we get ~983 steps
in 5 min. iter-003.5 at 983 steps (half its 10-min run) had val_bpb ≈ 1.75.
We should match or beat that.

**IF val_bpb < 1.80:** Codebase is faster AND quality matches. Run for 10 min → expect ~1.50.
**IF val_bpb 1.80-1.95:** Throughput gain is real but the 005 architectural changes (x0 highway, log-uniform A_log, vertical carry) hurt quality. Strip them.
**IF val_bpb > 1.95:** Throughput gain is an illusion or there's another bug. Debug.

### Config B: seq=2048 with safe LR
```
N=4, seq=2048, LR=0.02, warmdown=2400, tied_emb=yes, stateful=yes
```
**Hypothesis:** Longer context helps if LR is safe (0.02, not 0.03).
2400 warmdown is appropriate for ~1400 steps expected at seq=2048.

**IF val_bpb < Config A:** Longer context wins. Use for 8×H100.
**IF val_bpb > Config A:** seq=1024 throughput advantage dominates. Stick with 1024.

### Config C: N=2 max throughput + 2048 context
```
N=2, seq=2048, LR=0.02, warmdown=2400, tied_emb=yes, stateful=yes
```
**Hypothesis:** N=2 gives maximum steps. Combined with seq=2048 for context.
The earlier cfg_c got 1.936 at 5 min — with tied embeddings and fixed LR, could improve.

**IF val_bpb < Config A and B:** Throughput > depth. Submit this.
**IF val_bpb > Config A:** N=2 is too shallow. Depth matters more than steps.

## Decision Tree After Sweep

```
IF Config A wins:
├── Run Config A for 10 min → expect ~1.50 BPB
├── IF 10-min < 1.40: go to 8×H100 → project ~1.15-1.20
└── IF 10-min > 1.50: stick with iter-004's 1.320 on 8×H100

IF Config B or C wins:
├── Run winner for 10 min
├── Compare to iter-003.5's 1.600
└── Take the absolute best to 8×H100

IF all configs > 1.95 (005.5 codebase is broken):
├── Revert to iter-003.5's EXACT code (the .archive version)
├── Run that for 10 min as sanity check
└── Submit iter-004's 1.320 as non-record
```

## The 8×H100 Decision

We go to 8×H100 when we have a config that:
1. Gets < 1.70 BPB on 1×H100 in 10 min (proven scaling: 1×→8× drops ~0.3 BPB)
2. Is numerically stable (no NaN, no loss climbing)
3. Has an artifact < 16 MB

Current best 8×H100 result: **1.320 BPB** (iter-004).
To justify another 8×H100 run, we need a 1×H100 result below ~1.55.
