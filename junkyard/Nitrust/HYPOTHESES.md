# Nitrust Program — Hypothesis Backlog (NGRAM-Free)
Date: 2026-03-27

## Mission
Build foundational, hardware-first architecture upgrades above the crawler line that improve:
1. Model-only quality (`val_bpb`, no ngram mixing)
2. Artifact efficiency (bytes at fixed or better quality)
3. Throughput (step time / tokens-per-second)

## Hard Rules (Nitrust Phase 1)
1. Ignore all ngram paths for training and eval.
2. Compare only model outputs (`final_int6_roundtrip`, `final_int6_sliding_window`).
3. Keep export/legal path simple while architecture is changing.

### NGRAM-Off Guardrail
Use these defaults for all Nitrust runs unless explicitly overridden:
- `NGRAM_EVAL_ORDER=0`
- `NGRAM_EVAL_ADAPTIVE=0`
- `NGRAM_DIRICHLET=0`
- `PHRASE_CACHE=0`
- `REGIME_TRACKER=0`
- `NGRAM_ENTROPY_SHIFT=0`
- `TRIGRAM=0`

## Baseline First (NIT-00)
Before every new injection, re-run a stable baseline with the exact same wallclock budget and seed policy.

Success baseline record should include:
- `step@cap`, `val_bpb@cap`
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `Serialized model int6+*` bytes
- step average ms

---

## Ordered Hypotheses (Low -> High Complexity)

| ID | Complexity | Hypothesis | Architecture Injection | Hardware Rationale | Success Gate | Kill Gate |
|---|---:|---|---|---|---|---|
| NIT-01 | 1 | Hopper shape locking improves throughput without quality loss. | Lock dims/head dims to tensor-core-friendly multiples; remove odd shapes in recurrent path. | Fewer kernel variants, better matmul occupancy/fusion. | >=8% faster step time, `val_bpb` delta <= +0.01 | <3% speed gain or `val_bpb` worse by >0.02 |
| NIT-02 | 2 | Loop-conditioned low-rank adapters fix shared-block regime mismatch. | Shared core stays fixed, per-loop `W_k = W + A_k B_k` (small rank). | Keeps parameter compression while giving each loop a cheap specialization path. | Better `final_int6_sliding_window` by >=0.02 at <=15% artifact growth | No quality gain or artifact growth >20% |
| NIT-03 | 3 | Split sharing (shared attention, loop-specific MLP modulation) beats fully shared blocks. | Share attention weights; add tiny per-loop channel gates or low-rank MLP deltas. | Attention kernels stay reusable; cheap MLP modulation handles loop-specific distributions. | >=0.02 BPB improvement vs NIT-00 with <=20% slower step time | Regresses both speed and BPB |
| NIT-04 | 4 | Bucketed adaptive loop budget improves quality-per-compute. | Two static paths: short-loop and long-loop based on confidence bucket at sequence/window level. | Preserves static-ish execution while reducing unnecessary deep passes. | Same or better BPB with >=15% faster average step time | Control overhead removes speed gain |
| NIT-05 | 5 | Latent funnel recurrence dominates flat+bottleneck at same bytes. | Downsample sequence in bottleneck (`T -> T/2`), run recurrent core there, upsample back. | Shifts work to denser GEMMs and lowers KV bandwidth pressure. | >=0.03 BPB gain or >=20% speedup at comparable artifact size | Training instability or quality collapse |
| NIT-06 | 6 | Persistent memory tokens make recurrence actually cumulative. | Add small memory token bank carried across loops and rewritten each loop. | Small fixed memory adds global workspace without large parameter cost. | >=0.02 BPB gain over NIT-05 with <=10% speed hit | No measurable gain after two seeds |
| NIT-07 | 7 | Dual-rate recurrent superblock wins the frontier. | Heavy attention every 2 loops, lightweight update each loop (multi-rate core). | Cuts expensive attention frequency while keeping iterative refinement depth. | Better BPB and speed-vs-quality tradeoff than NIT-05/06 | Scheduling complexity causes compile/runtime fragility |

---

## Execution Order
1. NIT-00 baseline freeze
2. NIT-01 shape locking
3. NIT-02 low-rank loop adapters
4. NIT-03 split sharing
5. NIT-04 adaptive loop buckets
6. NIT-05 latent funnel
7. NIT-06 memory tokens
8. NIT-07 dual-rate superblock

## Notes
- Do not introduce ngram-dependent compensators while validating core architecture signal.
- Any candidate that wins only with ngram is considered unproven for Nitrust Phase 1.
