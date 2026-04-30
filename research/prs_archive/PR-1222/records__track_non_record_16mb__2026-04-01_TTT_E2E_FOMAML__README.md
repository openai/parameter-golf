# Prime MLP Test-Time Training: Naive vs E2E (FOMAML)

Two studies on test-time training with prime MLP adapters. **Naive TTT gives
-0.079 BPB for free (eval-only). E2E FOMAML gives -0.097 total but costs
44% of the training budget.**

## Motivation

All 25 prior naive TTT attempts failed because they perturbed GPTQ'd int5/int6
weights. Prime MLPs are separate bf16 parameters — they don't touch GPTQ'd weights.

## Architecture

Rank-256 prime MLPs on all 11 blocks, running before the main MLP:

```
h = h + attn(norm(h))
h = h + prime_MLP(prime_norm(h))   # bf16, adapted via SGD at eval time
h = h + MLP(mlp_norm(h))           # GPTQ'd int5/int6, frozen
```

Down projection zero-init (model starts unchanged). Score-first eval is legal.

## Results

### Full-data training (40 shards, MLP 3.5x, 10K steps, 1x L40S)

| Config | val_bpb | Delta |
|--------|---------|-------|
| **Baseline (EMA, no TTT)** | **1.3696** | — |
| **TTT lr=0.1, all 11 layers** | **1.2906** | **-0.079** |

### Sweep summary (5K chunks, full-data model)

| Experiment | val_bpb | Delta |
|------------|---------|-------|
| Baseline | 1.3696 | — |
| lr=0.03 | 1.3670 | -0.003 |
| lr=0.1 | 1.3636 | -0.006 |
| lr=0.3 | 1.3601 | -0.010 |
| lr=1.0 | 1.3550 | -0.015 |
| rank=64 (3 layers) | 1.3661 | -0.004 |
| rank=512 (3 layers) | 1.3638 | -0.006 |
| layer=[10] only | 1.3669 | -0.003 |
| layer=[6..10] | 1.3609 | -0.009 |
| **layer=all (11)** | **1.3242** | **-0.045** |
| momentum=0.9 | 1.3574 | -0.012 |

### Key findings (Study 1)

1. **Layer count >> rank.** All 11 layers (-0.045) crushes rank=512 on 3 layers (-0.006)
2. **Higher LR is better** up to 1.0 (still improving, ceiling not found)
3. **Full eval compounds** — 60K chunks gives ~1.8x the 5K-chunk delta
4. **Effect scales with model quality** — -0.079 on strong model vs -0.073 on weak
5. **Momentum=0.9 helps** (+2x at same LR on 3 layers)

### 1-shard control (earlier experiment)

| Config | Baseline | TTT | Delta |
|--------|----------|-----|-------|
| 1 shard, 7200 steps | 1.5019 | 1.4288 | -0.073 |
| **40 shards, 10K steps** | **1.3696** | **1.2906** | **-0.079** |

## Artifact size for PR 1105

| Config | Prime MLP size | Fits 16 MB? |
|--------|---------------|-------------|
| rank=256, all 11 layers | 5.75 MB | No |
| rank=64, all 11 layers | 1.41 MB | Yes |
| rank=32, all 11 layers | 0.70 MB | Yes |

rank=64 all-layers fits and rank barely matters vs layer count.

---

## Study 2: E2E TTT (FOMAML Meta-Learning)

### Method

Phase 2 FOMAML joint training on the strong 40-shard checkpoint. Base model at
0.0003 LR, prime MLPs at 0.003 LR. Inner loop: K=1 SGD step on prime weights.
Outer loop: both base and prime get gradients. 3000 steps.

### Results

| Stage | val_bpb | Delta vs orig baseline |
|-------|---------|------------------------|
| Original baseline (no prime MLPs) | 1.3696 | — |
| FOMAML baseline (prime at zero) | 1.5185 | — |
| **Post-FOMAML (no TTT)** | **1.2588** | **-0.111** |
| E2E TTT (meta-learned init, lr=0.1) | 1.2656 | -0.104 |
| Naive TTT (zero-init on FOMAML model) | 1.2776 | -0.092 |

Joint FOMAML massively improves the model even without TTT (-0.260 from FOMAML
baseline). But TTT on top of FOMAML **slightly hurts** — the meta-learned init
is already tuned and SGD overshoots.

### TTT LR sweep on FOMAML model (5K chunks)

| Config | val_bpb | Delta vs FOMAML no-TTT |
|--------|---------|------------------------|
| FOMAML, no TTT | 1.2732 | — |
| + TTT lr=0.001 | 1.2731 | -0.000 |
| + TTT lr=0.01 | 1.2726 | -0.001 |
| + TTT lr=0.1 | 1.2720 | -0.001 |

TTT adds only -0.001 on top of FOMAML. The meta-learning already captured the
adaptation value during training.

### Key findings (Study 2)

1. **Joint FOMAML makes training better** — -0.260 BPB from the FOMAML baseline, even standalone
2. **TTT is nearly redundant after FOMAML** — only -0.001 additional benefit
3. **The base model co-adapts** — this isn't just adapter training, the whole model improves

---

## Head-to-head: Naive TTT vs E2E FOMAML

| Approach | Baseline | Best | Total Δ | Training cost |
|----------|----------|------|---------|---------------|
| **Naive TTT (eval-only)** | 1.3696 | 1.2906 | **-0.079** | **0** |
| FOMAML + TTT | 1.3696 | 1.2720 | -0.097 | 3000 steps (~44% budget) |

**Naive TTT is the practical winner** — zero training cost, 81% of FOMAML's benefit.
FOMAML is worth it only if the 44% training budget can be absorbed.

---

## Next steps

- [ ] 8xH100 validation on actual PR 1105 model (1.1125 BPB)
- [ ] Combine lr=1.0 + all layers + momentum=0.9 (untested combination)
- [ ] rank=64 all-layers full eval (fits 16 MB budget)

## Files

- `train_ttt_e2e.py` — Model with prime MLPs + FOMAML + TTT eval
- `train_e2e_proper.py` — Proper E2E training (Phase 1 + Phase 2 joint)
- `sweep_naive_ttt.py` — Naive TTT LR/chunk/reset sweep
- `sweep_v2.py` — LR/rank/layer/momentum sweep
