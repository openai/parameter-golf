# exp105a: Meta-TTT Ablation — FOMAML Off (from exp101)

**Parent**: 11L XSA-all · BigramHash 4096×64 pos-conditional (ws/non-ws split) · trigram · VE7-10 · FOMAML every=4 · SGD+cosine TTT · int6 GPTQ+lzma (legal_ttt **1.11588**)
**Single change**: `META_TTT_ENABLED=1 → 0`
**Result**: legal_ttt = **1.11624** | int6 = **1.13956** | model = **14.94 MB** (15.66 MB w/ code)

---

## 1. Motivation

### Experiment lineage

```
BigramHash 10240×128 · VE9-10  · FOMAML every=8 (first meta-TTT attempt)           →  legal_ttt 1.1156
BigramHash 4096×64   · VE7-10  · FOMAML every=4 · TTT AdamW+flat (size-opt)        →  legal_ttt 1.1169  ← worse
BigramHash 4096×64   · VE7-10  · FOMAML every=4 · pos-cond bigram + trigram (ours) →  legal_ttt 1.1159  ← current parent
+ copy head wired into FOMAML outer loop                                             →  legal_ttt 1.1214  ← much worse
```

The pattern "more meta-TTT intensity → worse bpb" appeared three times but was never
tested causally. All comparisons confounded meta-TTT with other architectural changes.

**This experiment** isolates meta-TTT: identical architecture, identical schedule, one
flag changed. Every other hyperparameter is byte-identical to exp101 (including
`TRIGRAM=0`, which was the exp101 variant that achieved 1.1159).

### What meta-TTT is doing in exp101

Meta-TTT (FOMAML) runs every 4 training steps. For each meta-step it:
1. Copies the current bank parameters into detached clones `banks'`
2. Runs one SGD inner step on the current batch: `banks' ← banks - α·∇L(banks; x)`
3. Evaluates the outer loss with adapted banks: `L_meta = L(banks'; x)` (same batch)
4. Accumulates `∇_banks L_meta` into the regular bank gradient

The goal: shape bank initializations so that a single TTT step at eval time moves them
further toward the test distribution.

The cost: ~3% extra compute per training step (one extra forward + backward + parameter
clone every 4 steps).

---

## 2. Maths

The FOMAML objective as implemented in exp101:

$$
\theta' = \theta - \alpha \nabla_\theta \mathcal{L}(\theta;\, x_\text{batch})
$$

$$
\mathcal{L}_\text{meta} = \mathcal{L}(\theta';\, x_\text{batch})
$$

$$
g_\text{meta} = \nabla_\theta \mathcal{L}_\text{meta}
\approx \nabla_{\theta'} \mathcal{L}(\theta';\, x_\text{batch})
\quad \text{(first-order: Jacobian of inner step dropped)}
$$

$g_\text{meta}$ is added (scaled by `META_TTT_LOSS_WEIGHT=0.5`) to the standard
gradient before the Muon/Adam update.

Note that **inner and outer use the same batch $x_\text{batch}$**. This is the key
design flaw that exp106 addresses.

---

## 3. Implementation

Single change in `run.sh`:

```bash
# exp101
export META_TTT_ENABLED=1

# exp105a (this experiment)
export META_TTT_ENABLED=0
```

All other env vars are unchanged. The `META_TTT_INNER_LR`, `META_TTT_EVERY`,
`META_TTT_LOSS_WEIGHT`, `META_TTT_FREEZE_BLOCKS` vars are still exported but have
no effect when `META_TTT_ENABLED=0`.

No `train_gpt.py` changes — the env var guard is already in exp101's codebase.

---

## 4. Analysis

### Results table

| Metric | exp101 (meta-TTT ON) | exp105a (meta-TTT OFF) | Δ |
|---|---|---|---|
| Steps completed | 7020 / 7500 | 7226 / 7500 | — |
| val_bpb @ step 3000 | 1.2254 | 1.2264 | +0.0010 |
| val_bpb @ step 6000 | 1.1474 | 1.1524 | +0.0050 |
| val_bpb @ final step | 1.1349 | 1.1351 | +0.0002 |
| Post-EMA val_bpb | 1.1352 | 1.1353 | +0.0001 |
| **Int6 val_bpb (exact)** | **1.13930** | **1.13956** | **+0.0003** |
| **legal_ttt val_bpb (exact)** | **1.11588** | **1.11624** | **+0.00036** |
| TTT delta (int6 → TTT) | −0.02342 | −0.02331 | +0.00011 |
| Model size (int6+lzma) | 15,689,152 B (14.97 MB) | 15,659,520 B (14.94 MB) | — |
| Total submission size | 15,804,196 B (15.08 MB) | 15,774,564 B (15.05 MB) | — |
| Peak GPU memory | 23,044 MiB | 23,043 MiB | — |
| late_qat fired at step | 5384 | 5557 | — |
| SWA started at step | 5600 | 5750 | — |

*Submission size = int6+lzma weights + train_gpt.py code (122,683 B).*

### Key observations

**1. Training-time loss: identical.**
Post-EMA bpb 1.1352 vs 1.1353 — difference of 0.0001, well within seed noise.
Meta-TTT does not impair or improve training convergence.

**2. TTT delta: identical.**
Both models improve by ~0.0233 bpb from int6 baseline to legal_ttt (0.02342 vs 0.02331).
The meta-training did not cause the banks to generalize better under TTT.

**3. Net meta-TTT value: +0.00036 bpb at ~3% compute cost.**
This is noise-level (sub-0.001 bpb). The ablation verdict: **meta-TTT in its exp101
formulation adds no meaningful value.**

**4. exp105a is actually slightly faster per step.**
Without the FOMAML overhead, exp105a completed 7226 steps vs exp101's 7020 in the same
80-minute wallclock — 206 extra steps (~3% more training) from eliminating meta-TTT.

### Why the FOMAML signal is ineffective

The root cause is the **same-batch inner/outer** design:

- Inner step: `banks' ← banks - α·∇L(banks; x_batch)` adapts to `x_batch`
- Outer evaluation: `L(banks'; x_batch)` also evaluated on `x_batch`

The meta-gradient is rewarding banks that can "recover" from one SGD step on a batch
they just saw. This is trivially solved by having banks with small gradient norms —
i.e., banks that are *already* well-converged on the training distribution. The
FOMAML signal is not asking banks to generalize to new data; it's asking them not to
move much under SGD.

At eval time, TTT adapts to a new test chunk the model has never seen. The meta-
training objective does not match this deployment regime.

### Weight-space analysis (exp101 vs exp105a)

See `../META_TTT_ANALYSIS.md` for the full 5-analysis comparison. Summary:

| Analysis | Finding |
|---|---|
| Weight deltas | Banks near-orthogonal element-wise (rel_L2 ≈ 1.37, cosine ≈ 0.07) — Muon trajectories diverged |
| Quantization sensitivity | Essentially identical (ratio 0.9989) — meta-TTT does NOT reduce quant error |
| Spectral regularizer | Condition number −8.2% (5.6 vs 6.1) — only real signal from meta-TTT |
| Subspace overlap | kv_bank avg cos 0.955 — same principal subspace despite orthogonal element-wise weights |
| Linear mode connectivity | Midpoint norm ratio 0.799 — borderline different basins |

---

## 5. Conclusion

Meta-TTT as formulated in exp101 (FOMAML, same-batch inner/outer) provides **+0.0003
bpb** post-TTT improvement at **~3% training compute overhead**. The ablation
verdict is clear: the current formulation is not worth the cost.

The fundamental issue is objective misalignment: same-batch FOMAML trains banks to
resist SGD updates on seen data, not to adapt to unseen test-time data. The two regimes
(training distribution vs test distribution) are different enough that the meta-signal
is near-zero.

**This motivates exp106**, which addresses three concrete redesign points:
- **(A)** Cross-chunk split: inner/outer use different documents from the batch
- **(B)** Δ-loss outer: explicitly reward improvement from the inner step
- **(C)** MetaSGD: learn per-layer-per-bank inner-loop LR scales (~66 params, excluded from export)

---

## Run

```bash
bash records/phase3/exp105a_no-metattt_from_exp101/run.sh
```

Hardware: **1× H100 80 GB SXM**, `MAX_WALLCLOCK_SECONDS=4800` (80-minute cap).
A single H100 running for 80 minutes = 4800 GPU-seconds, matching the throughput
of the competition's standard 8×H100 @ 10-minute budget at substantially lower cost.
Steps completed: **7226 / 7500** — 206 more steps than exp101 because eliminating
FOMAML overhead freed ~3% compute per step.

---

## TL;DR

Disabling FOMAML meta-TTT entirely changes legal_ttt by only +0.00036 bpb (1.11624 vs exp101's 1.11588) — noise level. The meta-training objective in exp101 was fundamentally misaligned: same-batch inner/outer FOMAML trains banks to resist SGD updates on data they've already seen, not to generalize to unseen test chunks. This ablation confirms meta-TTT at its current formulation adds no meaningful value, and motivates the three-part redesign in exp106. The run used a single H100 for 80 minutes (= 4800 GPU-seconds, iso-compute with the competition's 8×H100 @ 10-min budget) and completed 7226 steps — 206 more than exp101 due to the eliminated FOMAML overhead.
