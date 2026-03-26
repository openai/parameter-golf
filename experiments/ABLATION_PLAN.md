# Ablation Plan — Isolating Transferable Findings

Generated 2026-03-24. All experiments use EXPLAIN mode (one variable per run).

---

## RC-0: Baseline Anchors (must run first)

### RC-0a: Frugendorff v2 reproduction
- **parent:** none (anchor)
- **config:** 6L × 2 loops, dim=640, 10H/5KV, MLP 4x, fixed cadence 2, per-row quant, no gate, no VE
- **script:** train_gpt_frugendorff_v2.py
- **purpose:** Establish symmetric baseline number at full scale (600s, 8xH100)
- **existing result:** 1.1478 sliding (1 seed, unverified)
- **status:** NEEDS VERIFICATION at full scale

### RC-0b: Micro crawler clean baseline
- **parent:** RC-0a
- **config:** 4f+2c×2, dim=640, fixed cadence 2, per-row quant, no gate, no VE, random loop pos
- **variable vs RC-0a:** topology only (symmetric → asymmetric)
- **hypothesis:** Topology change alone accounts for most of the 0.010 gap
- **expected:** 1.140–1.143 if topology matters, ~1.147 if not
- **status:** NOT RUN

### RC-0c: Flat-only control (no recursion)
- **parent:** RC-0a
- **config:** 8 unique flat layers, dim=640, no crawler, no looping, per-row quant
- **variable vs RC-0a:** remove all recursion, match effective depth
- **hypothesis:** Establishes what 8 unique layers can do without any weight sharing
- **purpose:** If this beats both Frugendorff and crawler, recursion is net negative
- **status:** NOT RUN

---

## H1: EMA Instability from Parameter Reuse

**Claim:** Frequent double-firing creates weight oscillation that EMA can't track. EMA gap scales with reuse frequency.

**Prior evidence:** Cadence ablation at 0.25 scale — EMA gap 0.105 (cad1) → 0.053 (cad4). Strong, monotonic.

### H1a: Full-scale cadence 4 vs cadence 2
- **parent:** RC-0b
- **config:** RC-0b + fixed cadence 4
- **variable:** cadence (2 → 4)
- **hypothesis:** Cadence 4 reduces EMA gap and improves sliding BPB at full scale
- **expected:** 0.002–0.005 BPP improvement over cadence 2
- **failure risk:** 0.25-scale ranking may not hold at 600s
- **metrics:** sliding_bpb, post_ema_bpb, val_bpb_at_stop, quant_gap, steps_completed

### H1b: Cadence infinity (crawler fires single only, never double)
- **parent:** RC-0b
- **config:** RC-0b + cadence=999999 (never a C step, always single-fire N)
- **variable:** cadence (2 → inf)
- **hypothesis:** If cad-inf beats cad4, the crawler's double-fire is pure overhead
- **expected:** If true, crawler recurrence has zero value. If cad4 > cad-inf, there's a sweet spot.
- **failure risk:** Single-fire crawler may be equivalent to flat layers (wasted architecture)
- **metrics:** same as H1a + compare to RC-0c (flat-only)

### H1c: Per-group EMA decay (fix, not avoid)
- **parent:** RC-0b (cadence 2)
- **config:** RC-0b + separate EMA decay for flat params (0.997) vs crawler params (0.999)
- **variable:** EMA decay (uniform → split)
- **hypothesis:** Slower EMA on crawler params reduces the oscillation damage without reducing cadence
- **expected:** Reduces EMA gap at cadence 2 by 0.02–0.04
- **failure risk:** Slower EMA may also delay convergence tracking
- **metrics:** post_ema_bpb, ema_gap (val_at_stop vs post_ema)

---

## H2: Training Dynamics → Quantization Robustness

**Claim:** Cadence controls quantization gap independently of float quality. Heavy reuse creates multi-modal weight distributions with outliers that break fixed-point quantization.

**Prior evidence:** Quant gap 0.030 (cad1) → 0.006 (cad4) on H1. 5× reduction.

### H2a: Weight distribution analysis (diagnostic, no training)
- **parent:** H1a, H1b completed runs
- **config:** Load saved checkpoints from cad1, cad2, cad4, cad-inf
- **variable:** none (analysis only)
- **purpose:** Plot per-layer weight histograms, measure kurtosis, outlier rate, entropy. Confirm mechanism: does heavy reuse actually produce multi-modal distributions?
- **metrics:** per-layer kurtosis, outlier fraction (>3σ), histogram entropy, GPTQ reconstruction error by layer

### H2b: Quantize float-matched models
- **parent:** H1a, RC-0b completed runs
- **config:** Take the FLOAT checkpoint (pre-EMA) from cadence 2 and cadence 4 runs. Apply identical GPTQ to both.
- **variable:** none (controlled comparison)
- **purpose:** If quant gap difference persists even on float checkpoints (not EMA), the effect is in the weight distribution, not EMA quality.
- **metrics:** quant_gap on float checkpoint, quant_gap on EMA checkpoint

---

## H3: Bidirectional Learned State vs Detached Buffers

**Claim:** Gradients must flow both IN and OUT of shared state for multi-path communication to work. Detached buffers kill the signal.

**Prior evidence:** Run 8 (bidir PD) 1.1355 vs Run 6 (detached PD) 1.1375. Promising but confounded with cadence change.

### H3a: Detached PD at fixed cadence 2
- **parent:** RC-0b
- **config:** RC-0b + PD gate with detached EMA consensus_ref, fixed cadence 2
- **variable vs RC-0b:** add detached PD gate
- **hypothesis:** Detached PD at fixed cadence gives modest or no improvement
- **expected:** ~neutral to +0.001 improvement
- **metrics:** sliding_bpb, delib_scale trajectory, cosine similarity between loop outputs

### H3b: Bidirectional PD at fixed cadence 2
- **parent:** RC-0b
- **config:** RC-0b + PD gate with learned nn.Parameter consensus_ref, fixed cadence 2
- **variable vs H3a:** consensus_ref (detached buffer → learned Parameter)
- **hypothesis:** Bidirectional gradient flow is strictly better than one-way
- **expected:** 0.001–0.003 improvement over H3a
- **failure risk:** The improvement may be too small to detect in 1 seed
- **metrics:** same as H3a + consensus_ref gradient norm over training

### H3c: Bidirectional PD at cadence 4
- **parent:** H1a (cadence 4 baseline)
- **config:** H1a + learned consensus_ref PD gate
- **variable vs H1a:** add bidirectional PD
- **hypothesis:** PD helps even at reduced crawl frequency
- **expected:** If PD still helps at cad4, the mechanism is real. If not, PD only matters when crawling is frequent enough to create conflict.
- **metrics:** sliding_bpb, quant_gap

---

## H4: Selective ±1 Pruning

**Claim:** Zeroing low-impact ±1 quantized values improves compression without meaningful quality loss.

**Prior evidence:** Implemented in streaker, functional, but never isolated for quality impact.

### H4a: Pruning impact on quality (sweep)
- **parent:** Any completed full-scale run with GPTQ
- **config:** Take a fixed quantized checkpoint. Apply pruning at 0%, 5%, 10%, 25%, 50% of ±1 values. Evaluate each.
- **variable:** pruning fraction
- **purpose:** Map the Pareto curve: how much quality do we lose per byte saved?
- **metrics:** sliding_bpb at each pruning level, artifact_bytes, compression_ratio
- **note:** No training needed. Pure post-hoc analysis.

### H4b: Pruning vs re-quantization
- **parent:** H4a
- **config:** Compare pruning N values vs reducing clip range vs increasing block size to hit the same artifact target
- **variable:** compression method
- **purpose:** Is pruning actually better than just tuning GPTQ parameters?
- **metrics:** sliding_bpb at matched artifact sizes

---

## H5: Compute Consistency vs Scheduling

**Claim:** Fixed computational load per step beats varying it during training.

**Prior evidence:** Fixed cadence beats tapered cadence in the 0.25-scale sweep. But confounded with EMA instability.

### H5a: Tapered cadence 2/4/6 vs fixed cadence 3
- **parent:** RC-0b
- **config A:** RC-0b + tapered cadence (early=2, main=4, late=6)
- **config B:** RC-0b + fixed cadence 3 (all phases)
- **variable:** cadence schedule (tapered vs fixed)
- **hypothesis:** Fixed cadence 3 beats tapered 2/4/6 despite same average crawl frequency
- **expected:** Fixed wins by 0.001–0.003 due to EMA consistency
- **failure risk:** If tapered wins, the "vary compute" principle has nuance
- **metrics:** sliding_bpb, ema_gap, steps_completed

---

## H6: Asymmetric Parameter Allocation

**Claim:** More unique + fewer shared parameters beats balanced sharing.

**Prior evidence:** 4f+2c×2 beats 3f+3cx2 by 0.019 at cad4 (0.25 scale). Consistent across all cadences.

### H6a: 5f+1cx2 (extreme asymmetric)
- **parent:** RC-0b
- **config:** 5 flat + 1 crawler × 2 = 7 effective, dim=640
- **variable vs RC-0b:** architecture (4f+2c → 5f+1c)
- **hypothesis:** Even more asymmetric is even better, up to some limit
- **expected:** If 5f+1c > 4f+2c, the trend continues. If worse, 4f+2c is the sweet spot.
- **metrics:** sliding_bpb, flat_params vs crawler_params ratio

### H6b: 6f+0c (no crawler, all flat)
- **parent:** RC-0c (flat-only control)
- **config:** 6 unique flat layers, dim=640, no sharing
- **purpose:** Same as RC-0c — the "recursion has zero value" test
- **note:** This is the same run as RC-0c. Listed here for lineage clarity.

---

## Execution Priority

**Phase 1 — Anchors (must run first, ~30 min on 8xH100):**
1. RC-0a (Frugendorff reproduction)
2. RC-0b (clean crawler baseline)
3. RC-0c (flat-only control)

**Phase 2 — Highest value ablations (~60 min):**
4. H1a (full-scale cad4)
5. H1b (cad-inf — does recursion help at all?)
6. H3a + H3b (detached vs bidirectional PD — isolate the confound)

**Phase 3 — Mechanism analysis (cheap, post-hoc):**
7. H2a (weight distribution analysis — no GPU needed)
8. H4a (pruning sweep — no training needed)
9. H2b (quant gap on float vs EMA checkpoints)

**Phase 4 — Secondary ablations (~60 min):**
10. H1c (per-group EMA decay)
11. H3c (PD at cadence 4)
12. H5a (tapered vs fixed cadence)
13. H6a (5f+1cx2 extreme asymmetric)

**Total estimated H100 time for phases 1-2:** ~90 min (9 runs × 10 min each)
**Total for full plan:** ~3 hours H100 + local analysis time

---

## Rules

- One variable per run. No exceptions unless marked OPTIMIZE.
- Save every checkpoint. Copy final_model.pt to unique name.
- Record: run_id, parent, variable, hypothesis, all standard metrics.
- If a result surprises, stop and investigate before continuing the plan.
- Do not combine winners until all Phase 2 ablations are complete.
