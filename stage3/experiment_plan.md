# Stage 3 Experiment Plan

## 1. Objective and Hard Constraints

**Objective**: Minimize val_bpb on fineweb validation set.

**Current**: 1.1631 BPB (rank 19). **Frontier**: 1.1453 (rank 1). **Gap**: 0.018.

| Budget | Value |
|--------|-------|
| Compute | 8xH100 SXM, 600s wallclock per decision run |
| Artifact | 16MB (code + compressed model) |
| Decision | ~4 full 8xH100 decision runs affordable. Screens are cheap (1-GPU, 180s). |

## 2. Score Decomposition into Causal Lanes

```
final_bpb = pre_quant_quality + quant_penalty - eval_lift
```

| Lane | Mechanism | Cheap Signal | Expensive Confirmation |
|------|-----------|-------------|----------------------|
| **A: Training dynamics** | Optimizer, init, architecture, schedule | 1-GPU 180s val curve vs matched control | 8xH100 600s final val_bpb |
| **B: Export quality** | Quantization scheme, compression, SWA | Same checkpoint, multiple export variants | Full train + export roundtrip |
| **C: Eval policy** | Sliding window, doc-isolation, TTT | Same checkpoint, multiple eval configs | Full pipeline end-to-end |
| **D: Systems throughput** | FA3, compile mode, warmup reduction | ms/step measurement, step count | Full run step count vs control |

Lanes B and C can share one checkpoint. Lane A and D require training runs. Do not mix lanes in the same screen pack.

## 3. Attack Surface Map

| Surface | Editable Region | Causal Role | Effect | Cheap to Test? |
|---------|----------------|-------------|--------|---------------|
| Optimizer law | Muon step, Newton-Schulz | Per-step convergence quality | 0.005-0.015 | Yes (train-screen) |
| Weight init | _init_weights, module constructors | Early convergence speed | 0.003-0.005 | Yes (train-screen) |
| Architecture | GPT forward, new modules | Representation capacity | 0.005-0.010 | Yes (train-screen) |
| Quantization | CastedLinear.forward, export code | Compression penalty | 0.002-0.015 | Partially (need export roundtrip) |
| LR schedule | lr_mul function | Convergence trajectory | 0.001-0.003 | Yes (train-screen) |
| Training loop | Main loop, SWA, aux losses | Optimization extras | 0.002-0.005 | Partially (SWA needs export) |
| Eval policy | eval_val, sliding window, TTT | Post-hoc score lift | 0.005-0.015 | Yes (checkpoint bakeoff) |
| Systems throughput | torch.compile, attention kernel, warmup | Steps per wallclock | 0.002-0.008 | Yes (ms/step) |
| Loss function | cross_entropy call | Training signal quality | 0.001-0.005 | Yes (train-screen) |
| Compressor | zlib vs zstd | Artifact bytes | 0.002 (indirect) | Yes (one-shot) |

## 4. Mechanism Families (from Discovery Phase)

| ID | Family | Causal Story | Est. BPB Gain |
|----|--------|-------------|---------------|
| OPT | Optimizer discipline | NorMuon + MuonWD + grad clip → better convergence + smaller weights → better quant | 0.005-0.015 |
| BIG | Bigram injection | SmearGate + BigramHash → cheap bigram features the model can't learn in 10min | 0.005-0.010 |
| QAT | Quant-aware training | STE fake int6 in forward → model learns to survive quantization | 0.002-0.005 |
| INIT | Init quality | OrthoInit + muP → faster early convergence | 0.003-0.005 |
| CAP | Capacity reallocation | 10L + int5/int6 + zstd-22 → more model in 16MB | 0.005-0.010 |
| SWA | Checkpoint ensemble | Average warmdown checkpoints → smoother weights → better quant | 0.002-0.005 |
| SYS | Systems throughput | FA3 + compile autotune + warmup reduction → more steps in 600s | 0.002-0.008 |
| EVAL | Eval lift | Stride=64 + doc-isolation + TTT tuning → better post-hoc scoring | 0.005-0.015 |
| LSMOOTH | Label smoothing | Softer targets → smaller logits → better quant | 0.001-0.005 |
| ANTI | Throughput-first | Simpler model + 25% more steps — challenges the technique-stacking narrative | 0-0.010 |

## 5. Coverage Matrix

| Family | Optimizer | Init | Architecture | Quant/Export | Schedule | Loop | Eval | Throughput | Loss |
|--------|-----------|------|-------------|-------------|----------|------|------|------------|------|
| OPT | **primary** | none | none | secondary | none | none | none | none | none |
| BIG | none | none | **primary** | secondary | none | none | none | none | none |
| QAT | none | none | none | **primary** | none | none | none | none | none |
| INIT | none | **primary** | none | secondary | none | none | none | none | none |
| CAP | none | none | **primary** | **primary** | none | none | none | none | none |
| SWA | none | none | none | **primary** | none | **primary** | none | none | none |
| SYS | none | none | none | none | none | none | none | **primary** | none |
| EVAL | none | none | none | none | none | none | **primary** | none | none |
| LSMOOTH | none | none | none | secondary | none | none | none | none | **primary** |
| ANTI | none | none | secondary | none | none | none | none | **primary** | none |

## 6. Set-Level Gaps and Course Corrections

**Inspection**:

| Surface | Coverage | Status |
|---------|----------|--------|
| Optimizer | OPT (primary) | Covered |
| Init | INIT (primary) | Covered |
| Architecture | BIG, CAP (primary) | Covered |
| Quant/Export | QAT, CAP, SWA (primary); OPT, INIT, BIG, LSMOOTH (secondary) | **Over-covered** — 3 primary + 4 secondary |
| Schedule | None | **UNCOVERED** |
| Loop | SWA (primary) | Covered |
| Eval | EVAL (primary) | Covered |
| Throughput | SYS, ANTI (primary) | Covered |
| Loss | LSMOOTH (primary) | Covered |

**Gaps found**:

1. **Schedule surface is uncovered.** No family directly tests warmdown length, cosine vs linear warmdown, or LR warmup. However, these are knob-level changes within an established mechanism. **Decision**: defer to follow-on tuning after the main mechanism families are validated. Not worth a slot.

2. **Quant/export is over-covered.** QAT, CAP, SWA, and LSMOOTH all affect quant quality. **Decision**: this is fine because they attack compression through different causal mechanisms (train-time adaptation, size reallocation, weight smoothing, logit regularization). Keep all.

3. **OPT mixes three distinct changes (NorMuon, MuonWD, grad clip).** These have the same causal direction (tighter optimizer discipline) but different mechanisms. **Decision**: split OPT into OPT-a (NorMuon + MuonWD as one unit, since community always pairs them) and keep grad clip as an env-var baseline change applied to all candidates.

4. **ANTI overlaps with SYS.** Both claim "more steps via throughput." **Decision**: merge. SYS subsumes ANTI's throughput mechanism. Drop ANTI as a separate family. If SYS wins on throughput, the question "does simpler model + more steps beat technique stacking?" can be answered by comparing the SYS-only candidate against the composite winner.

5. **No wildcard from a distant region.** All families are community-validated. **Decision**: LSMOOTH serves as the wildcard — no community PR tests label smoothing, so it provides coverage of an untested region.

**Course-corrected family list (9 families)**:

| ID | Family | Lane |
|----|--------|------|
| OPT | NorMuon + MuonWD 0.04 | Train-screen |
| BIG | SmearGate + BigramHash | Train-screen |
| QAT | STE Int6 QAT | Train-screen |
| INIT | OrthoInit + muP | Train-screen |
| SWA | Checkpoint averaging (every 50 steps during warmdown) | Export bakeoff |
| SYS | FA3 + compile autotune + warmup=5 | Train-screen (ms/step) |
| CAP | 10L + int5/int6 + zstd-22 | Full aligned run |
| EVAL | Stride=64 + doc-isolation | Eval bakeoff |
| LSMOOTH | Label smoothing 0.05 | Train-screen |

## 7. Hypothesis Table

### Lane A: Training Screens (1-GPU, 180s)

All candidates apply `GRAD_CLIP_NORM=0.3` as a baseline change (free env-var, no attribution needed).

| Slot | Family | What Changes | Matched Control | Cheap Signal | Kill Rule | Promote If |
|------|--------|-------------|----------------|-------------|-----------|-----------|
| R0A | Control | SOTA stack baseline (seq2048, MLP 3x, int6, fp16 embed, stride=256) | R0B | val curve | — | — |
| R0B | Control repeat | Identical to R0A | R0A | val curve | — | R0A-R0B delta gives noise floor |
| H1 | OPT | NorMuon + MuonWD=0.04 | R0A | val_bpb at 180s, ms/step | Kill if val_bpb worse than R0A by > noise floor | Best val_bpb delta vs control |
| H2 | BIG | SmearGate + BigramHash | R0A | val_bpb at 180s | Kill if val_bpb worse or ms/step > 20% slower without val recovery | Best val_bpb delta vs control |
| H3 | QAT | STE Int6 fake quant in CastedLinear.forward | R0A | val_bpb at 180s (pre-quant may look worse — check post-quant if possible) | Kill if pre-quant val_bpb worse by > 0.01 (QAT hurts pre-quant, must check quant gap) | If pre-quant within 0.005 of control (gap recovers at export) |
| H4 | INIT | OrthoInit + muP 1/sqrt(2*layers) on proj weights | R0A | val_bpb at steps 50-100 (early convergence) and at 180s | Kill if early curve is worse AND final is worse | val_bpb beats control at 180s |
| H5 | LSMOOTH | label_smoothing=0.05 in cross_entropy | R0A | val_bpb at 180s | Kill if val_bpb worse by > noise floor | val_bpb beats control |
| H6 | SYS | FA3 + compile mode="max-autotune" + WARMUP_STEPS=5 | R0A | ms/step, total steps reached at 180s | Kill if ms/step is SLOWER (FA3 package issue or compile regression) | ms/step improves by > 5% |

### Lane B: Export Bakeoff (on best Lane A checkpoint, no retraining)

| Variant | What Changes | Matched Control | Signal | Kill Rule |
|---------|-------------|----------------|--------|-----------|
| B-ctrl | int6 + zlib-9 (current) | — | post-quant BPB | — |
| B-zstd | int6 + zstd-22 | B-ctrl | post-quant BPB, artifact size | Kill if bigger and not better |
| B-swa50 | SWA every 50 steps during warmdown + int6 + zstd-22 | B-ctrl | post-quant BPB | Kill if post-quant BPB worse |
| B-swa200 | SWA every 200 steps + int6 + zstd-22 | B-ctrl | post-quant BPB | Kill if post-quant BPB worse than B-swa50 |
| B-int5mlp | int5 MLP / int6 attn + zstd-22 | B-ctrl | post-quant BPB, artifact size | Kill if post-quant BPB worse despite smaller artifact |

Note: SWA requires training integration but the comparison is on the same training run — just whether to average checkpoints at export time. The training run for the best Lane A winner should collect checkpoints during warmdown for the SWA bakeoff.

### Lane C: Eval Bakeoff (on best Lane A+B checkpoint, no retraining)

| Variant | What Changes | Matched Control | Signal | Kill Rule |
|---------|-------------|----------------|--------|-----------|
| C-ctrl | stride=256, standard eval | — | val_bpb | — |
| C-s64 | stride=64 | C-ctrl | val_bpb, eval time | Kill if eval time > 5min (eats into budget) |
| C-s128 | stride=128 | C-ctrl | val_bpb, eval time | Kill if worse than both C-s64 and C-ctrl |
| C-dociso | doc-isolated eval + stride=64 | C-s64 | val_bpb | Kill if no improvement over C-s64 |
| C-ttt | TTT LoRA with tuned hyperparams (rank=16, lr=0.005) | C-s64 | val_bpb, eval time | Kill if eval time > 8min or gain < 0.001 |

## 8. Screen Pack Layout

### Pack 1: Lane A Training Screen (8 GPUs, 180s each)

```
GPU 0: R0A  (control)
GPU 1: R0B  (control repeat)
GPU 2: H1   (OPT: NorMuon + MuonWD=0.04)
GPU 3: H2   (BIG: SmearGate + BigramHash)
GPU 4: H3   (QAT: STE Int6)
GPU 5: H4   (INIT: OrthoInit + muP)
GPU 6: H5   (LSMOOTH: label_smoothing=0.05)
GPU 7: H6   (SYS: FA3 + compile autotune + warmup=5)
```

**Decision after Pack 1**:
- Compute noise floor from R0A vs R0B
- Kill any candidate worse than control by > noise floor
- Promote top 2-3 survivors

### Pack 2: Composite Screen (8 GPUs, 180s)

Stack the survivors from Pack 1 into composites. Example (assuming H1, H2, H4 survive):

```
GPU 0: S0A  (composite control: best single winner)
GPU 1: S0B  (composite control repeat)
GPU 2: S1   (winner1 + winner2)
GPU 3: S2   (winner1 + winner2 + winner3)
GPU 4: S3   (all survivors stacked)
GPU 5: S4   (all survivors + QAT, if QAT was ambiguous at 180s)
GPU 6: S5   (all survivors + SYS throughput, if SYS won on speed)
GPU 7: S6   (wild: all survivors + LSMOOTH, if LSMOOTH showed signal)
```

**Decision after Pack 2**:
- Identify best composite
- Promote to full decision run

### Pack 3: Lane B Export Bakeoff (no GPU training needed)

Run on the best Pack 2 composite's checkpoint:
- B-ctrl, B-zstd, B-swa50, B-swa200, B-int5mlp
- All 5 are just export variations on the same checkpoint
- Takes minutes, not hours

### Pack 4: Lane C Eval Bakeoff (1-2 GPUs, minutes each)

Run on the best Pack 3 artifact:
- C-ctrl, C-s64, C-s128, C-dociso, C-ttt
- Each is just a different eval pass on the same artifact

### Pack 5: Full Decision Run (8xH100, 600s)

The complete stack: best composite + best export + best eval policy.
Run 3 seeds for statistical significance (p < 0.01 vs current best).

## 9. Run Sequence

```
Step 1: Code Implementation (~2-3 hours)
  - Implement all 6 Lane A candidates in train_gpt.py
  - Each gated by env var so they can be toggled independently
  - Verify all pass a local smoke test (10 steps)

Step 2: Pack 1 — Lane A Screen (30 min total)
  - Launch 8 parallel 1-GPU 180s training runs
  - Collect: val_bpb at 180s, ms/step, steps reached, learning curve
  - Decision: kill/promote based on rules in Section 7

Step 3: Pack 2 — Composite Screen (30 min total)
  - Stack survivors from Pack 1
  - Launch 8 parallel 1-GPU 180s runs
  - Decision: identify best composite

Step 4: Full Training Run of Best Composite (10 min)
  - 8xH100 600s decision run
  - Collect checkpoints during warmdown (every 50 steps) for SWA bakeoff
  - Save final checkpoint

Step 5: Pack 3 — Export Bakeoff (5 min)
  - Run 5 export variants on the saved checkpoint
  - Decision: identify best export config

Step 6: Pack 4 — Eval Bakeoff (10 min)
  - Run 5 eval variants on the best artifact
  - Decision: identify best eval config

Step 7: Final 3-Seed Run (30 min)
  - Best composite + best export + best eval
  - 3 seeds on 8xH100 600s
  - Compute mean, std, p-value vs frontier

Total wall time: ~2 hours (excluding implementation)
Total GPU-hours: ~25 H100-hours
```

## 10. Interpretation Rules

### For Lane A screens (180s, 1-GPU)

**Good signals**:
- Delta val_bpb at matched wallclock (most reliable)
- Delta val_bpb at matched step count (controls for speed differences)
- Learning curve shape in first 100 steps (early convergence for INIT)
- ms/step (for SYS)

**Bad signals**:
- Absolute val_bpb value (noisy at 180s, use relative delta)
- Post-quant BPB at 180s (quant gap may not be observable in short runs for QAT)
- Final-step-only comparison (could be schedule artifact)

**Noise calibration**:
- R0A vs R0B delta gives the noise floor
- Any candidate improvement must exceed this floor by 2x to be credible
- If R0A-R0B delta > 0.005, the screen is too noisy — extend to 300s

### For Lane B export bakeoffs

- Compare post-quant BPB, not pre-quant
- Also compare artifact size — a smaller artifact at same BPB is a win (can fund more params later)
- SWA benefit is primarily visible in post-quant BPB, not pre-quant

### For Lane C eval bakeoffs

- Compare final val_bpb after full eval pass
- Also check eval wall time — must fit within the eval budget
- Stride=64 eval takes ~4x longer than stride=256. Verify it fits.

### Composability assumptions

- OPT + INIT likely compose well (different mechanisms: optimizer vs init)
- OPT + QAT likely compose well (WD helps quant + QAT helps quant — different routes)
- BIG + INIT may partially overlap (both improve early representations)
- SYS composes with everything (pure throughput, orthogonal)
- SWA composes with everything (post-training averaging, orthogonal)
- LSMOOTH + QAT: unclear interaction — label smoothing changes weight distributions, QAT adapts to quantization. Could help or conflict.

### When to stop

- If the best composite at 600s on 8xH100 beats 1.145, we're done — submit.
- If the best composite is 1.145-1.150, run 3 seeds for significance.
- If the best composite is 1.150-1.160, we closed the gap but didn't win. Consider CAP (10L + int5/int6) as the next move.
- If the best composite is > 1.160, the composite didn't compose well. Debug which interaction failed.

---

## Follow-On Tuning (after mechanism validation)

These are knob-level changes within validated mechanisms. Do not test before the mechanism family survives.

| Mechanism | Tuning Knob | Range | When |
|-----------|------------|-------|------|
| OPT | MuonWD value | 0.02, 0.03, 0.04, 0.05 | After OPT survives Pack 1 |
| OPT | matrix_lr (higher with WD) | 0.02, 0.025, 0.03 | After OPT survives Pack 1 |
| BIG | BigramHash bucket count | 2048, 4096, 8192 | After BIG survives Pack 1 |
| BIG | BigramHash projection dim | 64, 128, 256 | After BIG survives Pack 1 |
| QAT | Quantization bit-width | 5, 6, 7 | After QAT survives Pack 1 |
| SWA | Checkpoint frequency | 25, 50, 100, 200 | After SWA survives Pack 3 |
| SYS | torch.compile mode | default, reduce-overhead, max-autotune | After SYS survives Pack 1 |
| LSMOOTH | Smoothing value | 0.01, 0.05, 0.1 | After LSMOOTH survives Pack 1 |
| EVAL | Stride | 32, 64, 128, 256 | After EVAL survives Pack 4 |
| EVAL | TTT rank | 4, 8, 16, 32 | After TTT survives Pack 4 |
| Schedule | warmdown_iters | 2000, 3000, 4000 | After best composite identified |
| Schedule | warmdown shape | linear, cosine | After best composite identified |
