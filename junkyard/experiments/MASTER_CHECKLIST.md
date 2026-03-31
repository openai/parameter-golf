# Master Checklist
_2026-03-31 | Target: #1_

---

## Active Right Now

- [ ] **BW5 seed=300** — confirmation run firing
  - `SEED=300 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_V/run.sh`
  - Pass: both seeds confirm 1.18672 range → ready to submit updated crawler PR

---

## Next Up

- [ ] **BWV-Cannon gate** — single GPU, 4 arms (none/scalar/channel/rmsnorm)
  - `bash experiments/Bandit_Wagon_V_Cannon/gate_1gpu.sh`
  - Pass criteria: any cannon arm beats BWVC-00 control
  - If passes → `Bandit_Wagon_V_Cannon` full run on 8×H100

---

## Crawler Submission Pipeline

- [x] Leg 3 submitted (PR #1140) — 1.18742 mean
- [x] BW5 seed=444 — 1.18672 (-0.00070 vs submission)
- [ ] BW5 seed=300 — confirmation pending
- [ ] Update PR #1140 with confirmed 2-seed result
- [ ] If cannon promotes: `Bandit_Wagon_V_Cannon` → new PR

---

## Inference Acceleration

| Tier | What | Status |
|------|------|--------|
| 1 | COMPILE_FULLGRAPH=1 | **DONE — baked into BW5** |
| 2 | CUDA graph for sliding window eval | Not started — 64s → ~50s target |
| 3 | Nitrust Rust crawler module | Not started — open questions on scope |

**Tier 2 prerequisite:** BW5 confirmed first.

---

## Future Architecture (post-cannon)

- [ ] **Delta Farce (BDF series)** — per-loop dynamic causal anchoring
  - 5 arms: BDF-00 through BDF-04
  - Prerequisite: cannon (BWE/BWVC) validated first
- [ ] **Tap (BWT series)** — static encoder anchors
  - Not designed yet

---

## SOTA Garage

| Track | Model | BPB | Size | Status |
|-------|-------|-----|------|--------|
| Neural | Rascal II | 1.10987 | 15.44MB | Submitted |
| **Crawler** | **BW5 seed=444** | **1.18672** | **8.61MB** | **Confirming** |
| Compression | FX_WING_DELTA | 0.2233 | — | Model lost — needs re-run |

---

## Archive

All superseded Bandit_Wagon experiments moved to `experiments/archive/`.
Active: `Bandit_Wagon_V`, `Bandit_Wagon_V_Cannon`.
