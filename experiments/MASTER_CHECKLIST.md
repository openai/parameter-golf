# Master Progress Checklist
_Updated: 2026-03-31_

---

## ACTIVE: BW5 Production Run

- [x] BW4 gate — baseline 74.80ms/step, 1.18730643 int6_sw_bpb (seed=444)
- [x] Tier 1 gate — COMPILE_FULLGRAPH=1 validated (74.51ms, 0 graph breaks, 2.77× faster roundtrip eval)
- [x] BW5 run.sh — BW4 + COMPILE_FULLGRAPH=1 (one variable, verified)
- [x] **BW5 seed=444 production run** — **1.18672385** ✓
  - 8035 steps, raw_bpb 1.1987, quant_gap -0.0120, 8.61MB
  - vs BW4: **-0.00058** | vs Leg 3 SOTA: **-0.00074**
- [x] Record BW5 seed=444 results in `experiments/Bandit_Wagon_V/RESULTS.md`
- [ ] BW5 seed=300 confirmation run (after seed=444 lands)
  - Runner: `SEED=300 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_V/run.sh`
  - Or full multi-seed: `NPROC_PER_NODE=8 bash experiments/Bandit_Wagon_V/run_multi_seed.sh`
- [ ] Record BW5 seed=300 results

---

## Crawler SOTA Submission (when BW5 confirmed)

- [ ] Confirm int6_sw_bpb beats Leg 3 SOTA (1.18746 reference)
- [ ] Verify bytes ≤ 16MB
- [ ] Create submission branch from BW5 best seed
- [ ] Build submission.json + logs + README (per submission checklist)
- [ ] PR to Open-parameter-golf-1 → openai/parameter-golf

---

## Cannon Ablations (BWE series)
_These were run on BW3 arch (pyramid + 9,1,1). Need re-gate on BW5 base if cannon promotes._

- [x] BWE-01 — cannon arm 1 (results in)
- [x] BWE-02 — cannon arm 2 (results in)
- [ ] BWE-03 — rmsnorm cannon (results pending — pod may have completed)
- [ ] Evaluate cannon signal: does it survive to full 600s?
- [ ] If cannon promotes: gate on BW5 base (1 variable, isolated test)

---

## Inference Acceleration — 3-Tier Plan

### Tier 1 — COMPILE_FULLGRAPH=1
- [x] Gate validated: -0.28ms/step, 2.77× faster roundtrip, 12.5% faster sliding window, 0 graph breaks
- [x] Baked into BW5 production config
- [ ] Confirm at full production scale (step_avg should settle below 74.52ms)

### Tier 2 — CUDA Graph for Sliding Window Eval (Python, no Rust)
- [ ] Scope: capture `forward_logits()` with fixed window shape, replay for all 961K eval windows
- [ ] Guard: `CUDA_GRAPH_EVAL=1` env var
- [ ] Implementation location: `train_gpt.py` eval loop
- [ ] Gate: compare sliding window eval time (baseline 64,274ms post-fullgraph)
- [ ] Expected: 50–60s range (depends on CPU vs GPU split)
- [ ] **Prerequisite**: BW5 production results confirmed first

### Tier 3 — Nitrust Crawler Module (Rust, longer term)
- [ ] Decide: training-time, inference-time, or both?
- [ ] Decide: tch-rs matmuls vs raw cuBLAS FFI?
- [ ] Confirm: Rust toolchain + CUDA toolkit on pods?
- [ ] Write `Nitrust/rust/Cargo.toml`
- [ ] Write `Nitrust/rust/src/lib.rs` (module entry)
- [ ] Write `Nitrust/rust/src/crawler.rs` (3-loop forward, RoPE precompute)
- [ ] Write `Nitrust/rust/src/weights.rs` (weight struct preloader)
- [ ] Integration: `NITRUST_ENABLE=1` path in train_gpt.py
- [ ] Gate vs Python fallback

---

## Future Architecture (post-BWCS, post-BWE validation)

### Delta Farce (BDF series) — Dynamic Per-Loop Causal Anchoring
_Prerequisite: cannon (BWE) validated first. Do not combine unvalidated components._
- [ ] BDF-00: control repin
- [ ] BDF-01: anchor_dim=32, loop→loop only (minimal)
- [ ] BDF-02: anchor_dim=64, loop→loop only
- [ ] BDF-03: anchor_dim=32, symmetric (send+catch all boundaries)
- [ ] BDF-04: anchor_dim=32 + tap-seeded (combined)

### Tap (BWT series) — Static Encoder Anchors
_Status: concept stage, no scripts yet_
- [ ] Define tap architecture
- [ ] Gate on BW5 base

---

## Garage Status

| Model | BPB | Size | Status |
|-------|-----|------|--------|
| Rascal II (neural) | 1.10987 | 15.44MB | SOTA — submitted |
| Leg 3 (crawler) | 1.18746 | 8.84MB | Previous SOTA |
| BW4 (battery, no choke) | 1.18731 | 8.97MB | Current crawler SOTA |
| **BW5 seed=444 (BW4 + fullgraph)** | **1.18672** | **8.61MB** | **NEW CRAWLER SOTA** |
| FX_WING_DELTA (compression) | 0.2233 | — | SOTA — model lost, needs re-run |

---

## Standing Rules (not tasks — always apply)

- ONE variable per ablation. Always. Non-negotiable.
- Gate (2k steps) before every 8×H100 full run. No exceptions.
- Copy final_model.pt to unique checkpoint name after every run.
- Never run from TEST_LAB branch for submissions. Dedicated branch only.
- $15/race budget. Every run needs a validated reason.
