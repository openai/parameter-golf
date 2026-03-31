# Crawler Science Board

Track: Bandit_Wagon lineage · Goal: best int6_sw_bpb + smallest artifact ≤ 16MB
Champion: **1.18672385 BPB** (seed 444) · **8.61MB** · `crawler/2026-03-29_BW5/`

Legend: → PROMOTED · ✓ PASS · ✗ FAIL · ⏳ PENDING · — n/a

---

## Thread: Baseline — Crawler loops=3 / mlp=6.0

Established in Leg 3. This is the root config all BW legs descend from.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-29 | Leg 3 (CL3-01) | loops=3, mlp=6.0, SKIP_GPTQ=1, 600s wallclock | — | — | 1.18720 | 8.84MB | → PROMOTED | Extra time > GPTQ tricks. Quant gap nearly closed. 7MB headroom. |
| 2026-03-29 | BW4 | unknown | — | — | 1.18731 | 8.97MB | → PROMOTED | +0.00011 vs Leg 3 seed=444. Reference parent for BW5. |

---

## Thread: Compile / Fullgraph

Hypothesis: `torch.compile(fullgraph=True)` eliminates graph breaks → faster step → more steps in budget → lower BPB.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-29 | **BW5** (CHAMPION) | BW4 + COMPILE_FULLGRAPH=1 | — | ✓ 74.52ms | **1.18672385** | **8.61MB** | → PROMOTED | −0.00058 vs BW4. 0 graph breaks. Roundtrip eval 2.77× faster. Seed=300 ⚠️ +0.00012 vs Leg 3 (mean still better). |

Notes: BW5 seed=300 does NOT individually confirm vs Leg 3. Mean is better (1.18715 vs 1.18743). A future leg should try to close this seed disparity.

---

## Thread: Cannon (gate feed-forward type)

Hypothesis: scalar cannon feed-forward gives the crawler loop a faster compiled path and calibrated output scale.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | Full Run BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|------------------------|------|---------|-------------|
| 2026-03-31 | BW5_Cannon | BW5 + CRAWLER_CANNON_TYPE=scalar | ✓ | ✓ speed | 1.18692423 | 8.44MB | ✗ DOES NOT PROMOTE | Gate signal (−0.00016) reversed at full run (+0.00020 vs BW5). Cross-run variance swamped cannon signal. No step overhead. Size actually −179KB vs BW5 at full run. |

8GPU gate detail (seed=444, 2000 steps):
- BWVC-00 control: 74.84ms · raw_bpb 1.28870981 · int6_sw_bpb 1.28787686
- BWVC-01 scalar cannon: 74.81ms · raw_bpb 1.28854887 · int6_sw_bpb 1.28820687
- Delta: −0.03ms speed (passes) · raw_bpb −0.00016 · size +343KB

Full run detail (seed=444, 600s, 8034 steps):
- BW5_Cannon: 74.69ms · raw_bpb 1.1990 · int6_sw_bpb **1.18692423** · 8,845,120 bytes
- BW5 champion: 74.68ms · raw_bpb 1.1987 · int6_sw_bpb **1.18672385** · 9,024,399 bytes
- Delta: +0.01ms · +0.00020 BPB · −179KB (zstd artifact smaller despite cannon)

---

## Thread: MLP Choke Architecture

Hypothesis: pyramid-shaped MLP bottleneck (CRAWLER_MLP_CHOKE_DIM=512) gives the loop
more representational pressure at each recurrence → better quality.

VERDICT: Current implementation (CHOKE_DIM=512) is **incompatible**. Concept not dead — see notes.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-31 | BW5_Pyramid | BW5 + CHOKE_DIM=512, shape=pyramid, groups=8 | ✓ −0.00987 int6_sw_bpb | SKIPPED | — | — | ✗ CONCEPT DEFERRED | Proxy inflation trap. 1GPU signal strong but 8GPU proxy run (via PyramidCannon control) shows cold param burden dominates at 2000 steps. |

1GPU gate detail (500 steps, seed=444, grad_accum=8):
- BWVP-00 (flat): step_avg 583.99ms · int6_sw_bpb 1.44668780
- BWVP-01 (pyramid): step_avg 611.21ms · int6_sw_bpb 1.43681894
- Delta: +27.22ms (÷8 ≈ +3.4ms real) · −0.00987 bpb (MISLEADING — see PyramidCannon)

Pyramid future paths (if revisited):
- Smaller choke dim (128 or 256) — less cold param burden
- Warm initialization of bottleneck weights
- Dedicated LR schedule for choke layers
- Investigate benefit at very long training (>>8000 steps)

---

## Thread: Combined — Pyramid + Cannon

Two-variable combined test. Cannon already validated for speed; pyramid sought quality synergy.

| Date | Leg | Change vs Parent | 1GPU Gate | 8GPU Gate | BPB (seed 444) | Size | Verdict | Key Finding |
|------|-----|-----------------|-----------|-----------|----------------|------|---------|-------------|
| 2026-03-31 | BW5_PyramidCannon | BW5 + CHOKE_DIM=512 + CANNON_TYPE=scalar | ✓ −0.0091 | ✗ +0.03440 int6_sw_bpb | — | — | ✗ DOES NOT PROMOTE | Hard failure. Proxy passed but 8GPU decisive regression. Root: 1.57M cold choke params compound over time. Crossover ~step 500, diverges through step 2000. |

8GPU gate detail (seed=444, 2000 steps):
- BWVPC-00 control: 74.40ms · raw_bpb 1.3069 · int6_sw_bpb 1.28787686 · 9,415,826 bytes
- BWVPC-01 pyramid+cannon: 79.33ms · raw_bpb 1.3283 · int6_sw_bpb 1.32227987 · 10,408,358 bytes
- Delta: +4.93ms · +0.0214 raw_bpb · **+0.03440 int6_sw_bpb** · +993KB

---

## Planned Hypotheses (post-pyramid/cannon screw)

| Priority | Hypothesis | Thread | Prerequisite | Rationale |
|----------|-----------|--------|-------------|-----------|
| 1 | BW6 — Skipgrams in crawler | Ngram/Bigram | BW5 champion | NGRAM_EVAL_ORDER or BIGRAM in crawler context. Neural track uses this; crawler doesn't yet. |
| 2 | Delta Anchor | Recurrence | BW5 champion | Per-loop causal time state. Battery differentiates reading, delta anchor differentiates writing. |
| 3 | Warmdown tuning | Schedule | BW5 | BW5 seed=300 ⚠️ — warmdown length or learning rate taper may close the seed gap. |
| 4 | ~~Cannon on BW5 full run~~ | Cannon | CLOSED | Full run ran: +0.00020 vs BW5. Does not promote. See PIPELINE.md for future cannon paths. |

---

## All-Time Reference

| Leg | BPB (seed 444) | Size | Mean BPB | Status |
|-----|----------------|------|----------|--------|
| Leg 3 | 1.18720 | 8.84MB | 1.18743 (3-seed) | Former champion |
| BW4 | 1.18731 | 8.97MB | — | Superseded |
| **BW5** | **1.18672** | **8.61MB** | **1.18715** | **CHAMPION** |
