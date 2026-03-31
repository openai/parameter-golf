# Bandit_Wagon_V_Cannon — Gate Results

## Gate: Single GPU, 500 steps, seed=444

Base: BW5 (CHOKE_DIM=0, COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1)
Variable: CRAWLER_CANNON_TYPE

| ARM | Type | raw_bpb | int6_sw_bpb | vs control | bytes |
|-----|------|---------|-------------|------------|-------|
| BWVC-00 | control (none) | 1.4413 | 1.44236 | — | 6,788,121 |
| BWVC-01 | scalar (3 params) | 1.4407 | 1.44261 | +0.00025 | 6,794,463 |
| BWVC-02 | channel (1.5K) | 1.4422 | 1.44296 | +0.00060 | 6,729,386 |
| BWVC-03 | rmsnorm (1.5K) | 1.4408 | 1.44428 | +0.00192 | 6,776,903 |

## Verdict: ~~DOES NOT PROMOTE~~ — CORRECTED. See 8GPU gate below.

**Correction:** The original verdict was based solely on int6_sw_bpb at 500 proxy steps (unreliable at that scale).
Scalar cannon raw_bpb (1.4407) was better than control (1.4413). Speed was also faster on 1GPU.
8GPU gate was required and has now been run.

---

## Gate: 8×H100, 2000 steps, seed=444

Base: BW5. Arms: control (none) vs scalar cannon only (best 1GPU arm).
Pass criteria: scalar step_avg < control step_avg.

| ARM | Type | step_avg | val_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|------|----------|---------|-------------|-------------|------------|
| BWVC-00 | control (none) | 74.84ms | 1.3080 | 1.31294609 | 1.28870981 | 9,169,530 |
| BWVC-01 | scalar cannon (3 params) | **74.81ms** | 1.3082 | **1.31256407** | **1.28854887** | 9,512,901 |
| delta | | **-0.03ms** | +0.0002 | **-0.00038** | **-0.00016** | **+343,371** |

### Verdict: SPEED GATE PASSES (barely). Quality positive. Size regression.

- **Speed:** scalar 74.81ms < control 74.84ms → **PASSES** (-0.03ms, marginal)
- **int6_sw_bpb:** scalar wins by -0.00016 → positive quality signal
- **int6_rt_bpb:** scalar wins by -0.00038 → positive quality signal
- **Size:** scalar is +343KB larger despite only 3 extra params — quantization behavior differs

**Finding:** Scalar cannon is real signal. Tiny speed gain, tiny quality gain, but notable size cost.
Proceed to `Bandit_Wagon_V_PyramidCannon` — the combined pyramid+cannon test is the next gate.

---

## Full Production Run: 8×H100, 600s, seed=444

| Metric | BW5_Cannon | BW5 Champion | Delta |
|--------|-----------|--------------|-------|
| steps | 8034 | 8035 | −1 |
| step_avg | 74.69ms | 74.68ms | +0.01ms |
| raw_bpb | 1.1990 | 1.1987 | +0.0003 |
| int6_sw_bpb | **1.18692423** | **1.18672385** | **+0.00020** |
| quant_gap | −0.0121 | −0.0120 | −0.0001 |
| size_bytes | 8,845,120 (8.44MB) | 9,024,399 (8.61MB) | −179KB |
| checkpoint | `BW5Cannon_s444_20260331_221134_bpb1.18692423.pt` | — | — |

## Verdict: DOES NOT PROMOTE

**int6_sw_bpb is +0.00020 worse than BW5.** The 2000-step gate showed −0.00016 (positive), but the signal did not compound — it reversed at production scale.

**Step time matched BW5 exactly (74.69ms vs 74.68ms).** Cannon adds no overhead.

**Size:** −179KB smaller than BW5 (8.44MB vs 8.61MB). Counterintuitive given the +343KB at 2000 steps — the quant_gap tightened slightly (−0.0121 vs −0.0120) which reduced the zstd artifact.

**Root cause:** Scalar cannon's 3-param output scale gives no meaningful benefit at production training length. The gate signal was real noise riding within the cross-run variance band (~0.0003 BPB). The cannon's architectural concept (output calibration per loop) requires a stronger mechanism — channel-level or coupled with a larger structural change — to show signal above noise at 8000+ steps.

**Cannon concept notes for future:**
- Channel cannon (1.5K params) was never tested at 8GPU full run — may have stronger signal
- Cannon may be most useful as a pairing with another architectural change that creates amplitude mismatch (e.g., delta anchor, wider choke)
- The +343KB size regression at gate → −179KB at full run suggests quant behavior changes significantly across training length
