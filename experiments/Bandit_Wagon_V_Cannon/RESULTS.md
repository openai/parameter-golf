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

## Verdict: DOES NOT PROMOTE

No cannon arm beats control on int6_sw_bpb. All arms are worse.

**Why:** The cannon was designed to calibrate per-loop output magnitude variance introduced by the pyramid choke. On the flat BW5 base (no choke), loop outputs are already well-balanced — cannon has nothing to fix and adds noise.

**Finding:** Cannon is architecture-dependent. It may add value in pyramid configurations but not on the flat+battery+fullgraph stack.

No full 8×H100 run needed.
