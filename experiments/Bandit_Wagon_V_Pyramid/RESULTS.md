# Bandit_Wagon_V_Pyramid — Gate Results

## Architecture

BW5 + `CRAWLER_MLP_CHOKE_DIM=512` (pyramid shape).
Single validated change vs BW5 control.

- `CRAWLER_MLP_CHOKE_DIM`: 0 (flat) vs 512 (pyramid)
- `CRAWLER_MLP_CHOKE_SHAPE`: flat vs pyramid
- `CRAWLER_MLP_CHOKE_GROUPS=8`
- All other flags identical to BW5

---

## Gate: Single GPU, 500 steps, seed=444

Base: BW5 (CHOKE_DIM=0, COMPILE_FULLGRAPH=1, ROPE_SCALES=9,1,1)
Variable: CRAWLER_MLP_CHOKE_DIM (0=flat vs 512=pyramid)

| ARM | CHOKE_DIM | step_avg | raw_bpb | vs control |
|-----|-----------|----------|---------|------------|
| BWVP-00 | 0 (flat) | ?ms | ? | — |
| BWVP-01 | 512 (pyramid) | ?ms | ? | ? |

Pass criteria: pyramid raw_bpb < control raw_bpb at 500 steps.
Also watch: step_avg cost of adding pyramid.

## Verdict: PENDING

---

## Gate: 8×H100, 2000 steps, seed=444

*Run only if 1GPU gate passes.*

| ARM | CHOKE_DIM | step_avg | val_bpb | int6_rt_bpb | int6_sw_bpb | size_bytes |
|-----|-----------|----------|---------|-------------|-------------|------------|
| BWVP-00 | 0 (flat) | | | | | |
| BWVP-01 | 512 (pyramid) | | | | | |
| delta | | | | | | |

## Verdict: PENDING
