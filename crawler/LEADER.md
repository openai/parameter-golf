# Crawler SOTA — Current Leader

Score:  1.18672385 BPB (seed 444) | 1.18715 mean (2-seed)
Size:   8.61MB (seed 444) | 8.82MB (seed 300)
Date:   2026-03-29
Leg:    crawler/2026-03-29_BW5/
Run:    bash crawler/2026-03-29_BW5/run.sh

## Architecture
Bandit_Wagon_V — BW4 + COMPILE_FULLGRAPH=1
CRAWLER_MLP_CHOKE_DIM=0 (no choke) | CRAWLER_LOOP_ROPE_SCALES=9,1,1
74.68ms/step on 8xH100 | SWA at step 7650 | 8035 steps (seed 444)

## Seeds
| Seed | BPB exact       | Size    | vs Leg 3     |
|------|-----------------|---------|--------------|
| 444  | 1.18672385      | 8.61MB  | -0.00058 ✓   |
| 300  | 1.18758156      | 8.82MB  | +0.00012 ⚠️  |
| mean | **1.18715**     | 8.82MB  | -0.00027 ✓   |

Note: seed=300 does NOT individually confirm vs Leg 3. Mean is better.
Leg 3 reference: 1.18743 mean, 9.36MB.

## Promotion Gate
Beat 1.18672385 on seed 444 → confirm on seed 300 → update this file.
One variable changed per leg. Gate (1-GPU, 2000 steps) before any 8x run.
