# Iso-Compute Bake-Off Results

**Target step time:** 700 ms/step (baseline plain ternary)

## Calibrated Dimensions

| Config | MODEL_DIM | ms/step | Params | Δ |
|--------|-----------|---------|--------|---|
| A_plain_ternary | 256 | 700 | 3085840 | +0.0% |
| B_feedback_engram | 156 | 716 | 1314298 | +2.2% |
| C_shared_blocks | 144 | 686 | 702312 | -2.0% |
| D_koopman_koopcaps | 144 | 667 | 1157426 | -4.7% |

## Race Results

| Config | DIM | Params | Steps | Sliding BPB | TTT BPB | N-gram BPB | Best |
|--------|-----|--------|-------|-------------|---------|------------|------|
| C_shared_blocks | 144 | 702312 | 2000 | 2.1606 | 2.16040829 | 2.0916 | 2.0916 |
| D_koopman_koopcaps | 144 | 1157426 | 2000 | 2.1526 | 2.15251835 | 2.0848 | 2.0848 |

## Convergence Curves (val_bpb vs step)

### C_shared_blocks (dim=144, 702312 params)

| Step | Val BPB |
|------|---------|
| 200 | 2.5099 |
| 400 | 2.4193 |
| 600 | 2.3858 |
| 800 | 2.3781 |
| 1000 | 2.3307 |
| 1200 | 2.2979 |
| 1400 | 2.2664 |
| 1600 | 2.2889 |
| 1800 | 2.1875 |
| 2000 | 2.1578 |
| 2000 | 2.1578 |
| 2000 | 2.1606 |
| 2000 | 2.1604 |
| 2000 | 2.1604 |
| 2000 | 2.0916 |

### D_koopman_koopcaps (dim=144, 1157426 params)

| Step | Val BPB |
|------|---------|
| 200 | 2.5205 |
| 400 | 2.4120 |
| 600 | 2.3791 |
| 800 | 2.3725 |
| 1000 | 2.3207 |
| 1200 | 2.2943 |
| 1400 | 2.2576 |
| 1600 | 2.2720 |
| 1800 | 2.1825 |
| 2000 | 2.1490 |
| 2000 | 2.1490 |
| 2000 | 2.1526 |
| 2000 | 2.1525 |
| 2000 | 2.1525 |
| 2000 | 2.0848 |


## Winner: `D_koopman_koopcaps` — Best BPB = **2.0848**

At MODEL_DIM=144, 1157426 params, 466 ms/step
