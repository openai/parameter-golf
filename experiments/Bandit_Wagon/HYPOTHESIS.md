# Bandit_Wagon — Pure Crawler Headroom Ablation

## Hypothesis

**Can systematic isolation of architecture levers (width, depth, loop config) drive the
pure neural crawler to ≤1.15 BPB at ~10 MB submission size — without any ngram oracle?**

The ngram oracle has been removed. This series establishes the crawler's true standalone
capacity and finds the best use of the 16 MB budget for a pure neural model.

## Config Baseline (post-CL1/Ablations_v1 optimal)

All arms share these locked settings, derived from research:

| Setting | Value | Source | Gain |
|---------|-------|--------|------|
| `CRAWLER_LOOPS` | 3 | CL1-01 | −0.088 BPB |
| `CRAWLER_MLP_MULT` | 5.0 | CL1-07 | −0.098 BPB |
| `CRAWLER_QUANT_INT8` | 1 | CL1-08 | mandatory (+0.197 if off) |
| `LOOP_AWARE_GPTQ` | 1 | Ablations_v1-B | −0.040 BPB |
| `COMPILE_FULLGRAPH` | 1 | Ablations_v1-E | −0.026 BPB |

## Ablation Arms

| ID | Lever | Config | Status |
|----|-------|--------|--------|
| BW-00 | Anchor | dim=512, 4F+1C×3, mlp=5.0 | pending |
| BW-01 | Width (narrow) | dim=576, 4F+1C×3, mlp=5.0 | pending |
| BW-02 | Width (wide) | dim=640, 4F+1C×3, mlp=5.0 | pending |
| BW-03 | Depth (shallow) | dim=512, 5F+1C×3, mlp=5.0 | pending |
| BW-04 | Depth (deep) | dim=512, 6F+1C×3, mlp=5.0 | pending |

Size estimates are approximate pending BW-00 anchor run.

## Hypotheses

**H-width:** Width is the validated signal from proxy sweeps. Wider dim → better base model
quality, harder tokens handled without oracle assistance. BW-02 (dim=640) is the maximum
width affordable near 10 MB.

**H-depth:** More unique flat layers increase representational diversity before the crawler
loop. Orthogonal mechanism to width. Cost per flat layer ~1.68 MB.

**Decision rule:** Confirm BW-00 anchor BPB first. Then promote the arm closest to 1.15
BPB at ≤10 MB for multi-seed confirmation. If both width and depth beat anchor → consider
576+5F combo after.

## Run Commands

```bash
# BW-00 anchor — run this first to establish new baseline
SEED=1337 bash experiments/Bandit_Wagon/run.sh

# BW-01 width narrow
MODEL_DIM=576 SEED=1337 bash experiments/Bandit_Wagon/run.sh

# BW-02 width wide
MODEL_DIM=640 SEED=1337 bash experiments/Bandit_Wagon/run.sh

# BW-03 depth shallow
NUM_FLAT_LAYERS=5 SEED=1337 bash experiments/Bandit_Wagon/run.sh

# BW-04 depth deep
NUM_FLAT_LAYERS=6 SEED=1337 bash experiments/Bandit_Wagon/run.sh
```

## Results

| ID | Seed | Base SW BPB | Int6 SW BPB | Size | Delta | Notes |
|----|------|-------------|-------------|------|-------|-------|
| BW-00 | 1337 | TBD | TBD | TBD | — | anchor |
| BW-01 | 1337 | TBD | TBD | TBD | TBD | dim=576 |
| BW-02 | 1337 | TBD | TBD | TBD | TBD | dim=640 |
| BW-03 | 1337 | TBD | TBD | TBD | TBD | 5F+1C×3 |
| BW-04 | 1337 | TBD | TBD | TBD | TBD | 6F+1C×3 |

**Target:** int6 SW BPB ≤1.15 at submission size ≤10 MB.

## Prior Reference (context only — oracle-assisted, not comparable)

| System | Base SW BPB | Ngram9 BPB | Size | Notes |
|--------|-------------|------------|------|-------|
| Bandit (with oracle) | 1.1867 | 0.4961 | 9.35 MB | 3-seed mean — oracle removed |
| Rascal II (flat model) | 1.1099 | — | 15.44 MB | current best legal base |
