# Bandit_Wagon — Pure Crawler Headroom Ablation

## Hypothesis

**Can systematic isolation of architecture levers (width, depth, loop config) drive the
pure neural crawler to ≤1.15 BPB at ~10 MB submission size — without any ngram oracle?**

The ngram oracle has been removed. This series establishes the crawler's true standalone
capacity and finds the best use of the 16 MB budget for a pure neural model.

## Config Baseline (post-CL3 proven)

All arms share these locked settings, derived from CL3 3-seed confirmation (1.18742 mean BPB):

| Setting | Value | Source | Notes |
|---------|-------|--------|-------|
| `CRAWLER_LOOPS` | 3 | CL1-01 | −0.088 BPB vs loops=4 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 | beats mlp=5.0 at 600s (1.18742 vs 1.19593) |
| `CRAWLER_QUANT_INT8` | 1 | CL1-08 | mandatory (+0.197 if off) |
| `SKIP_GPTQ` | 1 | CL3 | extra training time beats LOOP_AWARE_GPTQ overhead at 600s |
| `COMPILE_FULLGRAPH` | 0 | CL3 | proven config; fullgraph gains absorbed by longer training |

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

## Prior Reference

| System | Int6 SW BPB | Size | Notes |
|--------|:-----------:|------|-------|
| CL3 (loops=3 mlp=6.0 SKIP_GPTQ) | 1.18742 | 8.84 MB | **3-seed mean — proven config, this experiment's base** |
| CL2-02 (loops=3 mlp=5.0 LOOP_AWARE_GPTQ) | 1.19593 | 9.84 MB | single seed, 350s — superseded by CL3 |
| Rascal II (flat model) | 1.1099 | 15.44 MB | current best legal base |
