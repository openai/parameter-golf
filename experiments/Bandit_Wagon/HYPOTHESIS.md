# Bandit_Wagon — Crawler Width & Depth Headroom

## Hypothesis

**Can increasing model width (dim) or depth (flat layers) push the crawler below 1.18 BPB
at ≤10 MB — building on the CL3 proven 1.18742 baseline?**

## Locked Base Config (CL3, 3-seed mean 1.18742 BPB)

| Setting | Value | Source |
|---------|-------|--------|
| `CRAWLER_LOOPS` | 3 | CL1 |
| `CRAWLER_MLP_MULT` | 6.0 | CL3 |
| `CRAWLER_QUANT_INT8` | 1 | CL1 (mandatory) |
| `SKIP_GPTQ` | 1 | CL3 |
| `COMPILE_FULLGRAPH` | 0 | CL3 |
| `SKIP_EMA` | 1 | Ablations_v1 |

## Ablation Arms

| ID | Lever | Config | Status |
|----|-------|--------|--------|
| BW-00 | Anchor | dim=512, 4F+1C×3, mlp=6.0 | pending |
| BW-01 | Width narrow | dim=576, 4F+1C×3, mlp=6.0 | pending |
| BW-02 | Width wide | dim=640, 4F+1C×3, mlp=6.0 | pending |
| BW-03 | Depth +1 | dim=512, 5F+1C×3, mlp=6.0 | pending |
| BW-04 | Depth +2 | dim=512, 6F+1C×3, mlp=6.0 | pending |

## Hypotheses

**H-width:** Wider embedding dim → more representational capacity in flat layers.
BW-02 (dim=640) is near the 10 MB ceiling. Cost per 64-dim step ~1 MB compressed.

**H-depth:** More flat layers → more unique parameters before the shared crawler loop.
Orthogonal to width. Cost per flat layer ~1.68 MB compressed.

**Decision rule:** BW-00 anchor first. If anchor ≈ CL3 (1.187), the config is verified.
Promote the arm with the best delta for multi-seed confirmation.

## Run Commands

```bash
# BW-00 anchor
SEED=1337 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run.sh

# BW-01 width narrow
MODEL_DIM=576 SEED=1337 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run.sh

# BW-02 width wide
MODEL_DIM=640 SEED=1337 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run.sh

# BW-03 depth +1
NUM_FLAT_LAYERS=5 SEED=1337 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run.sh

# BW-04 depth +2
NUM_FLAT_LAYERS=6 SEED=1337 NPROC_PER_NODE=8 bash experiments/Bandit_Wagon/run.sh
```

## Results

| ID | Seed | Int6 SW BPB | Size | Delta | Notes |
|----|------|:-----------:|------|-------|-------|
| BW-00 | 1337 | TBD | TBD | — | anchor |
| BW-01 | 1337 | TBD | TBD | TBD | dim=576 |
| BW-02 | 1337 | TBD | TBD | TBD | dim=640 |
| BW-03 | 1337 | TBD | TBD | TBD | 5F+1C×3 |
| BW-04 | 1337 | TBD | TBD | TBD | 6F+1C×3 |

**Target:** int6 SW BPB < 1.187 (beat CL3 mean), ≤10 MB.

## Reference

| System | Int6 SW BPB | Size | Notes |
|--------|:-----------:|------|-------|
| CL3 (dim=512, 4F, mlp=6.0) | 1.18742 | 8.84 MB | 3-seed mean — this experiment's baseline |
| Rascal II (flat model) | 1.1099 | 15.44 MB | best legal base, different architecture |
