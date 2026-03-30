# Bandit_Wagon — Headroom Ablations

## Context

Bandit (ClownCar crawler + X-WING ngram9 oracle) scored **0.4961 BPB** at **9.35 MB mean** across 3 seeds.
16 MB budget leaves **~6.65 MB / ~9.3M params of unused headroom**.

This series answers: what is the best use of that headroom?
One variable per arm. No combos until winners are confirmed.

## Ablation Arms

| ID | Lever | Config | Size delta | Est. total | Status |
|----|-------|--------|------------|------------|--------|
| BW-00 | Baseline anchor | dim=512, 4F+1C×4 | — | ~9.35 MB | anchor |
| BW-01 | Width (narrow) | dim=576, 4F+1C×4 | +2.28 MB | ~11.6 MB | pending |
| BW-02 | Width (wide) | dim=640, 4F+1C×4 | +4.83 MB | ~14.2 MB | pending |
| BW-03 | Depth (shallow) | dim=512, 5F+1C×4 | +1.68 MB | ~11.0 MB | pending |
| BW-04 | Depth (deep) | dim=512, 6F+1C×4 | +3.36 MB | ~12.7 MB | pending |

All arms hold fixed: `DELTA_NET_HEADS=0`, `CRAWLER_LOOPS=4`, `INST_DIM=32`, `CRAWLER_MLP_MULT=4.0`, `COMPLEMENT_ALPHA=0.5`, `NGRAM_EVAL_ORDER=9`, `CUBRIC_CADENCE=32`.

## Hypotheses

**H-width:** Width is the validated signal (0.033 BPB from Frug2 proxy sweep). At full 600s + oracle scale, wider dim should improve base model quality → harder tokens handled better → lower oracle-combined BPB. BW-02 (dim=640) is the maximum affordable width bet.

**H-depth:** More unique flat layers increase representational diversity before the crawler. Each flat layer costs ~1.68 MB. Two extra layers (BW-04) uses 3.36 MB — same budget as a smaller width increase, but orthogonal mechanism. Unknown at crawler+oracle scale.

**Decision rule:** If width beats depth → promote BW-02 for multi-seed confirmation. If depth beats width → promote BW-04. If both beat baseline → consider 576+1F combo run after.

## Run Commands

```bash
# BW-00 anchor (should reproduce Bandit)
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

Smoke first (boots, memory, compile):
```bash
bash Nitrust/scripts/spark_bandit_wagon_smoke.sh
```

## Results

| ID | Seed | SW BPB | Ngram9 BPB | Size | Notes |
|----|------|--------|------------|------|-------|
| BW-00 | 1337 | TBD | TBD | TBD | anchor |
| BW-01 | 1337 | TBD | TBD | TBD | dim=576 |
| BW-02 | 1337 | TBD | TBD | TBD | dim=640 |
| BW-03 | 1337 | TBD | TBD | TBD | 5 flat |
| BW-04 | 1337 | TBD | TBD | TBD | 6 flat |

## Baseline Reference

| System | Base SW BPB | Ngram9 BPB | Size | Notes |
|--------|-------------|------------|------|-------|
| Bandit (BW-00) | 1.1867 | **0.4961** | 9.35 MB | 3-seed mean, std=0.0003 |
| X-WING Cubric | 1.1199 | **0.4820** | 15.58 MB | flat 11L, prior SOTA |
