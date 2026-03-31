# Bandit — ClownCar Crawler + X-WING Ngram Oracle

## Hypothesis

X-WING (PR #800) uses a flat transformer + shared ngram9 oracle + 3D Cubric to score 0.4818 BPB.
Our ClownCar crawler (Medusa_VII DN=0) scores 1.1823 SW BPB as a pure model.

Crawler is stronger than X-WING's flat model on long-range / novel contexts.
Ngram oracle handles the predictable tokens regardless of base model.
Combined: crawler handles hard tokens better, ngram handles easy tokens the same.

Target: beat X-WING's 0.4818 BPB.

## Architecture

- **Base model**: Medusa_VII crawler (4 flat + 1 crawler × 4 loops, inst_dim=32 FLOW)
  - DN=0 (no DeltaNet — causality fix applied)
  - EMA_START_STEP=4400, EMA_DECAY=0.99, LOOP_AWARE_GPTQ=1
- **Oracle**: X-WING ngram9 eval stack
  - Shared tables: all ranks see identical token ranges (full 62M token picture)
  - 3D Cubric: 54 warm-start adaptive cells (order × entropy_bin × count_bin)
  - Entropy-adaptive alpha: 0.20–0.75 via sigmoid on model entropy
  - Complementary training: COMPLEMENT_ALPHA=0.5 (downweight bigram-predictable tokens)

## Baseline references

| System | Base SW BPB | Ngram9 BPB | Notes |
|--------|-------------|------------|-------|
| X-WING (PR #800) | 1.1196 | **0.4818** | flat model, our prior run |
| Medusa_VII DN=0 | 1.1823 | ??? | crawler, no oracle |
| **Bandit** | 1.18~ | **TBD** | crawler + oracle |

## Results

| Seed | SW BPB (model only) | Ngram9 BPB | Size | Notes |
|------|---------------------|------------|------|-------|
| 1337 | TBD | TBD | TBD | |
| 300 | TBD | TBD | TBD | |
