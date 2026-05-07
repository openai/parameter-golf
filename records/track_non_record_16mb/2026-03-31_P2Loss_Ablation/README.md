# P2 Loss Ablation Study

This submission presents a controlled ablation study comparing standard cross-entropy (CE) training with P2-style loss reweighting under the naive baseline configuration.

## Motivation

P2 loss reweights token-level cross-entropy based on model confidence:

[
L = -(1 - p)^\gamma \log p
]

where (p) is the predicted probability of the correct token and (\gamma) controls emphasis on lower-confidence (harder) tokens.

This formulation has been proposed to:

* allocate more gradient to difficult tokens
* accelerate early learning
* potentially improve final compression (BPB)

This study evaluates whether these benefits hold under parameter-golf constraints.

---

## Experimental Setup

* Baseline: naive `train_gpt.py`
* Evaluation: standard CE (`F.cross_entropy`) for all runs
* Hardware: 8×H100
* Seed: 1337 (single-seed controlled comparison)
* Time cap: 600 seconds (wallclock-limited)

All runs are matched in:

* architecture
* optimizer
* data
* schedule

The only difference is the training loss.

---

## Implementation Details

The naive baseline `train_gpt.py` was minimally modified to support loss ablation via two parameters:

* `LOSS_TYPE` ∈ {`ce`, `p2`}
* `P2_GAMMA` (float)

No other changes were made to:

* architecture
* optimizer
* data pipeline
* training schedule
* evaluation (always standard CE)

This ensures a controlled comparison where the loss function is the only variable.

---

## Loss Variants

* **CE (baseline)**
  [
  L = -\log p
  ]

* **P2 loss**
  [
  L = -(1 - p)^\gamma \log p
  ]

Evaluated at:

* γ = 0.5
* γ = 1.0
* γ = 2.0

---

## Results

| Loss | Gamma | Step 1000 BPB | Step 3000 BPB |  Final BPB | Stop Step |
| ---- | ----: | ------------: | ------------: | ---------: | --------: |
| CE   |   0.0 |        1.3850 |        1.3010 | **1.2271** |     13725 |
| P2   |   0.5 |        1.3892 |        1.3075 |     1.2307 |     13701 |
| P2   |   1.0 |        1.4029 |        1.3201 |     1.2408 |     13719 |
| P2   |   2.0 |        1.4240 |        1.3489 |     1.2675 |     13736 |

---

## Key Finding

Across matched runs on the naive baseline, P2-style loss reweighting (γ = 0.5, 1.0, 2.0) consistently underperforms standard cross-entropy under canonical evaluation. The degradation is observed across multiple training stages (step 1000, step 3000, and final), indicating that P2-style reweighting does not improve over uniform CE under parameter-golf constraints.

---

## Interpretation

* Mild reweighting (γ = 0.5) comes closest to CE but does not surpass it
* Increasing γ degrades performance further
* Emphasizing harder tokens appears to reduce overall compression efficiency under fixed compute and parameter budgets

This suggests that uniform gradient allocation (standard CE) is more effective than P2-style reweighting in this regime.

---

## Logs

* `train.log` → CE baseline
* `train_p2_0_5.log` → P2 γ = 0.5
* `train_p2_1_0.log` → P2 γ = 1.0
* `train_p2_2_0.log` → P2 γ = 2.0

All logs are included for reproducibility.

---

## Notes

* This is a **non-record submission**
* Single-seed comparison (controlled ablation, not variance study)
* Evaluation is unchanged and fully comparable to the baseline
