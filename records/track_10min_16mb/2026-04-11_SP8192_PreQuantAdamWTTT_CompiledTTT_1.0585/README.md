# SP8192 + Pre-Quant AdamW TTT + Compiled TTT (3-seed mean 1.05850 BPB)

Joshua-owned SAFE_SUBMISSION reproduction of the PR #1539 recipe, reconciled from pulled TensorPool artifacts for `run036-safe016` / `j-5x7kcly8yl`.

## Headline result

- **SAFE_SUBMISSION authority:** `final_int6_sliding_window_exact`
- **3-seed mean:** **1.05850131 BPB**
- **3-seed std:** **0.00181649 BPB**
- **Best seed:** **1.05690014 BPB** (seed `2024`)
- **Worst seed:** `1.06047528 BPB` (seed `42`)
- **Total submission size:** `15,457,982` to `15,504,058` bytes across seeds
- **Legality lane:** **SAFE_SUBMISSION** — all clean-lane artifacts stayed below the `16,000,000` byte cap

## Per-seed clean-lane results

| Seed | Post-EMA BPB | Final int6 roundtrip | Final int6 sliding-window exact | Serialized int6+brotli | Total submission size |
|------|--------------:|---------------------:|--------------------------------:|-----------------------:|----------------------:|
| 42   | 1.10470000 | 1.07087282 | 1.06047528 | 15,366,526 | 15,504,058 |
| 1337 | 1.10270000 | 1.06836276 | 1.05812851 | 15,320,450 | 15,457,982 |
| 2024 | 1.10220000 | 1.06682712 | 1.05690014 | 15,346,751 | 15,484,283 |
| **Mean** | **1.10320000** | **1.06868757** | **1.05850131** | **15,344,575.67** | **15,482,107.67** |
| **Std** | — | — | **0.00181649** | — | — |

## Why this matters

- Improves Joshua's prior fork submission branch (`submission-run021-safe001-1.0745`) by **0.01601181 BPB**.
- Reproduces the strongest clean-lane stack currently on hand using Joshua-owned infrastructure and artifacts.
- Keeps legality lanes explicit: the static int6 artifact above is the submission authority; SLOT numbers remain frontier-only telemetry.

## Technique stack

1. **SP8192 tokenizer** with 11-layer, 512-dim, 8-head / 4-KV-head architecture.
2. **Depth recurrence** over layers 3-5 after step 3000 (14 virtual layers total).
3. **Parallel residuals** from layer 7 onward.
4. **Compiled pre-quant AdamW TTT** (`6` epochs, `lr=5e-4`, freeze first `2` blocks) before GPTQ.
5. **QK-Gain 5.25**, EMA `0.9965`, tuned Muon/AdamW hypers, late QAT.
6. **Int6 GPTQ + brotli** packaging, with no pruning needed on any seed.

## Legality notes

This record is **SAFE_SUBMISSION** because the scored artifact is the fixed int6 model produced after pre-quant TTT and quantization. There is **no eval-time adaptation** in the submission authority reported above. The same logs also contain `final_slot_exact` results around `0.8597 BPB`, but those belong to **FRONTIER_ONLY** and are intentionally excluded from the submission score.

## Reproduction notes

The attached `train_gpt.py`, `run_seed.sh`, and `run_all_seeds.sh` mirror the canonical launch bundle used for `run036-safe016`. Authoritative metrics come from pulled artifact logs under `~/parameter-golf-project/state/tp-pulls/run036-safe016/artifacts/train_seed*.log`.
