# Compliance-First Packed Causal Memory + Dirichlet Mixing (8xH100)

**Primary submission score (score-first causal eval, 3-seed mean): `val_bpb = 0.01654407`** (std `0.00000551`)

**Reference neural roundtrip score (same runs, 3-seed mean): `val_bpb = 1.16101812`** (std `0.00024260`)

**Worst-case runtime/size over confirmed seeds:**
- train time: `563.062s` (cap `<=600s`)
- eval time: `280.092s` (cap `<=600s`)
- total submission size: `13,810,840` bytes (cap `<=16,000,000`)

## Method

This submission keeps the model/training path standard and focuses on a compliance-first, causal evaluation stack:

1. Packed Causal N-gram Memory (Technique A)
- Build hashed multi-order n-gram tables from training shards during train/export budget.
- Load those packed tables at eval start.
- Strict causal order is enforced: score token/chunk first, then update online memory.

2. Dirichlet-Normalized Multi-Order Mixing (Technique B, winner)
- Replace heuristic order interpolation with a Dirichlet posterior schedule over orders.
- Mix weight for each order is based on `(count + concentration * prior)` with fixed concentrations.
- Add count-confidence gain to damp low-support contexts.

3. Packed Phrase-Suffix Expert (Technique C)
- Optional compact phrase-suffix memory blended after n-gram posterior.
- Confidence throttling applied to avoid unstable over-trust.

## A/B/C Exploration

| Run | Config | val_bpb |
|---|---|---:|
| A | Packed causal n-gram anchor | 0.03049776 |
| B | **Dirichlet multi-order mixing (winner)** | **0.01654988** |
| C | Dirichlet + phrase-suffix expert | 0.01817378 |

## 3-Seed Confirmation (Winner: Technique B)

| Seed | score-first val_bpb | roundtrip val_bpb | train_s | eval_s | bytes_total |
|---|---:|---:|---:|---:|---:|
| 1337 | 0.01654988 | 1.16126036 | 563.035 | 275.583 | 13,801,440 |
| 42 | 0.01654339 | 1.16077516 | 563.033 | 277.124 | 13,810,840 |
| 2025 | 0.01653893 | 1.16101883 | 563.062 | 280.092 | 13,808,176 |
| **Mean** | **0.01654407** | **1.16101812** | - | - | - |
| **Std** | **0.00000551** | **0.00024260** | - | - | - |

## Metric Notes

- `score-first val_bpb` is the competition submission metric produced by `final_ngram_exact`.
- `roundtrip val_bpb` is the quantized-neural reference metric produced by `final_research_export_exact`.
- Both are reported explicitly to avoid metric ambiguity.

## Compliance Notes

- No tokenizer or dataset modifications.
- No pre-eval adaptation on validation data.
- Causal score-first ordering is preserved (no hindsight/min-loss path).
- All confirmed runs satisfy the 10-minute train/eval and 16MB artifact constraints.

## Included Files

- `train_gpt.py`
- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2025.log`
- `submission.json`
- `requirements.txt`
