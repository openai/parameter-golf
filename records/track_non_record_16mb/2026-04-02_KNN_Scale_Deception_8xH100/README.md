# Non-Record: KNN Hidden State Retrieval — When Eval-Time Augmentation Helps Weak Models but Hurts Strong Ones

## TL;DR

**KNN Hidden State Retrieval** is an eval-time technique that stores hidden states from scored tokens and uses nearest-neighbor retrieval to augment neural predictions. It shows strong improvements on weak models (-2% to -4% BPC) but **hurts competition-quality models (+1.5% BPB).**

This is a definitive demonstration of **scale deception** — the same phenomenon we documented in PR #1227 with SSM hybrids. Techniques that help at small scale can hurt at competition scale, and the crossover happens silently.

**Key numbers:**

| Model Quality | Training | KNN Effect | Direction |
|--------------|----------|------------|-----------|
| Very weak (local, 1500 steps) | AdamW, dim=192 | **-2.34%** | Helps |
| Weak (1×H100, 2K steps) | Muon, dim=512 | **-1.57%** | Helps |
| Medium (1×H100, 2K steps, export) | Muon, GPTQ int6 | **-5.21%** | Helps (weak export) |
| **Strong (8×H100, 5665 steps, SOTA stack)** | **Muon, EMA, GPTQ int6** | **+1.47%** | **Hurts** |

## Method

### KNN Hidden State Retrieval

At eval time, for each scored token, we store its final hidden state and the actual next token in a growing datastore. For each new position:

1. Query the datastore with the current hidden state
2. Find K=8 nearest neighbors by L2 distance
3. Build an empirical distribution from neighbors' successor tokens
4. Mix with neural prediction: `P = 0.88 × P_neural + 0.12 × P_knn`
5. Score with the mixed distribution
6. AFTER scoring: add current hidden state to datastore

### Vectorized Implementation

Per-token Python loops are too slow for 62M tokens. We use batch `torch.cdist` to compute all distances per chunk:

```python
dists = torch.cdist(queries.half(), stored_h[:n_stored], p=2).pow(2)  # (C, N)
topk_d, topk_i = dists.topk(K, dim=1, largest=False)                  # (C, K)
knn_dist = torch.zeros(C, V, device=device)
knn_dist.scatter_add_(1, stored_tok[topk_i], weights)                  # build distribution
mixed = (1 - lam) * neural_probs + lam * knn_dist                     # mix
```

With `subsample=4` and `max_stored=8M`, KNN eval completes in **168 seconds on 8×H100** — well within the 600s eval budget.

### Legality

Score-first protocol, properly normalized, causal, zero artifact cost. Same protocol as TTT (explicitly legal). See PR #1227 for detailed legality analysis.

## 8×H100 Results (Competition Scale)

### Training
```
Profile:    full_8gpu_600s
Preset:     merged_leader (11L, 512d, LeakyReLU², XSA-all, BigramHash, EMA, Muon)
Steps:      5665 in 600s
Pre-export: val_bpb = 1.1446
Export:     GPTQ int6, LZMA preset 9
Artifact:   15,826,144 bytes (under 16MB)
```

### Evaluation
| Eval Method | val_loss | val_bpb | vs Neural |
|-------------|----------|---------|-----------|
| Neural (roundtrip, exported model) | 1.9473 | **1.1533** | baseline |
| **KNN (k=8, λ=0.12, subsample=4)** | **1.9758** | **1.1702** | **+1.47% worse** |

**KNN hurts by 1.47% on the competition-quality model.**

### KNN Eval Timing
- 8 GPUs, each processing 1/8 of 62M tokens
- `subsample=4`: store every 4th hidden state (max 8M vectors per rank)
- `chunk_size=1024`: batch distance computation
- **Total: 168 seconds** — fits in 600s eval budget

## Scaling Analysis

### Why KNN Helps Weak Models

On a 2000-step model (BPC ~1.9), the neural predictions are noisy. Many positions have high entropy (model is uncertain). KNN provides an alternative signal — if the hidden state is similar to a previously seen context, the empirical distribution from that context is informative. The model's hidden states are discriminative enough for retrieval but the predictions are poor enough that KNN can improve them.

### Why KNN Hurts Strong Models

On a 5665-step competition model (BPB ~1.15), the neural predictions are already well-calibrated. The model correctly assigns high probability to the right tokens in most contexts. KNN introduces noise because:

1. **Nearest neighbors aren't close enough.** With 512-dimensional hidden states, L2 distance is a crude similarity measure. Two states can be "nearest neighbors" but represent very different linguistic contexts.

2. **The empirical distribution is sparse.** With K=8 neighbors, the KNN distribution places probability mass on only 3-8 distinct tokens. The neural distribution spreads probability more appropriately across the full vocabulary.

3. **Mixing degrades calibration.** Even 12% KNN weight is enough to move probability mass away from the correct neural prediction toward the noisy KNN estimate.

### The Crossover Point

Based on our data, the crossover (where KNN goes from helping to hurting) happens approximately at:
- BPC ~2.5 (local model at ~3000 steps)
- BPB ~1.4 (H100 model at ~2000 steps with Muon)

Below these thresholds, KNN helps. Above them, it hurts. Competition-quality models (BPB ~1.15) are well past the crossover.

## Comparison with Other Scale Deception Findings

| Technique | Local Result | Competition Result | Reversal |
|-----------|-------------|-------------------|----------|
| S4D-Lin SSM (PR #1013) | -18% CE | +2.7% BPB | 180° flip |
| **KNN Hidden State** | -4.6% BPC | +1.5% BPB | Sign flip |
| QAT NF5 | -0.66% CE | (untested at scale) | Unknown |
| Self-distillation | -9.24% CE | (untested at scale) | Unknown |

This is now the **second technique** where we've demonstrated definitive scale deception in this competition. The pattern is consistent: techniques that compensate for model weakness don't help (and actively hurt) when the model is strong.

## Implications for Other Competitors

1. **Don't trust local eval-time improvements.** If your technique helps a weak model, it may hurt a strong one. The only reliable test is at competition scale.

2. **Eval-time augmentation has diminishing (then negative) returns.** The stronger the base model, the less room for eval-time tricks. At BPB ~1.15, the model's predictions are hard to improve by mixing in external signals.

3. **The eval budget IS underutilized, but not for prediction mixing.** Our KNN used 168s of the 600s budget effectively. The compute is there — the challenge is finding eval-time techniques that actually help strong models. TTT (gradient-based adaptation) may be more promising because it adapts the model itself rather than mixing in an external signal.

## Hardware and Cost

| Phase | Hardware | Time | Cost |
|-------|----------|------|------|
| Local experiments (28+) | Mac Mini M4 | 5 days | $0 |
| 1×H100 validation | Single H100 | ~4 hours | ~$12 |
| 8×H100 record attempt | 8×H100 SXM | ~2 hours | ~$43 |
| **Total** | | | **~$55** |

## Code

The KNN implementation is integrated into `train_gpt.py` as `eval_knn()`, following the same pattern as the competition's `eval_ttt()` and `eval_ngram()`. Enable with `KNN_ENABLED=1`.

Key files:
- `knn_eval_patch.py` — standalone KNN module
- `apply_knn_patch.py` — script to patch train_gpt.py
- `h100_knn_eval_submission.py` — standalone KNN eval for checkpoints

## Training Log (seed 42, 8×H100)

```
step:0/20000 val_loss:6.9301 val_bpb:4.1044
step:4000/20000 val_loss:2.0157 val_bpb:1.1938
stopping_early: wallclock_cap train_time:600062ms step:5665/20000
DIAGNOSTIC post_average val_loss:1.9325 val_bpb:1.1446
Serialized model research_export: 15826144 bytes
Total submission size research_export: 16018423 bytes
final_research_export_roundtrip val_loss:1.9473 val_bpb:1.1533
final_knn val_loss:1.9758 val_bpb:1.1702 eval_time:168181ms k:8 lam:0.12
```

---

*Self-funded research. Mac Mini M4 + RunPod H100. Total GPU spend: ~$55.*

*Author: Himanshu Dongre (@himanshudongre) — also author of PR #1227 (28 Experiments), PR #1013 (SSM Hybrid), PR #1012 (JEPA-LM).*
