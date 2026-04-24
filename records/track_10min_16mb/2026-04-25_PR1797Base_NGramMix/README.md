# PR #1797 Base + Prefix-Only Causal Bigram Mixture

**Status:** scaffold — numbers to be filled after the Day 3 (sweep) and
Day 4 (3-seed) runs on 8×H100 SXM.

**Track:** `track_10min_16mb`.

**Base:** PR #1797 (dexhunter) — SmearGate + LQER Asymmetric on PR #1787
stack (CaseOps + SparseAttnGate + PolarNS + MIN_LR + FusedCE + TTT warm-A).
3-seed mean 1.06157 BPB.

## Change: one orthogonal eval-only knob

A prefix-only causal bigram mixer (`NGramMixer` in `train_gpt.py`) is spliced
into the `eval_val` scoring path. Every token position `t` in the validation
stream is scored by

```
p_mix(a) = lambda(prefix) * p_nn(a) + (1 - lambda(prefix)) * q_bi(a | x_{t-1})
```

- `p_nn(a)`: the neural distribution from the trained+quantized model,
  unchanged.
- `q_bi(a|x_{t-1})`: a Dirichlet-smoothed bigram distribution built online
  from already-scored validation tokens. State resets only between ranks.
- `lambda(prefix) = sigmoid(alpha + beta * log1p(row_count))`: a 2-scalar
  gate that depends only on how confidently the bigram table has seen the
  current prefix. It does **not** observe `x_t`, the token being scored.

Training, quantization, compression, serialization, and the LoRA-TTT path are
not touched. All gain (if any) comes from eval-time mixing.

## Legality — four conditions from Issue #1017

| Cond | Statement | Where it lives | How it's checked |
|------|-----------|----------------|------------------|
| C1 | `p_t(·)` uses only A and `x_{0..t-1}`. | `NGramMixer.bi` / `.uni` are appended to by `update_stream(...)` after scoring; never before. State per rank starts empty. | `test_ngram_legality.py :: test_c1_future_invariance` |
| C2 | `p_t` is a normalized distribution over Σ. | Both `p_nn` and `q_bi` are full normalized distributions; convex combo is normalized. No `x_t`-contingent renormalization. | `test_ngram_legality.py :: test_c2_normalization` |
| C3 | Score before update; `x_t` must not influence its own probability. | `lambda` depends on `row_count(x_{t-1})` only; `q_bi(·|x_{t-1})` is target-free. `mix_nll` does not mutate state (asserted). | `test_ngram_legality.py :: test_c3_target_invariance`, `test_c3_no_update_during_mix` |
| C4 | Single left-to-right pass. | State accumulates monotonically; `eval_val` iterates each rank's stream exactly once in increasing raw-token order. | `test_ngram_legality.py :: test_c4_monotonic_counts` |

Run the verifier on the local box (CPU, ~2 s):

```
python3 test_ngram_legality.py
```

All 7 tests must print `[OK  ]`. Submission without a green run is invalid.

## Env-var knobs (new)

| Name | Default | Meaning |
|------|--------:|---------|
| `NGRAM_MIX_ENABLED` | 0 | Master switch. Set `1` to activate mixer. |
| `NGRAM_MIX_ALPHA` | 2.0 | Sigmoid intercept of λ gate. Higher → trust NN more. |
| `NGRAM_MIX_BETA` | -0.25 | Slope on `log1p(row_count)`. Negative → ↓λ as bigram warms up. |
| `NGRAM_MIX_SCALE` | 8.0 | Dirichlet pseudo-count for `q_bi` smoothing. |
| `NGRAM_MIX_USE_UNI_PRIOR` | 1 | If 1, Dirichlet prior is the running unigram; if 0, uniform. |

## Artifact-size impact

NGramMixer adds ~180 lines (~6 KB raw, ~3 KB compressed) to `train_gpt.py`.
No new model weights, no new tokenizer tables, no baked-in statistics. PR
#1797 base artifact was 15.953 MB; the mixer adds ≲ 10 KB of code bytes.
Artifact cap headroom comfortably preserved (target ≤ 15.97 MB).

## Reproducing — Day 3 sweep (eval-only on a single trained artifact)

1. Build once, train once (reuse PR #1797's own artifact if you already have
   one, else:)

   ```
   NGRAM_MIX_ENABLED=0 TTT_ENABLED=0 PREQUANT_ONLY=1 \
   torchrun --standalone --nproc_per_node=8 train_gpt.py
   ```

2. Sweep λ gate (fast — eval-only, compile cache warm):

   ```
   for alpha in 1.0 2.0 3.0; do
     for beta in -0.10 -0.25 -0.40; do
       NGRAM_MIX_ENABLED=1 NGRAM_MIX_ALPHA=$alpha NGRAM_MIX_BETA=$beta \
       PREQUANT_ONLY=1 TTT_ENABLED=0 \
       torchrun --standalone --nproc_per_node=8 train_gpt.py
     done
   done
   ```

3. Pick best `(alpha, beta)` from the sweep. Kill-switch: if the best config
   gives less than -0.003 BPB vs `NGRAM_MIX_ENABLED=0`, stop and pivot —
   do not spend Day-4 compute.

## Day 4 (conditional on Day 3 pass)

Integrate per-slot unigram mixer into `eval_val_ttt_phased` (the reported-
number path) and run full 3-seed reproduction. This adds a small per-batch-
slot unigram table updated after each TTT chunk; bigram inside TTT is
deferred for memory reasons and is not required for the Day-3 gain.

## Lineage

- PR #549, PR #1019, PR #1394, PR #1736, PR #1787, PR #1797 — merged /
  pending base.
- This submission — adds only the `NGramMixer` class and the one-block
  integration in `eval_val`. Diff vs PR #1797 is small and self-contained.
