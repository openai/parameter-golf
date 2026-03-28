# Non-Record: CTW Eval-Time Augmentation + Full SOTA Stack

**val_bpb**: `TBD` (to be updated after 8×H100 validation)
**Artifact size**: `TBD`
**Hardware**: 8×H100 SXM, 600s training + sliding window eval
**Track**: Non-record (iteration build)

## Summary

This submission combines the full merged SOTA neural stack (PR #549 base) with **Context Tree Weighting (CTW)** as a novel eval-time augmentation — the first CTW-based entry in Parameter Golf. CTW provides Bayesian-optimal sequential probability assignment over variable-order Markov models, replacing the heuristic alpha mixing used by current n-gram submissions with provably minimax-optimal weighting. It adds **zero bytes** to the 16MB artifact (the suffix tree is built entirely at eval time from already-scored tokens) and is fully rule-compliant (backward-looking only).

## Novel Contribution: Context Tree Weighting

### What CTW Does
CTW (Willems, Shtarkov, Tjalkens, 1995) builds a suffix tree during evaluation and maintains at each node:
1. A **Krichevsky-Trofimov (KT) estimate** — the probability assuming that node is a leaf
2. A **weighted probability** — a 50/50 Bayesian mixture of the KT estimate and children's weighted probabilities

The root's weighted probability automatically averages over ALL possible tree source structures up to depth D, without any tunable hyperparameters.

### How It Integrates With the Neural Model
During sliding-window evaluation, after the neural model produces logits for each token:
1. CTW produces its own probability distribution from already-scored tokens
2. The two distributions are mixed in **log-odds space** (PAQ-style logistic mixing):
   `logit_mixed = w_neural * logit(p_neural) + w_ctw * logit(p_ctw)`
3. The mixed distribution is used for the final BPB calculation

### Implementation Details
- **Sparse, lazy M-ary CTW**: Nodes allocated on-demand to handle vocab_size=1024
- **Depth D=4**: Captures up to 4-gram context with Bayesian-optimal depth weighting
- **KT estimator**: Dirichlet-Multinomial with alpha=0.5 per symbol
- **Mixing weight**: w_ctw=0.1 (conservative, tunable)
- **~80 lines of code** added to eval, well within 1500-line limit

### Why This Is Novel
- As of March 28, 2026: **no Parameter Golf submission uses CTW** (confirmed via Issue #140)
- Current n-gram submissions use hand-tuned or entropy-adaptive alpha
- CTW replaces heuristics with provably minimax-optimal Bayesian weighting
- Estimated gain: 0.005-0.020 BPB over heuristic mixing

## Architecture (Full SOTA Stack)

- 11 layers, 512d, 8H/4KV GQA, 3x MLP with LeakyReLU²(0.5)
- U-Net skips, XSA on last 4 layers (GQA-aware)
- SmearGate + BigramHash(2048), Partial RoPE (16/64), LN Scale
- Value Residual Learning (VRL): 22 extra params
- Tied embeddings, logit softcap=30.0

## Training

- Muon (lr=0.025, WD=0.04, momentum 0.92->0.99) + AdamW (embed/scalar)
- OrthoInit, EMA(0.997) GPU-side + Tight SWA(50)
- seq_len=2048, batch=786,432 tokens, warmdown=3500

## Compression

- GPTQ-lite (runs during training budget, not eval — per Mar 24-25 enforcement)
- Int6 per-row (MLP+attn), Int8 (embeddings), zstd-22
- Late QAT (STE fake-quant at LR<0.15)

## Evaluation

- Sliding window (stride=64) + CTW augmentation in log-odds space
- TTT uses AdamW not SGD (PR #601: SGD hurts GPTQ models +0.030 BPB)
- Score-first only (backward-looking, rule-compliant)

## Reproduction

```bash
cd /workspace && git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
RUN_ID=anubhav_ctw_v1 SEED=1337 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 MAX_WALLCLOCK_SECONDS=600 \
  torchrun --standalone --nproc_per_node=8 \
    records/track_non_record_16mb/2026-03-28_CTW_VRL_LeakyReLU2_GPTQ/train_gpt.py
```

## References

- Willems, Shtarkov, Tjalkens (1995). "The Context-Tree Weighting Method: Basic Properties."
- Messias & Whiteson (2017). "Dynamic-Depth Context Tree Weighting." NIPS 2017.

## Acknowledgments

Built on signalrush (PR #414), abaybektursun (PR #549), and the Parameter Golf community.
