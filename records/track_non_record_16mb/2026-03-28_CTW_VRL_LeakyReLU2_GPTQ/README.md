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
### Logs of the run

Total parameters: 26,829,912

Starting training: batch=131,072 tokens, seq_len=2048, warmdown=3500
Step 1: Late QAT activated (lr_scale=0.0500)
step=1/500 | loss=6.9312 | lr=0.0500 | 92285ms/step | 0s/1200s | ETA=20.0min | qat=ON
step=2/500 | loss=6.7275 | lr=0.1000 | 46943ms/step | 92s/1200s | ETA=18.5min | qat=ON
step=3/500 | loss=6.1427 | lr=0.1500 | 31823ms/step | 94s/1200s | ETA=18.4min | qat=ON
step=4/500 | loss=5.8644 | lr=0.2000 | 24261ms/step | 95s/1200s | ETA=18.4min | qat=ON
step=5/500 | loss=5.9709 | lr=0.2500 | 19725ms/step | 97s/1200s | ETA=18.4min | qat=ON
step=25/500 | loss=5.3123 | lr=0.0448 | 5209ms/step | 129s/1200s | ETA=17.9min | qat=ON
step=50/500 | loss=5.1736 | lr=0.0402 | 3398ms/step | 168s/1200s | ETA=17.2min | qat=ON
step=75/500 | loss=5.0655 | lr=0.0359 | 2794ms/step | 208s/1200s | ETA=16.5min | qat=ON
step=100/500 | loss=5.1499 | lr=0.0319 | 2490ms/step | 247s/1200s | ETA=15.9min | qat=ON
step=125/500 | loss=4.9813 | lr=0.0281 | 1584ms/step | 287s/1200s | ETA=9.9min | qat=ON
step=150/500 | loss=4.9898 | lr=0.0245 | 1585ms/step | 327s/1200s | ETA=9.2min | qat=ON
step=175/500 | loss=5.0093 | lr=0.0211 | 1588ms/step | 367s/1200s | ETA=8.6min | qat=ON
step=200/500 | loss=4.9182 | lr=0.0180 | 1590ms/step | 406s/1200s | ETA=8.0min | qat=ON
step=225/500 | loss=4.8912 | lr=0.0152 | 1593ms/step | 446s/1200s | ETA=7.3min | qat=ON
step=250/500 | loss=4.9016 | lr=0.0125 | 1593ms/step | 486s/1200s | ETA=6.6min | qat=ON
step=275/500 | loss=4.9626 | lr=0.0102 | 1590ms/step | 526s/1200s | ETA=6.0min | qat=ON
step=300/500 | loss=4.8947 | lr=0.0080 | 1590ms/step | 565s/1200s | ETA=5.3min | qat=ON
step=325/500 | loss=4.8771 | lr=0.0062 | 1590ms/step | 605s/1200s | ETA=4.6min | qat=ON
step=350/500 | loss=4.8561 | lr=0.0045 | 1589ms/step | 645s/1200s | ETA=4.0min | qat=ON
step=375/500 | loss=4.8868 | lr=0.0031 | 1589ms/step | 685s/1200s | ETA=3.3min | qat=ON
step=400/500 | loss=4.8844 | lr=0.0020 | 1588ms/step | 724s/1200s | ETA=2.6min | qat=ON
step=425/500 | loss=4.8747 | lr=0.0011 | 1588ms/step | 764s/1200s | ETA=2.0min | qat=ON
step=450/500 | loss=4.8459 | lr=0.0005 | 1588ms/step | 804s/1200s | ETA=1.3min | qat=ON
step=475/500 | loss=4.8406 | lr=0.0001 | 1589ms/step | 844s/1200s | ETA=0.7min | qat=ON
step=500/500 | loss=4.8392 | lr=0.0000 | 1592ms/step | 884s/1200s | ETA=0.0min | qat=ON

Applying EMA weights...
Applying SWA (10 checkpoints)...
  eval:   0.0% (0/969057) | 0.0s

## References

- Willems, Shtarkov, Tjalkens (1995). "The Context-Tree Weighting Method: Basic Properties."
- Messias & Whiteson (2017). "Dynamic-Depth Context Tree Weighting." NIPS 2017.

## Acknowledgments

Built on signalrush (PR #414), abaybektursun (PR #549), and the Parameter Golf community.
