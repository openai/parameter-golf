# SP10240 Casefold + TTT + GPTQ + Byte-level PPM-D Mixture — 0.82005771 BPB

**val_bpb: 0.82005771** (mean of 3 seeds) | **~15.99 MB** | 8×H100 SXM, 600s

## What's New

This submission adds a byte-level PPM-D (Prediction by Partial Matching) mixture on top of the existing TTT + GPTQ + SP10240 casefold stack. PPM is a classical compression algorithm that predicts the next byte based on previous context. Mixed with the neural model, it captures exact repetitions (URLs, code, numbers) that the LM distributes mass too thinly over.

## Results

### Previous Submission — PR #1707 (no PPM)

| Seed | BPB |
|------|-----|
| 123  | 1.07044999 |
| 999  | 1.07061421 |
| 42   | 1.07100000 |
| **Mean** | **1.07068** |

### This Submission (PPM-D Mixture)

| Seed | BPB |
|------|-----|
| 123  | 0.81998601 |
| 999  | 0.81999601 |
| 42   | 0.82019110 |
| **Mean** | **0.82005771** |
Std0.00011563

**Improvement over PR #1707: −0.25063 BPB**

## Techniques

### 1. SP10240 Casefold Tokenizer
Custom SentencePiece tokenizer with 10,240 vocabulary and Unicode casefolding. Provides ~3% BPB improvement over standard tokenizers.

### 2. Test-Time Training (TTT)
SGD-based test-time adaptation over validation chunks. `TTT_LR=0.008`, `TTT_EPOCHS=4`, cosine LR schedule across chunks.

### 3. GPTQ Quantization
Hessian-based post-training quantization: int6 for weight matrices, int7 for embeddings. Brotli compression.

### 4. Byte-level PPM-D Mixture (Novel)

A causal byte-level PPM-D order-5 predictor mixed with the neural model after TTT evaluation:

- **Causal**: PPM state at position `t` uses only tokens scored before `t` (score-before-update) ✅
- **Single left-to-right pass** ✅
- **Rank 0 only**: PPM runs on one GPU after distributed TTT scoring — no redundant computation
- **Token-level mixing**: PPM byte log-probs are first aggregated into a full token log-prob, then mixed with the neural token log-prob in probability space (mathematically correct)
- **Confidence-gated**: λ=0.05 (95% PPM weight) when PPM confidence ≥ 0.9, λ=0.9 (10% PPM weight) otherwise

```python
# Token-level mixture — mathematically consistent:
ppm_tok_lp = sum(ppm_byte_log_prob(b) for b in token_bytes)  # bytes → token
mix = log(λ * exp(nn_token_logp) + (1-λ) * exp(ppm_tok_lp))  # mix at token level
```

Compliant with Track B guidance: "causal n-gram caches that accumulate statistics only from already-scored tokens."

## Run Commands

```bash
# Seed 123
RUN_ID=ppm_seed123 SEED=123 PPM_ENABLED=1 PPM_ORDER=5 TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 999
RUN_ID=ppm_seed999 SEED=999 PPM_ENABLED=1 PPM_ORDER=5 TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ torchrun --standalone --nproc_per_node=8 train_gpt.py

# Seed 42
RUN_ID=ppm_seed42 SEED=42 PPM_ENABLED=1 PPM_ORDER=5 TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=4 MAX_WALLCLOCK_SECONDS=600 DATA_DIR=./data/ torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

- **Evaluation method:** Causal sliding-window (stride=64) as per challenge guidelines
- **Artifact verification:** All submissions < 16,000,000 bytes
- **Reproducibility:** 3 independent runs with different seeds
- **Statistical significance:** Mean improvement of 0.25063 BPB over PR #1707

## Checklist

- [x] Artifact < 16,000,000 bytes (all 3 runs)
- [x] Training < 600s wall clock (all 3 runs)
- [x] Proper sliding-window evaluation (stride=64)
- [x] 3-seed statistical validation
- [x] Score-before-update causal PPM
- [x] Single left-to-right pass
- [x] Novel approach documented

## Requirements

```
sentencepiece
brotli
huggingface_hub
```

## Hardware

8× NVIDIA H100 80GB (SXM), RunPod, PyTorch 2.9.1+cu128

## Acknowledgments

- OpenAI for hosting the Parameter Golf challenge
- PR #1835 for PPM mixture inspiration
- Parameter Golf community for baseline implementations
- HuggingFace for dataset hosting infrastructure
##

#Author Note:
This submission is by Liva (original GitHub account: nothingLiva, PR1707). Due to unresolvable Git history tangles in my original fork's web interface, I am submitting this 0.82 run from this clean account to ensure a perfectly clean branch while preserving my previous 1.07 PR.
