# Record: Split-LR + N-gram Agreement + Full Hessian GPTQ + Brotli

**val_bpb: 1.1078** (3-seed mean, std 0.0009) | **1.8752 nats** | **~15.86 MB** | 8xH100 SXM, 600s train + 449s eval

Built on [PR #1179](https://github.com/openai/parameter-golf/pull/1179) by @dexhunter (training) and [PR #1145](https://github.com/openai/parameter-golf/pull/1145) by @AnirudhRahul (n-gram agreement evaluation).

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | Steps | ms/step | Sliding BPB | **N-gram BPB** | Artifact |
|------|-------|---------|-------------|----------------|----------|
| 1337 | ~6780 | 88.0 | 1.1110 | **1.1083** | 15,853,466 |
| 42 | ~6780 | 88.0 | 1.1095 | **1.1068** | 15,857,705 |
| 2025 | ~6780 | 88.0 | 1.1112 | **1.1085** | 15,846,914 |
| **Mean** | | | **1.1106** | **1.1078** | |

SOTA (PR #1019, 3-seed mean): **1.8822 nats**. This run: **1.8752 nats**. Delta: **-0.00697 nats**. Clears the 0.005-nat threshold.

### Timing Budget

| Phase | Time |
|-------|------|
| Training (wallclock cap) | ~591s |
| GPTQ calibration (reserved) | ~7s |
| Post-EMA eval | ~2s |
| Int6 roundtrip eval | ~7s |
| Sliding window eval (stride=64) | ~78s |
| **N-gram agreement eval** | **~449s** |
| **Total eval** | **~536s** |

## What's New vs PR #1019

### Training improvements (from PR #1179)
1. **Split-LR** — different learning rates for early (0.025) vs late (0.030) layers
2. **BigramHash(2816x160)** — wider projection (160 vs 112), fewer buckets
3. **Sigmoid-gated U-Net** — learnable gates on encoder-decoder skip connections
4. **Soft-round QAT** — temperature-controlled rounding (alpha 1->16) replacing STE
5. **Brotli-11 + byte-shuffle** — saves ~400KB vs LZMA
6. **Coprime-stride data loader** — better data shuffling and coverage

### Evaluation improvement (from PR #1145)
7. **Online n-gram agreement** — 3 causal experts (token n-gram, within-word, word-start) with agreement boosting. Adjusts LLM probabilities via properly normalized exponential tilting. Contributes **-0.0028 BPB**.

## N-gram Agreement: How It Works

Three online n-gram experts predict the next token using only already-scored (past) tokens:
- **Token n-gram** (16-gram context, hash table): predicts based on raw token patterns
- **Within-word continuation**: predicts next subword within the current word
- **Word-start hints**: predicts first token of next word based on previous word context

For each position, the expert with highest expected gain is selected. When 2+ experts agree on the same token, their boost is increased. The LLM's probability is adjusted via exponential tilting:

```
p_adjusted = (scale * p_true) / (1 - p_hint + scale * p_hint)
```

This produces a properly normalized distribution (sums to 1.0). The adjustment is:
- **Causal**: each expert predicts BEFORE updating its state with the target token
- **Score-first**: runs under `torch.inference_mode()`, no model parameters modified
- **Properly normalized**: exponential tilting with correct partition function

## Legality

- Standard F.cross_entropy for training
- N-gram agreement: causal, score-first, properly normalized (exponential tilting)
- No training on validation data
- No SLOT, no multi-epoch TTT
- GPTQ calibration within training budget
- Artifact < 16,000,000 bytes (all seeds)
- Training <= 600s, eval <= 600s (all seeds)

## Architecture

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) |
| MLP | 3x (1536) with LeakyReLU(0.5)^2 |
| Attention | XSA on all 11 layers |
| BigramHash | 2816 x dim=160 |
| Split-LR | early=0.025, late=0.030, bank_split=5 |
| Skip connections | Sigmoid-gated U-Net |
| QAT | Soft-round (alpha ramp 1->16) |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| SmearGate | Position-mixing gate |
| Weight avg | EMA(0.997) + SWA(every 50) |
| Quantization | Full Hessian GPTQ int6 |
| Compression | Brotli quality=11 + byte-shuffle |
| Optimizer | Parallel Muon + Parameter Banking |
| Eval | Online n-gram agreement (token 16-gram + within-word + word-start) |

## Run Command

```bash
# Training (3 seeds)
pip install brotli
for SEED in 1337 42 2025; do
  BIGRAM_DIM=160 SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_seed${SEED}.log
  cp final_model.int6.ptz checkpoints/final_model_seed${SEED}.int6.ptz
done

# N-gram agreement evaluation (per seed)
gcc -O3 -march=native -shared -fPIC -o libonline_ngram_state.so online_ngram_state.c
for SEED in 1337 42 2025; do
  BIGRAM_DIM=160 CHECKPOINT=checkpoints/final_model_seed${SEED}.int6.ptz \
  torchrun --standalone --nproc_per_node=8 eval_ngram_on_checkpoint.py
done
```

## Credits

- **Training scaffold**: [PR #1179](https://github.com/openai/parameter-golf/pull/1179) by @dexhunter (built on PR #1019 by @abaybektursun)
- **N-gram agreement eval**: [PR #1145](https://github.com/openai/parameter-golf/pull/1145) by @AnirudhRahul
- **Full Hessian GPTQ**: [PR #535](https://github.com/openai/parameter-golf/pull/535) by @raahilshah
- **XSA-all**: [PR #478](https://github.com/openai/parameter-golf/pull/478) by @gowtham0992

## Included Files

- `train_gpt.py` — training + quantization + sliding window eval
- `online_best_agree_eval.py` — n-gram agreement evaluation
- `online_ngram_state.c` — native n-gram hash table (compiled at eval time)
- `eval_ngram_on_checkpoint.py` — helper to run n-gram eval on saved checkpoints
- `train_seed{1337,42,2025}.log` — training logs
- `submission_ngram_seed{1337,42,2025}.log` — n-gram eval logs
- `submission.json` — leaderboard metadata
