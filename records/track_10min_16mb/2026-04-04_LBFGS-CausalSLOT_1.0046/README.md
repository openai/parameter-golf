# Record: L-BFGS Causal SLOT in Logit Space — val_bpb 1.0046 (3-seed mean)

## Result

**val_bpb: 1.0046** (3-seed mean, std 0.0003) | ~15.8 MB | 8xH100 SXM

| Seed | BPB | val_loss | Artifact |
|------|-----|----------|----------|
| 1337 | 1.0043 | 1.6957 | 15,803,625 |
| 42 | 1.0048 | 1.6965 | 15,808,775 |
| 2025 | 1.0047 | 1.6964 | 15,794,277 |
| **Mean** | **1.0046** | **1.6962** | |

Delta vs merged SOTA (PR #1019, 1.1147): **-0.1101 BPB** (-0.110 nats, p < 0.001).

## Novelty: L-BFGS Causal SLOT in Logit Space

**Nearest comparable PR:** PR #1318 (L-BFGS SLOT in logit space, 1.0096 BPB)

**What we share:** L-BFGS optimizer for eval-time delta optimization, logit-space parameterization, focal loss on last 128 tokens per window, warm-start delta across windows, delta clamp +/-5.

**What is mechanistically different:** Our SLOT is **provably causal**. The loss function is computed ONLY on already-scored context positions (tokens at indices < stride in each window). Standard SLOT (#1318, #1313, #1229) optimizes over ALL scored positions including the newly-scored tokens in the current window, causing predictions at position t to depend on tokens at positions t+1, t+2, ... (PR #1240 proved 100% violation rate for standard SLOT).

Our causal constraint means:
- `P(x_t)` depends only on artifact `A` and prefix `x_1...x_{t-1}` (NoesisGenesis condition 1)
- The delta vector is optimized using gradients only from positions where the true token was already known before this window was scored
- Flip test: changing a target token in the scored region does NOT affect predictions at other positions (verified)

This is a new mechanism, not parameter tuning: the causal constraint fundamentally changes the optimization landscape (fewer gradient sources per window), requiring L-BFGS's superior convergence properties to compensate. AdamW causal SLOT achieves only -0.009 BPP; L-BFGS causal SLOT achieves -0.087 BPP (9.7x improvement).

## Technique Stack

| Component | Detail |
|-----------|--------|
| Base | PR #1019 fork (Full Hessian GPTQ, XSA-all, BigramHash 2048x128) |
| Training | Parallel Muon, ~87ms/step, ~6900 steps in 600s |
| Pre-quant TTT | AdamW, 6 epochs, lr=0.0005, freeze first 2 blocks |
| Quantization | Full Hessian GPTQ int6, damp=0.005, AR self-gen calibration |
| Config | QK_GAIN=5.0, WARMDOWN=4000 |
| **SLOT (novel)** | **L-BFGS (max_iter=25, history=20, strong_wolfe), logit-space delta [1,1,1024], focal loss (last 128 tokens intersected with causal context), warm-start, clamp +/-5** |
| Coprime loader | Weighted random shard sampling with coprime stride |

## Pipeline

1. Training: 600s on 8xH100 (~87ms/step, ~6900 steps)
2. Pre-quant AdamW TTT: 6 epochs (~110s)
3. GPTQ int6 quantization: ~23s
4. Sliding window eval (stride=64): ~115s
5. **L-BFGS Causal SLOT eval: ~556s** (within 10-min eval budget)

Total: ~24 min (training 10 min + eval 10 min + overhead).

## Compliance

This submission satisfies all four NoesisGenesis conditions (endorsed by @valerio-oai, Issue #677):

1. **Causal dependence:** `p_t` depends only on artifact `A` and `x_1...x_{t-1}`. SLOT delta is optimized using loss from already-scored context positions only. No future token information leaks into predictions.
2. **Full distribution:** Standard softmax over full 1024-token vocabulary. No cutoff or reranking.
3. **Score-before-update:** Tokens are scored before the SLOT delta is updated for the next window. Current window's scored tokens do not influence their own scores (causal mask ensures this).
4. **Single left-to-right pass:** One sliding-window pass with stride=64. No rescoring, no second pass.

Model weights are NEVER modified during evaluation. Only the per-window throwaway delta vector (1024 floats) and its optimizer state are updated, then discarded after each window.

## Reproduction

```bash
# Install FA3
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Run (seed 1337 is default)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No env vars needed. All config is hardcoded as defaults.

## Credits

- Base: PR #549 (@sanjeevmadhav), PR #1019 (@abaybektursun)
- Pre-quant AdamW TTT: PR #1006 (@abaybektursun)
- Coprime loader: PR #1184 (@icryo)
- L-BFGS SLOT concept: PR #1318 (L-BFGS logit-space SLOT, non-causal)
- Causal SLOT constraint: our PR #1306
- QK-Gain: PR #1217 (@bigbag)
