# Record: Shallow Recurrence + LoRA + LeakyReLU(0.9)^2 + QK-Gain 4.0 + TTT

**val_bpb: TBD** (awaiting compute -- 3-seed runs pending) | **~15.9 MB artifact** | 8xH100 SXM

## Status

**WIP** -- This submission describes a concrete approach combining independently validated hyperparameter improvements with a novel architectural contribution (shallow recurrence with low-rank corrective adapters). Training results will be added once compute credits are available.

## Novel Contribution: Shallow Recurrence with LoRA Corrections

The primary contribution of this submission is **bounded-depth recurrence with per-pass low-rank corrections** -- a technique that simulates deeper networks within the 16MB parameter budget.

### Background

Weight sharing (depth recurrence) has been extensively explored in this competition and mostly fails due to **quantization error amplification** -- int6 quantization noise compounds multiplicatively through shared layers, causing catastrophic degradation at 3+ cycles (+4.3 BPB, PR #363). However, PR #686 demonstrated that **shallow recurrence (1-2 repeats)** survives quantization and reaches 1.1182 BPB with just 2 repeated layers.

### Our approach

We take layers 4 and 5 (middle of the 11-layer U-Net encoder) and run each through a second pass, yielding 13 virtual layers from 11 physical layers. Critically, the second pass applies:

1. **RMSNorm before repeat** -- Normalizes the residual stream before re-entry, preventing distribution drift
2. **Low-rank attention corrections** (rank-2 LoRA on Q, K, V, O projections only) -- Provides per-pass specialization while keeping the shared MLP weights intact. The corrections act as error correctors that compensate for the fact that pass 2 sees a different input distribution than pass 1
3. **Learnable scaling factor (alpha)** -- Initialized at 0.4, allows the model to control correction magnitude. Prevents overcorrection early in training

### Why this works (mechanistically)

- **Bounded recurrence (2 passes)** stays in the "safe zone" where quantization error doesn't compound catastrophically
- **LoRA corrections are attention-only** -- MLPs are left shared because (a) they're the largest parameter cost and (b) MLP sharing is more stable than attention sharing
- **Parameter overhead is negligible**: ~14K params (28KB fp16) = 0.18% of the 16MB budget. These stay as fp16 passthrough, immune to int6 quantization noise
- **Net effect**: the model gets 2 extra virtual layers of depth at near-zero parameter cost, which can improve representation quality for the hardest tokens

### Design choices

- **Rank 2** (not higher): At 512d, rank-2 LoRA provides 4 directions of correction per matrix. Higher ranks risk overfitting per-pass corrections and destroying shared structure
- **Layers 4, 5** (encoder middle): These layers process intermediate representations where recurrence is most beneficial. Early layers handle token embedding (low value from recurrence), late decoder layers have U-Net skip connections that already provide representational flexibility
- **No MLP LoRA**: MLPs at 3x width (1536) are already the dominant parameter cost. Adding LoRA to MLPs would bloat the budget without proportional gain, and MLPs are empirically more stable under sharing than attention

## Hyperparameter Improvements

### 1. LeakyReLU(0.5)^2 -> LeakyReLU(0.9)^2 (expected ~0.010 BPB)

Community sweep by @MatoTeziTanka (issue #140, comment 13) tested 7 slopes (0.1-0.9) and found monotonic improvement with higher negative slope. Slope 0.9 beat 0.5 by 0.013 BPB, and the trend had not yet peaked.

### 2. QK-Gain 1.5 -> 4.0 (expected ~0.006 BPB)

Per-head learnable scalar multiplier on Q vectors after RoPE and QK-norm. Validated in PR #1176 sweep (originally from PR #1125).

### 3. Score-First Muon-TTT (expected ~0.003 BPB)

Legal test-time training variant: score each validation chunk first under `torch.inference_mode()`, record NLL, then update model parameters on already-scored tokens using Newton-Schulz orthogonalized SGD. 3 epochs, lr=0.002, cosine decay, first 2 blocks frozen.

### 4. Extended Warmdown 3500 -> 4000 iterations (expected ~0.001 BPB)

Longer cosine warmdown phase for more stable weight averaging.

## Architecture

- 11 physical layers -> **13 virtual layers** (layers 4,5 repeated with LoRA)
- 512d, 8 GQA heads / 4 KV heads
- MLP 3x (1536) with LeakyReLU(0.9)^2
- BigramHash(2816x160)
- Partial RoPE (16/64 dims)
- XSA on all 11 layers
- U-Net skip connections + SmearGate
- QK-Gain 4.0 (per-head learnable)
- EMA(0.997) + SWA(every 50)
- Parallel Muon optimizer with Split-LR (early=0.025, late=0.030)
- Full Hessian GPTQ with AR self-generated calibration
- Brotli-11 + byte-shuffle compression

## Run Command

```bash
NEGATIVE_SLOPE=0.9 QK_GAIN_INIT=4.0 TTT_ENABLED=1 \
BIGRAM_VOCAB_SIZE=2816 BIGRAM_DIM=160 WARMDOWN_ITERS=4000 \
RECUR_LAYERS=4,5 RECUR_LORA_RANK=2 RECUR_ALPHA_INIT=0.4 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation Plan (post-compute)

1. Baseline: hyperparameters only (no recurrence) -- establish control
2. Recurrence without LoRA -- confirm parity with PR #686
3. Recurrence + LoRA (rank 2) -- measure LoRA contribution
4. Alpha sweep: [0.3, 0.5, 0.7] -- find optimal correction strength
5. Layer selection sweep: [3,4], [4,5], [5,6] -- find optimal recurrence position
6. LeakyReLU slope sweep: 0.90, 0.95, 0.98

## Credits

- **Base stack**: PR #1179 by @dexhunter
- **Shallow recurrence insight**: PR #686 by @msisovic (demonstrated 2-repeat survival)
- **LeakyReLU slope sweep**: @MatoTeziTanka (issue #140)
- **QK-Gain**: PR #1125/#1176 sweep
- **TTT implementation**: PR #549 by @abaybektursun
