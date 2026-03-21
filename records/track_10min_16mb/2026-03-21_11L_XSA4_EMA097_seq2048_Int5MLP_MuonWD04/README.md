# 11L + XSA4 + EMA(0.997) + seq2048 + Int5-MLP + MuonWD=0.04 + LateK-FP16

**val_bpb: 1.1357** (seed=1337, sliding window stride=64, post int5/int6+zstd-22 quantization roundtrip)
Hardware: 8×H100-80GB-SXM5 | Steps: ~8903 | Wallclock: 600s | Artifact: ~15.79MB

## Approach

This submission stacks eight techniques on the baseline, building directly on PRs #135, #162, #264, #265, and #287. All changes are incremental over those prior submissions — no new external dependencies are required.

### 1. Per-Row Int6 Quantization + Int5-MLP + zstd-22 (PR #264, extended)

Attention weight matrices are quantized to int6 (symmetric, 63 levels, per-row scaling). MLP weights (`c_fc`, `c_proj`) use int5 (32 levels, [-16, 15]) instead of int6, which compresses ~15% better at the cost of a small precision penalty. This saves approximately 1.9MB relative to uniform int6, which directly funds the 11th transformer layer while staying within the 16MB artifact limit. Straight-through estimator (STE) fake-quantization is applied during training for both int6 and int5. The tied token embedding is kept in fp16. Compressed with zstandard at level 22.

### 2. Late-K FP16 on Final Layer

The key projection (`c_k.weight`) of the last transformer block is kept in fp16 rather than int6. This avoids quantization noise in the most context-sensitive attention keys, at a cost of ~131KB — a favorable quality/size trade-off.

### 3. 11-Layer U-Net Architecture

The model uses 11 transformer blocks (5 encoder + 6 decoder) with U-Net skip connections, up from the 9-layer baseline. The 11th layer becomes feasible due to the byte savings from int5-MLP quantization.

### 4. Exclusive Self-Attention (XSA) on Last 4 Layers (PR #265)

XSA projects each value vector out of the attention output before the projection layer, preventing the model from trivially attending to each token's own value. Applied to the deepest 4 layers (`xsa_last_n=4`). Implemented as an efficient GQA-aware subtraction with no repeat_interleave. Confirmed neutral vs 3 layers at seq_len=2048 with EMA (three independent runs). The implementation closely follows PR #265.

### 5. EMA Weight Averaging, decay=0.997 (PR #287)

An exponential moving average of all model parameters is maintained throughout training: `ema = 0.997 * ema + 0.003 * param`. The EMA weights are substituted into the model before the final quantization and evaluation. This replaces Stochastic Weight Averaging (SWA), which was found to exceed the 16MB artifact budget when combined with 11 layers and seq2048 batch tokens. EMA adds zero artifact size (the shadow copy is discarded after training) and provides smoother, better-regularized weights for quantization. Based on PR #287 (which uses the same decay=0.997).

### 6. Sequence Length 2048

Training sequence length extended from 1024 to 2048. This increases gradient token count per step (same batch token budget), giving each update richer long-range context. The warmdown schedule is tuned to 2000 steps (rather than the 3500 default) to match the step budget at seq2048 training speed.

### 7. SmearGate + BigramHash(2048) + OrthoInit (PR #135)

Three techniques from PR #135 are included unchanged:
- **SmearGate**: a learned per-dimension sigmoid gate that blends each token embedding with the preceding token, injecting lightweight bigram context at the embedding layer.
- **BigramHash**: a 2048-bucket hash table (dim=64, projected to 512) that maps consecutive token pairs to learned embeddings via `xor(36313*t[i], 27191*t[i-1]) % 2047`. The bucket count is reduced to 2048 (from the PR #135 default of 4096) to fit the artifact budget with 11 layers. The embedding dimension is 64 rather than 128, maintaining the ~0.5MB FP16 footprint.
- **OrthoInit**: all large (≥64×64) weight matrices are initialized with `nn.init.orthogonal_(gain=1.0)`. Output projections are additionally scaled by `1/sqrt(2*num_layers)` following muP conventions.

### 8. Muon Optimizer with Weight Decay = 0.04 (PR #162, tuned)

PR #162 introduced decoupled weight decay for Muon. We tune the decay from the PR #162 default: sweep over {0.01, 0.02, 0.04, 0.1} shows 0.04 as optimal (+0.0006 BPB over 0.02; 0.1 catastrophic). Muon momentum is fixed at 0.95 with a warmup from 0.85 over 500 steps. RoPE base is 500K (confirmed −0.0036 BPB over the default 10K).

## Hyperparameters

| Parameter | Value |
|---|---|
| num_layers | 11 |
| model_dim | 512 |
| num_heads | 8 |
| num_kv_heads | 4 (GQA) |
| mlp_mult | 3 (hidden=1536) |
| train_seq_len | 2048 |
| train_batch_tokens | 524,288 |
| warmdown_iters | 2000 |
| warmup_steps | 20 |
| matrix_lr | 0.02 |
| scalar_lr | 0.02 |
| tied_embed_lr | 0.03 |
| muon_momentum | 0.95 (warmup from 0.85 over 500 steps) |
| muon_weight_decay | 0.04 |
| rope_base | 500,000 |
| eval_stride | 64 |
| ema_decay | 0.997 |
| xsa_last_n | 4 |
| bigram_vocab_size | 2048 |
| bigram_dim | 64 |
| late_k_fp16_layers | 1 |
| use_int5_mlp | True |
| use_int6_qat | True |
| quantization | int5 (MLP) + int6 (attn) + fp16 (embed, last c_k) |
| compression | zstd level 22 |

## Results

| Seed | val_bpb | steps | wallclock |
|---|---|---|---|
| 1337 | **1.1357** | 8903 | 600s |

Single-seed result. Multi-seed validation pending.

## Technique Attribution

| Technique | Source |
|---|---|
| SmearGate, BigramHash, OrthoInit, tied-embed LR, matrix/scalar LR | PR #135 |
| Muon weight decay | PR #162 |
| Int5-MLP, Int6 QAT, Late-K FP16 | PR #264 |
| Exclusive Self-Attention (XSA) | PR #265 |
| EMA weight averaging (decay=0.997) | PR #287 |
| Warmdown tuning for seq2048, MuonWD=0.04, BigramVocab=2048 | this submission |
