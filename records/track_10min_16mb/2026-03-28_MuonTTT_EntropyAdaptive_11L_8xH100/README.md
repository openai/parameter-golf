# 11L Muon TTT + Entropy-Adaptive Epochs

**val_bpb: 1.1179** (3-seed mean, std 0.0002) | **~15.9 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)

| Seed | step_avg | steps | Pre-TTT bpb | Post-TTT bpb | TTT gain | TTT time | Artifact |
|------|----------|-------|-------------|-------------|----------|----------|----------|
| 1337 | 83.5ms | 7,189 | 1.1214 | **1.11765** | -0.0037 | 477s | 15,944,410 |
| 42   | 83.5ms | 7,177 | 1.1217 | **1.11813** | -0.0035 | 485s | 15,873,826 |
| 2025 | 83.5ms | 7,175 | 1.1217 | **1.11790** | -0.0038 | 479s | 15,879,042 |
| **Mean** | **83.5ms** | **~7,180** | **~1.1216** | **1.1179 (std 0.0002)** | **-0.0037** | **~480s** | |

## Key Innovation 1: Muon as TTT Optimizer

Every prior TTT submission uses SGD in the test-time training loop. This submission replaces SGD with Newton-Schulz orthogonalized gradient updates -- the same Muon principle that dominates the training leaderboard, now applied to test-time adaptation.

### Why this works

Muon constrains each matrix update to the space of orthogonal transformations, normalizing the gradient direction. For TTT this means:
- No gradient blowup: updates only rotate weight matrices, cannot inflate them
- Better gradient signal: Newton-Schulz whitens the gradient, removing scale correlation between rows that SGD accumulates
- Faster per-epoch convergence: each TTT epoch moves farther in the useful direction

The result: +0.0037 TTT gain per seed vs SOTA's +0.0025 (SGD, same 3 epochs), with total TTT time remaining under 600s.

### Implementation

Replaces optimizer.step() in the TTT loop:

    with torch.no_grad():
        for p in ttt_params:
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if g.ndim >= 2:
                g = zeropower_via_newtonschulz5(g, steps=3)
            p.data.add_(g.to(p.dtype), alpha=-cos_lr)

zeropower_via_newtonschulz5 is already present in every train_gpt.py. 3 NS steps balance orthogonalization quality vs eval wall-clock (5 steps exceeded the 600s eval budget; 3 steps complete in ~480s).

## Key Innovation 2: Entropy-Adaptive TTT Epochs

All prior TTT submissions use a fixed epoch count per chunk. This submission dynamically assigns 2, 3, or 4 TTT epochs per chunk based on the model's measured uncertainty on that content.

After SCORE phase, the per-chunk NLL is globally synchronized across all DDP ranks (critical: per-rank NLL gives different epoch counts per rank -> different number of dist.all_reduce calls per chunk -> NCCL collective mismatch -> watchdog timeout at 600s). The global NLL gates epoch selection:

    cls_t = torch.tensor(chunk_loss_sum, device=device, dtype=torch.float64)
    ctc_t = torch.tensor(chunk_token_count, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(cls_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(ctc_t, op=dist.ReduceOp.SUM)
    chunk_nll = (cls_t / ctc_t).item()

    if chunk_nll > 2.1:    # hard content (code, math): 4 epochs
        effective_epochs = 4
    elif chunk_nll < 1.75: # easy content (prose): 2 epochs
        effective_epochs = 2
    else:
        effective_epochs = 3

This concentrates adaptation budget where it helps most. Average epochs ~3.0; total TTT time unchanged vs fixed-3-epoch baseline.

## Legal TTT Protocol

Score-first TTT following PR #461 framework:

1. Val tokens split into 1,893 non-overlapping 32K-token chunks
2. For each chunk:
   - SCORE: Sliding window eval under torch.inference_mode() -- no gradients, no weight mutation
   - TRAIN: Muon-style update on already-scored chunk. Entropy-adaptive 2/3/4 epochs, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on
4. Chunk N scored by model adapted only on chunks 0..N-1

### TTT Hyperparameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 32,768 tokens |
| Optimizer | Muon-style (Newton-Schulz NS=3 + LR step) |
| Learning rate | 0.002 (cosine decay across chunks) |
| Epochs per chunk | 2/3/4 entropy-adaptive (H_HIGH=2.1, H_LOW=1.75 nats) |
| Frozen blocks | None (all blocks adapt) |
| Gradient clip | 1.0 |

## Training Architecture

Full SOTA stack from PR #399 and PR #414:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3x with LeakyReLU(0.5)^2 |
| BigramHash | 1536 |
| XSA | Last 4 layers |
| RoPE | Partial (16/64 dims) |
| LN Scale | 1/sqrt(layer+1) |
| VE128 | Layers 9-10 |
| Weight avg | EMA(0.997) + Tight SWA(every 50) |
| Quantization | GPTQ-lite int6 + lzma (preset=7) |
| Optimizer | Parameter Banking + Parallel Muon |

## Run Command

    NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4
    EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50
    ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15
    VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
    TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768
    TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0
    TTT_MUON=1 TTT_NS_STEPS=3 TTT_ENTROPY_ADAPT=1
    TTT_ENTROPY_HIGH=2.1 TTT_ENTROPY_LOW=1.75
    MUON_WD=0.04 ADAM_WD=0.04
    MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
    MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92
    MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500
    ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=599 EVAL_STRIDE=64
    SEED=1337
    torchrun --standalone --nproc_per_node=8 train_gpt.py

## Timing Budget

| Phase | Time |
|-------|------|
| Training | ~600s (<=10 min) |
| Standard eval (int6 roundtrip + sliding window) | ~82s |
| Legal TTT (score-first + adaptation) | ~480s |
| Total eval | ~562s (< 10 min) |

## Credits

- LeakyReLU^2 activation: PR #493 by @parinzee, PR #518 by @sofiabod
- Optimizer (Parameter Banking + Parallel Muon): PR #399 by @abaybektursun
- TTT recipe (score-first framework): PR #461 by @Christopher-Lee-McClendon
- Base architecture: PR #414 by @signalrush
- SOTA base adapted from: @abaybektursun (val_bpb 1.1194)
