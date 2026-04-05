# Neuromodulatory Depth-Recurrent Transformer

**val_bpb: 1.3151** (sliding window, 1xH100 80GB) | **~12.87 MB** | 4000 steps

## What this is

A depth-recurrent transformer that shares weight banks across selected layers and uses learned FiLM (Feature-wise Linear Modulation) vectors to differentiate how each shared layer processes its input. We keep the full PR #549 SOTA stack underneath and add two things on top: partial weight sharing and per-iteration conditioning.

The model has 9 physical transformer blocks but runs 11 virtual layers. Two blocks are each executed twice, with a small scale/shift vector applied between iterations so the model can learn different behavior at each pass. This saves 4.7M parameters (17% reduction) while slightly improving BPB in controlled ablations.

We also implemented a modified TTT protocol where only the FiLM vectors are updated at test time for shared blocks, avoiding the gradient compounding problem that breaks standard TTT with weight sharing. This crashed on a one-line bug before we could get results. The fix is trivial but we ran out of GPU credits.

## Why this architecture

I'm a medical student, so I think about this through the lens of neuroanatomy.

The neocortex uses the same six-layer cortical column circuit roughly 150,000 times. Visual cortex, motor cortex, prefrontal cortex: same wiring, different function. What makes them different is not the circuitry but the neuromodulatory environment. Acetylcholine, dopamine, noradrenaline, and serotonin change how neurons respond to inputs without changing the connections between them. A cortical column in V1 receiving cholinergic input from the basal forebrain processes visual features differently than the same canonical circuit in prefrontal cortex receiving dopaminergic input from VTA.

The analogy here is direct:

- Shared transformer blocks are cortical columns. Same weights, reused at different depths.
- FiLM conditioning vectors are neuromodulatory signals. Small per-layer scale and shift parameters that modulate the block's output without changing its weights.
- FiLM-only TTT is analogous to adjusting neuromodulatory tone rather than rewiring synapses. You adapt quickly by changing the chemical environment, not by restructuring circuits.

This is not just a metaphor. The parameter efficiency argument matches too. The brain cannot afford 150,000 unique circuits, and a parameter-golfing competition cannot afford 11 unique transformer blocks when some of that parameter budget could be better spent elsewhere. The question is whether you can get away with reuse plus modulation, and the answer appears to be yes.

## The TTT problem with shared weights

Test-time training updates model weights via gradient descent on already-scored validation chunks. When those weights are shared across multiple forward passes (depth recurrence), a single gradient step effectively applies the same update at every layer that shares those weights. The gradients compound in a way they were never meant to.

The loveless2001 submission ran into this and documented it. Their conclusion was that TTT and recurrence are incompatible.

Our proposed fix: do not update the shared weights at test time. Instead, update only the FiLM conditioning vectors, which are unique per virtual layer and do not compound. For unique (non-shared) blocks, standard full-weight TTT proceeds as normal. This gives you the best of both worlds: recurrence for parameter efficiency and TTT for test-time adaptation, without the conflict.

We implemented this but hit a Python-level bug (`if p not in ttt_params` does element-wise tensor comparison instead of identity comparison; the fix is `if id(p) not in ttt_param_ids`). The crash happened after training and sliding window eval completed successfully but before TTT could run.

## Architecture details

Everything from PR #549 is preserved. The modifications are:

| Component | PR #549 | This submission |
|-----------|---------|-----------------|
| Physical blocks | 11 | 9 |
| Virtual layers | 11 | 11 |
| Weight sharing | None | Blocks 3-4 share, 9-10 share |
| FiLM conditioning | None | 4 scale/shift pairs on shared layers |
| Parameters | 26.9M | 22.2M |
| TTT target | All weights | FiLM only for shared blocks |

### Layer layout

```
Virtual layer:  0  1  2  3  4  5  6  7  8  9  10
Physical block: 0  1  2  3  3  4  5  6  7  8   8
FiLM applied:   -  -  -  Y  Y  -  -  -  -  Y   Y
```

Physical blocks 3 and 8 each run twice. After each execution of a shared block, a learned affine transform is applied: `x = scale_i * x + shift_i`. Scales start at 1.0, shifts at 0.0, so at initialization the two iterations of a shared block are identical. The model learns to differentiate them during training.

FiLM parameters go to the Adam/scalar optimizer, not Muon. They are tiny (4 x 512 x 2 = 4096 parameters total).

### What we kept from PR #549

LeakyReLU(0.5)^2, XSA on last 4 virtual layers, BigramHash(1536), EMA(0.997) + SWA(every 50), Partial RoPE (16/64 dims), LN Scale, VE128 on layers 9-10, int6 QAT + GPTQ-lite + lzma, sliding window eval (stride=64), Parameter Banking + Parallel Muon, U-Net skip connections.

## Results

### Ablation at 500 iterations (1x RTX 4090)

All three runs used identical hyperparameters and seeds. Only the architecture changed.

| Config | Params | val_bpb | Delta vs baseline |
|--------|--------|---------|-------------------|
| SOTA baseline (PR #549) | 26.9M | 1.7075 | -- |
| **Recurrence + FiLM** | **22.2M** | **1.6864** | **-0.0211** |
| FiLM only, no recurrence | 26.9M | 1.7446 | +0.0371 |

The recurrence is doing the work, not FiLM by itself. Adding FiLM to all 11 layers of the unmodified architecture actually hurts. The conditioning only helps when it has a job to do, namely telling apart two iterations of the same block. When every layer already has unique weights, FiLM just adds optimization noise.

### Full training (1x H100 80GB, 4000 iterations)

| Metric | Value |
|--------|-------|
| Raw val_bpb (step 4000) | 1.3410 |
| EMA val_bpb | 1.3134 |
| Int6 quantized val_bpb | 1.3371 |
| **Sliding window val_bpb** | **1.3151** |
| Artifact size | 12.87 MB |
| Step time | 792 ms/step |
| Total training time | 53 min |

TTT was not completed (bug described above). Based on the SOTA's TTT gains of -0.0025, we would expect roughly 1.312-1.313 with the fix applied.

### FiLM-only TTT (not completed)

The design:
- Unique blocks: full-weight TTT as in PR #549
- Shared blocks: freeze all block parameters, update only FiLM scale/shift
- SGD with momentum 0.9, lr 0.002, cosine decay, 3 epochs per chunk

Crashed at the parameter selection stage. One-line fix identified. Rerun pending.

## What I learned

The recurrence result is clean. 17% fewer parameters, better BPB, and the ablation rules out FiLM as the source of improvement. The parameter sharing itself is what matters.

The negative FiLM-only result (Exp 3) was the surprise. I expected adding per-layer modulation to be neutral at worst. Instead it degraded performance by 0.037 BPB. My best guess is that the extra degrees of freedom in the scale/shift vectors interfere with the Muon optimizer's ability to find good bank weight updates, since the FiLM params are on a separate optimizer (Adam) with different dynamics.

The thing I most wanted to test and could not: whether FiLM-only TTT actually resolves the recurrence/TTT conflict. The SOTA gets -0.0025 from TTT. If FiLM-only TTT on shared blocks recovers even half that, it would validate the core idea. The bug is genuinely trivial to fix, I just ran out of credits at the worst possible moment.

Things worth trying that I did not get to:
- Different recurrence patterns. We shared blocks at positions 3-4 and 9-10, which was a heuristic choice. Sharing the middle blocks (positions 4-6) might work better since they process more abstract features.
- MLP-only sharing. Attention weights might need to be unique while MLP weights are more interchangeable.
- Reinvesting freed parameters. We saved 4.7M parameters but did not use them. A wider model (model_dim=576?) or deeper model (13 virtual layers from 9 physical) could be better.

## Limitations

- 1x H100, not 8x H100 SXM. Step time and total training time are not comparable to leaderboard submissions.
- 4000 iterations vs the ~7200 that the SOTA achieves in 10 minutes on 8xH100.
- No TTT results due to the bug + credit exhaustion.
- Recurrence pattern was chosen by intuition, not searched.
- The 1.3151 sliding window BPB would differ on 8xH100 with full TTT.

## Run command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
RECUR_ENABLED=1 FILM_ENABLED=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=4000 MAX_WALLCLOCK_SECONDS=0 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## References

- Dehghani et al. (2019). Universal Transformers. ICLR.
- Perez et al. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI.
- Doya, K. (2002). Metalearning and neuromodulation. Neural Networks, 15(4-6), 495-506.
- PR #549: LeakyReLU^2 + Legal TTT + Parallel Muon (@abaybektursun)
- PR #461: TTT recipe (@Christopher-Lee-McClendon)
- PR #399: Parameter Banking + Parallel Muon (@abaybektursun)
- PR #414: Base model stack (@signalrush)
- PR #493: LeakyReLU^2 activation (@parinzee)

## Credits

Built on top of PR #549 by @abaybektursun, which integrates work from PRs #399, #414, #461, and #493. The novel parts are the depth recurrence pattern, FiLM conditioning, and FiLM-only TTT design. By Nir Mathur (@nirmathur), medical student at King's College London.
