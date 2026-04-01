# Recurrent Depth with Progressive Pass Growth + Error Feedback

**val_bpb: 1.1163** (3-seed mean, std 0.0013) | **~15.96 MB** | 8×H100 SXM

A non-record submission targeting significant improvement over [PR #549](https://github.com/openai/parameter-golf/pull/549) (LeakyReLU² baseline, 1.1194 mean bpb). Achieves **-0.0031 bpb** vs that baseline. For an in-depth analysis of depth recurrence in this competition, see [PR #363](https://github.com/openai/parameter-golf/pull/363). I targeted 549 when I started building this solution, after I finished evaluation the new improved model has been published to the leaderboard. However I believe the experiments here can be applied to any model to improve performance, with the largest benefit for submissions using TTT since the recurrance makes use of the 10 available minutes of evaluation time very effectively. 

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)


| Seed     | step_avg   | steps     | Pre-TTT bpb | **Post-TTT bpb**        | TTT gain    | TTT time  | Artifact   |
| -------- | ---------- | --------- | ----------- | ----------------------- | ----------- | --------- | ---------- |
| 1337     | 83.5ms     | 6,328     | 1.1353      | **1.1157**              | -0.0196     | 566s      | 15,909,018 |
| 42       | 83.5ms     | 6,334     | 1.1372      | **1.1177**              | -0.0195     | 579s      | 15,897,530 |
| 2025     | 83.4ms     | 6,334     | 1.1351      | **1.1155**              | -0.0197     | 588s      | 15,995,558 |
| **Mean** | **83.5ms** | **6,332** | **1.1359**  | **1.1163 (std 0.0013)** | **-0.0196** | **~578s** |            |

We significantly beat the [PR #549](https://github.com/openai/parameter-golf/pull/549) LeakyReLU² baseline (1.1194 mean bpb) by **-0.0031 bpb** across all three seeds, achieving the goal we set out with.

## Progressive Recurrence Architecture

```
   ┌───────────┐               ┌───────────┐                 ┌───────────┐
   │           │               │           │                 │           │
   │   Tail    │               │   Tail    │                 │   Tail    │
   │  [7-10]   │               │  [7-10]   │                 │  [7-10]   │
   │           │               │           │                 │           │
   ├───────────┤               ├───────────┤╮                ├───────────┤╮
   │           │  4500 steps   │           ││   1000 steps   │           ││
   │   Core    │ ───────────>  │   Core    ││  ──────────>   │   Core    ││
   │  [4-6]    │               │  [4-6]    │2x               │  [4-6]    │3x
   │           │               │           ││                │           ││
   ├───────────┤               ├───────────┤╯                ├───────────┤╯
   │           │               │           │                 │           │
   │   Stem    │               │   Stem    │                 │   Stem    │
   │  [0-3]    │               │  [0-3]    │                 │  [0-3]    │
   │           │               │           │                 │           │
   └───────────┘               └───────────┘                 └───────────┘

   11 layers                   14 layers                     17 layers
   (steps 0-4499)              (steps 4500-5499)             (steps 5500+, eval)
```

## The Problem: Depth Recurrence Fails Under Competition Constraints

[PR #363](https://github.com/openai/parameter-golf/pull/363) demonstrated that depth recurrence — reusing a shared block of transformer layers multiple times — saves parameters but *hurts* bpb under the 10-minute / 16MB competition constraints. Their controlled experiments showed a **+0.025 bpb gap** (looped worse) due to two compounding taxes:

1. **Quantization error amplification.** When shared weights are quantized to int6, the quantization error is injected at every pass. After K passes through the same core, the cumulative error grows superlinearly. Additionally hidden state magnitudes tend to explode with to many recurrent passes through a block if we do not stabilize this. 
2. **Step time overhead.** Each additional recurrence pass adds forward/backward compute. With 4 passes, +32ms/step translates to ~1200 fewer training steps in the 600s budget.

## Our Solution: Late Growth + Contractive Stabilization

We address both taxes by growing recurrence depth progressively during training and stabilizing the recurrent dynamics.

### Progressive Pass Schedule (Late Growth)

The key insight: **start training with 1 pass and gradually add passes late in training**. This preserves fast step times for the majority of training (83.5ms/step at 1-pass vs ~95ms at 3-pass), maximizing the total number of gradient updates within the 600s wallclock budget. The schedule:


| Step range | Passes | Effective layers | step_avg |
| ---------- | ------ | ---------------- | -------- |
| 0–4499     | 1      | 11               | ~83.5ms  |
| 4500–5499  | 2      | 14               | ~85.5ms  |
| 5500–6328  | 3      | 17               | ~91ms    |


This reduces the step/capacity trade-off that normally makes recurrence impractical under competition constraints. We get ~6,330 training steps (vs ~7,180 for the flat LeakyReLU baseline), but the final model has 17 effective layers at eval vs the baseline's 11.

We also tested training with 4 recurrence passes. While 4-pass shows better per-step loss, the additional step time cost (~105ms/step) means fewer total steps within the wallclock budget. Under the competition's 600s constraint, **3-pass wins the step/capacity trade-off**, the extra training steps from the faster 3-pass schedule outweigh the marginal per-step quality gain from 4 passes.

### Learnable Residual Scaling

Per-pass learnable scalars contract the residual update, preventing hidden state magnitude growth across passes:

$$h_{k+1} = h_k + \alpha_k \cdot F(h_k + c_k)$$

where $\alpha_k$ is initialized to 0.5 and learned during training. This ensures the recurrent dynamics are contractive — later passes refine rather than amplify.

### Error Feedback Module

A low-rank correction compensates for accumulated error before each recurrence pass:

$$e_k = U(V^\top h_k), \qquad c_k = \mathrm{diag}(d) \cdot e_k$$

where $U, V \in \mathbb{R}^{d \times r}$ with rank $r=2$ and $d \in \mathbb{R}^d$ is a learnable diagonal. The correction is zero on pass 0 (no prior error to correct) and active on subsequent passes. Total parameter overhead: **2,560 params** (negligible vs 26.7M model params).

The feedback module is important but not strictly required — we confirmed that stable training is possible without it, and even running eval-only without feedback works, at a cost of ~0.001 bpb higher. The feedback module's main contribution is providing the recurrent passes with an error signal about the previous iteration's residual.

### Jacobian Proxy Loss (Stabilizer)

A regularization term penalizes hidden state growth ratio above 1.0, enforcing contractive dynamics without computing the full Jacobian:

$$\mathcal{L}*J = \lambda \cdot \mathrm{ReLU}\left(\frac{h*{k+1} - h_k}{h_k + \epsilon} - 1\right)^{2}$$

with $\lambda = 0.01$. This is a cheap finite-difference proxy for the spectral norm of the Jacobian $\partial h_{k+1}/\partial h_k$, encouraging it to stay below 1 (contractive map). The model learns to adhere to this quickly and it does not seem to effect early training dynamics. However we did see better results with 0.01 compared to 0.1 for Lambda, potentially since the restriction of 0.1 is to high, we don't always need contractive layers with only 3x recurrance, but we do need it to not explode. 

This loss term is critical for training stability. **Without it, gradient norms and hidden state magnitudes explode** during the multi-pass phases, destabilizing training. The proxy loss keeps the recurrent dynamics well-behaved without the computational cost of full Jacobian computation.

Note: the jacobian proxy loss is only added to the training loss — it does not affect evaluation scoring, which uses pure cross-entropy.

## Legal TTT Protocol

Score-first legal TTT following [PR #461](https://github.com/openai/parameter-golf/pull/461):

1. Val tokens split into 1,893 non-overlapping 32K-token chunks. Here 3 pass recurrance is vital since with 4 passes we must increase chunk size to fit within the time limit. 
2. **For each chunk**:
  - **SCORE**: Sliding window eval under `torch.inference_mode()` — no gradients, no weight mutation
  - **TRAIN**: SGD on the already-scored chunk. 3 epochs, all blocks unfrozen, cosine LR decay, grad clip 1.0
3. Last chunk scored but never trained on


| Parameter        | Value                             |
| ---------------- | --------------------------------- |
| Chunk size       | 32,768 tokens                     |
| Optimizer        | SGD + momentum(0.9)               |
| Learning rate    | 0.002 (cosine decay)              |
| Epochs per chunk | 3                                 |
| Frozen blocks    | None (all blocks adapt)           |
| Gradient clip    | 1.0                               |
| Eval passes      | 3 (matching final training phase) |


### Timing Budget


| Phase                                 | Time                 |
| ------------------------------------- | -------------------- |
| Training (wallclock cap)              | 600s (10 min)        |
| Standard eval (int6 + sliding window) | ~3s                  |
| Legal TTT (score-first + adaptation)  | ~578s                |
| **Total eval**                        | **~581s (< 10 min)** |


## Architecture

Built on the [PR #414](https://github.com/openai/parameter-golf/pull/414) stack with [PR #399](https://github.com/openai/parameter-golf/pull/399) Parallel Muon:


| Component               | Setting                                                     |
| ----------------------- | ----------------------------------------------------------- |
| Layers                  | 11 unique (512d, 8H, 4KV)                                   |
| Effective layers (eval) | 17 (4 stem + 3 core ×3 + 4 tail)                            |
| MLP                     | 3× with LeakyReLU(0.5)²                                     |
| BigramHash              | 512                                                         |
| XSA                     | Last 4 layers                                               |
| RoPE                    | Partial (16/64 dims)                                        |
| LN Scale                | 1/√(layer+1)                                                |
| VE128                   | Layers 9-10                                                 |
| Recurrence core         | Layers 4-6, progressive 1→2→3 passes                        |
| ResidualScale           | Per-pass learnable, init 0.5                                |
| Error Feedback          | Diagonal mode, rank 2, 2560 params                          |
| Jacobian proxy          | λ=0.01                                                      |
| Weight avg              | EMA(0.997) + SWA(every 50)                                  |
| Quantization            | Late QAT (threshold 0.15) + GPTQ-lite int6 + lzma           |
| Warmup precompilation   | All pass×QAT graph variants compiled during 20 warmup steps |
| Optimizer               | Parameter Banking + Parallel Muon                           |


## Run Command

```bash
cd records/track_non_record_16mb/2026-03-26_Stable_Growing_Recurrance
bash run_earlyqat.sh  # Single seed (set SEED env var)
```

Key flags:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py \
    --feedback-mode diagonal --feedback-rank 2 \
    --residual-scale-init 0.5 \
    --jacobian-proxy-weight 0.01 \
    --no-interpass-rmsnorm
```

## Tricks

### Graph Precompilation Warmup

`torch.compile` is lazy — it only compiles a new graph variant the first time it's encountered. With progressive recurrence (1→2→3 passes) and late QAT, this means the training loop would hit compilation stalls at step 4500 (2-pass), step 5500 (3-pass), and again when QAT enables. Under a 600s wallclock cap, these stalls are expensive.

The fix: **precompile all graph variants during warmup before training starts**. During the 20 warmup steps:

1. The last few warmup steps cycle through each `num_passes` variant (2-pass, 3-pass) and each with QAT toggled on
2. This forces `torch.compile` to eagerly compile every forward/backward graph that will appear during training
3. After warmup, model weights and optimizer states are restored to their initial values — the warmup steps have zero effect on the actual training run

This ensures the training loop runs at full speed from step 0 with no compilation jitter when passes change or QAT kicks in.

### Code Minification with python-minifier

The original training script was 88,253 bytes, which caused seed 2025 to exceed the 16MB submission limit (16,025,625 bytes). After removing dead code paths (eval-only mode, int8 quantization, unused feedback variants, verbose logging), the file was still too large.

[python-minifier](https://github.com/dflook/python-minifier) with `--no-rename-locals` shrinks the code aggressively (whitespace, docstrings, constant folding) while preserving local variable names — critical because the training script uses string-based lookups for `state_dict` keys and `named_parameters`. This brought the file from 68,435 bytes down to **58,186 bytes**, comfortably fitting all seeds under the 16MB decimal limit.

**Note:** The code was minified *after* all three seed runs completed, so the log files report `Code size: 88253 bytes` and correspondingly larger `Total submission size` values. The actual submission uses the minified 58,186-byte script — the correct per-seed totals are listed in `submission.json` and the results table above.

## Credits

- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **LeakyReLU² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Depth recurrence analysis**: [PR #363](https://github.com/openai/parameter-golf/pull/363) by @evangelinehelsinki

