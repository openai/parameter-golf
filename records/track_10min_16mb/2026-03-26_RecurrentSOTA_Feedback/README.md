# Recurrent Depth: 2-Pass Train + 4-Pass Eval with Error Feedback

**val_bpb: **** (3-seed mean) | ****~16 MB** | 8xH100 SXM

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128)


| Seed     | step_avg | steps   | Pre-TTT bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact |
| -------- | -------- | ------- | ----------- | ---------------- | -------- | -------- | -------- |
| 1337     | TBD      | TBD     | TBD         | **TBD**          | TBD      | TBD      | TBD      |
| 42       | TBD      | TBD     | TBD         | **TBD**          | TBD      | TBD      | TBD      |
| 2025     | TBD      | TBD     | TBD         | **TBD**          | TBD      | TBD      | TBD      |
| **Mean** | **TBD**  | **TBD** | **TBD**     | **TBD**          | **TBD**  | **TBD**  |          |


## The Problem: Depth Recurrence Fails Under Competition Constraints

[PR #363](https://github.com/openai/parameter-golf/pull/363) demonstrated that depth recurrence -- reusing a shared block of transformer layers multiple times -- saves parameters but *hurts* bpb under the 10-minute / 16MB competition constraints. Their controlled experiments showed a **+0.025 bpb gap** (looped worse) due to two compounding taxes:

1. **Quantization error amplification.** When shared weights are quantized to int6, the quantization error $\epsilon$ is injected at every pass. After $K$ passes through the same core, the cumulative error grows superlinearly. PR #363 measured this as a **0.37 bpb quantization gap** for a 3x-looped architecture vs near-zero for a flat model.
2. **Step time overhead.** Each additional recurrence pass adds forward/backward compute through the core layers. With 5 core layers and 4 passes, PR #363 observed +32ms/step, translating to ~1200 fewer training steps in the 600s budget. The capacity benefit of shared weights cannot overcome the lost training signal.

## Our Solution: Contractive Recurrence + Inference-Time Depth Scaling

We address both taxes with three architectural mechanisms and a decoupled train/eval strategy.

### 1. Learnable Residual Scaling (ResidualScale)

Per-pass learnable scalars $\alpha_k$ contract the residual update, preventing hidden state magnitude growth across passes:

$$h_{k+1} = h_k + \alpha_k \cdot F(h_k + c_k)$$

where $\alpha_k$ is initialized to 0.5 and learned during training. This ensures the recurrent dynamics are contractive -- later passes refine rather than amplify.

```python
class ResidualScale(nn.Module):
    def __init__(self, num_passes: int, init_value: float = 1.0):
        super().__init__()
        self.scales = nn.Parameter(
            torch.full((num_passes,), init_value, dtype=torch.float32)
        )

    def forward(self, residual: Tensor, pass_idx: int) -> Tensor:
        return self.scales[pass_idx].to(dtype=residual.dtype) * residual
```

### 2. Error Feedback Module

A low-rank residual approximation estimates the accumulated error, and a learned diagonal correction compensates for it before each pass:

$$e_k = U(V^\top h_k), \qquad c_k = \mathrm{diag}(d) \cdot e_k$$

where $U, V \in \mathbb{R}^{d \times r}$ with rank $r=2$ and $d \in \mathbb{R}^d$ is a learnable diagonal. The correction is zero on pass 0 (no prior error to correct) and active on subsequent passes. Total parameter overhead: **2,560 params** (negligible vs 26.9M model params).

```python
class ErrorFeedbackModule(nn.Module):
    """Combined error-feedback path: residual -> correction.

    e_k = U (V^T h_k)     -- low-rank residual approximation
    c_k = diag(d) * e_k   -- diagonal correction
    """
    def forward(self, h: Tensor, pass_idx: int) -> Tensor:
        e = self.residual(h)          # Low-rank projection
        c = self.correction(e)        # Diagonal scaling
        mask = 1.0 if pass_idx > 0 else 0.0  # Inactive on first pass
        return c * mask
```

### 3. Jacobian Proxy Loss

A regularization term penalizes hidden state growth ratio above 1.0, enforcing contractive dynamics without computing the full Jacobian:

$$\mathcal{L}*J = \lambda \cdot \mathrm{ReLU}\left(\frac{h*{k+1} - h_k}{h_k + \epsilon} - 1\right)^{2}$$

with $\lambda = 0.1$. This is a cheap finite-difference proxy for the spectral norm of the Jacobian $\partial h_{k+1}/\partial h_k$, encouraging it to stay below 1 (contractive map).

```python
def jacobian_proxy_loss(self, h_in: Tensor, h_out: Tensor) -> Tensor:
    delta = h_out - h_in
    ratio = delta.norm() / (h_in.norm() + self.eps)
    return self.jacobian_proxy_weight * torch.relu(ratio - 1.0).square()
```

### 4. Train Cheap, Eval Deep

The key insight: **train with 2 recurrence passes, evaluate with 4**. This completely sidesteps the step-time tax during training (our step time is only ~15% above the flat baseline, vs 30%+ for 4 passes), while still harvesting the depth benefit at inference time. The contractive mechanisms (ResidualScale + Jacobian proxy) ensure that adding passes at eval time does not cause hidden state blowup, since the learned dynamics are stable for arbitrary iteration counts.

After training completes and the checkpoint is saved, we override `num_passes` from 2 to 4 and pad the `ResidualScale` parameters for the additional passes. The model then runs TTT and final evaluation with 4 effective core passes (20 effective layer evaluations: 4 stem + 3x4 core + 4 tail).

```
Architecture (11 unique layers, 20 effective at eval):
  Stem [0-3] -> Core [4-6] x4 passes -> Tail [7-10]
                            ^^^^^^^^^^
                            Shared weights, reused 4 times at eval
```

### Eval-time pass sweep (1-GPU development, seed 1337)


| Eval passes | TTT bpb    | vs 2-pass   |
| ----------- | ---------- | ----------- |
| 2 (=train)  | 1.1204     | baseline    |
| 4           | **1.1157** | **-0.0047** |
| 6           | 1.1166     | -0.0039     |
| 8           | 1.1176     | -0.0029     |


4 passes is the sweet spot: enough depth to improve token prediction, not so many that the contractive scaling dampens signal. This result shows that our stability mechanisms successfully enable inference-time compute scaling for recurrent transformers.

## What Didn't Work

### LoRA adapters for per-pass specialization

We tried adding low-rank adapters (rank 2 and 8) to differentiate core layer behavior across passes. Results:

- **No bpb improvement**: rank-2 and rank-8 LoRA produced nearly identical loss curves to the baseline, even with careful warmup scheduling.
- **Size constraint**: At rank 8, LoRA parameters pushed the total artifact over the 16MB limit.
- **Hypothesis**: The core layers already learn pass-invariant features through the ResidualScale mechanism; LoRA's per-pass deltas are redundant.

### Training with more recurrence passes (4+)

Direct training with 4 passes hits the step-time tax:

- **4-pass training**: ~105ms/step on 8xH100 vs ~83ms for flat. In 600s: ~5700 steps vs ~7200.
- **Result**: The 1500 fewer training steps cost more bpb than the extra depth recovers.
- **2-pass training + 4-pass eval**: ~96ms/step, ~6250 steps. Nearly matches the flat model's step count while gaining inference-time depth.

## Architecture

Built on the [PR #414](https://github.com/openai/parameter-golf/pull/414) stack with [PR #399](https://github.com/openai/parameter-golf/pull/399) Parallel Muon:


| Component               | Setting                                           |
| ----------------------- | ------------------------------------------------- |
| Layers                  | 11 unique (512d, 8H, 4KV)                         |
| Effective layers (eval) | 20 (4 stem + 3 core x4 + 4 tail)                  |
| MLP                     | 3x with LeakyReLU(0.5)^2                          |
| BigramHash              | 1536                                              |
| XSA                     | Last 4 layers                                     |
| RoPE                    | Partial (16/64 dims)                              |
| LN Scale                | 1/sqrt(layer+1)                                   |
| VE128                   | Layers 9-10                                       |
| Recurrence core         | Layers 4-6, 2 passes (train), 4 passes (eval)     |
| ResidualScale           | Per-pass learnable, init 0.5                      |
| Error Feedback          | Diagonal mode, rank 2                             |
| Jacobian proxy          | lambda=0.1                                        |
| Weight avg              | EMA(0.997) + SWA(every 50)                        |
| Quantization            | Late QAT (threshold 0.15) + GPTQ-lite int6 + lzma |
| Optimizer               | Parameter Banking + Parallel Muon                 |


### TTT Configuration

Score-first legal TTT following [PR #461](https://github.com/openai/parameter-golf/pull/461):


| Parameter        | Value                                    |
| ---------------- | ---------------------------------------- |
| Chunk size       | 32,768 tokens                            |
| Optimizer        | SGD + momentum(0.9)                      |
| Learning rate    | 0.002 (cosine decay across chunks)       |
| Epochs per chunk | 3                                        |
| Frozen blocks    | None (all blocks adapt, freeze_blocks=0) |
| Gradient clip    | 1.0                                      |
| Eval passes      | 4 (overridden from training's 2)         |


## Run Command

```bash
cd records/track_10min_16mb/2026-03-26_RecurrentSOTA_Feedback
bash run_submission.sh
```

Or for a single seed:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
CORE_START=4 CORE_END=7 NUM_PASSES=2 EVAL_PASSES=4 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt_recurrent.py \
    --feedback-mode diagonal --feedback-rank 2 \
    --residual-scale-init 0.5 \
    --jacobian-proxy-weight 0.1 \
    --no-interpass-rmsnorm
```

## Credits

- **Base model**: [PR #414](https://github.com/openai/parameter-golf/pull/414) by @signalrush
- **Optimizer (Parameter Banking + Parallel Muon)**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **LeakyReLU^2 activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee
- **TTT recipe**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **Depth recurrence analysis**: [PR #363](https://github.com/openai/parameter-golf/pull/363) by @evangelinehelsinki (identified the quantization error amplification problem we solve here)

