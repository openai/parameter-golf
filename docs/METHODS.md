# Methods

A walk through each technique used in my two submissions. Ordered by how much it moved the final number, not alphabetically.

## Muon and Newton Schulz

The base optimizer for both my submissions. Muon in one sentence: instead of SGD or Adam, apply an orthogonalization operation to the gradient, getting a step that preserves the spectral properties of the weight matrix.

The core operation is a fifth-order Newton Schulz iteration. Input: gradient matrix G. Output: matrix X approximating the orthogonal projection `G (G^T G)^(-1/2)`. The upstream code reads:

```python
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X
```

Five iterations is a trade-off between orthogonalization quality and time. In my runs, five iterations gave spectral quality around 1e-3, each iteration adding roughly 2 ms per step.

Why this helps. Gradients in transformers are strongly anisotropic. Some directions carry 100x the weight of others. Adam compensates via per-parameter normalization, but loses matrix structure. Muon works on the full matrix. After orthogonalization, all directions update proportionally.

In my first submission I used basic Muon with momentum=0.99 and weight decay=0.04. Momentum warmup from 0.85 to 0.99 over 1500 steps removes instability early, where orthogonalization is poorly defined on small gradients.

In the second submission I used Turbo-Muon, a variant with parallel communication. The idea: instead of DDP all-reduce, do reduce-scatter of local gradients by weight banks, apply Newton Schulz locally to your bank, then all-gather the results. This saves communication and lets NS=4 work as well as NS=5 would.

Muon is the work of Keller Jordan from the modded-nanogpt project (https://github.com/KellerJordan/modded-nanogpt). In Parameter Golf it generally gives faster convergence than AdamW on the same configuration thanks to spectrally-balanced updates. Quantifying the gain in my setup is hard because I didn't keep controlled Muon vs AdamW runs on the same seed.

## EngramLite and BigramHash

Simply put: give the model cheap access to information about which pairs of tokens appeared recently, without spending attention compute on it.

Implementation. Keep a hash table of 10240 buckets (in my second submission). At each step through the sequence, for each pair (t-1, t) compute hash(t-1, t) and pull a 48-dim vector from the table. These vectors get added to regular token embeddings.

```python
class BigramHashEmbedding(nn.Module):
    def __init__(self, vocab_size, buckets, dim):
        super().__init__()
        self.table = nn.Embedding(buckets, dim)
        self.buckets = buckets
    def forward(self, tokens: Tensor):
        prev = F.pad(tokens[:, :-1], (1, 0))
        idx = (tokens * 31 + prev) % self.buckets
        return self.table(idx)
```

In the April submission this is called EngramLite, an extended version on two orders (bigram + trigram) with two separate heads.

Why it works. For a small model with vocab=1024 and dim=512, direct learning of bigram statistics needs a matrix of `1024 × 1024 × dim` weights, which is excessive. A hash table gives a compressed representation. Hash collisions get compensated by other parameters during training.

Gain in my measurements: roughly 0.01 to 0.015 bpb. Small absolutely, but in a regime where 0.001 bpb is a leaderboard position, it matters.

## SmearGate

A gate that slightly blurs (smears) information between neighboring tokens.

Formula:

```python
class SmearGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim))
    def forward(self, x: Tensor):
        shifted = F.pad(x[:, :-1], (0, 0, 1, 0))
        alpha = torch.sigmoid(self.gate)
        return x * (1 - alpha) + shifted * alpha
```

A learned scalar (per dim) parameter that mixes the current token with the previous one. Early in training the gate is near zero, meaning no mixing. As training progresses the model learns which dimensions are worth blurring.

The effect is a form of very cheap depthwise convolutional window of size 2 with no bias and no non-linearity. It saves attention compute on tasks that don't need long-range connections.

Gain: 0.005 to 0.01 bpb. Very cheap in parameters, 512 weights for the whole model.

## U-Net skip connections

Regular residual connections operate within a transformer block. U-Net adds connections across several blocks: the output of layer 3 gets skipped into layer 8 with a learned scalar sigmoid gate.

```python
# pseudocode
x0 = embed(tokens)
x1 = block1(x0); x2 = block2(x1); x3 = block3(x2); x4 = block4(x3)
x5 = block5(x4) + sigmoid(alpha_5) * x4
x6 = block6(x5) + sigmoid(alpha_6) * x3
x7 = block7(x6) + sigmoid(alpha_7) * x2
x8 = block8(x7) + sigmoid(alpha_8) * x1
x9 = block9(x8) + sigmoid(alpha_9) * x0
```

Idea from the original U-Net in computer vision, where encoder and decoder mirror each other and skip connections pass low-level information. In a transformer the analogy holds because early layers compute low-level patterns (token bigrams, frequencies), late layers try to use them for prediction. Direct skips let the residual stream avoid spending bandwidth carrying this information.

Gain in my runs: 0.01 to 0.02 bpb.

## ReLU² MLP vs SwiGLU

Short answer: in Parameter Golf, ReLU² wins.

Long answer. SwiGLU is `SwiGLU(x) = (Wx * sigmoid(Wx)) * Vx`, needing three matrix multiplications (W_gate, W_up, W_down) with an activation between them. ReLU² is `ReLU²(x) = max(0, x)²`, needing only two matmuls (W_up, W_down).

When you have exactly 600 seconds, a third matmul per step translates to about 30% slowdown. Over 11 layers × 6000 steps that's a 2-minute gap.

In my first submission (March) I used ReLU² with hidden=1536. In the second (April) it's LeakyReLU² with per-layer slopes (Adaptive Squared Units, ASQU v3 from PR #1089), 3.5x expansion. LeakyReLU² keeps gradient signal on negative activations, stabilizing training late.

My failed `020_ultimate` used SwiGLU. That was a mistake. Each step took 86 ms instead of 54 ms with ReLU². Over 600 seconds I got 6601 steps instead of 11070, a direct loss of 4469 training steps.

Takeaway. SwiGLU is a great choice for 70B+ models. For 16 MB in 10 minutes it's a luxury you can't afford.

## Partial RoPE

Rotary Positional Embedding applies not to the full head dimension, but only to the first K dims of head_dim. In my runs K=16 out of 64 (a quarter).

Why. RoPE encodes position via rotations in 2D planes. For short sequences (up to 2048) a small number of planes is enough. The remaining dims work as regular attention channels with no positional information.

Gain: computational (fewer sin/cos ops per step) and performance (the model doesn't overload early dims with positional info, can use them for content patterns).

In my runs this wasn't a big bpb win (0.003 to 0.005), but it saved 2 to 3 ms per step, which compounded over 6000 steps adds up.

## XSA (Exclusive Self-Attention)

A subtle tweak to standard attention. The math:

Standard self-attention: `softmax(QK^T / sqrt(d)) V`.

XSA: computes the same thing, but subtracts the self-value component from the output. Attention doesn't let a token give itself a value projection, only cross-token.

```python
def xsa_adjustment(attn_out, v_self_projection):
    return attn_out - v_self_projection
```

The idea: force the model to use information from other tokens explicitly, don't do a trivial identity projection.

In my April submission XSA sits on all 11 layers. This is unusually aggressive, most work applies XSA only on the last few layers. In my test, all-layers XSA gave better results on short sequences (seq_len=2048), but I didn't verify this is stable for longer contexts.

Gain: about 0.01 bpb.

## ValueEmbedding

An extra embedding applied at the value projection of attention, not at the input.

```python
class ValueEmbedding(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.ve = nn.Embedding(vocab, dim)
    def forward(self, tokens, v_proj):
        return v_proj + self.ve(tokens)
```

Intent: give the model a direct way to recover token identity at each layer, not relying on the residual stream. Useful on late layers where the residual stream is already loaded with contextual information.

In my April submission VE sits on layers 8, 9, 10. Widening it from [9, 10] (the PR #1089 default) to [8, 9, 10] gave about 0.003 bpb.

## SWA vs EMA

Stochastic Weight Averaging vs Exponential Moving Average. Both techniques average model weights across training history, but they do it differently.

EMA: `W_ema = decay * W_ema + (1 - decay) * W_current`, updated every step, newer weights dominate (with decay=0.999 the last 1000 steps carry most of the weight).

SWA: save snapshots every N steps in the tail of training, average them uniformly at the end.

For Parameter Golf, SWA works better. The reason is quantization. EMA does a smooth transition between weights, the final values are a weighted sum biased toward the late steps. Late steps are warmdown with small LR, and the gradients there are no longer exploring the loss surface.

SWA captures instantaneous snapshots from different warmdown stages and averages them uniformly. You get a flatter optimum in weight space. Flat optima handle quantization better (weights quantization is effectively adding noise to weights; a flat optimum is resilient to noise).

In my first submission: SWA every 50 steps from 50% of training, ending with 115 snapshots. In the second: SWA every 50 steps after warmdown threshold, 18 snapshots plus EMA with decay=0.997.

Effect of SWA on the quantization gap in my runs: down from 0.08 without SWA to 0.02 with SWA. This is the most important technique in my set.

## STE Quantization Aware Training

Straight-Through Estimator is a way to make quantization differentiable.

Problem: quantization is a step function, zero gradient almost everywhere. If you apply quantization on the forward pass during training, backward can't update weights.

STE solution: apply round on forward, pretend the op wasn't there on backward.

```python
class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, n_levels):
        q = torch.round(x / scale).clamp(-n_levels, n_levels) * scale
        return q
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None, None
```

In my first submission STE was on from step 0 (int6 for all weights from the very first step). In the second, Late QAT: regular training up to a threshold step, then smooth STE activation via a sigmoid ramp (threshold 0.15 of total steps).

Late QAT works better on short runs. The model explores in fp32 space for the first 70% of training, then learns to live with quantization in the last 30%. This is analogous to curriculum learning: easy task first, hard task later.

## GPTQ with Hessian and Cholesky

For the April submission, the main quantization is GPTQ, not STE.

GPTQ is a one-shot calibration: take the trained fp32 model, collect calibration activations (256 batches in my case), compute Hessian matrices `H = X^T X` where X is the activations, quantize weights one at a time with error compensation through Cholesky decomposition.

```python
# pseudocode, simplified
H = collect_hessian(model, calib_data)
for W in weights:
    L = cholesky(H + damping * I)
    for column in W.columns:
        quant_column = quantize(column)
        error = column - quant_column
        # error compensation: distribute error to remaining columns
        remaining_columns -= error @ L_inverse_column
```

At our scale GPTQ takes about 9 seconds out of the 600-second budget. In exchange it lets us compress all weights to int5 with a loss of less than 0.02 bpb.

Implementation adapted from PR #1089 and follows the original GPTQ paper (Frantar et al., 2022, https://arxiv.org/abs/2210.17323).

## Mixed Precision int5 / int6 / int8

Different weight types have different sensitivity to quantization. Embeddings are more sensitive than attention weights, attention is more sensitive than MLP.

My scheme:
- Embeddings: int8
- Attention weights (Q, K, V, out): int6
- MLP weights: int5 in the April submission, int6 in the March one

April: 66 weight groups, all int5 after GPTQ, none promoted to int6. This means the bitrate is so tight that selective pruning (20.5% of ±1, ±2 values) is needed to fit the budget.

The bitrate choice is made automatically at calibration. If even int5 quantization doesn't fit 16 MB, selective pruning kicks in: values with the smallest gradient contribution (±1 and ±2 on the quantized scale, the smallest absolute values) get zeroed.

## Compression: zstd-22 and Brotli-11

After quantization you have an array of int values. It needs another pass of compression to save bytes.

zstd level 22 (max) is what I used for the first submission. It compresses an int array roughly 3.8x relative to packed bytes. Compression is slow (22 is hard-compression mode), but we only do it once at the end.

Brotli-11 (also max level) with byte-shuffle preprocessing is used in the second submission. Byte-shuffle is a regrouping of an int16/int32 array: all high bytes first, then all low bytes. This makes the sequence more uniform, Brotli compresses it 3 to 5% better than without shuffle.

Both approaches give roughly the same result. Brotli is slightly better on this data type, zstd is faster.

## Sliding Window Evaluation

Not a training technique, but an important evaluation trick.

Standard val_bpb evaluation walks the sequence in blocks of 1024 tokens: first block tokens 0 to 1023, second block 1024 to 2047, etc. The model predicts tokens independently for each block.

Problem: the first token of each block has 0 context tokens, the second has 1, and so on. Average context 512 tokens.

Sliding window with stride=64: blocks overlap, each shifted by 64 tokens from the previous. We only score the last 64 tokens of each block (they have 960 tokens of context). Average context close to 960.

Gain: 0.02 to 0.03 bpb. Pure eval trick, no training changes.

Technique from PR #50 upstream. I didn't invent it, I applied it honestly after reading it.

## Logit softcap

```python
logits = soft_cap * torch.tanh(logits / soft_cap)
```

With soft_cap=30 the logits are bounded in magnitude by 30. This prevents numerical issues under mixed precision (bf16 handles very large logits poorly).

The bpb effect is minimal, but it noticeably stabilizes training in late stages. Without softcap I occasionally saw divergence in the last 100 steps.

## LN Scale 1/sqrt(layer+1)

The residual stream in a transformer grows layer by layer roughly as sqrt(layer). Without normalization, late layers see inputs with much larger norms than early ones.

LN Scale multiplies the input of each RMSNorm by 1/sqrt(layer+1), compensating for this accumulation.

The gain is not absolute, but it stabilizes training and allows a higher learning rate.

## What I didn't use

For honesty: a list of popular techniques I tried and rejected, or just never got around to.

**SwiGLU MLP.** Tried in `020_ultimate`, turned out too slow for the 600-second budget. The third matmul eats 30% of step time.

**Mamba / State Space Models.** Interesting architecture for Parameter Golf in theory. Didn't try because of the setup complexity and my unfamiliarity with flash_linear_attention.

**Mixture of Experts.** At a 16 MB budget MoE makes no sense, the expert weights all go into the budget anyway.

**Knowledge distillation from a large model.** Prohibited by contest rules.

**Test-Time Training (TTT).** Tried in `v5_ttt_killer` locally, didn't get to a working run. The code has a LoRA scaffold but it never gave stable results for me.

**Depth recurrence.** Tried (experiment 010 in the non-record track), gave +0.05 bpb over baseline. Doubling depth via weight reuse halves the number of training steps in the same wall-clock time, which outweighs any capacity gain.

## Sources

Core works I lean on:

- modded-nanogpt (Muon, speedrun): https://github.com/KellerJordan/modded-nanogpt
- GPTQ (Frantar et al., 2022): https://arxiv.org/abs/2210.17323
- SWA original (Izmailov et al., 2018): https://arxiv.org/abs/1803.05407
- RoPE (Su et al., 2021): https://arxiv.org/abs/2104.09864
- Straight-Through Estimator (Bengio et al., 2013): https://arxiv.org/abs/1308.3432
- FineWeb dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Parameter Golf challenge: https://github.com/openai/parameter-golf

Links to specific upstream PRs I used as a base or drew inspiration from:

- PR #1089, Turbo-Muon stack (base of my April submission)
- PR #50, Sliding window eval
- PR #162, Int6 + SmearGate + BigramHash
- PR #1477, 3x3090 adaptation
- PR #1564, GDN hybrid architecture (didn't use, but studied)

Locally all these PR versions sit in `parameter-golf-scripts/`, if you want them.
