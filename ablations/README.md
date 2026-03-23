# Ablation Experiments

5 ablations on top of the SOTA record (11L EMA + GPTQ-lite, val_bpb=1.1233).
Each file is a self-contained copy of the baseline with a targeted modification.

## Baseline

```bash
# Run the unmodified SOTA as control
torchrun --nproc_per_node=8 baseline.py
```

## Ablation 1: SwiGLU MLP

**File:** `ablation1_swiglu.py`
**Hypothesis:** SwiGLU (silu-gated linear unit, used in LLaMA/Gemma) outperforms relu-squared at same param count.

**What changes:** Replaces `relu(fc(x))^2` MLP with `silu(gate(x)) * up(x)`. Uses 2/3 hidden dim to keep param count equal (hidden=1024 vs baseline 1536, but 3 projections vs 2).

**Why it might help:** SwiGLU gives each neuron an independent learned gate, providing more expressive nonlinearity per parameter. Every top-performing LLM uses SwiGLU over relu-squared.

**Why it might hurt:** relu-squared has stronger sparsity (many exact zeros), which may help quantization. The 3-matrix structure has slightly more overhead.

```bash
torchrun --nproc_per_node=8 ablation1_swiglu.py
```

No new env vars needed.

---

## Ablation 2: Sliding Window Attention (Early Layers)

**File:** `ablation2_sliding_window.py`
**Hypothesis:** Early layers only need local context; restricting them to a window saves FLOPs that translate to more training steps.

**What changes:** First N layers use sliding-window attention (window_size tokens left + right). Deep layers retain full causal attention. Uses FlashAttention 3's native `window_size` parameter (zero overhead).

**Why it might help:** Encoder layers (0-4) primarily learn local patterns (syntax, n-grams). Full O(n^2) attention there is wasteful. Window attention is faster, so we get more training steps in 600s.

**Why it might hurt:** Some early layers may benefit from global context for topic-level features. The U-Net skip connections may partially mitigate this.

```bash
# Default: 256-token window on first 5 layers
torchrun --nproc_per_node=8 ablation2_sliding_window.py

# Tune window size and number of layers
SW_WINDOW_SIZE=128 SW_NUM_LAYERS=3 torchrun --nproc_per_node=8 ablation2_sliding_window.py
SW_WINDOW_SIZE=512 SW_NUM_LAYERS=7 torchrun --nproc_per_node=8 ablation2_sliding_window.py

# Disable (equivalent to baseline)
SW_WINDOW_SIZE=0 torchrun --nproc_per_node=8 ablation2_sliding_window.py
```

---

## Ablation 3: Register/Sink Tokens

**File:** `ablation3_register_tokens.py`
**Hypothesis:** Learnable register tokens prepended to every sequence act as attention sinks and global memory banks, improving attention quality.

**What changes:** Adds N learnable vectors (shape [1, N, 512]) prepended to every sequence before the transformer layers. All real tokens can attend to these registers. Stripped before logit computation.

**Why it might help:** From "Vision Transformers Need Registers" (ICLR 2024) — register tokens absorb the "attention sink" phenomenon (where tokens dump attention on position 0 when they have nothing relevant to attend to). This frees up real token positions to carry more useful information. Also provides a global memory that every token can read from.

**Why it might hurt:** Increases effective sequence length by N tokens (slightly more FLOPS), and the register embeddings use ~N*512*4 = ~8KB of params. May not matter at seq_len=2048.

```bash
# Default: 4 register tokens
torchrun --nproc_per_node=8 ablation3_register_tokens.py

# Tune
NUM_REGISTERS=2 torchrun --nproc_per_node=8 ablation3_register_tokens.py
NUM_REGISTERS=8 torchrun --nproc_per_node=8 ablation3_register_tokens.py

# Disable
NUM_REGISTERS=0 torchrun --nproc_per_node=8 ablation3_register_tokens.py
```

---

## Ablation 4: Gated Value Normalization

**File:** `ablation4_head_temperature.py`
**Hypothesis:** Adding RMS normalization to value vectors (with a learned gate) constrains output scale, improving training stability and quantization-friendliness.

**What changes:** After computing V, applies a learnable gate per KV-head that interpolates between raw V and RMS-normed V: `v = gate * rms_norm(v) + (1-gate) * v`. Gate initialized at 0 (sigmoid(0)=0.5, so starts at 50/50 mix).

**Why it might help:** Q and K are already normalized, but V isn't. Normalizing V constrains the attention output scale, reducing the dynamic range of activations flowing through the network. This should: (1) improve quantization (smaller weight ranges = less quantization error), (2) stabilize training in deep layers, (3) reduce the need for careful lr tuning.

**Why it might hurt:** Normalizing V discards magnitude information that may carry useful signal. The gate should learn to preserve it where needed, but adds a small overhead.

```bash
torchrun --nproc_per_node=8 ablation4_head_temperature.py
```

No new env vars needed. Only adds 4 learnable scalars (one per KV head).

---

## Ablation 5: Mixture of Softmax (MoS)

**File:** `ablation5_mixture_of_softmax.py`
**Hypothesis:** A single softmax is a rank-bottleneck (rank <= model_dim=512 for vocab=1024). Mixing K softmaxes breaks this bottleneck.

**What changes:** Instead of `p = softmax(W @ h)`, computes `p = sum_k pi_k * softmax(W @ tanh(proj_k(h)))` where pi_k are input-dependent mixing weights. The projection creates K diverse "views" of the hidden state, each producing a different probability distribution.

**Why it might help:** Yang et al. 2018 "Breaking the Softmax Bottleneck" shows that a single softmax can't represent the true data distribution when vocab > model_dim. With vocab=1024 and dim=512, we're right at the bottleneck. K=3 experts increase the effective rank to ~1536.

**Why it might hurt:** The mos_proj adds K * dim * dim parameters (~786K for K=3). This is significant in a 16MB budget. The extra params may not pay for themselves if the bottleneck isn't binding. Also, `torch.compile` may struggle with the MoS computation graph.

```bash
# Default: 3 experts
torchrun --nproc_per_node=8 ablation5_mixture_of_softmax.py

# Tune number of experts
MOS_NUM_EXPERTS=2 torchrun --nproc_per_node=8 ablation5_mixture_of_softmax.py
MOS_NUM_EXPERTS=4 torchrun --nproc_per_node=8 ablation5_mixture_of_softmax.py

# Disable (equivalent to baseline)
MOS_NUM_EXPERTS=1 torchrun --nproc_per_node=8 ablation5_mixture_of_softmax.py
```

---

## Expected Priority

Based on analysis of what's likely to help most:

| # | Ablation | Expected Impact | Risk |
|---|----------|----------------|------|
| 1 | SwiGLU | Medium (+) | Low — well-validated in literature |
| 2 | Sliding Window | Medium (+/-) | Medium — depends on FLOP savings vs info loss |
| 3 | Register Tokens | Small-Medium (+) | Low — very cheap, few params |
| 4 | Gated V-Norm | Small (+) | Low — minimal params, may help quant gap |
| 5 | MoS | High (+/-) | High — large param cost, but directly targets BPB |

**Recommended run order:** 1, 4, 3, 2, 5 (cheapest/safest first, MoS last due to param cost concerns)
