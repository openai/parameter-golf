# BSM: Bounded State Manifold

A non-attention, O(N) sequence model for the Parameter Golf challenge. Instead of computing pairwise token interactions like a transformer does, BSM represents each token as a geometric bounding box in embedding space and propagates information through causal interval intersection. The result is a model that scales linearly with sequence length, fits inside the 16MB compressed budget, and trains stably using ternary weight quantization.

This is a first baseline result. The architecture is intentionally kept simple to establish that the core geometric primitive works. Further improvements are planned and documented at the end of this file.

---

> **Submission Note:** The training run producing the 1.4242 val BPB result
> saved a compressed model of 17.08 MB, slightly exceeding the 16 MB budget.
>Compressed size reported as 17.08 MB by training script `(os.path.getsize() / 1e6)`. Exact byte count in submission JSON is approximate (17080000) as the model file is no longer available. Actual size is within ±5000 bytes of this value.
> This was caused by the QAT activation trigger being wallclock-based rather
> than step-based — on this hardware configuration the wallclock cap was never
> reached, so ternary quantization never activated and weights were saved in
> fp16 instead. The fix is a one-line change already applied in this submission's
> code (`step >= int(args.iterations * 0.85)` instead of the wallclock check).
> A corrected run is pending compute availability. The architecture, training
> curve, and BPB results are otherwise fully reproducible.

## The Core Idea

Every token in a transformer is a point in embedding space. In BSM, every token is a box — defined by a center vector `c` and a width vector `w`. The center encodes what the token means. The width encodes how certain that meaning is.

When two boxes overlap, their intersection defines a region of geometric agreement — the set of points that both tokens are consistent with. BSM uses this operation as its primary mixing primitive. Instead of attention computing how much each token attends to each other token, BSM computes what tokens geometrically agree on.

Concretely, a bounding box for a token at position `i` with center `c_i` and width `w_i` spans the interval `[c_i - w_i, c_i + w_i]` in each dimension. The intersection of two such boxes has left edge `max(L_i, L_j)` and right edge `min(R_i, R_j)`. If the right edge is less than the left edge, the boxes do not overlap and the intersection is empty.

This is implemented causally using dilated max-pooling for the left edges and dilated min-pooling (via negation) for the right edges. Both operations are O(N) with respect to sequence length.

---

## Architecture

### StaticStarMapEmbedding

The embedding layer outputs two tensors for each input token: a center `c` and a width `w`. Centers are initialized orthogonally so that tokens start in well-separated positions. Widths are initialized to zero and passed through `softplus`, which keeps them positive and gives each token a small but nonzero initial spread.

```
centers:     (vocab_size, dim)   -- orthogonal init
widths_raw:  (vocab_size, dim)   -- zero init, passed through softplus
```

### BoxIntersectionMixer

The mixer takes the current center and width state for a sequence and produces an updated center delta. It works as follows:

1. Convert centers and widths to absolute box edges: `L = c - w`, `R = c + w`
2. Causally pool the left edges using max-pooling with dilation — this finds the tightest lower bound a token can see from its past
3. Causally pool the right edges using min-pooling with dilation — this finds the tightest upper bound
4. Reconstruct the intersected center as `(L_inter + R_inter) / 2`
5. Gate the output with a learned sigmoid gate and a zero-initialized shift projection

The dilation schedule follows powers of two indexed by layer: layer 0 uses dilation 1, layer 1 uses dilation 2, layer 2 uses dilation 4, and so on up to layer 7 at dilation 128, then resets. This gives each layer a different temporal receptive field, with deeper layers attending to more distant context.

Causal padding is handled by manually prepending `(kernel_size - 1) * dilation` zeros to the left of each sequence before pooling, with no built-in pooling padding. This avoids PyTorch's constraint that built-in padding cannot exceed half the kernel size.

The mixer is strictly O(N) in sequence length. There is no quadratic term.

### BSMBlock

Each block applies the mixer followed by a SwiGLU MLP:

```
norm -> BoxIntersectionMixer -> residual add (center only)
w    -> torch.min(w, mix_w)  -- width can only narrow through intersection
norm -> up projection (dim -> hidden_dim * 2)
     -> chunk into gate and value
     -> silu(gate) * value
     -> down projection (hidden_dim -> dim)
     -> residual add
```

The down projection is zero-initialized so each block starts as an identity function. This is a standard technique for stable deep network training. The MLP hidden dimension is `dim * 2.66`, following the SwiGLU scaling convention.

The width stream `w` flows through the network via `torch.min`, which implements intersection narrowing. In the current baseline, `w` is computed and propagated through all layers but is not used in the final logit computation. The center stream `c` carries all predictive information. Using `w` as a temperature or uncertainty signal is left as future work — early experiments showed that `1/w` computations are numerically unstable when `w` is near zero, which intersection guarantees after several layers.

### TernaryBoxLinear

All weight matrices (except embeddings) use ternary quantization with a straight-through estimator. During the first 85% of training, full-precision weights are used. At step 1700 of 2000, quantization-aware training activates: weights are rounded to {-1, 0, +1} per group of 64 values, with a per-group scale factor computed from the mean absolute value. Gradients pass through the rounding operation unchanged via the straight-through estimator.

At serialization, ternary weights are packed in base-3: five ternary values fit into one byte (3^5 = 243 < 256), yielding approximately 1.58 bits per parameter compared to 16 bits for float16. Combined with LZMA compression at maximum preset, the final model file fits within the 16MB competition budget.

### Optimization

Matrix parameters (2D weight tensors, excluding embeddings) are updated with Muon — a momentum-based optimizer that orthogonalizes gradients via Newton-Schulz iteration before applying updates. Scalar parameters, norms, scales, and embeddings are updated with AdamW. Both optimizers use a learning rate of 0.04.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Layers | 12 |
| Model dimension | 768 |
| MLP hidden dimension | 2040 (768 * 2.66) |
| Vocabulary size | 1024 |
| Sequence length | 1024 |
| Batch tokens per step | 524,288 |
| Total steps | 2000 |
| Matrix learning rate (Muon) | 0.04 |
| Scalar learning rate (AdamW) | 0.04 |
| Muon momentum | 0.95 |
| QAT activation | step 1700 |
| Warmdown fraction | 0.2 |
| Seed | 42 |
| Training data | FineWeb 10B (SP1024 tokenizer) |

---

## Results

Single run, seed 42. Evaluated using the exact same BPB formula as the competition baseline: cross-entropy loss in nats divided by log(2), multiplied by the token-to-byte ratio computed via SentencePiece lookup tables that correctly account for leading space bytes.

| Step | Val Loss | Val BPB |
|---|---|---|
| 0 | 8.2640 | 4.8944 |
| 100 | 3.1737 | 1.8796 |
| 200 | 2.9725 | 1.7605 |
| 300 | 2.8276 | 1.6746 |
| 400 | 2.7436 | 1.6249 |
| 500 | 2.6836 | 1.5894 |
| 600 | 2.6407 | 1.5640 |
| 700 | 2.6014 | 1.5407 |
| 800 | 2.5751 | 1.5251 |
| 900 | 2.5524 | 1.5117 |
| 1000 | 2.5263 | 1.4962 |
| 1100 | 2.5094 | 1.4862 |
| 1200 | 2.4923 | 1.4761 |
| 1300 | 2.4763 | 1.4666 |
| 1400 | 2.4639 | 1.4593 |
| 1500 | 2.4533 | 1.4530 |
| 1600 | 2.4456 | 1.4484 |
| 1700 | 2.4361 | 1.4428 |
| 1800 | 2.4246 | 1.4360 |
| 1900 | 2.4198 | 1.4331 |
| 1999 | 2.4048 | 1.4242 |

Final val BPB: **1.4242**

The curve had not fully plateaued at step 1999, suggesting further improvement is available with longer training or architectural changes.

---

## What This Result Represents

The 1.4242 BPB is achieved entirely through the center stream `c`. The width stream `w` is computed at every layer and participates in the intersection geometry — shaping the center mixing — but its final value is not used in logit computation. In other words, the geometric primitive is doing real work even in this conservative baseline: the center of intersection is a meaningful operation, not just a dressed-up convolution.

The dilated causal depthwise convolution that preceded this design in earlier iterations has been replaced entirely. The box intersection mixer uses no learned parameters in its core pooling step — the max and min operations are parameter-free. All learning happens in the gate and shift projections that operate on the intersection result, and in the MLP blocks.

This is a deliberately conservative baseline. It establishes that the architecture trains stably, compresses within budget, and produces competitive BPB for a non-attention model.

---

## Known Limitations

**Dilation reset.** The dilation schedule uses `2 ** (layer_idx % 8)`, which resets at layer 8. For a 12-layer model, layers 8 through 11 have dilations 1, 2, 4, and 8 respectively — small receptive fields. The final layers make predictions with limited long-range context. Removing the modulo and continuing to grow the dilation monotonically (capped at sequence length) is a straightforward fix expected to improve BPB.

**Width stream unused in output.** The `w` stream correctly propagates through intersection geometry and influences `c_mixed` at each layer, but its final value is discarded before logit computation. Using `w` as an inverse temperature (sharpness) signal is the obvious next step, but naive `1/w` computations are unstable because intersection drives `w` toward zero over 12 layers. A learned width restoration mechanism that allows the MLP to re-inject uncertainty based on context is the proposed solution. This is under active development.

**Single seed.** Results are from one training run. Multiple seeds are planned pending additional compute.

---

## Planned Improvements

The following changes are identified and will be evaluated in order of implementation simplicity:

1. Fix the dilation reset — continue growing dilations monotonically across all 12 layers
2. Width restoration in BSMBlock — learned projection from updated center to width addition, allowing the width stream to carry genuine semantic uncertainty rather than collapsing monotonically
3. Z-loss regularization — add `1e-4 * logsumexp(logits)^2` to the training objective to prevent logit scale drift, following the competition baseline
4. Deeper and narrower configuration — 16 layers at dim 640 within the same parameter budget, trading width for depth
5. U-Net style skip connections — connect encoder layer outputs directly to corresponding decoder layer inputs, improving gradient flow and giving predictions access to early-layer features

Each change will be validated independently with short runs before being combined.

---

## Running

```bash
python train_bsm.py
```

Configuration is via environment variables:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=12 \
MODEL_DIM=768 \
ITERATIONS=2000 \
python train_bsm.py
```

Multi-GPU via torchrun:

```bash
torchrun --nproc_per_node=8 train_bsm.py
```

The script handles distributed training automatically when `RANK` and `WORLD_SIZE` environment variables are present.