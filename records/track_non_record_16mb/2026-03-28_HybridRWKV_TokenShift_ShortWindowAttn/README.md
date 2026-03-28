# Hybrid RWKV Token-Shift + Short Window Attention

**Author:** Dillon Blake
**val_bpb:** 1.2252 (3-seed mean) | **Artifact size:** ~15.86 MB | **Hardware:** 8x H100 SXM | **Params:** 17.0M

## Results

| Seed | Steps | val_loss | val_bpb | Artifact Size |
|------|-------|----------|---------|---------------|
| 0    | 14,683 | 2.0689  | 1.2253  | 15,859,562 B  |
| 1337 | 14,716 | 2.0682  | 1.2249  | 15,863,546 B  |
| 42   | 14,736 | 2.0691  | 1.2254  | 15,862,343 B  |
| **Mean** | | **2.0687** | **1.2252** | |

## Architecture

- **Layers:** 11 (3 attention + 8 token-shift)
- **Model dim:** 512
- **Heads:** 8 (4 KV heads, GQA)
- **MLP:** 3x expansion, LeakyReLU squared
- **Sequence length:** 1024
- **Vocab:** 1024 (SentencePiece BPE)
- **Quantization:** Int6 + zlib compression

## Approach

Recently, I have become increasingly interested in hybrid transformer architectures. With this, I wanted to apply some of what I have learned and experimented with to parameter golf. Some of my favorite techniques for hybrid architectures are mamba and gated delta nets. In my early experiments I tried replacing the vast majority of attention layers with these more efficient alternatives. However, at this scale (especially when keeping sequence lengths short) I found flash attention 3 to be much faster. However, I did observe that loss per training step was lower with the hybrid architectures. This makes sense as research suggests that many layers actually just focus on more local context. I believe that with more training time, a classic transformer with softmax attention would have learned this, but the inductive bias of knowing that only some layers needed long range dependencies allowed the model to skip the step in training where it learns not to attend to much earlier tokens in the context for most layers. However, due to the speed of flash attention 3, the increased steps in training and more data seen were more important.

With these observations, my next step was to explore how I could make the mechanisms behind the improved per step loss faster in order to match or exceed the number of steps completed with flash attention 3. A notable trial was differential attention. I theorized that the mechanism that improved the loss per step for hybrid models was that models need to learn to forget. I believed that differential attention would allow me to emulate the lossy memory of hybrid layers while still using the well optimized flash attention 3 on the hopper architecture. This produced mixed results. I tried a mixture of 1-1 differential attention, 1 differential head for every 3 normal heads, 1 for every four normal heads, and even isolating differential attention to certain layers, but it did not compellingly replicate the improved loss per step. I think at a larger scale this could be interesting potentially on scenarios like having models admit when they don’t know, but I did not find it helpful at this small scale. 

After this, I tried to narrow down the fastest potential ways to apply local mixing with extreme efficiency. I ended up with two candidates that seemed to work well. The first method was simply having attention with short windows. I found that leaving only a few layers using full context attention with the rest using very short (128 token) windows worked very well and sped up training. This makes sense, as methods like Mamba hybrid models use an approximately 1:9 ratio of attention to Mamba layers, and the short attention can be thought of as an approximation of the Mamba layer. However, I believe the more interesting finding was actually the benefits of using RWKV style local mixing. Using all RWKV 6 layers with the flash linear attention library demonstrated the need for attention, with its slower learning. However, more notably it greatly slowed down training even with a sequence length of 8192. 
RWKV has two components to each layer: token mixing and channel mixing. Inspired by some of the other methods like SmearGate, I chose to try adapting just the token mixing part to replace attention. To do this, I use learned per-dimension interpolation weights (passed through sigmoid) to create a weighted blend of the current token and the previous token. Two separate interpolation vectors are learned: one for the receptance (R) branch and one for the key (K) branch. The blended K representation is projected and passed through squared ReLU, then projected again to produce a value. The R branch produces a sigmoid gate that controls how much of this value passes through to the residual stream. Because it only looked one token back, this operation was extremely fast. I tested looking 3 and 5 tokens back as well, but looking 1 back was found to be optimal. In addition, as an ablation I removed the FFN after the token mixing and found the mlp does indeed still provide important transformations before the next layer. 
Although this did not produce a record, I believe this token mixing approach continues to show the strengths of hybrid models. Although decode speed was not part of this competition, I believe this architecture would be very fast and require significantly less overhead. I did a short, definitely not thorough, sweep of a couple configurations of classic transformer layers vs token mixing layers and I ended up landing on having most layers use this simple token mixing method inspired from RWKV, with only 3 of 11 layers retaining quadratic attention (spaced across the middle and back at positions 4, 7, and 10). For the attention layers, I used short window attention (128 tokens) for layers 4 and 7, while the last attention layer (layer 10) kept full context.

I would like to continue to explore these hybrid architectures because I think there is a lot of room for growth still. I will submit a request for the medium tier of Runpod credits so hopefully I can continue this work. Although most of this work focused on ultra cheap hybrid architectures, I did have a few other minor findings:
1. Removing the Unet style connections between layers improved performance
2. Longer context helped, but only to a point. I found the sweet spot to be around 4096-8192 tokens in earlier experiments, though the final submission used a sequence length of 1024.

In addition to the hybrid architecture, the submission incorporates a number of other techniques that contributed to the final result:
- **Bigram hash embedding**: A hashed bigram embedding that captures local token-pair context and is added to the token embeddings.
- **SmearGate**: After the bigram embedding is added and RMS norm is applied, a learned gate blends each token's representation with the previous token's before the first layer.
- **Value embedding**: A shared value embedding table projected into KV-head space, applied with a learned per-layer scale to attention layer 10. (Layer 9, a token-shift layer, also receives the value embedding but its TokenShiftMix ignores it.)
- **XSA (cross-head suppression of attention)**: Enabled on attention layers within the last 4 layer indices (i.e., layers 7 and 10, since layer 4 falls outside this range), this projects the attention output away from the value direction to encourage heads to learn diverse representations.
- **Partial RoPE**: Only 16 out of 64 head dimensions receive rotary position embeddings, leaving the rest position-agnostic.
- **LeakyReLU squared activation**: The MLP uses LeakyReLU (slope 0.5) followed by squaring, rather than the more standard SwiGLU.
- **Muon optimizer**: A momentum-based optimizer using Newton-Schulz orthogonalization for matrix-shaped parameters, paired with Adam (with weight decay) for scalar and embedding parameters.
- **Int6 quantization with zlib compression**: Weights are quantized to int6 range (stored as int8) with per-row scales and zlib compressed to meet the 16MB size constraint.
- **EMA with late QAT**: Exponential moving average of weights with quantization-aware training applied late in training.
- **Logit softcapping**: Output logits are soft-capped at 30.0 to stabilize training.

## Run Command

```bash
SEED=1337 \
TRAIN_SEQ_LEN=1024 \
TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
EMBED_LR=0.05 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Thanks for putting on this competition! I hope to continue some more experiments and share more if I am able to get some more Runpod credits.


