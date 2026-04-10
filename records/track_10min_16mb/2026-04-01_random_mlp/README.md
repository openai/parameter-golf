# Non-Record: Partially Random MLP

Not a record run, but a proof of concept that partially random MLP layers are competitive — and likely more so with a cleaner implementation stack than the one I had time to build.

**val_bpb: 1.1527**: (3-seed mean, std. 0.0021) | ~**15.95MB** under zlib | 8xH100 SXM, 600s | No TTT

One oversight worth calling out: only one of my three submissions uses zstd for compression. Switching from `zlib` to `zstd` is worth roughly ~1.49MB of artifact headroom - enough to fit one or two additional fully learned layers. The average compression figure above uses the larger zlib artifacts for fairness, but this is low-hanging fruit for anyone building on this.

## Results

| Seed     | Steps | ms/step | Pre-quant BPB | **Sliding BPB** |  Artifact  |
|--------- |-------|---------|---------------|-----------------|------------|
| 1337     | 4,962 | 120.93  | 1.1661        | **1.1516**      | 15,979,644 |
| 2026     | 4,973 | 120.66  | 1.1656        | **1.1508**      | 15,979,644 |
| 999      | 4,598 | 130.50  | 1.1687        | **1.1558**      | 14,454,547 |
| **Mean** |       |         |               | **1.1527**      |            |

---

## Core Idea
Any parameter that is computable at initialization time is effectively free — it costs nothing in the artifact budget. This observation is obvious in hindsight, but it took a detour through HRM and TRM experimentation to arrive at it clearly.\
While it's much harder to guess correct initialization values for the attention subsystem, the MLP blocks are much better understood in the sense, that we have at least some intuition as to what they're doing - some mix of content-addressable memory and feature extraction.\
Since the up-projection of the MLP is what determines the features to be detected, there's less of an argument that it needs to be fully learned from scratch.
Any well-chosen random basis should work as a fixed feature extractor and leaving down-projection and surrounding machinery to do the adaptation.
The resulting construction is simple: at initialization, selected MLP up-projections are sampled from a random matrix and frozen. Only the seed is stored - the weights are recomputed at load time and never saved. Each frozen layer additionally gets a learnable per-feature gain vector (initialized to all ones), which gives the model a cheap learned scaling on top of the fixed basis.
The weight saving ends up anecdotally being around ~0.7MB per random layer, depending on training progress, and a working `zstd` stack may be able to squeeze in one or two more fully trainable layers.
The freed parameter budget was reinvested in model depth, landing at 12 layers total: 5 random, 7 learned.

Note that this does not reduce compute — it trades parameter storage for additional depth under a fixed budget.


### Initializing the random projections
I experimented with several initialization schemes, scaled normal, Rademacher, and QR.
QR won consistently on my local 3090 iso-step ablations, and it's simple to construct: sample a random matrix using a fixed seed (or generator when doing this over multiple layers), compute the QR-decomposition, scale the resulting Q by `sqrt(d_in)` and use it as the up-projection.
My intuition for this is structural - QR yields an almost guaranteed orthogonal basis (okay, in all but pathological cases, and then you could still use rejection sampling to guarantee it) meaning the random features are well behaved - maximally space-covering without any redundancy. The right prior for a feature extractor you lock in at init and never update. And still having a learned down-projection means that mixing between features is fully learnable.

One interesting observation from local ablations on my 3090 under iso-steps: In early training, between roughly steps 400–1000 (bsz=16k, after initial loss settling), models with random MLP layers *temporarily outperform* fully learned models. My interpretation is that the frozen projections provide a stable feature basis that the rest of the model can organize around quickly - learned layers then route through this fixed scaffold rather than having to discover a useful basis from scratch. The advantage narrows as learned layers catch up and eventually both variants end up with quite similar loss-curves, but it doesn't necessarily hurt.

### Additional constructions around random layers
During early experimentation, without QR-init, the thought came up that potentially the randomly initialized matrices were rank-deficient, or ill-conditioned in the sense that some learned features may be co-linear. To address this problem, I added what I call a "mini-MoE" construction, where multiple random up-projections are performed and the model learns a token-dependent router that adjusts their relative weights. This construction did improve performance (even with QR-init), but I removed it during my final H100 runs because under an iso-wallclock setting they didn't help. If someone can reduce the throughput-cost, this remains a viable direction.

## How I got here: HRM/TRM
Before arriving at random reservoirs, I spent time experimenting with [HRM](https://arxiv.org/abs/2506.21734) and [TRM](https://arxiv.org/abs/2510.04871) inspired looped/repeated layer constructions, motivated by interest in what happens when you push the number of inner settling steps large enough that the model stops doing fixed-point iteration and starts exploring the latent space more freely.

**TLDR**: they didn't work well for this challenge, and I don't think that's surprising in retrospect. Looped constructions seem to favor settings where algorithmic, per-token "reasoning" effort pays off.
The canonical example for the HRM paper is something like Sudoku, where you can fit the entire problem space into the model's latent dimension and iterative refinement of a pre-existing solution works out. General language modeling however does not fit well into this structure.

The underlying issue here, and why I wanted to experiment with decoupled settling steps in the first place, may be gradient flow.
In traditional looped language model constructions, even outside of HRM/TRM, e.g. [Ouro](https://arxiv.org/abs/2510.25741), [MobileLLM](https://arxiv.org/abs/2402.14905) or the original [Universal transformers](https://arxiv.org/abs/1807.03819) gradients flow through the repeated layers and there's optimization pressure that pushes intermediate solutions directionally towards the final solution.
I believe this yields convergent, and in the case of HRMs explicitly fixed-point iteration-like behavior with steadily decaying residuals - to be fair, I don't know if this aspect applies to looped language models as well, but it does to HRMs.
Pushing the non-gradiented steps may be a way of forcing the model into a compositional instead of iterative regime, resulting in actual latent-space exploration, beyond the pressures of intermediate steps having to directionally align with the intended target.
I didn't have the compute budget to test this seriously, and beyond that I'm still somewhat sceptical if this is really the way to go about this - you do need to allow intermediate layers more freedom, but having this arise from repetition requires "traces of composability" to be hidden within the model that get amplified by training. And then again those traces have to be strong enough that gradient descent can lock onto and amplify them.

Cross-attention coupling (as an alternative to elementwise-additive mixing in HRM/TRM) did improve performance over additive coupling on my local 3090 - but showed no clear advantage over a dense baseline at the scales relevant to this competition. The experiments were useful for ruling out that direction quickly.

---

## Other components
Beyond the random MLP architecture, I ported a number of tricks from [PR #414](https://github.com/openai/parameter-golf/pull/414) the current SOTA submission at the time of my experimentation. In short:

- MLP 3x
- efficient XSA on the last 4 layers, tested and won against XSA-2 and XSA-3
- partial RoPE
- LN Scale
- VE 128 on the last 3 layers, no significant differences seen vs. last 4
- BigramHash(2048) + Smeargate
- int6 QAT with STE
- EMA + late SWA

---

## TTT

Legal test-time training was enabled but did not improve post-quantization performance. This appears to be a known issue - the new SOTA submission at the time of writing, [PR #1019](https://github.com/openai/parameter-golf/pull/1019) mentions similar behavior.
Although it's unclear if they saw the same catastrophic results as I did, in my experimentation loss increased substantially and while it did show signs of decreasing it still ends up being higher than at TTT start (bpb jumped from 1.186077 to 1.428367).
There's still time budget remaining in eval, so this is likely fixable for someone willing to debug it carefully. If it's useful to someone, I can share the final non- and quantized models for further experimentation.

## Run Command

```bash
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=786432 \
TRAIN_SEQ_LEN=2048 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=786432 \
WARMDOWN_ITERS=3500 \
NUM_LAYERS=12 \
TRAIN_LOG_EVERY=10 \
MLP_MULT=3 \
RAND_PROJ_LAYERS="0,1,2,3,4" \
RAND_GAIN=1 \
RAND_INIT_QR=1 \
MINI_MOE_EXPERTS=1 \
VE_LAYERS="9,10,11" \
VE_DIM=128 \
XSA_LAYERS="8,9,10,11" \
BIGRAM_VOCAB_SIZE=2048 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

An explanation on unusual hyperparameters:
- RAND_PROJ_LAYERS configures which layers use the random up-projection
- RAND_GAIN - enables (1) or disables the learnable per-feature weighting after projection
- RAND_INIT_QR - enables/disables the QR-init, falls back to normal init otherwise
- MINI_MOE_EXPERTS - the number of up-projections to generate per random MLP at 1 it removes the router and expert-gating altogether


## Future directions
The most natural extension of this idea is going all in on the `Learning adapters on random linear maps` angle this idea falls into.
My current bet is that random up- **and** down-projections may work for some layers if you stack the feature weighting and potentially some LoRA style adapters ontop of them - cheap, potentially expressive, early feature detectors that have little to no learned parameters.
A potential constraint that makes this interesting - you can theoretically compute the up- and down-projections to be pseudo-inverses, then learn diagonal scaling (the current random-gain, per-feature weighting) and a low-rank correction ontop.


Aside from this, there's TTT debugging and further exploration of the mini-MoE idea.
If anyone wants to debug the TTT further, I can share the trained model checkpoints. That should make it possible to isolate whether the failure is in the quantization interaction, the random layer gradient issue, or something else entirely.

Since I likely won't have the time to run more experiments (and running experiments on 8xH100s is quite expensive), feel free to expand and build off of the ideas here!
