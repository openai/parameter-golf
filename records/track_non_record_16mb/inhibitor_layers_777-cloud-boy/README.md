Non-Record: Inhibitory Layers on PR #1851 Stack
val_bpb = 1.06438 (single seed, post-TTT) | 15,996,198 bytes | 8×H100 SXM 80GB
Summary
This is a non-record submission that adds a novel architectural primitive — inhibitory layers — on top of the PR #1851 stack by @aquariouseworkman, as well as PR #1855 @codemath3000
The contribution is a small, low-rank gating mechanism applied to attention and MLP residual paths, providing the transformer with a subtractive primitive it otherwise lacks. The mechanism is biologically motivated: cortical circuits are ~20% inhibitory neurons, and the fly mushroom body uses a single inhibitory interneuron (APL) to enforce sparsity. Modern transformers have no native subtractive operation — every layer writes additively to the residual stream, and "removing" a feature requires a downstream layer to learn an equal-and-opposite contribution. We add a minimal inhibitory primitive and ablate it against the PR #1851 baseline.
This submission does not claim a new SOTA. Single-seed post-TTT val_bpb is 1.06438, ~0.0033 above the current SOTA's 3-seed mean of 1.06108. Based on the reference 3-seed standard deviation of 0.00068 and an observed seed spread of 0.00133 across the existing reproduction, a 3-seed mean of this configuration would plausibly land within touching distance of SOTA. We offer the inhibitory primitive as a small, easy-to-port architectural addition that may compose with other improvements.
Mechanism
The inhibitory layer is a tiny data-dependent gate added to transformer blocks. It has two low-rank MLPs: one gates the attention residual path, one gates the MLP residual path.
Each gate maps d_model → rank → d_model, then applies a sigmoid. The output is a per-channel multiplier on the existing residual scale:
attn_out = attn_scale * inhibitor_attn(attn_normed) * attn(...)
mlp_out  = mlp_scale  * inhibitor_mlp(mlp_normed)   * mlp(...)
Initialization is critical. The inhibitor is initialized so the sigmoid output is ~0.95 (5% suppression at init) rather than near-identity. Earlier configurations using ~0.98 init (2% suppression) produced gates with weak gradient signal — the bias was so far into sigmoid saturation that the inhibitor weights barely updated during training. 5% init breaks the gradient-dead regime without destabilizing early training.
Configuration:

inhibitor_rank: 22
Init: final projection = 0, bias chosen so sigmoid output ≈ 0.95
Inhibitor weights serialized as row-int8 (gate_int8_row) to keep artifact under 16MB
Applied per block (attention + MLP residual paths)

Results
Stageval_bpbPre-quantization, post-EMA1.06830Quantized1.07734Quantized + post-TTT phased1.06438
MetricValueTrain time (training-data-access)592.1sHessian collection3.5sTotal training-data-access595.6s (< 600s ✓)Artifact size (quantized + brotli)15,996,198 bytes (< 16MB ✓)TTT eval time795.6s
Architecture
Inherits the PR #1851 stack: 11L × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence: layers 3–5 looped ×2 (activated at frac=0.35). Parallel residuals from layer 8. XSA on all 11 layers. SmearGate window=12.
Reproduction
bashSEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
INHIBITOR_ENABLED=1 \
INHIBITOR_RANK=22 \
SPELLINGBEE_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
Hardware: 8×H100 SXM 80GB (RunPod)
Limitations

Single seed; no statistical significance testing performed against the SOTA baseline
Did not run full ablation isolating inhibitor contribution from SpellingBee removal
5% init was selected over 2% empirically; intermediate values not swept

Credits

@aquariouseworkman — PR #1851 (base stack this builds on)
@nprime06 — PR #1787 (base architecture)
@romeerp — PR #1729 (CaseOps)
@dexhunter — PR #1797 (SmearGate + LQER asymmetric quantization)
@cocohearts — BOS document boundary bug identification
@abaybektursun — PR #549 (score-first TTT)
@clarkkev — PR #1394 (GPTQ + SP8192)
@Christopher-Lee-McClendon — PR #1855 (3-seed compliance reproduction)
@cloud-777-boy - Lafayette Compton — Inhibitory layers (this submission)