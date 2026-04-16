# WIP: Mixture of Depths + Adaptive Computation Time (ACT)

**Status: In progress — compute grant pending. Results TBD.**

Target: sub-1.076 val_bpb | 8xH100 SXM | ≤16MB artifact

---

## Approach

This submission combines two ideas that are architecturally underexplored in the current leaderboard:

### 1. Mixture of Depths (MoD) Token Routing
Rather than applying uniform depth to all tokens, a lightweight per-layer router (single linear projection) decides which tokens participate in each transformer block. Easy tokens (punctuation, common n-grams, whitespace) are routed around layers; hard tokens (rare words, syntactically complex positions) receive full depth.

This is fundamentally different from the current depth recurrence submissions (PR #1394, #1437), which loop fixed layer pairs uniformly over all tokens. MoD is *input-conditional* — the model learns which tokens need more compute.

Parameter cost: ~1 linear layer per routed layer (~512 params each), negligible against the 16MB budget.

### 2. Adaptive Computation Time (ACT) with Learned Halting
Inspired by Graves (2016) and PonderNet (Banino et al., 2021), each token accumulates a halting probability across recurrence steps. The model learns to halt early on easy tokens and continue on hard ones. Unlike fixed-loop recurrence, the halting threshold is a learned scalar, optimized jointly with the LM objective via a ponder cost regularizer.

The combination: MoD handles *layer-wise* routing (which layers to apply), ACT handles *recurrence-wise* routing (how many passes to take). Together they form a 2D adaptive compute budget over the token sequence.

---

## Planned Architecture

Building on the SP8192 + GPTQ + parallel residuals stack (PR #1394, #1412):

- **Base**: 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64), tied embeddings
- **MoD routing**: Top-k token selection per layer (k = 75% of sequence), learned router, straight-through gradient
- **ACT halting**: Sigmoid halting unit per recurrence step, ponder cost λ=0.01, max steps=3
- **Quantization**: SP8192 vocabulary, int6 GPTQ (SDClip), int8 embeddings, Brotli-11 compression
- **Optimizer**: MuonEq-R + AdamW for embeddings/scalars

---

## Hypothesis

The current SOTA spends the same compute on every token regardless of difficulty. Natural language is highly non-uniform — most tokens in FineWeb are predictable given context. MoD+ACT should improve bits-per-byte by concentrating the fixed parameter budget on the positions that actually carry information, while spending less on trivial continuations.

Expected gain over the current depth recurrence baseline: 0.005–0.015 bpb, based on MoD results in Raposo et al. (2024) scaled to this parameter regime.

---

## Background

- Solo ML researcher / founder (VectaBind — AI drug discovery platform)
- Background in adaptive computation: implemented PonderNet/ACT-style architecture for multi-game reasoning (Othello, MiniGrid, drone navigation) with learned halting over latent recurrence steps
- Built SE(3)-equivariant EGNN + ESM2-3B cross-attention model trained on multi-source dataset (BindingDB 3M, PDBBind 2020, CrossDocked2020, TDC, GDSC, CTRP, DrugComb) — experience with efficient training on constrained compute (single NVIDIA L4)
- GitHub: ChipGlitch

---

## Progress

- [x] Reviewed full leaderboard and SOTA stack
- [x] Cloned and ran baseline locally
- [x] Designed MoD router architecture
- [x] Designed ACT halting mechanism
- [ ] Local smoke test on L4 (in progress)
- [ ] H100 validation runs (pending compute grant)
- [ ] Hyperparameter sweep
- [ ] 3-seed statistical validation
- [ ] Final submission

---

## References

- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks. arXiv:1603.08983
- Banino et al. (2021). PonderNet: Learning to Ponder. arXiv:2107.05407
- Raposo et al. (2024). Mixture of Depths: Dynamically allocating compute in transformer models. arXiv:2404.02258
- Clark et al. — SP8192 + GPTQ SDClip (PR #1394)
- Robby955 — Parallel residuals (PR #1412)
- dexhunter — Depth recurrence (PR #1437)

---

## Planned Files (final submission)

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`
