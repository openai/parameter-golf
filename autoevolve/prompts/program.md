# Parameter Golf Auto-Evolve: Agent Instructions

## Challenge Overview
Train the best language model that:
- Fits in a **16,000,000 byte** artifact (code + zlib/zstd-compressed model)
- Trains in under **10 minutes on 8xH100 SXM** GPUs
- Minimizes **val_bpb** (bits per byte) on FineWeb validation set
- Evaluation must also complete within 10 minutes (separate from training)
- No external downloads or network calls during eval - fully self-contained

## Leaderboard Context (as of 2026-03-23)
| Score  | Key Innovation                                                      |
|--------|---------------------------------------------------------------------|
| 1.1194 | LeakyReLU^2 + legal score-first TTT + Parallel Muon                 |
| 1.1228 | 11L EMA + GPTQ-lite + warmdown3500                                  |
| 1.1248 | 11L Partial RoPE + LN Scale + EMA + XSA4                            |
| 1.1271 | 11L XSA4 + EMA + Int6 MLP3x                                         |
| 1.1307 | 11L Efficient Partial XSA                                           |

## Current SOTA Techniques (Already in Baseline — DO NOT RE-IMPLEMENT)
All of the following are already in `train_gpt.py`. Do NOT propose these as new ideas.

- **11 transformer layers** at **512 dim**, 8 heads, 4 KV heads, with U-Net-style skip wiring
- **3x MLP** with **LeakyReLU(0.5)^2** activation
- **Parameter banking + Parallel Muon** optimizer path for the large matrix banks
- **XSA on the last 4 layers**
- **Partial RoPE** (16/64 dims) + **LN scale** factor `1/sqrt(layer+1)`
- **Value Embeddings** (`VE_DIM=128`) on layers 9 and 10
- **SmearGate** residual mixing
- **BigramHash** baseline recipe centered around the latest top-record stack
- **EMA (0.997)** plus **tight SWA** every 50 steps
- **GPTQ-lite int6** post-training quantization with **lzma** compression
- **Legal score-first TTT** support in evaluation
- **warmdown=3500**, **iterations=9000**, **eval_stride=64**
- Artifact target is still extremely tight: roughly **15.95 MB**

## 1xH100 Proxy Mode Priorities
When the runner is in 1xH100 proxy mode, optimize for transfer to the official 8xH100 / 600s competition setting.

- Preserve official-like evaluation behavior and artifact accounting, so local wins are meaningful before 8xH100 promotion.
- Prefer robust changes that should survive the move from 1xH100 long proxy runs to 8xH100 final validation.
- Use cheap smoke tests only as a filter; do not overvalue tricks that win solely because proxy hardware is single-GPU.
- If the dossier shows repeated non-kept outcomes from the same family, pivot families before making a more complex variant of the same idea.

## Ideas to Explore — Full-Fidelity / Promotion Track

### Tier 1: High Impact (expected delta > -0.005 BPB)

1. **Improved Activation: SwiGLU / GeGLU**
   Replace relu² with SwiGLU: out = (W1·x * silu(W3·x)) · W2.
   Used in LLaMA-3, Gemma, Mistral. Consistently ~0.3-0.5 ppl improvement.
   Need to adjust hidden dim to keep param count: hidden = floor(2/3 × 3 × 512) = 1024.
   The parameter savings from 1024→1024 (vs current 1536) can fund an 11th layer.

2. **KV Head Count**
   Try 2 KV heads (from current 4) for an MQA-like setup.
   This can save projection bytes and runtime, which is especially proxy-friendly.

3. **Longer Training with Warmdown Tuning**
   Try warmdown tuning or a gentler LR tail without adding per-step compute.
   More careful late training can be a cheap probe for optimization-limited regimes.

4. **Weight Pruning Increase**
   Current 3% magnitude pruning. Try 5-8% with a gradual schedule.
   With zstd compression, sparse weights compress better and can fund capacity elsewhere.

5. **Quantization-Aware Training (QAT) — INT5/INT6 STE**
   Inject fake int5/int6 quantization noise during the LAST 10-15% of training via
   Straight-Through Estimator (STE). The model learns to pack weights into integer ranges.
   Record #2 already uses this. Expected gain: -0.003 to -0.008 BPB in final-validation mode.
   Key: match the quantization scheme used at eval (int5 for MLP, int6 for attn).
   Proxy warning: this is expensive and should not be the first family you try unless the long local proxy is already stable.

6. **Better Compression: Block Quantization + zstd**
   Switch from per-row to per-block INT5/INT6 quantization (block size 32-64).
   Smaller scale factors = more bits available for actual weights.
   Combine with zstd level 22 for maximum compression.
   Current artifact is 15.9MB — any saved bytes can fund extra capacity.

7. **Depth Recurrence / Looped Transformer**
   Share transformer block weights and forward through them 2× per input position.
   Effectively doubles depth for zero extra parameters.
   Key research: Universal Transformers, DEQ.
   Risk: throughput halves (may lose training steps), gradient flow trickier.
   Try: 5 unique blocks × 2 loops = 10 effective layers.

8. **Larger Vocabulary with Subword BPE**
   The 1024 vocab is very small. A 2048 or 4096 vocab reduces token sequence length,
   more context per step, better byte-level compression.
   But embedding table grows: 2048×512×2bytes = ~2MB FP16. Tight on 16MB.
   Try: 2048 vocab with compressed INT8 embeddings or factored embedding.

9. **Test-Time Training (TTT) with Low-Rank Updates**
   Fine-tune rank-1 LoRA adapters per document during evaluation.
   Submission #4 got 1.1928 without combining with SOTA architecture.
   Combined: potentially -0.015 or more. Risk: eval budget.

### Tier 2: Medium Impact (expected delta -0.002 to -0.005 BPB)

10. **11th Layer via Aggressive Quantization**
   Current artifact: ~15.9MB. Extra layer = ~1MB params.
   But if we drop MLP to 2.5x + int4 for least-important heads = break even.
   Mixed-precision: int4 for 2-3 shallow layers, int6 for deep layers.

11. **Attention Score Bias / ALiBi or FIRE Positional Bias**
   Replace RoPE with ALiBi or relative position encoding.
   More generalizable to longer sequences. Saves embedding params (no RoPE cache).

### Tier 3: Experimental / Lower Confidence

12. **Mixture of Experts (MoE)**: Sparse MLP with top-2 of 4 experts.
    Effectively 2x capacity at ~1.5x compute. Complex routing overhead.
13. **Flash Attention with longer context**: Train at 4096 seq_len.
    Current 2048 already good but longer helps with sliding window eval.
14. **Gradient Checkpointing** to enable larger model at same memory.
15. **FP8 Training**: H100s support FP8 natively. Could speed up matmuls ~2x.
16. **Embedding Factorization**: Low-rank decomp E = UV (vocab×r × r×dim).
    With rank 64, embeds compress from 1MB to 0.13MB — spend on more layers.
17. **Curriculum Learning**: Start on shorter seqs, increase. May hurt or help.
18. **Low-rank MLP via SVD**: Decompose W=UΣVᵀ with rank reduction to save bytes.
19. **Deeper QK Normalization**: Normalize keys+queries separately per layer.
20. **Alternative tokenizers**: Character-level or byte-level with n-gram hashing.

## Compression Budget Analysis (Critical — Read Before Every Proposal)
- Current: ~19.2M params → ~15.9MB INT5/INT6/zstd → ~0.1MB headroom only!
- Headroom is ALMOST GONE. Any new capacity MUST come from compression gains.
- If you add a layer or widen the model, you MUST compress elsewhere.
- Options: higher pruning, INT4 for some layers, factored embeddings, vocab changes.
- Always estimate compressed size BEFORE proposing capacity changes.

## Parameter Budget Rules
- 1 transformer layer ≈ 4 × 512² × 2 × (attn + mlp) ≈ ~2M params ≈ ~1MB INT6
- Embedding table (1024 vocab, 512 dim) = 0.5M params ≈ 0.5MB FP16 = 1MB
- BigramHash (10240 × 128 + proj) ≈ 1.4M params ≈ 0.7MB INT8
- Budget: total must compress to ≤16MB with code overhead (~3KB)

## PROVEN NEGATIVE RESULTS — DO NOT REPEAT THESE
These have been tested and CONFIRMED to hurt. Do not re-propose them.

- **Layer recurrence / weight sharing**: Tested on 1×5090. Doubling depth via weight reuse
  halves training steps → net NEGATIVE: +0.051 BPB worse. The throughput cost outweighs
  the capacity gain at 600s wallclock. Only reconsider if you can avoid throughput loss.
- **Weight decay at small scale**: Tested at WD=0.01, no benefit for short training runs.
  (Note: WD=0.04 IS in the SOTA baseline and does help — this finding is about adding MORE WD.)
- **Warmdown schedule bug**: If warmdown_iters > total_steps_achieved, LR decays from step 1.
  Current SOTA uses warmdown_iters=3000 with wallclock stop. On 1xH100 this could be
  an issue if only ~2-3k steps complete. The SOTA baseline handles this via time-based
  warmdown fraction — but verify this is still correct after any modifications.

## Common Pitfalls
- **Artifact over 16MB**: Do NOT add params without detailed budget math first.
- **Training too slow**: Each extra ms/step on 8xH100 costs real training steps.
- **SWA + warmdown interaction**: SWA collects from warmdown. Don't break this.
- **zstd dependency**: Use `try: import zstandard` with zlib fallback already in baseline.
- **Multi-GPU bugs**: DDP requires careful handling of per-rank ops.
- **Numerical instability**: Muon with high momentum (0.99) + aggressive LR → NaN risk.
- **QAT noise timing**: Too early = destabilizes training; too late = no benefit.
- **1×H100 proxy vs 8×H100 final**: longer local runs are useful for idea discovery, but transfer is imperfect.
  Promote only the strongest proxy wins to final validation.

## How to Reason About Each Proposal
1. **Parameter budget**: Calculate params added vs removed. Will it fit in 16MB?
2. **Step throughput**: Will this slow the forward/backward pass? By how much?
3. **Theoretical backing**: Is there peer-reviewed evidence this helps at small scale?
4. **History check**: Has this or something similar been tried? What happened?
5. **Implementation risk**: Is this a risky CUDA change or a simple Python change?
6. **Expected gain**: Be honest. If expected delta < -0.001, pick something bolder.
