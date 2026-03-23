# Parameter Golf Auto-Evolve: Agent Instructions

## Challenge Overview
Train the best language model that:
- Fits in a **16,000,000 byte** artifact (code + zlib/zstd-compressed model)
- Trains in under **10 minutes on 8xH100 SXM** GPUs
- Minimizes **val_bpb** (bits per byte) on FineWeb validation set
- Evaluation must also complete within 10 minutes (separate from training)
- No external downloads or network calls during eval - fully self-contained

## Leaderboard Context (as of 2026-03-20)
| Score  | Key Innovation                                                  |
|--------|----------------------------------------------------------------|
| 1.1428 | Int5 MLP + Int6 Attn + BigramHash(10240) + SWA(0.4) + WD=0.04 |
| 1.1458 | 3x MLP + SmearGate + BigramHash + OrthoInit + Muon WD + SWA   |
| 1.1502 | 11L, 3x MLP, Int6 QAT, zstd-22, WD=0.04, sliding eval         |
| 1.1556 | SmearGate + BigramHash + 3x MLP + int6 STE QAT + sliding eval  |
| 1.1586 | Int6 QAT + zstd-22, MLP 1344, Muon 0.99, sliding eval          |

## Current SOTA Techniques (Already in Baseline — DO NOT RE-IMPLEMENT)
All of the following are already in `train_gpt.py`. Do NOT propose these as new ideas.

- **10 transformer layers** with U-Net skip connections
- **512 model dim**, 8 heads, 4 KV heads (GQA), **3x MLP expansion** (hidden=1536), relu² activation
- **seq_len=2048** (training), batch=786K tokens
- **SmearGate**: sigmoid-gated residual mixing with layer-position schedule
- **BigramHash(10240, dim=128)**: consecutive token pair embeddings, projected to model_dim
- **Orthogonal init** with muP-scaled output projections
- **Mixed Int5/Int6 quantization**: Int5 [-16,15] for MLP (best compression), Int6 [-32,31] for attention, FP16 for tied embeddings + last-layer K projections
- **zstandard compression** (zstd-22 level) for serialized weights
- **SWA** (Stochastic Weight Averaging): collect from last 40% of warmdown, every 50 steps
- **Muon optimizer** (matrix_lr=0.02, momentum=0.99, WD=0.04) + AdamW (embed_lr=0.6, scalar_lr=0.02, WD=0.04)
- **Muon momentum warmup**: 0.92 → 0.99 over 1500 steps
- **QK normalization** with learned per-head gain (init 1.5)
- **RoPE** positional encoding, rope_base=10000
- **Logit softcap** at 30.0 (tanh-based)
- **CastedLinear** (fp32 weights, bf16 compute)
- **Sliding window eval** (stride=64, eval_seq_len=2048)
- **3% magnitude weight pruning** before serialization
- **warmdown=3000**, warmup=20 steps, grad_clip=0.3
- ~19.2M params → ~15.9MB compressed artifact (very tight: ~0.1MB headroom)

## Scout Mode Priorities
When the runner is in scout mode on 1xH100, optimize for completed runs and good telemetry.

- Prefer low-overhead probes first: activation swaps, optimizer/warmdown tuning, KV head count, eval-safe architectural simplifications, and other changes that preserve compile/step throughput.
- Treat throughput-heavy quantization alignment ideas as promotion-track ideas until the scout loop has produced completed runs with trustworthy timing telemetry.
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
   This can save projection bytes and runtime, which is especially scout-friendly.

3. **Longer Training with Warmdown Tuning**
   Try warmdown tuning or a gentler LR tail without adding per-step compute.
   More careful late training can be a cheap probe for optimization-limited regimes.

4. **Weight Pruning Increase**
   Current 3% magnitude pruning. Try 5-8% with a gradual schedule.
   With zstd compression, sparse weights compress better and can fund capacity elsewhere.

5. **Quantization-Aware Training (QAT) — INT5/INT6 STE**
   Inject fake int5/int6 quantization noise during the LAST 10-15% of training via
   Straight-Through Estimator (STE). The model learns to pack weights into integer ranges.
   Record #2 already uses this. Expected gain: -0.003 to -0.008 BPB in full-fidelity mode.
   Key: match the quantization scheme used at eval (int5 for MLP, int6 for attn).
   Scout warning: this is expensive and should not be the first family you try when runs are not finishing.

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
- **1×H100 vs 8×H100**: On 1×GPU, grad_accum_steps=8, so ~8× fewer steps in 600s.
  Relative improvements should still transfer, but absolute BPB will be higher.

## How to Reason About Each Proposal
1. **Parameter budget**: Calculate params added vs removed. Will it fit in 16MB?
2. **Step throughput**: Will this slow the forward/backward pass? By how much?
3. **Theoretical backing**: Is there peer-reviewed evidence this helps at small scale?
4. **History check**: Has this or something similar been tried? What happened?
5. **Implementation risk**: Is this a risky CUDA change or a simple Python change?
6. **Expected gain**: Be honest. If expected delta < -0.001, pick something bolder.
