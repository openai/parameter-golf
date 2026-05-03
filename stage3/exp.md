# Meta-Prompt: Single-Shot LM Training Script Optimization

## Usage

Feed this prompt to a frontier LLM along with the baseline `train_gpt.py` as input. The output should be a fully rewritten, competition-ready training script.

---

## The Prompt

```
You are an expert ML systems engineer competing in a constrained language model training challenge. Your task is to rewrite the provided baseline training script to achieve the lowest possible validation bits-per-byte (val_bpb).

## Constraints

- Hardware: 8× H100 SXM GPUs, 80GB each
- Wall clock: 600 seconds (10 minutes) hard cap
- Artifact size: 16 MB maximum (model weights + code)
- Metric: val_bpb (bits per byte) on a held-out FineWeb validation set, lower is better
- The script must be self-contained: training, quantization, export, and evaluation in one file
- No external data access after training starts (no val/train data for calibration)
- Tokenizer can be changed but evaluation is tokenizer-agnostic (BPB, not loss)

## Output Structure (mandatory)

You must produce your output in THREE sequential phases. Do not skip phases. Do not merge them.

### Phase 1: Modification Spec Sheet

Before writing any code, produce a numbered list of every modification you intend to make. For each modification, write:

**M-{N}: {Name}**
- Mechanism: One sentence explaining WHY this helps (the causal claim, not just "improves performance")
- Math: The core equation or operation. Use tensor notation with explicit shapes. Example:
  > v̂ = v / ‖v‖₂ along dim=-1, shape [B, T, H_kv, D]
  > y_out = y - (y · v̂) v̂, broadcast over GQA groups: reshape y to [B, T, H_kv, G, D]
- Placement: Exactly where in the forward pass this goes (after which operation, before which operation, in which module method)
- Shapes: Input and output tensor shapes, noting any GQA grouping, head dim splits, or dtype requirements
- Init: How new parameters are initialized (zero, orthogonal, constant, normal with what std)
- Coefficients: Any magic numbers, with the value you'll use and WHY that value (not "empirically good" — what principle constrains it)
- Failure modes: The most likely implementation bug for this modification (wrong dim in normalize, missing dtype cast, breaking torch.compile, etc.)

If you cannot write the Math line for a modification, you do not understand it well enough to implement it. Drop it and replace with something you can specify precisely.

### Phase 2: Implementation

Write the complete training script. For each modification from Phase 1, include a brief inline anchor comment referencing the spec:

```python
# M-3: XSA — y = y - proj(y, v_hat), shapes [B,T,Hkv,G,D]
```

This forces you to cross-reference every code block against its spec. If you find yourself writing code that doesn't match any spec entry, STOP — either add it to the spec or remove the code.

### Phase 3: Self-Review Checklist

After the complete script, produce a verification table:

| Mod | Spec matches code? | Shapes verified? | Init correct? | dtype safe? | compile safe? |
|-----|---------------------|-------------------|---------------|-------------|---------------|
| M-1 | ✓ explain | ✓ explain | ✓ explain | ✓ explain | ✓ explain |
| ... | | | | | |

For each cell, write 3-5 words proving you actually checked. "Yes" is not acceptable. "dim=-1 on [B,T,H,D]" is.

If any cell fails, go back and fix the code. Do not submit a script with known spec-code mismatches.

## Common Implementation Traps

These are the most frequent ways that correct ideas become wrong code. Check each one:

1. **Normalization dimension**: F.normalize(v, dim=-1) vs dim=-2 vs dim=1. The correct dim depends on whether v is [B,T,H,D] or [B,H,T,D] or [B,T,D]. Flash Attention 3 outputs [B,T,H,D]. PyTorch SDPA outputs [B,H,T,D]. Get this wrong and the operation is mathematically meaningless.

2. **GQA broadcast**: With 8 heads and 4 KV heads, you have groups of 2. An operation on v [B,T,4,D] applied to y [B,T,8,D] requires reshaping y to [B,T,4,2,D] and broadcasting. Using repeat_interleave works but is 3× slower than reshape+broadcast.

3. **dtype mismatch**: New nn.Parameter defaults to float32. If the model runs in bfloat16, the parameter will silently upcast the activation to float32, breaking torch.compile's fullgraph and slowing everything down. Cast explicitly: `self.gate.to(dtype=x.dtype)`.

4. **torch.compile breaking patterns**: Python-level `if` on tensor values, in-place ops on views of compiled tensors, dynamic list appends, and data-dependent control flow all break `torch.compile(fullgraph=True)`. Test your modifications mentally for these.

5. **Gate initialization**: A sigmoid gate initialized at 0 outputs 0.5 (half-strength), not 0. For "start disabled, learn to enable," init the bias to -4.0. For "start enabled, learn to disable," init to +4.0. Getting this backwards is a silent 50% effect size reduction.

6. **Projection vs component removal**: "Subtract the projection of y onto v" is `y - (y·v̂)v̂`. This is NOT `y - v`. The first removes one direction. The second subtracts a potentially large vector. Getting this wrong doesn't crash — it just produces garbage.

7. **Scale factors**: `max(1, rows/cols)**0.5` is the Muon scale correction. Omitting it silently halves the effective learning rate for non-square matrices. The Newton-Schulz output is orthogonal, and this factor corrects for the aspect ratio.

8. **Warmdown math**: The LR schedule needs to transition from 1.0 to 0.0 based on remaining wallclock time, not step count. Using step count makes warmdown length hardware-dependent. Use `remaining_ms / warmdown_ms` clamped to [0, 1].

9. **GPTQ column ordering**: actorder sorts by diagonal of H (descending), quantizes most-sensitive columns first. If you sort ascending, you're quantizing least-sensitive first, which means error propagation accumulates into the sensitive columns — exactly backwards.

10. **Async communication ordering**: reduce-scatter must complete before Newton-Schulz on that shard. all-gather must complete before using the updated weights. Mixing up the wait points doesn't crash — it uses stale data silently.

## Optimization Axes

You must simultaneously optimize across four independent axes. Each axis contributes independently to the final score:

### Axis 1: Training Efficiency (steps × quality per step)
The score is determined by how many gradient steps you complete AND how much each step improves the model. Optimize both:

- **Throughput**: Minimize ms/step. Use kernel-level optimizations (Flash Attention 3, fused ops), overlap communication with computation, eliminate synchronization barriers. Every saved millisecond = more training steps.
- **Model capacity**: Maximize learnable capacity within the throughput budget. More layers, wider MLPs, larger sequence lengths — but only if ms/step stays competitive.
- **Optimizer quality**: The optimizer must extract maximum improvement per step. Consider Muon (Newton-Schulz orthogonalized updates) with momentum warmup, weight decay, gradient clipping, and learning rate warmdown schedules tuned for the short-horizon regime.
- **Auxiliary signals**: Add cheap training signals that improve generalization without slowing the loop. Bigram/n-gram hash embeddings, position-mixing gates, value embeddings at specific layers, residual connections across the U-Net encoder-decoder structure.
- **Weight averaging**: EMA and/or SWA to smooth the optimization trajectory and reduce generalization gap.

### Axis 2: Architecture Design
Design choices that improve the model's expressiveness per parameter:

- **Attention**: GQA (grouped query attention), QK normalization with learnable gain, partial RoPE (apply rotary to subset of head dims), cross-sequence attention (XSA — subtract self-value projection to force cross-position information flow).
- **MLP**: LeakyReLU with large negative slope (0.5) squared — preserves gradient flow through negative activations while maintaining the sparsity benefit of squaring.
- **Normalization**: Per-layer scaling on norm outputs (1/√(layer+1)) to stabilize deep networks.
- **Depth**: U-Net skip connections (encoder stores, decoder reuses in reverse). Consider depth recurrence (looping a subset of layers 2-3× to create virtual depth from shared parameters).
- **Embeddings**: Larger vocab tokenizers (SP4096/SP8192) compress text into fewer tokens, improving per-byte modeling. Hash-based bigram embeddings inject local context at zero attention cost.

### Axis 3: Quantization & Export
The 16MB artifact limit means aggressive quantization is mandatory. The quantization method is as important as the model architecture:

- **Format**: int6 (6-bit, ±31 range) for large matrices, int8 for embeddings, fp16 passthrough for small control tensors.
- **Method**: Full Hessian GPTQ — collect H = X^T X via forward hooks, Cholesky decomposition, block-wise error propagation with column reordering (actorder). This is strictly superior to naive percentile clipping.
- **Calibration**: The model must generate its own calibration data autoregressively (no external data access). Generate ~64 sequences of ~2048 tokens at temperature 0.8.
- **Size fitting**: After quantization, if the artifact exceeds the size limit, selectively prune ±1 quantized values sorted by reconstruction error (scale²). Binary search for the minimum pruning needed.
- **Compression**: LZMA or Brotli compression on the serialized checkpoint.
- **Late QAT**: Enable quantization-aware training (straight-through estimator) during the warmdown phase to pre-adapt weights to quantization grid.

### Axis 4: Evaluation Policy
The evaluation method itself affects the score. Optimize how the artifact is scored:

- **Sliding window**: Instead of evaluating each sequence independently, use overlapping windows with a stride (e.g., 64 tokens). Each token is scored with the maximum available context. This recovers ~0.02 BPB over standard evaluation.
- **Sequence length**: Evaluate at the training sequence length or longer. Separate EVAL_SEQ_LEN from TRAIN_SEQ_LEN.
- **Compilation**: torch.compile the inference path (forward_logits) for faster eval.

## Engineering Requirements

- **No DDP wrapper**: For the main model, handle gradient communication manually. Use async reduce-scatter → local computation → async all-gather to overlap communication with optimizer work.
- **Parameter banking**: Stack all block-level matrix weights into contiguous 3D tensors (e.g., qo_bank[2*N, dim, dim]). This enables batched Newton-Schulz and efficient gradient communication.
- **Separate Hessian model**: For GPTQ, create a parallel non-banked model with standard nn.Linear layers that can attach forward hooks. Load the trained weights into it, collect Hessians, then quantize.
- **All configuration via environment variables**: The script should be fully configurable without code changes.

## Output Format

Follow the three-phase structure above (Spec Sheet → Implementation → Self-Review). The implementation must:
1. Train the model for up to 600 seconds on 8× H100
2. Apply EMA/SWA weight averaging
3. Run full Hessian GPTQ with self-generated calibration
4. Export a compressed artifact under 16MB
5. Evaluate with sliding window and report val_bpb
6. Be runnable via `torchrun --standalone --nproc_per_node=8 train_gpt.py`

Maximize density and correctness. Every line of code must trace back to a spec entry.

## Baseline Script

<paste baseline train_gpt.py here>
```

---

## Why This Works

The prompt is structured around five key insights:

1. **Spec-before-code kills the idea-implementation gap.** The most common failure in LLM-generated optimization code is that the idea is right but the implementation is subtly wrong — wrong normalization dim, wrong gate init, wrong broadcast. Forcing the LLM to write the math FIRST, with explicit shapes and placement, means the code has a contract to satisfy. Without this, the LLM "vibes" its way through implementation and gets details wrong.

2. **The self-review table forces verification.** LLMs are bad at spontaneously checking their own work but good at checking when given a specific rubric. The per-modification checklist (shapes, init, dtype, compile safety) catches the exact class of bugs that silently degrade results without crashing.

3. **The trap list is a negative-example prior.** Instead of hoping the LLM avoids common mistakes, we enumerate the 10 most frequent ones with explanations of what goes wrong and why. This is more effective than positive instructions because it targets the specific failure modes that distinguish a 0.01 BPB win from a 0.005 BPB loss.

4. **Explicit axis decomposition**: Most people think "make the model better." This prompt forces the LLM to think about training efficiency, architecture, quantization, and eval as independent optimization surfaces. Each axis has ~0.01-0.03 BPB of headroom, and they stack.

5. **Anchor comments create traceability.** The `# M-3: XSA — ...` comments in the code force every block to reference its spec. Orphaned code (modifications that appeared in the implementation but weren't in the spec) becomes visible. This prevents the LLM from adding "while I'm here" changes that aren't grounded in any mechanism claim.

## Limitations

- This prompt assumes knowledge of specific techniques (Flash Attention 3, Muon, GPTQ). A truly general version would need to describe the mathematical foundations.
- The technique list is frozen to ~Apr 2026 SOTA. New techniques (depth recurrence details, SLOT, discriminative TTT) would need to be added as they emerge.
- Single-shot generation of 2000+ lines of correct CUDA-aware distributed training code is at the frontier of current LLM capability. Expect to need 1-2 rounds of debugging.
- The prompt encodes the *winning* stack. It does not encode the search process that discovered it. For novel competition settings, you'd need a different meta-prompt focused on exploration rather than exploitation (see `orchestrator_meta.md`).
- The three-phase structure increases output length significantly. On models with tight output limits, the self-review phase may get truncated. In that case, prioritize Phase 1 (spec) and Phase 2 (code) — the spec alone catches ~60% of implementation bugs by forcing the LLM to think before coding.
