# Makora Beta Feedback — Parameter Golf Competition (March 2026)

## Context

Using Makora to generate fused Triton kernels for a competitive ML training challenge (OpenAI Parameter Golf). Target hardware: NVIDIA H100 SXM 80GB. Model: ~15-22M parameter transformer, bf16 training, 8xH100 distributed.

Makora CLI v1.0.3 on Windows 11, Python 3.13. Also used web app in parallel.

## Jobs Submitted

Three problem files targeting H100, Triton language:

| Problem | Session (CLI) | Session (Web) | Reference Time | Best Kernel | Speedup |
|---------|--------------|---------------|----------------|-------------|---------|
| Fused RMSNorm + QKV projection | c1215f27 | c4bb51fa | 0.314ms | 0.194ms | **1.48x** |
| Batched LoRA forward (rank-8) | 9d615014 | e245c74e | 0.091ms | 0.066ms | **1.40x** |
| Fused LM head + softcap + CE loss | 15da3aab | 9ca1921f | 0.788ms | 0.260ms | **1.17x** |

## What Worked Well

**Generation quality:** All three kernels eventually produced valid, faster-than-PyTorch solutions. The iterative refinement process (failing validation → retrying → improving) works. The RMSNorm+QKV kernel went through ~47 failed attempts before landing valid kernels, then consistently produced 1.40-1.48x variants. That's impressive autonomous optimization.

**CLI experience:** `makora generate --file problem.py -d H100 -l triton` is clean. Job submission, monitoring with `makora jobs`, and pulling results with `makora kernels <session_id> <kernel_id>` all work well.

**Parallel runs:** Running CLI and web app simultaneously gave different solutions — the web app found a 1.17x LM head kernel while CLI only managed 1.00x on the same problem. Useful to run both.

**Benchmark reporting:** The per-kernel timing breakdown (vs eager, vs torch.compile) is exactly what you need to decide whether to integrate.

## Issues Encountered

### 1. Generated kernels produce incorrect results at integration time

This is the biggest issue. Both the RMSNorm+QKV (1.48x) and LoRA (1.40x) kernels passed Makora's validation but produced **incorrect results** when integrated into the actual training pipeline:

- **RMSNorm+QKV:** `CUDA error: illegal memory access` on 8xH100. The kernel assumes specific alignment (M % 256 == 0, K % 128 == 0) but the fallback path with masking still crashed. Likely an out-of-bounds write in the masked kernel variant.

- **Batched LoRA:** Passed forward validation but produced wrong numerical results during test-time training evaluation. Post-quant eval went from val_bpb=1.296 (correct, PyTorch) to val_bpb=1.657 (wrong, Makora kernel). The packed weight layout (`_pack_weights` with rank-16 padding) may have a subtle transpose or indexing bug that doesn't show up in single-pass validation but accumulates over iterative LoRA updates.

**Root cause hypothesis:** Makora validates correctness with a single forward pass on random inputs, but integration contexts involve:
- Autocast (bf16 compute with fp32 accumulation)
- Gradient computation through the output
- Iterative application (LoRA weights updated between calls)
- Non-standard tensor strides from DDP/torch.compile

**Suggestion:** Offer an option to validate with gradient flow (backward pass) and with multiple sequential calls using updated parameters.

### 2. Windows CLI encoding issues

`makora info`, `makora check`, and other commands crash on Windows with:
```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2717'
```

The Rich library tries to output Unicode checkmarks/crosses that cp1252 (Windows default) can't handle. Workaround: `PYTHONIOENCODING=utf-8 makora ...`. Should be fixed in the CLI by setting the console encoding or using ASCII fallbacks.

### 3. `expert-generate` vs `generate` confusion

I initially used `makora expert-generate` (which takes an existing solution and improves it) when I meant to use `makora generate` (which creates a solution from a problem file). `expert-generate` silently accepted the problem file as if it were a solution, echoed it back unchanged, and reported "No relevant optimization patterns found."

**Suggestion:** `expert-generate` should detect when it receives a problem file (has `Model` class + `get_inputs()`) instead of a solution file (has `ModelNew` class) and error with a helpful message.

### 4. Device naming inconsistency between docs and CLI

Skill docs say `nvidia/H100`, CLI requires just `H100`. Minor but caused a failed attempt.

## Feature Requests

1. **Multi-pass correctness validation:** Validate kernel output across multiple sequential calls with parameter updates between them (critical for training/TTT use cases).

2. **Gradient validation:** Option to verify backward pass produces correct gradients, not just forward output. Training kernels that break autograd are useless even if forward is correct.

3. **Integration template generation:** Given a problem file, generate not just the kernel but a drop-in replacement function with proper dtype casting, contiguity checks, and fallback path. The boilerplate around `ensure weights are bf16`, `handle non-contiguous tensors`, `fall back if dimensions don't align` is where most integration bugs live.

4. **Batch generation:** Submit multiple problems in one command and get results for all. Would have saved time vs 6 separate submissions.

## Bottom Line

Makora's kernel generation quality is genuinely good — 1.48x on fused RMSNorm+QKV is a real win that I couldn't easily hand-write. The problem is the gap between "passes Makora validation" and "works correctly in a real training pipeline." If that gap closes, this tool becomes indispensable for ML competitions and production optimization.

**Would use again.** The unlimited beta credits made it practical to explore kernel optimization as a competition strategy, even though the kernels ultimately couldn't be used in the final submission due to correctness issues.

---

*Anthony Maio — March 2026*
*Competition: OpenAI Parameter Golf (github.com/openai/parameter-golf)*
*Submission: Depth recurrence + kitchen sink stack*
