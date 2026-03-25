# Non-Record: Negative Results — Hardware Alignment & Quantization on 8×H100

30+ experiments attempting to improve the 11L d=512 transformer beyond PR #593 (1.1171 BPB). Every kernel optimization, quantization trick, and architectural change that did NOT help.

**Base:** PR #593 — 1.1171 BPB, 83ms/step, 7189 steps, Parallel Muon + Full GPTQ, 8×H100 SXM

---

## Kernel-Level Optimization (All Dead)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| CUTLASS SM90 TMA+WGMMA GEMM | 2.5× slower than cuBLAS | cuBLAS heuristics beat default CUTLASS for 98304×512×1536. Built a working kernel — correct results, wrong speed. |
| Fused Triton GEMM + LeakyReLU² | 1.82× faster fwd, **2.7× slower** fwd+bwd | `torch.autograd.Function` bypasses Inductor. Backward runs in eager mode, 2-3× slower than Inductor's auto-generated Triton backward. |
| `torch.library.triton_op` for GEMM | Compile error | FakeTensor can't provide `data_ptr()` — GEMM kernels incompatible with triton_op tracing. |
| Custom CUDA C++ fused activation | 6% slower | PyTorch's `vectorized_elementwise_kernel` is already highly optimized for pointwise ops. |
| Fused norm+residual (Triton) | Ties torch.compile exactly | 0.136ms ours vs 0.136ms Inductor-generated. torch.compile already fuses this pattern. |
| FP8 training (TransformerEngine) | No speedup (90 vs 89ms) | At d=512, attention GEMMs are already memory-bound (AI=170-255). FP8 doubles peak FLOPS but also doubles the ridge point, making MORE ops memory-bound. |
| QKV fusion (8Q/4KV GQA) | 3-17% slower | Fused (512→1024) GEMM is slightly faster, but splitting output into non-contiguous Q(512)/K(256)/V(256) tensors costs more than the GEMM savings. |

**Conclusion:** torch.compile (PyTorch 2.9.1) already fuses CE+softcap+tanh, LeakyReLU²+residual, RMSNorm+backward, and all pointwise chains. cuBLAS is at the hardware limit for K=512 (~48% roofline, pipeline depth limitation). The 82ms step is 95%+ optimized.

## torch.compile Gotchas

| Issue | Impact | Mechanism |
|-------|--------|-----------|
| Late QAT recompilation | OOM with larger models | Flipping `_qat_enabled` mid-training changes the forward graph → torch.compile recompiles → memory spike exceeds 80GB |
| `torch.autograd.Function` | 2-3× slower backward | Custom Functions bypass Inductor entirely. Backward runs uncompiled eager Python ops. |
| H100 memory compression | 25-50% inflated benchmarks | Synthetic data (cudaMemset, BlockFillRandom, zeros) compresses in HBM hardware. Only `torch.randn` gives real numbers. |

## Quantization Experiments (Diminishing Returns)

| Approach | BPB | Delta | Why It Failed |
|----------|-----|-------|---------------|
| SpinQuant (Hadamard rotation before GPTQ) | 1.1151 | −0.0002 | GPTQ's actorder + Cholesky already handles outliers. Rotation adds little on top. Artifact slightly larger (rotated weights compress worse). |
| Mixed-precision int5/int8 per-layer | 1.1209 | +0.006 | int5 (31 levels) is too coarse. Boundary layers at int8 can't compensate for middle layers losing half their precision. |
| Soft-Round QAT (differentiable rounding) | 1.1151 | −0.0002 | `soft_round(x,T) = x + 0.5*tanh(T*(x-round(x)-0.5))/tanh(T/2)`. Identical to standard STE — the ~500 QAT steps aren't enough for the temperature annealing to have effect. |
| Selective ±1 pruning at 28-37% | 1.1198-1.1204 | +0.004-0.005 | Too aggressive. Only <10% pruning is loss-neutral. The #609 approach works because their base artifact is smaller (BigramHash 2048). |

## Architecture & Training (All Negative)

| Approach | BPB | Delta | Why It Failed |
|----------|-----|-------|---------------|
| XSA on all 11 layers (vs last 4) | worse at 100s | +0.014 | 2.9ms/step overhead from 7 additional XSA ops. In our Parallel Muon stack, the slower step time costs more than XSA gains. Works in #609's stack but not ours. |
| Value Residual Learning | 1.1179 | +0.0008 | VRL conflicts with VE128 in our stack — both inject identity information into deep attention layers. Redundant. |
| Gated Attention | 1.1197 | +0.0026 | 4% slower step time (86.7 vs 83ms). Per-head sigmoid gates add overhead that isn't compensated by quality improvement. |
| Weight decay 0.08 (vs 0.04) | 1.1235 | +0.008 | Better at 100s (2.191 vs 2.207 val_loss), WORSE at 600s. Over-regularization prevents learning fine-grained patterns during warmdown. **Early loss does not predict final post-quant BPB.** |
| Batch size 1M tokens | 1.1197 | +0.003 | Fewer steps (5,526 vs 7,189) hurt more than better gradients help. At this scale, step count dominates. |
| Train bigger d=576 + int5 | 1.1233 | +0.006 | 110ms/step = 24% fewer steps. Scaling law gain (~0.019 from more params) can't compensate for 1,700 fewer training steps. |
| Shard ordering (hard→easy) | 1.1162 | +0.0009 | Per-shard loss spread is only 0.024 (0.3% relative). Ordering disrupts natural data diversity, net negative. |
| Legal TTT (22 experiments) | 1.1177 best | +0.0006 | Score-first constraint means model adapts too late — early tokens get no benefit. 400-1600s eval time for zero or negative gain. |
| Hessian all-reduce across GPUs | 1.1169 | −0.0002 | 256 calibration batches per GPU already provide sufficient Hessian statistics. |

## Meta-Lessons

1. **The step is 95%+ optimized.** torch.compile handles all fusion, cuBLAS is at hardware limit, FA3 already in use. No kernel-level headroom.

2. **H100 is massively overprovisioned** for this model. 21.5GB of 80GB GPU used. 99% of NVLink idle. The hardware constraints don't bind — the 16MB artifact limit does.

3. **The competition is bits-per-parameter, not FLOPS-per-second.** The quantization gap (0.022 BPB) is 10× larger than any kernel optimization. Reducing it is the only path.

4. **Stale processes from nohup+torchrun** accumulate silently, causing 2-3× performance degradation and false experimental results. Always verify `nvidia-smi` shows 0 MiB before experiments.

5. **Early training loss direction doesn't predict final BPB.** WD=0.08 looks better at 100s but worse at 600s after warmdown + EMA + GPTQ. Fast A/B tests can filter out clearly bad ideas but cannot confirm good ones.
