# Elite Universal Transformer: Implementation Compression

This document summarizes the technical evolution and final configuration of the **Elite Universal Transformer** for the 10-minute Parameter Golf challenge.

## 1. The Core Architecture (Universal Recursive Depth)
- **Model**: 12-step depth recurrence with shared weights (Block-Tying).
- **Stabilizer ($1/12$)**: Restored the recursive gradient division (`p.grad.div_(12)`) for all parameters. This prevents the residual stream from exploding over 12 steps.
- **Deep State Refactor (v22.8)**: Removed all internal and initial normalization bottlenecks.
    - **Logic**: Transitioned from Post-Norm to **Strict Pre-Normalization** and removed the post-embedding `RMSNorm`.
    - **Impact**: Allows the hidden state to accumulate depth and complexity across the 12-step chain without being "reset" to unit variance at every step. The residual stream remains untouched from embedding to `final_norm`.
- **Subtle Stochastic Depth (4%)**: Restored a minimal 4% drop rate in the recursive update branch to provide "quieting" regularization for the residual stream.

## 2. Evaluation & Adaptation (TTT)
- **Legal LoRA TTT**: Implemented a "Score-then-Adapt" loop that adapts ~150k LoRA parameters to the validation set at runtime.
- **TTT Cooling (4e-4)**: Downscaled the Test-Time Training learning rate to prevent "Catastrophic Forgetting" and validation BPB climbs observed at higher rates.
- **Conditional Stride Evaluation**: 
    - **Mid-Run**: `stride=256` (1x speed) to maintain training throughput.
    - **Final Step**: `stride=64` (4x precision) to capture a "warm context" BPB boost (~0.12 reduction).

## 3. Training Dynamics & Optimizer
- **"Safe-Speed" Muon (0.012)**: Settled on a balanced matrix learning rate that provides rapid convergence without the instability seen at higher pressures.
- **Optimizer Routing**:
    - **Muon**: Internal, dense 2D matrices (Attention and MLP projections).
    - **AdamW**: Sparse/Scalar parameters (Word Embeddings, Step Embeddings, Norms, Gains).
- **EMA Shadow Weights**: Fixed the `module.` key mismatch in DDP, ensuring validation always uses the stable 0.99 decay shadow weights.

## 4. Final Submission Specification (v22.8)
| Parameter | Value |
| :--- | :--- |
| **Matrix LR** | 0.012 |
| **TTT LR** | 4e-4 |
| **Warmup** | 20 Steps |
| **Drop Rate** | 0.04 (Recursive) |
| **Logit Softcap** | 10.0 |
| **Label Smoothing** | 0.05 |
| **Cosine Target** | 600s (Competition Deadline) |

## 5. Verification Results
- **v22.7 Trial**: Confirmed stable, monotonic BPB descent. The "BPB Climb" at Step 19 has been neutralized, holding sub-3.41 effectively during the stabilization window.
