# QAT + NTK-4096 Eval + Cosine Warmdown + Aggressive SWA

**Status: Incomplete** — RunPod terminated the pod during evaluation on all 8xH100 attempts. Best run completed training (6606 steps, 600s) but no final roundtrip val_bpb.

**Pre-quant val_bpb: 1.1702** (step 6606) | **1xH100 roundtrip val_bpb: 1.2890** (872 steps)

## Approach & Changes from Baseline

This submission modifies the baseline `train_gpt.py` by integrating proven architectural optimizations from the community alongside my own quantization and training strategies.

**1. Architecture Updates**
* **Scaled Capacity:** Increased to 10 layers with 3x MLP expansion (hidden=1536) and a phase-transition sigmoid init for the residual mix.
* **SmearGate & BigramHash:** Added a learned gate to blend consecutive token embeddings for better local context, and a 10240-bucket (dim=128) bigram hash embedding. *Adapted from approaches by [@raahilshah](https://github.com/raahilshah) and [@thwu1](https://github.com/thwu1).*
* **Skip Connections & Init:** Introduced learnable U-Net skip connections and orthogonal weight initialization with gain-scaled projections.

**2. Training & Optimization**
* **Cosine Warmdown:** Replaced the baseline's linear warmdown with a cosine schedule (`0.5 * (1 + cos(πt))`) to sustain higher learning rates longer.
* **Aggressive SWA:** Implemented Stochastic Weight Averaging starting at 35% of warmdown, collecting every 25 steps (averaging 48 checkpoints in the best run) for smoother weight distribution.
* **Muon Optimizer:** Maintained Muon with 0.04 weight decay and 0.3 gradient clipping, plus momentum warmup (0.92→0.99 over 1500 steps).

**3. Quantization-Aware Training (QAT)**
* Unlike the baseline's post-training `int8` quantization, I implemented QAT using a Straight-Through Estimator (STE).
* `CastedLinear` layers fake-quantize weights during the forward pass (int5 for MLPs, int6 for Attention). This forces the model to learn robustness against quantization noise *during* training, minimizing the final compression penalty.

**4. Evaluation & Compression**
* **NTK-Aware RoPE & Sliding Window:** To bridge the gap between the 2048 training length and the 4096 evaluation length, I dynamically rescale RoPE frequencies (NTK-aware) and use a sliding-window evaluation (stride=64) so scored tokens see near-full context. *Building on work by [@matthewjli](https://github.com/matthewjli).*
* **Aggressive Compression:** Replaced `zlib` with `lzma` (PRESET_EXTREME). Applied 5% magnitude pruning and packed weights using the mixed int5/int6 QAT scheme, fitting the artifact well under 16MB.

## Feature Comparison

| Feature | `train_gpt.py` | This |
|---------|----------|-----------------|
| **Layers & MLP** | 9 layers, 2x MLP | 10 layers, 3x MLP (relu²) |
| **Context Mix** | None | SmearGate + BigramHash (10240 buckets) |
| **Skip Connections** | None | U-Net style (learnable weights) |
| **Weight Init** | Default | Orthogonal + phase-transition resid_mix |
| **Quantization** | Post-training (int8 + zlib) | QAT via STE (mixed int5/int6) + lzma |
| **Warmdown** | Linear | Cosine |
| **SWA** | None | Yes (frac=0.35, every=25 steps) |
| **Pruning** | None | 5% magnitude pruning |
| **Eval Setup** | Standard (seq_len = train) | Sliding window (stride=64), NTK RoPE (4096) |

## Run Attempts

| Attempt | Hardware | Steps | Last val_bpb | Outcome |
|---------|----------|-------|-------------|---------|
| leaderboard-8xh100-v1 | 8xH100 SXM | 6606 | 1.1702 (pre-quant) | Pod killed during eval |
| fail2 | 8xH100 SXM | 2000 | 1.2789 (pre-quant) | Pod killed mid-training |
| leaderboard_v1 | 1xH100 | 872 | 1.2890 (roundtrip) | Completed, too few steps |
