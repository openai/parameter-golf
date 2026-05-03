Context
The OpenAI Parameter Golf competition optimizes a language model to minimize BPB (bits-per-byte) under tight constraints: 16MB artifact, 600s training on 8xH100, 600s eval. The current SOTA is 1.1194 BPB (Python/PyTorch). This plan rewrites the entire stack in Rust + CUDA to gain performance advantages from zero-overhead NCCL overlap, pre-compiled binary (no torch.compile warmup), fused custom kernels, and smaller artifact serialization.
Decisions Made

Deployment: Modal for submission, keep RunPod/local access for daily dev iteration (Modal H100:8 queue times can be 5-30 min)
Backward pass: Manual backward kernels (no autodiff library) — reused for both training and TTT
Hedge Mixer: Excluded for now — add as Phase 6b (~200 lines CPU-only Rust) if legality confirmed
GEMM strategy: cuBLASLt as default for all GEMM on H100. Run microbenchmark in Phase 1 for shapes (512×512, 512×1536, 1536×512 at batch ~384); only use CubeCL matmul if it matches within 5%. CubeCL reserved for fused element-wise kernels.

Architecture Decision: cudarc + CubeCL (not Burn)
Why not Burn? The SOTA code deliberately avoids nn.Linear — all weights live in 3D "parameter banks" (qo_bank [22, 512, 512], kv_bank [22, 256, 512], etc.) sliced per-layer and passed to F.linear. The Parallel Muon optimizer does async NCCL reduce-scatter → Newton-Schulz → all-gather on these banks. Burn's module system would fight both patterns.
Stack:

cudarc 0.19.3 — device management, cuBLASLt (GEMM), cuDNN 9.12 (flash attention), NCCL (multi-GPU), cuRAND
CubeCL 0.6.0 — fused element-wise kernels only (RMSNorm, RoPE, activations, XSA, STE, cross-entropy)
half 2.x — bf16/f16 types
zstd 0.13 — level-22 compression
rayon — CPU-side parallelism (GPTQ-lite clip search, data loading)
sentencepiece-sys — tokenizer for BPB metric

Cargo Workspace
parameter-golf-rs/
  Cargo.toml                     # workspace
  crates/
    pg-core/src/                 # GpuTensor, DevicePool, NCCL wrapper, multi-stream
    pg-kernels/src/              # All CUDA kernels (CubeCL + cudarc wrappers)
    pg-model/src/                # GPT with banks, U-Net forward, config
    pg-optim/src/                # Parallel Muon, AdamW, WSD scheduler, EMA
    pg-data/src/                 # Shard reader (mmap), TokenStream, BPB LUTs
    pg-quant/src/                # Int6 GPTQ-lite, unbank/rebank, zstd-22
    pg-eval/src/                 # Sliding window inside TTT loop, BPB
    pg-compat/src/               # PyTorch weight loading (safetensors/pickle), bank reshaping, dtype conversion
    pg-train/src/main.rs         # Entry point: distributed init, training loop
  modal_deploy.py                # Modal deployment script
  Dockerfile                     # CUDA 12.x + Rust binary
pg-compat is critical for the verification strategy: load known-good PyTorch checkpoints into Rust at any phase to isolate bugs. Also enables mid-training checkpoint injection if the Rust training diverges.
Implementation Plan
Phase 1: Core Infrastructure
Files: pg-core/src/{tensor.rs, buffer_pool.rs, nccl.rs, streams.rs, lib.rs}

GpuTensor struct: CudaSlice<u8> + shape/strides/dtype/offset, zero-copy slicing for bank views
Arena buffer pool with explicit memory budget:

Per-layer activations for backward: Q, K, V projections + attn output + MLP intermediates ≈ 6 × 48 × 2048 × 512 × 2B = ~600MB/layer
Activation checkpointing for layers 3–8: recompute during backward, only store layers 0–2 and 9–10 (saves ~3.6GB → use for larger micro-batch)
U-Net skip connections: layers 0–4 activations must persist until layers 6–10 consume them — arena must NOT reclaim these during forward
Optimizer states: Muon momentum buffers for each bank, AdamW first/second moments for scalar params
NCCL scratch: padded_grad, shard, full_update per bank (from SOTA lines 156–167)
Total estimate: ~40GB of 80GB HBM. Comfortable margin.


NCCL wrapper: init_process_group(rank, world_size), async reduce_scatter, all_reduce, all_gather returning NcclWork handles with .wait()
Three-stream management (not two):

Compute stream — GEMM, kernels, forward/backward
NCCL stream — all collective communication
Memcpy stream — host↔device transfers (data loading, checkpoint I/O)


Synchronize with CUDA events, not stream synchronize (which blocks CPU thread)


H100 GEMM microbenchmark: Run cuBLASLt vs CubeCL matmul for shapes [512×512, 512×1536, 1536×512] at batch=384 in bf16. Decide GEMM backend before proceeding.

Phase 2: Kernels
Files: pg-kernels/src/{rms_norm.rs, qk_rope_fused.rs, activations.rs, attention.rs, newton_schulz.rs, xsa.rs, quantize.rs, smear_gate.rs, bigram_hash.rs, cross_entropy.rs}
KernelImplementationNotesRMSNormCubeCLFuse with LN scale 1/sqrt(layer+1)Fused QK-norm+RoPE+q_gainCubeCLSingle kernel: RMSNorm(Q,K) → partial RoPE(16/64) → q_gain scale. Saves ~15μs launch overhead × 11 layers × 9000 steps = 1.5sLeakyReLU(0.5)²CubeCLy = leaky_relu(x, 0.5); y*y — single kernel, forward+backward fusedFlash AttentioncuDNN 9.12GQA (8Q/4KV), causal, head_dim=64, bf16. Fallback: compile FA2 CUDA to PTX, load via ctx.load_module() if cuDNN graph API too painful (budget 2-3 days)Newton-Schulz 5cuBLASLt strided batched GEMM5 iters, constants (3.4445, -4.7750, 2.0315), bf16 throughoutXSACubeCLSubtract self-value projection with GQA-aware reshape [B,T,Hkv,group,D]Int6 STE (fwd+bwd)CubeCLForward: abs→amax→scale→round→clamp→mul. Backward: grad_out * (abs(q_unclamped) <= 32.0) — fused into single kernel, not separate fwd/bwd launchesSmearGateCubeCL(1-σ(g))*x + σ(g)*x_prev — validate this helps before implementing (see note below)BigramHashCubeCLXOR hash lookup + embedding gatherCross-Entropy + softcapCubeCLFused classifier (llm.c trick): 30*tanh(logits/30) → softmax → log → NLL without materializing full logits tensor. Vocab=1024 so savings are moderate but still avoids a global memory round-trip
SmearGate validation note: SmearGate appears in the SOTA code but is not mentioned in any merged record's ablation table. Before implementing, run the Python SOTA with SmearGate disabled to confirm it contributes >0.001 BPB. If not, skip it — every unvalidated component is a bug surface.
Phase 3: Model Forward Pass + PyTorch Weight Loading
Files: pg-model/src/{config.rs, model.rs, block.rs, attention.rs, embeddings.rs}, pg-compat/src/{loader.rs, convert.rs}
Exact architecture from SOTA submission:

Config: 11 layers, d=512, 8H/4KV, MLP 3x, vocab 1024, seq 2048, partial RoPE 16/64, XSA last 4, VE128 on layers 9-10, BigramHash(1536, 128), logit softcap 30
Banks: qo_bank [22, 512, 512], kv_bank [22, 256, 512], mlp_up [11, 1536, 512], mlp_down [11, 512, 1536]
Forward: tok_emb + bigram → RMSNorm → SmearGate → U-Net (5 encoder + 6 decoder with skip connections) → final_norm → tied linear → softcap
Attention: QKV from banks → fused QK-norm+RoPE+q_gain → flash_attn → XSA(last 4) → VRL(v0 from layer 0) → output proj
MLP: x → linear(up_w) → LeakyReLU(0.5) → square → linear(down_w)
VE128: Value Embedding on layers 9-10 only. Verify if VRL-all-layers gives even 0.001 BPB improvement — if so, use it everywhere (negligible cost).
BigramHash(1536, 128): 197K params, not 4096 buckets. Smaller table compresses better with zstd.

CRITICAL: Load PyTorch SOTA weights immediately after forward pass is complete, before writing any backward kernels. This gives a known-good checkpoint to validate every subsequent phase against. Compare Rust logits vs PyTorch logits on identical input (max abs diff < 1e-3 in bf16).
Phase 4: Training Loop + Manual Backward Pass
Files: pg-optim/src/{muon.rs, adamw.rs, scheduler.rs, ema.rs}, pg-train/src/main.rs
Manual backward kernels (~15 distinct ops):

linear (dX = dY @ W, dW = dY^T @ X)
RMSNorm backward
LeakyReLU² backward: grad_input = grad_output * 2 * leaky_relu(x, 0.5) * d_leaky_relu(x, 0.5)
Flash Attention backward (cuDNN backward graph)
SmearGate backward (if validated)
BigramHash embed backward (sparse gradient accumulation)
Softcap cross-entropy backward
XSA backward (projection subtraction gradient)
VRL backward (simple linear combination gradient)

Parallel Muon 3-phase pipeline:

After backward: launch async reduce-scatter for all 4 banks (sorted by size desc)
While RS in flight: all-reduce scalar grads + AdamW step on non-bank params
For each bank: wait RS → NS5 on local shard → async all-gather; AG overlaps with next bank's NS5

Hyperparameters (exact from SOTA):

MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035, HEAD_LR=0.008
MUON_MOMENTUM=0.99, warmup 0.92→0.99 over 1500 steps, MUON_WD=0.04
AdamW β1=0.9, β2=0.95, WD=0.04
WSD: warmup 20 steps, stable until step 5500, warmdown 3500 steps to 0
Batch: 786K tokens/step, seq_len=2048, ~9000 iterations, 600s wall cap
EMA(0.997) every step
SWA only during warmdown (step 5500+), every 50 steps. NOT during stable phase — consecutive checkpoints are too different for averaging to help.

Gradient accumulation: 786K tokens / 8 GPUs / 2048 seq = ~48 seqs/GPU. Verify this fits in HBM with activation checkpointing. If not, use 2 micro-batches of 24 seqs. NCCL allreduce after accumulation, not per micro-batch.
Late QAT: Activate int6 STE fake quantization when LR scale drops below 0.15 (~last 1400 steps). Adds ~70% per-step overhead but critical for post-quantization quality.
Failure recovery: Save checkpoint to host RAM at step 5500 (end of stable phase, before warmdown). If training diverges during warmdown, restart from this checkpoint with different warmdown hyperparameters without retraining from scratch. Host RAM checkpoint is fast (~1s), no disk I/O.
Phase 5: Quantization + Serialization
Files: pg-quant/src/{int6.rs, banking.rs, compress.rs, serialize.rs}

Export EMA weights
Unbank: 3D banks → individual 2D weight tensors
GPTQ-lite int6: for each row, try 5 clip percentiles {0.999, 0.9995, 0.9999, 0.99999, 1.0}, pick min MSE. Parallelize with rayon.
Quick eval validation: Run 100-document eval on quantized model. If BPB degraded >0.003 from FP EMA checkpoint, QAT threshold was too conservative (lower from 0.15 to 0.20).
Custom binary format (no pickle): [magic][version][num_tensors][per-tensor: name_len, name, ndim, shape, dtype, **CRC32**, data] — CRC32 per tensor catches silent bit corruption in compressed data
zstd level 22 compression
Verify artifact_size ≤ 16MB

Phase 6: Evaluation
Files: pg-eval/src/{ttt.rs, bpb.rs}
CRITICAL: Sliding window is INSIDE the TTT loop, not a separate phase. The correct structure:
for chunk in validation.chunks(32768) {
    // Phase 1: SCORE this chunk with stride-64 sliding window (no gradients)
    for window in chunk.sliding_windows(2048, stride=64) {
        let logits = model.inference_forward(window);
        record_bpb(logits, only_last_stride_tokens);
    }

    // Phase 2: TRAIN on already-scored chunk (SGD, 3 epochs)
    // All 8 GPUs: data-parallel with allreduced gradients
    for epoch in 0..3 {
        let loss = model.train_forward(chunk);
        loss.backward();
        nccl.all_reduce(gradients);  // TTT needs multi-GPU gradient sync
        sgd_step(lr=cosine_decay(chunk_idx), momentum=0.9, grad_clip=1.0);
    }
}
// Last chunk: scored but never trained on
TTT GPU parallelism: All 8 GPUs must see identical model weights throughout TTT. The scoring phase partitions windows across GPUs (data-parallel inference). The training phase runs data-parallel SGD with allreduced gradients. Budget NCCL overhead for the per-chunk allreduce.
BPB Metric:

Load SentencePiece model, build byte-count LUTs (base_bytes, has_leading_space, is_boundary)
bpb = (loss / ln2) * (tokens / bytes) — tokenizer-agnostic

Phase 6b: Hedge Mixer (CONDITIONAL — if legality confirmed)
~200 lines CPU-only Rust, no GPU involvement. Low-risk late addition.

Count-min sketch (~4M XOR-hash buckets) for n-gram statistics (orders 2–7)
Multiplicative weights update (Hedge algorithm) mixing neural + n-gram experts
Worth 0.10–0.16 BPB at zero artifact cost

Phase 7: Modal Deployment
Files: modal_deploy.py, Dockerfile

Dockerfile: nvidia/cuda:12.4-devel-ubuntu22.04 base, copy Rust binary + data
Modal: @app.function(gpu="H100:8", timeout=1200) — 600s train + 600s eval
Modal volume for data: mount fineweb10B_sp1024 shards + tokenizer
Environment: RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
Cross-compile: cargo build --release --target x86_64-unknown-linux-gnu
Modal image: modal.Image.from_dockerfile("Dockerfile")

Rust Performance Advantages (Calibrated)

NCCL overlap: Python GIL serializes async handle management. Rust does true zero-overhead async NCCL + cuBLAS on separate streams. Est. 3–7% faster training (bottleneck is cuBLAS/cuDNN which are identical between Python and Rust; Rust wins in the gaps: launch overhead, handle management, CPU scheduling).
No torch.compile warmup: Python wastes 15–25s on first compilation (mode="max-autotune"). At 65ms/step, that's 230–380 extra steps.
Fused kernels: QK-norm+RoPE+q_gain (saves 1.5s total), Int6 STE (5 ops → 1), cross-entropy+softcap.
Smaller artifact: Custom binary format without pickle — delta is ~10–50KB, not transformative but helps at the margin.
Arena allocation: Zero CUDA malloc during training after init.

Verification Strategy

H100 GEMM benchmark: cuBLASLt vs CubeCL on target shapes before committing (Phase 1)
Forward pass parity: Load PyTorch weights → Rust forward → compare logits (max abs diff < 1e-3 in bf16) — do this immediately after Phase 3, before any backward work
Gradient parity: Compare Rust manual backward vs PyTorch autograd on small model (same input, same weights)
Training curve: Single-GPU 500-step run should match Python's loss trajectory within 2%
NCCL determinism: Verify allreduce produces bitwise-identical results across runs (required for 3-seed significance test)
Quantization round-trip: Rust quantize → compress → decompress → dequantize → eval BPB should match Python's within 0.001
Post-quant quick eval: 100-document sanity check after GPTQ-lite (Phase 5 step 4)
Multi-GPU: 8-GPU run should reach ~1.122 pre-TTT BPB
TTT: Post-TTT should reach ~1.119 BPB
Artifact size: Verify ≤ 16MB after zstd-22

Future Ablation Experiments
Once baseline Rust training matches Python SOTA:

Warmdown length: 3000 vs 3500 vs 4000 steps
QAT activation threshold: 0.10 vs 0.15 vs 0.20
EMA decay: 0.995 vs 0.997 vs 0.999
BigramHash buckets: 1024 vs 1536 vs 2048
Depth recurrence: shallow (layers 4+5 repeated once, 11→13 virtual layers) — achieved 1.1182 BPB in pending PRs but conflicts with TTT
VRL scope: layers 9-10 only vs all layers
Run as 500-step abbreviated sweeps (1/18th cost) to bracket before full 9000-step run

Plan Addendum (April 6, 2026 — Post-Professor Sync)
Section 0: Competition State Update
The competition has bifurcated since the original plan was drafted on March 24:
Pure neural track:

Best merged: 1.1194 BPB (abaybektursun)
Best non-TTT pending: 1.1154 BPB (PR #609 — XSA-all + Full GPTQ + Selective Pruning)
Floor appears to be ~1.11 BPB
Our target: ≤ 1.11 BPB on pure neural

Bandit / n-gram hybrid track:

Frontier dropped from sub-1.0 BPB → sub-0.50 BPB (PR #1083: 0.4961)
Uses X-WING shared n-gram tables, Cubric 3D per-order adaptive alpha, entropy-adaptive alpha 0.20–0.75
Not yet merged: legality remains under organizer review
Rust is especially well-suited here: lock-free count-min sketch, SIMD histograms, CPU-side n-gram updates parallel to GPU training

New techniques added since March 24 (all integrated below):

qTTT (arXiv:2512.13898) — query-only test-time training, cache K/V once, adapt only Q. Strictly better than SGD TTT inside the eval budget (2–3× more epochs possible).
LaCT (arXiv:2505.23884) — Large Chunk TTT, document-sized chunks, 70% GPU utilization vs <5%.
Complementary Training — bigram-predictable tokens get lower training loss weight; the neural model specializes on what n-grams cannot predict; enables higher eval-time n-gram α.
Order-Adaptive Entropy Gating (PR #774) — per-order entropy thresholds. Building block for sub-0.55 hybrid submissions.
Prune-then-quantize ordering (arXiv:2603.18426) — "Progressive Intensity Hypothesis": weaker perturbations (pruning) precede stronger (quantization). Zero-cost experiment.
SLOT eval-time augmentation (PR #1084) — 1.1185 BPB on the SOTA stack (3-seed std 0.0003).
SmearGate validation — confirmed beneficial only with OrthoInit as a co-dependency. Re-enable both together.

Implications for the plan:

Replace Phase 6 standard SGD TTT with qTTT as the default, falling back to LaCT for the chunk loop.
Bake Complementary Training into the loss-weight pipeline (Phase 4) — needed for both pure and hybrid tracks.
Move SmearGate from "optional / validate first" to "required, with OrthoInit co-dependency".
Add SLOT as an opt-in eval-time augmentation pass (Phase 6) — cheap, multiplicative.
Add prune-then-quantize ordering as an experiment in Phase 5.

Innovation 1: Quantization-Scheme Compiler (Phase 5b)
A #[derive(QuantKernels)] proc macro that takes a compile-time Scheme description and emits all the kernels needed to train, serialize, and evaluate that scheme:

fake_quant_fwd_<bits>_<block> — CubeCL forward STE
fake_quant_bwd_<bits>_<block> — CubeCL backward STE (clamp mask)
pack_<bits>_<block> — CPU-side bit packing
unpack_<bits>_<block> — CubeCL dequant kernel
gptq_lite_<bits>_<block> — CPU-side clip percentile search

Configuration space (≈1,800 raw, ~50 meaningful):

bit-width per group: int4/int5/int6/int7/int8
block size: 16 / 32 / 64 / 128 / per-row
clip strategy: percentile {0.999, 0.9995, 0.9999, 0.99999, 1.0}
attn vs mlp split: int5 MLP / int6 attn (current SOTA), inverted, uniform, graduated
embedding bit-width: int6 / int8 / fp16

Risk mitigation: Hand-write the first 3 schemes (int5/PerRow, int6/PerRow, int8/PerRow) before extracting the macro. The macro is a productivity tool, not a prerequisite. Generate CubeCL kernel source as string literals compiled at build time to avoid macro-in-macro pain.
Sweep harness:

~50 schemes × 30s quick-eval (100 docs) = ~25 min on 1 GPU
Filter by compressed_size <= 16MB first
rayon::par_iter() over the surviving configs
Output: Vec<(SchemeConfig, BPB, ArtifactBytes)> Pareto frontier

Innovation 2: Three Kernel Fusions (Phase 2 — start hand-written, fuse incrementally)
FusionPer-step savingsTotal over 9000 stepsComplexityA: XSA + residual + RMSNorm~50μs × 4 layers~1.8sMediumB: BigramHash + token embedding~30μs × 1~0.27sLowC: RMSNorm + QK projection + Partial RoPE + q_gain~20μs × 11 layers~1.98sMediumTotal~450μs~4.05s ≈ 62 extra steps
Implementation order (after baseline kernels work):

Fusion C first (simplest, biggest aggregate win, runs once per layer)
Fusion B second (lowest risk, small win)
Fusion A last (most ambitious — uses cuBLASLt epilogue for GEMM-fusion path; fallback is "XSA + residual + norm" single-kernel without GEMM fusion, which is the realistic version)

Fusion C is the headline: #[comptime] parameters bake rope_dims=16 and head_dim=64 into the kernel as immediates, eliminating loop bound checks.
Combined value: ~412 extra steps (fusions ~62 + warmup elimination ~300 + NCCL overlap ~50). Enough to potentially train a wider model (d=576) in the same wall budget — exactly the question that Innovation 3's sweep is designed to answer.
Innovation 3: Const-Generic Architecture Specialization (Phase 6c)
rustpub struct Arch<
    const D: usize,          // 384, 512, 576, 640
    const HEADS: usize,      // 6, 8, 10, 12
    const KV_HEADS: usize,   // 2, 4
    const LAYERS: usize,     // 9..16
    const MLP_MULT: usize,   // 2, 3, 4
    const ROPE_DIMS: usize,  // 8, 16, 24, 32
    const XSA_FROM: usize,   // 0..LAYERS  (LAYERS = disabled)
    const SEQ_LEN: usize,    // 1024, 2048, 4096
> { _phantom: PhantomData<()> }

pub type BaselineArch = Arch<512, 8, 4, 11, 3, 16, 7, 2048>;
pub type WideArch     = Arch<576, 8, 4, 11, 3, 16, 7, 2048>;
pub type DeepArch     = Arch<512, 8, 4, 13, 3, 16, 9, 2048>;
// ... 10 total variants
Every kernel that depends on architecture parameters takes them as #[comptime] CubeCL params. The compiler folds for d in 0..ROPE_DIMS to a fully unrolled loop of exactly 16 iterations, and if layer_idx >= XSA_FROM is per-layer constant-folded.
Sweep: 10 variants × (~2s CubeCL compile + ~32s 500-step run) = ~340s on 1 GPU. (Compare PyTorch's ~520s due to torch.compile overhead — Rust saves ~180s.)
Configurations of interest:

Baseline (512/11), Wide (576/11), Wide-NarrowMLP (576/11/2× MLP)
Deep-Narrow (384/16), Deep-Medium (448/14)
Full-XSA (XSA_FROM=0), More-RoPE (32 dims)
High-GQA (12H/3KV), Long-ctx (4096)
Aggressive-quant pair: (640/11 @ int4) — only viable if Innovation 1's sweep validates int4

Section 4: Per-Step Equivalence Validation (TA feedback)
Before any ablation runs, prove Rust ≡ PyTorch on a single training step:
rustfn validate_per_step_equivalence() {
    let weights = pg_compat::load_pytorch_weights("sota_checkpoint.pt");
    let batch = load_batch(seed=42, step=0);
    let (rust_loss, rust_grads) = rust_train_step(&weights, &batch);
    let (torch_loss, torch_grads) = pytorch_train_step(&weights, &batch);
    assert!((rust_loss - torch_loss).abs() < 1e-3);
    for (name, (rg, tg)) in rust_grads.iter().zip(torch_grads.iter()) {
        let rel_diff = (rg - tg).abs().max() / tg.abs().max().max(1e-7);
        assert!(rel_diff < 0.02, "{}: rel_diff={}", name, rel_diff);
    }
}

fn validate_loss_trajectory() {
    let rust_losses = rust_train(steps=200, seed=42);
    let torch_losses = pytorch_train(steps=200, seed=42);
    for step in 20..200 {
        let rel_diff = (rust_losses[step] - torch_losses[step]).abs() / torch_losses[step];
        assert!(rel_diff < 0.02);
    }
}
Known divergence sources to control:

cuBLASLt GEMM algorithm selection — force the same algorithm via cublasLtMatmulAlgoGetHeuristic with a fixed preference
cuDNN SDPA tile order — deterministic but implementation-specific; flag as "expected within edge tolerance"
NCCL allreduce ordering — force NCCL_ALGO=Ring on both sides

This is a verification gate sitting between Phase 4 (training loop) and Phase 5 (quantization). It runs in CI; if it fails, training is forbidden until parity is restored.
Section 5: Updated Implementation Timeline (weeks 2–6)
WeekInnovation workIntegrates with2Hand-write int5/int6/int8 STE kernels + packingpg-kernels, pg-quant3Fusion C (RMSNorm+QK+RoPE+q_gain post-GEMM)pg-kernels4Fusion B (BigramHash+embed)pg-kernels4Per-Step Equivalence Validation harnesspg-train, pg-compat5Extract #[derive(QuantKernels)] proc macropg-quant5Fusion A (XSA+residual+RMSNorm)pg-kernels5Quantization sweep (~50 configs, rayon parallel)pg-quant6Const-generic Arch<...> + 10-variant arch sweeppg-model, pg-train6qTTT + LaCT TTT replacementpg-eval6Complementary Training loss weightingpg-train6Prune-then-quantize ordering experimentpg-quant6SLOT eval-time augmentation hookpg-eval6Full ablation: Configs A, B, C, Dpg-train, pg-eval
Code Review Fix Plan
Batch 1 — P0 Correctness Bugs (must fix before any training)
Fix 1.1: Eval softcap missing (pg-eval/src/sliding.rs)

eval_sliding and score_chunk compute NLL directly from raw logits without applying softcap
buf.logits stores pre-softcap values; training loss applies softcap inside cross_entropy_forward
Fix: Apply cap * tanh(logit / cap) to each logit before the max_logit/sum_exp/NLL computation
Extract a helper fn nll_with_softcap(logits: &[f32], target: usize, softcap: f32) -> f32 and use it in both eval_sliding and score_chunk
Files: pg-eval/src/sliding.rs

Fix 1.2: SmearGate backward missing (pg-model/src/backward.rs)

Lines 637-642: backward through SmearGate is a comment stub — gradient passes through as identity
SmearGate forward: output[i] = (1-σ(g[d]))*x[i] + σ(g[d])*x_prev[i]
Backward needs:

grad_x_pre_smear[i] = grad_x[i] * (1 - σ(g[d])) (grad w.r.t. x input)
grad_x_prev[i] = grad_x[i] * σ(g[d]) (flows to shifted x, which is x[i-1])
grad_g[d] += grad_x[i] * σ(g[d]) * (1-σ(g[d])) * (x_prev[i] - x[i]) (gate param gradient)


Also need backward through initial RMSNorm between SmearGate output and embeddings
Fix: Implement full SmearGate backward in backward() between encoder backward and embedding scatter. Recompute x_pre_smear (pre-smeargate hidden state) from cache. Accumulate gate grad into grads.smear_gate. Then pass grad_x_pre_smear through initial RMSNorm backward before embedding scatter.
Need to save pre-smeargate x in ForwardCache (after initial RMSNorm, before SmearGate)
Files: pg-model/src/backward.rs

Fix 1.3: Bigram backward missing (pg-model/src/backward.rs)

Forward: x += bigram_scale * (bigram_proj @ bigram_embed[hash(tokens)])
Backward needs:

grad_bigram_proj_out = grad_x_pre_norm * bigram_scale (before initial RMSNorm)
grad_bigram_proj: weight grad via linear_backward_weight
grad_bigram_embed: scatter from linear_backward_input through projection
grad_bigram_scale += sum(grad_x_pre_norm * bigram_proj_out)


Requires saving bigram_proj_out (or recomputing) in cache
Fix: After RMSNorm backward in backward(), compute bigram backward chain. Accumulate into grads.bigram_embed, grads.bigram_proj, grads.bigram_scale.
Files: pg-model/src/backward.rs

Fix 1.4: Skip connection gradient not propagated to encoder (pg-model/src/backward.rs)

Lines 611-621: skip gradient is computed for grads.skip_weights but NOT propagated back through the encoder skip output
Forward: decoder_x += skip_weight * encoder_skip_output
Backward: grad_encoder_skip = grad_decoder_x * skip_weight — this must be added to the encoder layer's output gradient
Fix: Accumulate grad_skip = grad_x[j] * skip_weight[di] into a grad_encoder_skips: Vec<Vec<f32>> array (one per skip connection). After all decoder blocks are done, before encoder backward, add each grad_encoder_skips[enc_layer] to grad_x at the appropriate point.
Actually simpler: since encoder backward processes layers in reverse, and skip i connects encoder layer n_enc - 1 - i to decoder layer i, we need to inject the skip gradient between encoder layer backward calls. Restructure the encoder backward loop to add grad_skip after processing the encoder layer that produced it.
Files: pg-model/src/backward.rs

Batch 2 — P1 Correctness + Performance
Fix 2.1: XSA grad_v dropped (pg-model/src/backward.rs)

Lines 480-486: grad_v_xsa from XSA backward is computed but never added to the V gradient path
Fix: After XSA backward, add grad_v_xsa to grad_v_attn before the V projection backward:

  // After causal_attention_backward:
  for i in 0..t * hkv * hd {
      grad_v_attn[i] += grad_v_xsa[i];
  }

Files: pg-model/src/backward.rs

Fix 2.2: tok_emb grad overwrite fragility (pg-model/src/backward.rs:671)

linear_backward_weight overwrites (assigns, not accumulates) into grads.tok_emb
Currently works because grads are zeroed and this runs first, but fragile
Fix: Use a local scratch buffer grad_tok_emb_logits, compute via linear_backward_weight, then += into grads.tok_emb:

rust  let mut grad_tok_emb_logits = vec![0.0f32; vocab * d];
  linear_backward_weight(&grad_logits, &final_normed, &mut grad_tok_emb_logits, t, vocab, d);
  for i in 0..vocab * d {
      grads.tok_emb[i] += grad_tok_emb_logits[i];
  }

Files: pg-model/src/backward.rs

Fix 2.3: Double forward pass (pg-model/src/backward.rs)

forward_with_cache runs self.forward() then self.forward_capture() — two full forwards
Fix: Merge into a single forward_with_cache that captures the cache while also filling buf for loss computation. Delete the separate forward_capture method. Modify forward to optionally accept a &mut Option<ForwardCache> parameter, or better: have forward_with_cache be the primary implementation and have forward call it with None.
Concretely: refactor forward to call a shared forward_inner(input_ids, buf, cache: Option<&mut ForwardCache>) that conditionally saves layer states
Files: pg-model/src/backward.rs, pg-model/src/model.rs

Fix 2.4: Missing VE/bigram optimizer states (pg-train/src/main.rs)

No AdamW states for ve_embed, ve_proj, ve_scale, ve_layer_scales, bigram_scale
These parameters have gradient buffers but are never updated
Fix: Add AdamW states and step calls for all VE params and bigram_scale. Use embed_lr_scaled for VE embed, scalar_lr_scaled for the rest.
Files: pg-train/src/main.rs

Batch 3 — P2 Performance + Robustness
Fix 3.1: 140MB alloc per step for EMA (pg-train/src/main.rs:157)

flatten_params creates a new Vec every step
Fix: Pre-allocate flat_buf: Vec<f32> once before the loop. Use flatten_params_into(&model, &mut flat_buf) that writes into the existing buffer. Same for SWA.
Alternatively, update EMA in-place by iterating over each parameter group directly (avoids the flatten/unflatten entirely). This is cleaner:

rust  ema.update_group(&model.tok_emb);
  ema.update_group(&model.qo_bank);
  // ...
Where Ema tracks an internal offset and updates each group's shadow slice.

Chosen approach: Pre-allocate flat buffer + flatten_into. Simpler, and the EMA offset-tracking approach is error-prone with the mixed scalar/vector params.
Files: pg-train/src/main.rs

Fix 3.2: AdamW reconstructed every step (pg-train/src/main.rs:138)

Two new AdamW structs per step (one for embed LR, one for scalar LR)
Fix: Create persistent adamw_embed and adamw_scalar before the loop. Update adamw_embed.lr and adamw_scalar.lr each step instead of constructing new instances.
Files: pg-train/src/main.rs

Fix 3.3: TrainConfig::wsd_lr underflow (pg-model/src/config.rs)

warmdown_start = self.total_iterations - self.warmdown_iters — wraps on underflow
Fix: Use self.total_iterations.saturating_sub(self.warmdown_iters) matching scheduler.rs
Files: pg-model/src/config.rs

Fix 3.4: Per-token Vec alloc in cross_entropy backward (pg-kernels/src/cross_entropy.rs:82)

let mut exps = vec![0.0f32; vocab_size] allocates 4KB per token
With 2048 tokens per step: 8MB of throwaway allocations
Fix: Accept a pre-allocated scratch buffer parameter, or since vocab=1024 is small, use a fixed-size stack array:

rust  // vocab_size is always 1024, use stack allocation
  let mut exps = [0.0f32; 1024];
But this hardcodes vocab. Better: add a scratch: &mut [f32] parameter to cross_entropy_backward and allocate it once in the caller. Or use a thread-local scratch buffer.

Chosen approach: Add scratch: &mut Vec<f32> to backward, pre-allocated by caller. The Vec is reused across tokens within the same call since we process tokens sequentially.
Files: pg-kernels/src/cross_entropy.rs, callers in pg-model/src/backward.rs

Fix 3.5: ForwardBuffer reallocation in eval (pg-eval/src/sliding.rs:46-49)

Buffer is reallocated when window size changes
Fix: Always allocate for seq_len (the max window). Pass wlen as the actual token count to forward, which already uses buf.tokens. Just update buf.tokens = wlen instead of reallocating. Need to check that ForwardBuffer supports resizing tokens without reallocation — it should since all buffers are pre-allocated to max size.
Actually, ForwardBuffer::new allocates based on tokens. We need a buf.set_tokens(wlen) that just updates the tokens field without reallocating. All buffer accesses already use buf.tokens for bounds.
Fix: Add pub fn resize_tokens(&mut self, tokens: usize) that asserts tokens <= self.max_tokens and updates self.tokens. Allocate with max seq_len once.
Files: pg-eval/src/sliding.rs, pg-model/src/model.rs

Fix 3.6: token_stream.rs — next_batch reads data for all ranks (pg-data/src/token_stream.rs:108-109)

Each rank reads per_rank_span * world_size tokens then slices
Fix: Skip to the rank's offset in the stream. Each rank should call stream.skip(self.rank * per_rank_span) then stream.take(per_rank_span). Add a skip method to TokenStream.
Files: pg-data/src/token_stream.rs

Batch 4 — Minor / Cleanup
Fix 4.1: NS5 scratch allocations (pg-optim/src/muon.rs)

5 full-size scratch buffers allocated per step_bank call
Fix: Add pre-allocated scratch buffers to MuonBankState. Initialize them in Muon::new.
Files: pg-optim/src/muon.rs

Fix 4.2: block_forward returns Option<Vec<f32>> for dead VRL code

VRL is disabled in SOTA config; the raw_v return is always None
Fix: Remove the return value. If VRL is needed later, re-add it. The allocation and branch are unnecessary noise.
Files: pg-model/src/model.rs

Fix 4.3: gpu.rs memory estimate missing NS5 workspace

Fix: Add NS5 scratch to estimate_memory: for each bank [B,M,N], NS5 needs a_buf [B,M,M], aa [B,M,M], b_buf [B,M,M], new_x [B,M,N] — approximately 3*B*M*M + B*M*N elements × 2 bytes (BF16).
Files: pg-model/src/gpu.rs

Fix 4.4: ForwardCache should store pre-SmearGate and pre-RMSNorm states

Required by fixes 1.2 and 1.3 for SmearGate and bigram backward
Fix: Add x_post_embed: Vec<f32> (after embedding + bigram, before initial RMSNorm) and x_post_norm: Vec<f32> (after initial RMSNorm, before SmearGate) to ForwardCache. Save them during forward_with_cache.
Files: pg-model/src/backward.rs

Execution Order
All fixes applied in dependency order:

Fix 3.3 (wsd_lr underflow) — trivial one-liner, no deps
Fix 4.4 (ForwardCache new fields) — needed by 1.2, 1.3, 2.3
Fix 2.3 (merge double forward) — restructure forward_with_cache, needed by everything downstream
Fix 1.2 (SmearGate backward) — uses new cache fields
Fix 1.3 (Bigram backward) — uses new cache fields
Fix 1.4 (Skip gradient propagation) — restructure encoder backward
Fix 2.1 (XSA grad_v) — small change in block_backward
Fix 2.2 (tok_emb grad overwrite) — small change in backward_output_loss
Fix 1.1 (Eval softcap) — independent fix in pg-eval
Fix 2.4 (Missing VE optimizer states) — training loop addition
Fix 3.1 (EMA alloc) — training loop optimization
Fix 3.2 (AdamW reconstruct) — training loop optimization
Fix 3.4 (CE scratch buffer) — kernel + caller change
Fix 3.5 (ForwardBuffer resize) — eval optimization
Fix 3.6 (TokenStream skip) — data loader optimization
Fix 4.1 (NS5 scratch) — optimizer optimization
Fix 4.2 (Remove VRL return) — cleanup
Fix 4.3 (GPU memory estimate) — estimate fix

Verification
After all fixes:

cargo test --workspace — all existing tests pass
New test: test_numerical_gradient_tok_emb should have MUCH tighter tolerance (< 10% relative error) now that SmearGate/bigram/skip/XSA gradients flow correctly
New test: test_smear_gate_grad_nonzero — verify grads.smear_gate is non-zero after backward
New test: test_bigram_grad_nonzero — verify grads.bigram_embed, grads.bigram_proj, grads.bigram_scale are non-zero
New test: test_skip_grad_propagates — 2-layer model (1 enc + 1 dec), verify encoder layer gets gradient through skip
New test: test_eval_softcap_matches_training — compare eval NLL with cross_entropy_forward NLL on same input
Run a 10-step training loop on tiny model, verify loss decreases (smoke test that all gradients flow)

Implementation Order (Priority)

Code review fixes (Batches 1-4 above) — before any further implementation
pg-core + pg-data (read actual data shards, H100 GEMM benchmark)
pg-kernels (cuBLASLt wrappers, fused QK-norm+RoPE, RMSNorm, activations, attention)
pg-compat (PyTorch weight loader)
pg-model forward pass → immediately validate against PyTorch checkpoint
Manual backward kernels → validate gradients against PyTorch autograd
pg-optim single-GPU training → verify loss curve matches Python
Multi-GPU NCCL (Parallel Muon pipeline, NCCL determinism check)
EMA + SWA(warmdown-only) + Late QAT + WSD schedule + failure checkpoint at step 5500
pg-quant (GPTQ-lite, CRC32 serialization, compression, quick-eval validation)
pg-eval (TTT with sliding window inside loop, multi-GPU TTT with allreduced gradients)
Modal deployment + end-to-end test


# Technical Design Document: Rust-Enabled Innovations for Parameter Golf

**Author:** Cedric | **Date:** April 6, 2026 | **Status:** Implementation-ready

---

## 0. Competition State Update (as of April 6, 2026)

### What changed since our initial plan (March 24)

The competition has undergone a dramatic shift. The key developments:

**The leaderboard was restructured.** The official README leaderboard now shows only the naive baseline (1.2244 BPB). The record submissions still exist in the `records/track_10min_16mb/` directory — they haven't been deleted — but the README table was cleaned up, likely because the organizers are restructuring how records are displayed as the competition matures. All the merged records from March 17–23 (signalrush at 1.1228, jfprincz at 1.1248, abaybektursun at 1.1194, etc.) remain in the repo as merged directories.

**The n-gram bandit track has exploded.** The frontier has moved from sub-1.0 BPB to **sub-0.50 BPB**. The Issue #140 live commentary (last updated March 30) tracks submissions including:
- PR #1083: "ClownCar Crawler × Cubric Ngram9" at **0.4961 BPB** (9.9MB artifact)
- PR #795: 0.8881 BPB
- PR #788: 0.9059 BPB
- Several submissions in the 0.55–0.97 range using order-adaptive entropy gating with n-gram orders 2–11

These use "X-WING" shared n-gram tables (all 8 GPU ranks updating tables with the same tokens for a full 62M-token view), "Cubric" 3D per-order adaptive alpha, and entropy-adaptive alpha ranging from 0.20–0.75.

**The competition has bifurcated into two distinct tracks:**
1. **Pure neural track:** Best merged is 1.1194 BPB (abaybektursun). Best non-TTT pending is 1.1154 BPB (PR #609: XSA-all + Full GPTQ + Selective Pruning). Pure neural appears to have a floor around 1.11 BPB.
2. **Bandit/n-gram hybrid track:** Submissions reaching sub-0.50 BPB. These are not yet merged — legality remains under organizer review. The organizers have not made a definitive ruling on n-gram cache mixing.

**New techniques since March 24:**
- **Complementary Training:** Tokens predictable by bigram statistics get lower loss weight during training, so the neural model specializes on what n-grams can't predict. This enables higher eval-time n-gram alpha (20–75%).
- **Order-Adaptive Entropy Gating:** Per-order entropy thresholds from PR #774, used as building blocks for the sub-0.55 submissions.
- **qTTT (query-only test-time training):** From arXiv:2512.13898 — cache K/V once, adapt only Q projection weights. Enables 2–3× more TTT epochs within the eval budget.
- **LaCT (Large Chunk TTT):** From arXiv:2505.23884 (ICLR 2026 Oral) — document-sized chunks achieving 70% GPU utilization vs <5% for per-token TTT.
- **Prune-then-quantize ordering:** From arXiv:2603.18426 (ICLR 2026) — the "Progressive Intensity Hypothesis" suggests weaker perturbations (pruning) should precede stronger ones (quantization).
- **SLOT eval-time augmentation:** PR #1084 achieves 1.1185 BPB (3-seed mean, std 0.0003) on the SOTA stack.
- **SmearGate validated:** OrthoInit confirmed as critical co-dependency. SmearGate without OrthoInit hurts BPB.

**Implications for our plan:**
- Our pure-neural target of ≤1.11 BPB remains competitive for the merged leaderboard
- The n-gram hybrid track is the real frontier, and Rust is actually well-suited for it (CPU-side count-min sketch construction, lock-free concurrent hash tables, SIMD-accelerated histogram computation)
- qTTT should replace standard SGD TTT in our eval pipeline — it's strictly better within the eval time budget
- The prune-then-quantize ordering is a zero-cost experiment we should run
- Complementary Training is directly relevant to Innovation 1 (quantization-scheme compiler) — we should explore loss-weighting schemes as part of the search

---

## 1. Innovation 1: Quantization-Scheme Compiler via Procedural Macros

### 1.1 Problem Statement

The competition's quantization design space has the following dimensions:

| Dimension | Range | Current SOTA choice |
|-----------|-------|-------------------|
| Bit-width per weight type | int4, int5, int6, int7, int8 | int5 MLP / int6 attention |
| Block size for scaling factors | 16, 32, 64, 128, per-row | Per-row |
| Clipping strategy | Row-max, percentile (5 values), learned | Percentile (GPTQ-lite) |
| MLP vs attention allocation | Uniform or split | Split (int5/int6) |
| Embedding quantization | int6, int8, fp16 | int8 |
| Per-layer bit-width | Uniform or graduated | Uniform per type |

The total number of meaningful configurations is roughly 5 × 4 × 3 × 5 × 3 × 2 = **1,800**. Current competitors evaluate at most 5–10 by hand because each configuration requires rewriting:
1. The STE forward kernel (different clamping bounds per bit-width)
2. The STE backward kernel (different gradient mask)
3. The bit-packing serialization code
4. The dequantization kernel for inference
5. The GPTQ-lite clip search (different quantization levels per bit-width)

In Python/CUDA, that's 3–4 hours of kernel development per configuration.

### 1.2 Architecture

```
┌─────────────────────────────────────────────────┐
│            User-Facing Configuration             │
│                                                  │
│  QuantScheme {                                   │
│    attn_qkvo: IntConfig<6, Block::PerRow>,       │
│    mlp_gate_up: IntConfig<5, Block::B32>,        │
│    mlp_down: IntConfig<5, Block::PerRow>,         │
│    embeddings: IntConfig<8, Block::PerRow>,       │
│    clip: ClipStrategy::Percentile([0.999,         │
│           0.9995, 0.9999, 0.99999, 1.0]),        │
│  }                                               │
└───────────────────┬─────────────────────────────┘
                    │ compile-time
                    ▼
┌─────────────────────────────────────────────────┐
│         Procedural Macro Expansion               │
│                                                  │
│  #[derive(QuantKernels)]                         │
│  Generates:                                      │
│    • fake_quantize_fwd_int5_perrow()  (CubeCL)   │
│    • fake_quantize_bwd_int5_perrow()  (CubeCL)   │
│    • pack_int5_perrow()               (Rust)     │
│    • unpack_int5_perrow()             (CubeCL)   │
│    • gptq_lite_int5_perrow()          (Rust+SIMD)│
│    ... repeated for each unique config           │
└───────────────────┬─────────────────────────────┘
                    │ runtime
                    ▼
┌─────────────────────────────────────────────────┐
│         Generated Training + Eval Kernels        │
│                                                  │
│  Training:                                       │
│    forward → fake_quant_fwd per layer type       │
│    backward → STE identity + clamp mask          │
│                                                  │
│  Post-training:                                  │
│    GPTQ-lite per layer type (rayon parallel)     │
│    pack → serialize → zstd-22                    │
│                                                  │
│  Evaluation:                                     │
│    unpack → dequantize → inference               │
└─────────────────────────────────────────────────┘
```

### 1.3 Procedural Macro Design

The core macro is `#[derive(QuantKernels)]` applied to a configuration struct. It inspects the const generic parameters and generates specialized kernel code for each unique bit-width + block-size combination.

```rust
// pg-quant/src/config.rs

/// Marker types for bit-width (compile-time)
pub struct B4;
pub struct B5;
pub struct B6;
pub struct B7;
pub struct B8;

/// Marker types for block size
pub struct PerRow;
pub struct Block32;
pub struct Block64;
pub struct Block128;

/// A single quantization configuration for one weight group
pub struct IntConfig<Bits, BlockSize> {
    _bits: PhantomData<Bits>,
    _block: PhantomData<BlockSize>,
}

/// Full scheme specification — the proc macro reads this
#[derive(QuantKernels)]
pub struct Scheme {
    pub attn_q: IntConfig<B6, PerRow>,
    pub attn_k: IntConfig<B6, PerRow>,
    pub attn_v: IntConfig<B6, PerRow>,
    pub attn_o: IntConfig<B6, PerRow>,
    pub mlp_gate: IntConfig<B5, PerRow>,
    pub mlp_up: IntConfig<B5, PerRow>,
    pub mlp_down: IntConfig<B5, PerRow>,
    pub embed: IntConfig<B8, PerRow>,
}
```

The `#[derive(QuantKernels)]` macro generates an implementation block containing:

**For each unique (Bits, BlockSize) pair found in the struct:**

```rust
// AUTO-GENERATED by derive(QuantKernels)
impl Scheme {
    /// Fused STE forward: scale → round → clamp → rescale
    /// Generated with CLAMP_MIN=-16, CLAMP_MAX=15 for B5
    #[cube(launch_unchecked)]
    pub fn fake_quant_fwd_b5_perrow(
        weight: &Tensor<f32>,
        output: &mut Tensor<f32>,
    ) {
        // Per-row abs max → scale
        let row_max = weight.abs().max_dim_reduce(1);
        let scale = row_max / 15.0;  // (2^(5-1) - 1) = 15
        let q = weight / scale;
        let rounded = q.round();
        let clamped = rounded.clamp(-16.0, 15.0);  // -2^(5-1) to 2^(5-1)-1
        output = clamped * scale;
    }

    /// STE backward: identity with clamp mask
    #[cube(launch_unchecked)]
    pub fn fake_quant_bwd_b5_perrow(
        grad_output: &Tensor<f32>,
        q_pre_clamp: &Tensor<f32>,  // saved from forward
        grad_input: &mut Tensor<f32>,
    ) {
        // Gradient passes through where values weren't clamped
        let mask = (q_pre_clamp.abs() <= 16.0) as f32;
        grad_input = grad_output * mask;
    }

    /// CPU-side bit packing for serialization
    pub fn pack_b5_perrow(weights: &[f32], scales: &mut Vec<f32>) -> Vec<u8> {
        // Pack 8 int5 values into 5 bytes (40 bits = 8 × 5 bits)
        // ...
    }

    /// GPU dequantization for inference
    #[cube(launch_unchecked)]
    pub fn dequant_b5_perrow(
        packed: &Tensor<u8>,
        scales: &Tensor<f32>,
        output: &mut Tensor<f32>,
    ) {
        // Unpack 5-bit values, multiply by per-row scale
        // ...
    }
}
```

### 1.4 Sweep Infrastructure

```rust
// pg-quant/src/sweep.rs

/// All schemes to evaluate (generated from config space)
const SCHEMES: &[SchemeConfig] = &[
    // Baseline (current SOTA consensus)
    SchemeConfig { attn: (6, PerRow), mlp: (5, PerRow), embed: (8, PerRow) },
    // Inverted allocation
    SchemeConfig { attn: (5, PerRow), mlp: (6, PerRow), embed: (8, PerRow) },
    // Aggressive uniform
    SchemeConfig { attn: (5, PerRow), mlp: (5, PerRow), embed: (6, PerRow) },
    // Conservative uniform
    SchemeConfig { attn: (6, PerRow), mlp: (6, PerRow), embed: (8, PerRow) },
    // Per-layer graduated (deeper layers get fewer bits)
    SchemeConfig { attn: (6, PerRow), mlp_early: (6, PerRow), mlp_late: (5, PerRow), embed: (8, PerRow) },
    // Block quantization variants
    SchemeConfig { attn: (6, Block32), mlp: (5, Block32), embed: (8, PerRow) },
    // ... ~50 total configurations
];

/// Quick eval: quantize EMA checkpoint, evaluate on 100 docs (~30s)
fn quick_eval(scheme: &SchemeConfig, ema_weights: &Weights) -> f32 {
    let quantized = scheme.quantize(ema_weights);
    let compressed_size = zstd_compress(&quantized, 22).len();
    if compressed_size > 16_000_000 { return f32::MAX; }  // Over budget
    let bpb = eval_100_docs(&quantized);
    bpb
}

/// Full sweep: ~50 configs × 30s each = ~25 minutes on 1 GPU
fn sweep(ema_weights: &Weights) -> Vec<(SchemeConfig, f32, usize)> {
    SCHEMES.par_iter()  // rayon parallel
        .map(|scheme| {
            let bpb = quick_eval(scheme, ema_weights);
            let size = compressed_size(scheme, ema_weights);
            (scheme.clone(), bpb, size)
        })
        .collect()
}
```

### 1.5 Expected Outcomes

The search should answer several questions that the competition meta hasn't addressed:
- Is int5 MLP / int6 attention actually the Pareto-optimal split, or is it just the first thing that worked?
- Does block quantization (block size 32 or 64) with shared scaling factors compress better than per-row scaling under zstd-22?
- Is there value in graduated quantization (fewer bits for deeper layers, which have lower-magnitude weights)?
- Can int4 attention work if you compensate with int7 MLP (same total bits, different allocation)?

If the sweep finds a configuration that improves by even 0.001 BPB over int5/int6, that's a genuine contribution.

### 1.6 Risk Assessment

**Risk: Proc macro complexity.** Rust proc macros are powerful but debugging them is painful — errors manifest as cryptic compiler messages. **Mitigation:** Write the first 3 configurations (int5/PerRow, int6/PerRow, int8/PerRow) as hand-written kernels, verify they work, then extract the pattern into the macro. The macro is a productivity tool, not a prerequisite.

**Risk: CubeCL kernel generation from macro output.** CubeCL's `#[cube]` attribute is itself a proc macro. Generating CubeCL-annotated functions from another proc macro creates a macro-in-macro situation. **Mitigation:** Generate the CubeCL kernel source as string literals and compile them at build time via CubeCL's runtime compilation path, rather than trying to nest proc macros.

---

## 2. Innovation 2: Novel Kernel Fusions

### 2.1 Fusion A: XSA-in-Attention Epilogue

**Current state:** Flash Attention (cuDNN SDPA) writes attention output `y` to global memory. A separate kernel reads `y` and `v`, computes the XSA projection `z = y - (y·v / ||v||²) · v`, and writes `z` to global memory. This requires one extra read + write of a `[batch, seq, dim]` tensor per XSA layer.

**Memory traffic analysis:** At batch_per_gpu=48, seq=2048, dim=512, bf16: each read or write is 48 × 2048 × 512 × 2 = 100.7 MB. The XSA kernel does 2 reads (y, v) + 1 write (z) = 302 MB. At H100's 3.35 TB/s HBM bandwidth, that's 90μs. Over 4 XSA layers and 9000 steps: 4 × 90μs × 9000 = 3.24 seconds.

**Proposed fusion:** Instead of using cuDNN SDPA (which is a black box), write a custom attention kernel in CubeCL that performs the XSA projection as an epilogue before writing to global memory. The attention output lives in registers/shared memory; the XSA projection is a per-element operation (vector projection) that can be applied in-place.

**Implementation approach:**

This is the hardest fusion because Flash Attention is algorithmically complex (online softmax, tiling for shared memory). We do NOT rewrite Flash Attention from scratch. Instead:

1. Use cuDNN SDPA for the attention computation (it writes `y` to global memory)
2. Fuse the XSA projection with the *output projection* linear layer that follows attention

The output projection is a GEMM: `output = y @ W_o`. We use cuBLASLt with a **custom epilogue** that applies the XSA projection before the GEMM:

```rust
// Pseudo-code for fused XSA + output projection
fn fused_xsa_output_proj(
    attn_output: &GpuTensor,  // y from Flash Attention
    v_per_token: &GpuTensor,  // V vectors (need to be saved)
    w_o: &GpuTensor,          // Output projection weights
    output: &mut GpuTensor,
) {
    // Option A: cuBLASLt matmul with bias epilogue
    // Compute XSA projection as a "bias" applied before matmul
    // This requires XSA to be element-wise, which it is per-token

    // Option B: CubeCL fused kernel
    // For each row (token):
    //   1. Load y[i] and v[i] from global memory
    //   2. Compute z[i] = y[i] - (y[i]·v[i] / ||v[i]||²) * v[i]  (registers)
    //   3. Compute output[i] = z[i] @ W_o  (use tensor cores)
    //   4. Write output[i] to global memory
}
```

**Realistic assessment:** Option B (full fused kernel) is extremely ambitious — writing a competitive GEMM with a pre-processing epilogue in CubeCL would require deep expertise in Hopper tensor core scheduling. Option A (separate XSA kernel, but fuse it with the *residual add* that follows) is more practical:

```rust
// More practical fusion: XSA + residual add + RMSNorm
// Three operations on the same tensor → one kernel
#[cube(launch_unchecked)]
fn fused_xsa_residual_norm(
    attn_out: &Tensor<f32>,     // y from attention output projection
    v_self: &Tensor<f32>,       // per-token V vectors
    residual: &Tensor<f32>,     // residual stream
    rms_weight: &Tensor<f32>,   // RMSNorm parameters
    ln_scale: f32,              // 1/sqrt(layer_idx + 1)
    output: &mut Tensor<f32>,
    normed: &mut Tensor<f32>,
) {
    let pos = ABSOLUTE_POS;
    let y = attn_out[pos];
    let v = v_self[pos];

    // XSA projection (per-element within each token's vector)
    // Computed across the dim dimension using shared memory reduction
    // for dot product y·v and ||v||²
    let y_dot_v = /* shared mem reduction */;
    let v_norm_sq = /* shared mem reduction */;
    let z = y - (y_dot_v / (v_norm_sq + 1e-8)) * v;

    // Residual add
    let h = residual[pos] + z;
    output[pos] = h;

    // RMSNorm for next sublayer
    let rms = /* shared mem reduction of h² */;
    normed[pos] = (h / rms.sqrt()) * rms_weight[pos % dim] * ln_scale;
}
```

**Savings:** Eliminates 2 kernel launches and 1 full tensor read-write (the separate residual add). Estimated savings: ~50μs per layer, ~2.2 seconds total over 9000 steps for 4 XSA layers.

### 2.2 Fusion B: BigramHash + Token Embedding

**Current state:** Two separate lookups + one addition:
1. `tok_emb = embedding_table[token_ids]` → write to global memory
2. `bigram_emb = bigram_table[hash(prev, curr)]` → write to global memory  
3. `combined = tok_emb + projection(bigram_emb)` → write to global memory

That's 3 kernel launches and 3 writes to global memory for a `[batch, seq, dim]` tensor.

**Fused version:**

```rust
#[cube(launch_unchecked)]
fn fused_embed_bigram(
    token_ids: &Tensor<u32>,        // [batch * seq]
    embed_table: &Tensor<f32>,      // [vocab, dim]
    bigram_table: &Tensor<f32>,     // [num_buckets, bigram_dim]
    bigram_proj: &Tensor<f32>,      // [bigram_dim, dim]
    output: &mut Tensor<f32>,       // [batch * seq, dim]
) {
    let seq_pos = ABSOLUTE_POS_X;
    let dim_idx = ABSOLUTE_POS_Y;

    let tok = token_ids[seq_pos];
    let tok_emb_val = embed_table[tok * DIM + dim_idx];

    // BigramHash (skip for position 0)
    let bigram_val = if seq_pos > 0 {
        let prev_tok = token_ids[seq_pos - 1];
        let hash_idx = (prev_tok * 36313) ^ (tok * 27191);
        let bucket = hash_idx % NUM_BUCKETS;

        // Accumulate projection: sum over bigram_dim
        let mut proj_val = 0.0f32;
        for bd in 0..BIGRAM_DIM {
            proj_val += bigram_table[bucket * BIGRAM_DIM + bd]
                      * bigram_proj[bd * DIM + dim_idx];
        }
        proj_val
    } else {
        0.0f32
    };

    output[seq_pos * DIM + dim_idx] = tok_emb_val + bigram_val;
}
```

**Savings:** 2 kernel launches eliminated, 2 unnecessary global memory writes removed. Estimated savings: ~30μs per step, ~0.27 seconds total. Small but free.

**Backward pass:** The backward through this fused kernel is straightforward — `grad_embed_table[tok] += grad_output` (scatter-add), `grad_bigram_table[bucket] += grad_output @ bigram_proj.T`, `grad_bigram_proj += bigram_emb.T @ grad_output`.

### 2.3 Fusion C: RMSNorm + QK-Projection + Partial RoPE + q_gain

**Current state:** After the input RMSNorm, the Q and K projections are computed via cuBLAS GEMM. Then three separate kernels apply: (1) QK RMSNorm, (2) Partial RoPE, (3) q_gain scaling.

**Fused version:** Fuse operations (1), (2), (3) into a single post-GEMM kernel:

```rust
#[cube(launch_unchecked)]
fn fused_qk_post_gemm(
    q_raw: &Tensor<f32>,         // [batch, seq, num_heads * head_dim]
    k_raw: &Tensor<f32>,         // [batch, seq, num_kv_heads * head_dim]
    cos: &Tensor<f32>,           // [seq, rope_dims/2]
    sin: &Tensor<f32>,           // [seq, rope_dims/2]
    q_gain: &Tensor<f32>,        // [num_heads] learnable per-head scale
    q_out: &mut Tensor<f32>,
    k_out: &mut Tensor<f32>,
    #[comptime] rope_dims: u32,  // 16 — baked in at compile time
    #[comptime] head_dim: u32,   // 64 — baked in at compile time
) {
    // Per-head, per-token:
    // 1. RMSNorm the head vector
    // 2. Apply RoPE to first rope_dims dimensions
    // 3. Apply q_gain scaling (Q only)
    // All in registers, single write to global memory
}
```

**Savings:** 2 kernel launches eliminated (QK RMSNorm, RoPE are merged; q_gain is folded in). The `#[comptime]` parameters mean the loop bounds for RoPE dim splitting are baked in as constants — no runtime branching. Estimated savings: ~20μs per layer × 11 layers × 9000 steps = ~1.98 seconds.

### 2.4 Combined Fusion Impact

| Fusion | Per-step savings | Total savings (9000 steps) | Complexity |
|--------|-----------------|---------------------------|------------|
| XSA + residual + RMSNorm | ~50μs × 4 layers | ~1.8s | Medium |
| BigramHash + embedding | ~30μs | ~0.27s | Low |
| QK post-GEMM | ~20μs × 11 layers | ~1.98s | Medium |
| **Total** | **~450μs** | **~4.05s** | |

At 65ms/step, 4.05 seconds = **~62 additional training steps**. Not transformative alone, but combined with compile warmup elimination (~300 steps) and NCCL overlap (~50 steps), the total is ~412 extra steps — enough to potentially train at d=576 (which is ~7ms slower per step but benefits from wider representations).

**The real research question:** Can the ~412 extra steps be traded for a wider model (d=576 instead of d=512) that finishes ~8600 steps instead of ~9000 but achieves lower BPB due to increased capacity? The ablation (Config B vs C in the proposal) tests exactly this.

---

## 3. Innovation 3: Compile-Time Architecture Specialization

### 3.1 Problem Statement

Architecture search in Parameter Golf requires evaluating many configurations. In PyTorch, each configuration either runs in eager mode (slow — no kernel fusion) or triggers `torch.compile` (15–25s overhead per configuration). For a sweep over 20 architectures, that's 5–8 minutes of pure compilation overhead.

### 3.2 Const-Generic Architecture Struct

```rust
// pg-model/src/arch.rs

/// Architecture configuration as a const-generic type
/// Changing ANY parameter regenerates all kernels at compile time
pub struct Arch<
    const D: usize,          // hidden dim: 384, 512, 576, 640
    const HEADS: usize,      // attention heads: 6, 8, 10, 12
    const KV_HEADS: usize,   // KV heads: 2, 4
    const LAYERS: usize,     // depth: 9, 10, 11, 12, 13
    const MLP_MULT: usize,   // MLP expansion: 2, 3, 4
    const ROPE_DIMS: usize,  // RoPE dimensions: 8, 16, 24, 32
    const XSA_FROM: usize,   // XSA starts at layer: 0, 7, 8, 9, 11 (11 = disabled)
    const SEQ_LEN: usize,    // sequence length: 1024, 2048, 4096
> {
    _phantom: PhantomData<()>,
}

/// Type aliases for configurations to sweep
pub type BaselineArch = Arch<512, 8, 4, 11, 3, 16, 7, 2048>;
pub type WideArch = Arch<576, 8, 4, 11, 3, 16, 7, 2048>;
pub type DeepArch = Arch<512, 8, 4, 13, 3, 16, 9, 2048>;
pub type NarrowDeepArch = Arch<384, 8, 4, 16, 3, 12, 12, 2048>;
pub type FullXsaArch = Arch<512, 8, 4, 11, 3, 16, 0, 2048>;
pub type LongCtxArch = Arch<512, 8, 4, 11, 3, 16, 7, 4096>;
```

### 3.3 How Specialization Propagates

Every kernel that depends on architecture parameters uses `#[comptime]` parameters in CubeCL:

```rust
impl<const D: usize, const HEADS: usize, /* ... */> Arch<D, HEADS, /* ... */> {
    pub fn attention_kernel_config() -> AttentionConfig {
        AttentionConfig {
            head_dim: D / HEADS,            // computed at compile time
            rope_dims: ROPE_DIMS,           // baked in
            xsa_enabled: XSA_FROM < LAYERS, // constant-folded
            gqa_ratio: HEADS / KV_HEADS,    // baked in
        }
    }
}
```

When CubeCL sees `#[comptime]` values, it generates PTX with those values as immediate constants. The loop `for d in 0..ROPE_DIMS` becomes an unrolled loop of exactly 16 iterations — no runtime check needed. The branch `if layer_idx >= XSA_FROM` is resolved at compile time per-layer — XSA layers compile to kernels with XSA projection; non-XSA layers compile to kernels without it.

### 3.4 Sweep Protocol

```rust
// pg-train/src/sweep.rs

/// Abbreviated training: 500 steps (~32s) on 1 GPU
/// Enough to estimate convergence trajectory
fn abbreviated_train<A: ArchTrait>(config: A) -> (f32, f32) {
    let model = Model::<A>::new();
    let mut optim = Muon::new(/* same hyperparams */);

    for step in 0..500 {
        let lr = wsd_lr(step, 10, 500, 200, 0.025);  // proportionally scaled WSD
        let loss = train_step(&mut model, &mut optim, lr);
    }

    let train_loss = model.eval_loss(100_docs);
    let step_time = elapsed / 500.0;
    (train_loss, step_time)
}

// Sweep: compile 10 architecture variants, run 500 steps each
// Total time: ~10 × (compile ~2s + train ~32s) = ~340s on 1 GPU
// Compare: PyTorch would need ~10 × (compile ~20s + train ~32s) = ~520s
```

The Rust advantage is ~2s compile time per variant (CubeCL kernel compilation) vs ~20s for `torch.compile`. Over 10 variants, that's **180 seconds saved** — enough for 4 additional abbreviated runs.

### 3.5 Configurations to Sweep

| Config | D | L | Heads | KV | MLP | RoPE | XSA | Expected size (int5/6) |
|--------|---|---|-------|-----|-----|------|-----|----------------------|
| Baseline | 512 | 11 | 8 | 4 | 3× | 16 | 7+ | ~15.7 MB |
| Wide | 576 | 11 | 8 | 4 | 3× | 16 | 7+ | ~19.8 MB (over!) |
| Wide-narrow MLP | 576 | 11 | 8 | 4 | 2× | 16 | 7+ | ~15.2 MB |
| Deep-narrow | 384 | 16 | 6 | 3 | 3× | 12 | 12+ | ~15.5 MB |
| Deep-medium | 448 | 14 | 8 | 4 | 3× | 14 | 10+ | ~15.9 MB |
| Full-XSA | 512 | 11 | 8 | 4 | 3× | 16 | 0+ | ~15.7 MB |
| More-RoPE | 512 | 11 | 8 | 4 | 3× | 32 | 7+ | ~15.7 MB |
| High-GQA | 512 | 11 | 12 | 3 | 3× | 16 | 7+ | ~15.3 MB |
| Long-ctx | 512 | 11 | 8 | 4 | 3× | 16 | 7+ | ~15.7 MB |
| Aggressive-quant | 640 | 11 | 8 | 4 | 3× | 16 | 7+ | ~15.8 MB (int4!) |

The last configuration is particularly interesting — if int4 quantization doesn't destroy quality (which the quantization sweep from Innovation 1 will test), a d=640 model at int4 fits in the same budget as d=512 at int6. That's a 56% increase in hidden dimension.

---

## 4. Per-Step Equivalence Validation (TA feedback)

### 4.1 Methodology

Before running any ablation, we must confirm that Rust and PyTorch produce equivalent per-step learning:

```rust
// pg-train/src/validate.rs

/// Load the same data batch, same weights, compute one step in both stacks
fn validate_per_step_equivalence() {
    // 1. Load PyTorch SOTA checkpoint (via pg-compat)
    let weights = load_pytorch_weights("sota_checkpoint.pt");

    // 2. Load identical batch (deterministic from seed)
    let batch = load_batch(seed=42, step=0);

    // 3. Rust forward + backward
    let (rust_loss, rust_grads) = rust_train_step(&weights, &batch);

    // 4. PyTorch forward + backward (via subprocess)
    let (torch_loss, torch_grads) = pytorch_train_step(&weights, &batch);

    // 5. Compare
    assert!((rust_loss - torch_loss).abs() < 1e-3, "Loss mismatch");
    for (name, (rg, tg)) in rust_grads.iter().zip(torch_grads.iter()) {
        let max_diff = (rg - tg).abs().max();
        let rel_diff = max_diff / tg.abs().max().max(1e-7);
        assert!(rel_diff < 0.02, "Gradient mismatch in {}: {}", name, rel_diff);
    }
}

/// Multi-step validation: run 200 steps in both, compare loss curves
fn validate_loss_trajectory() {
    let rust_losses = rust_train(steps=200, seed=42);
    let torch_losses = pytorch_train(steps=200, seed=42);

    // Losses should track within 2% after step 20 (warmup noise is expected)
    for step in 20..200 {
        let rel_diff = (rust_losses[step] - torch_losses[step]).abs()
                      / torch_losses[step];
        assert!(rel_diff < 0.02,
            "Loss trajectory diverged at step {}: rust={}, torch={}",
            step, rust_losses[step], torch_losses[step]);
    }
}
```

### 4.2 Known Sources of Divergence

Even with identical algorithms, bf16 accumulation can diverge due to:
- **cuBLAS GEMM algorithm selection:** cuBLASLt may choose different tiling strategies than PyTorch's default. Fix: force the same algorithm via `cublasLtMatmulAlgoGetHeuristic` with a fixed preference.
- **Flash Attention tile ordering:** cuDNN SDPA processes tiles in a deterministic but implementation-specific order. The accumulated softmax statistics may differ at the edges.
- **NCCL allreduce ordering:** On multi-GPU, reduction order affects bf16 accumulation. Fix: verify with `NCCL_ALGO=Ring` forced on both sides.

For the single-GPU validation, GEMM algorithm selection is the main concern. We force the same algorithm and verify loss < 1e-3 on the first step.

---

## 5. Implementation Priority and Timeline

| Week | Innovation work | Integrates with |
|------|----------------|----------------|
| 2 | Hand-write int5, int6, int8 STE kernels + packing code | pg-kernels, pg-quant |
| 3 | Fused QK-post-GEMM kernel (Fusion C) | pg-kernels |
| 4 | Fused BigramHash+embedding kernel (Fusion B) | pg-kernels |
| 5 | Extract proc macro from hand-written kernels → quantization sweep | pg-quant |
| 5 | Fused XSA+residual+RMSNorm kernel (Fusion A) | pg-kernels |
| 5 | Quantization sweep (~50 configs, parallelized via rayon) | pg-quant |
| 6 | Const-generic architecture specialization + architecture sweep | pg-model |
| 6 | Full ablation: Configs A, B, C, D | pg-train, pg-eval |

The innovations are ordered by dependency and risk: quantization kernels are needed for basic training (week 2), fusions provide incremental throughput (weeks 3–5), the proc macro extracts patterns from working code (week 5), and the architecture sweep uses all of the above (week 6).
