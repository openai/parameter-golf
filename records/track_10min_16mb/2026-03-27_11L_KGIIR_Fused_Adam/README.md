# Kinematic Gated IIR (KGIIR) Trajectory Mixing

**Author:** Adam Jacuch  
**Base Architecture:** [Abay Bektursun](https://github.com/abaybektursun)
**Validation BPB:** 1.11837  
**Throughput:** 88ms / step (8xH100)

## The "KGIIR" Breakthrough
This submission introduces **Kinematic Gated IIR (KGIIR)** trajectory mixing. This architecture is built directly upon the high-performance base model developed by **Abay Bektursun**, utilizing his optimized Parallel Muon implementation, parameter banking, and Test-Time Training (TTT) recipe.

### What is KGIIR?
**KGIIR** stands for **Kinematic Gated Infinite Impulse Response** mixing. 

While standard architectures use discrete token shifts (FIR-like behavior) to handle local context, KGIIR treats the hidden state as a continuous physical signal with **Kinematic Momentum**. 

* **Kinematic:** It models the "velocity" of information across the sequence, ensuring that the influence of a token flows smoothly through the layers rather than jumping between discrete steps.
* **Gated:** Every dimension of the model has a per-channel learnable gate, allowing the network to dynamically decide whether to trust the current token or the momentum of the previous trajectory.
* **IIR (Infinite Impulse Response):** Unlike a standard windowed shift, the IIR filter allows information to persist across much longer ranges with zero additional parameter cost, using a recursive 4-tap analytical structure.

### Why it works for Parameter Golf
In the 16MB regime, Attention heads are too valuable to waste on local syntax "bookkeeping." By offloading temporal dependencies to the KGIIR filter, I achieve a superior **Pareto frontier**—reaching a deeper semantic resolution without sacrificing the 88ms/step throughput required for the 600s sprint.

### Controlled Experiment: The BPB Drop
To isolate the impact of KGIIR, this run was conducted as a strict controlled ablation. **The only architectural change made to the Abay Bektursun SOTA was the integration of the KGIIR trajectory layer.**

* **Abay Bektursun SOTA BPB:** 1.11923
* **KGIIR Augmented BPB:** **1.11837**
* **Net Improvement:** **-0.00086 BPB**

## Technical Innovations
* **Hardware-Fused Kernels:** The KGIIR trajectory is implemented as a single-pass fused mathematical expression. This maintains a blistering **88ms step time** on 8xH100 by keeping the mixing logic within the GPU's L2 cache.
* **BPB Progress:** This run successfully pushes into the 1.118x range, demonstrating that trajectory mixing is a viable path forward for ultra-constrained language models.

## Submission Transparency & Constraints
**SOTA Threshold:** This submission provides a single-seed verification. While the BPB improvement is clear, I acknowledge it does not yet clear the required 0.005 nat statistical threshold for a definitive SOTA record flip. 

**Compute Limits:** Due to restricted compute funding (Undergraduate Research), I am unable to provide the standard 3-run mean at this time. This is submitted as an **Architectural Record** to document the KGIIR primitive for the community.

## Reproduction Settings
```bash
# Exact HParams from Bektursun SOTA + KGIIR
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337
```

Special thanks to Abay Bektursun for the world-class baseline architecture and TTT implementation.
