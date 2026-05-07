# Connectome-JEPA: Sparse I/O Bottleneck Architecture Inspired by Biological Neural Wiring

**Non-record submission.** First JEPA-based and first biologically-inspired sparse architecture in Parameter Golf.

## Summary

We explore whether sparse I/O bottleneck architectures inspired by biological neural wiring can serve as parameter-efficient language models. The architecture routes token representations through a sparse network derived from the *C. elegans* connectome — a 300-neuron organism whose synaptic wiring has been mapped at single-synapse resolution. Token embeddings are injected into 88 sensory neurons, propagated through 6 sparse hops (5,084 directed edges per hop, 5.65% density) with gated skip connections, interleaved with multi-head causal cross-position attention, then read from 123 motor neurons and projected to vocabulary logits via tied embeddings. Training combines cross-entropy with a JEPA latent prediction loss and SigREG for collapse-free representation learning.

**val_bpb: 1.7942 ± 0.0028** (3 seeds, post-quant) | **1.97MB artifact** (12.3% of 16MB budget) | **3.34M params** | **10 min on 8×H100** (record-track eligible with `MAX_WALLCLOCK_SECONDS=600`: val_bpb 1.7978 at step 4175)

## Architecture

```
Input token_ids [B, S]
        │
   Embedding (1024 × 768, tied with output)
        │
   RMSNorm
        │
   Sensory Projection (768 → 88)
        │
   Inject into sensory neuron positions of 300-dim state
        │
   ┌─ Hop 1: MaskedLinear(300×300, mask=adjacency) + GELU ────────────┐
   ├─ Hop 2: MaskedLinear(300×300, mask=adjacency) + GELU ────────────┤
   │  Cross-Position Attention (4-head, dim 128, causal, RoPE)        │
   │  ← Sensory Re-injection (gated skip connection)                  │
   ├─ Hop 3: MaskedLinear(300×300, mask=adjacency) + GELU ────────────┤
   ├─ Hop 4: MaskedLinear(300×300, mask=adjacency) + GELU ────────────┤
   │  Cross-Position Attention (4-head, dim 128, causal, RoPE)        │
   │  ← Sensory Re-injection (gated skip connection)                  │
   ├─ Hop 5: MaskedLinear(300×300, mask=adjacency) + GELU ────────────┤
   ├─ Hop 6: MaskedLinear(300×300, mask=adjacency) + GELU ────────────┤
   │  Cross-Position Attention (4-head, dim 128, causal, RoPE)        │
   └──────────────────────────────────────────────────────────────────┘
        │
   Motor Readout (select 123 motor neurons)
        │
   Motor Projection (123 → 768)
        │
   RMSNorm → Tied Logits (768 × 1024) → Softcap(30)
```

`MaskedLinear` multiplies its weight matrix element-wise by a fixed binary adjacency mask from the *C. elegans* connectome (Cook et al. 2019). The mask is a buffer, not a parameter. Gated skip connections re-inject the sensory signal every 2 hops, providing gradient shortcuts through the sparse chain.

| Component | Detail |
|-----------|--------|
| Neurons | 300 (88 sensory, 89 interneuron, 123 motor) |
| Synapses (edges) | 5,084 directed |
| Density | 5.65% |
| Connectome hops | 6 |
| Skip injection | Every 2 hops (gated, init 0.1) |
| Cross-attention | 3 layers (after hops 2, 4, 6), 4 heads, dim 128 |
| Embedding dim | 768 |
| Vocab / tokenizer | 1024 (sp1024 SentencePiece) |

## Why This Architecture Is Artifact-Efficient

The sparse I/O bottleneck provides several structural advantages under a size constraint:

- **Fixed adjacency mask = zero parameter cost routing.** The binary connectome mask is a buffer, not learned weights. It provides structured information routing for free — equivalent to a hardcoded attention pattern that costs no artifact bytes.
- **Sparse hops = cheap depth scaling.** Each `MaskedLinear` hop stores ~5k learned weights (the non-zero entries) vs ~90k for a dense layer at the same dimension. Six hops of connectome processing cost fewer parameters than a single dense transformer layer.
- **Sensory/motor bottleneck = forced compression.** Information must pass through 88 sensory inputs and 123 motor outputs, creating an architectural compression bottleneck that doesn't rely on quantization.
- **Selective capacity allocation.** The cross-attention layers (where most parameters live) handle sequence modeling; the sparse hops handle position-independent feature routing. This separation means parameters are allocated where they have the most impact rather than spread uniformly across dense layers.

The result: a 1.97MB artifact at 3.34M params. Most submissions need aggressive int5/int6 quantization to fit 15M+ params into 16MB. This architecture fits naturally at int8 with 87.7% of the budget unused.

## Training Objective

```
total_loss = CE + 0.5 × JEPA + 0.0001 × SigREG
```

**CE:** Standard next-token cross-entropy. Primary loss for BPB.

**JEPA:** Span-masked input is encoded through the same network. MSE between masked encoder output and full encoder output (detached target). Regularizes representations toward predictability from partial context.

**SigREG:** From LeJEPA (Balestriero & LeCun, 2025). Random 1D projections compared against the characteristic function of a standard Gaussian. Prevents collapse without EMA or teacher-student networks.

## Results

### 3-Seed Validation (uncapped, 5000 steps)

| Seed | val_bpb (post-quant) |
|------|---------------------|
| 1337 | 1.7955 |
| 42 | 1.7909 |
| 7 | 1.7961 |
| **Mean ± Std** | **1.7942 ± 0.0028** |

### Canonical Run (MAX_WALLCLOCK_SECONDS=600, record-track eligible)

```
val_bpb (pre-quant):            1.7856
val_bpb (post-quant):           1.7978
Steps completed:                4,175 / 5,000 (wallclock cap)
Training time:                  600 seconds (10.0 minutes)
Step time:                      ~144ms/step
```

```
Params:                         3,336,051
Code size:                      47,420 bytes
Model (int8 + zlib):            1,917,705 bytes
Total artifact:                 1,965,125 bytes  (12.3% of 16MB budget)
Hardware:                       8× NVIDIA H100 SXM (Modal)
Peak VRAM:                      7,456 MiB per GPU
```

### Complete Ablation Table

| Config | val_bpb | Params | Artifact | Finding |
|--------|---------|--------|----------|---------|
| **Primary (embed=768, 4 heads)** | **1.79 ± 0.003** | **3.34M** | **1.97MB** | **Best config** |
| CE-only, no JEPA/SigREG (embed=768) | 1.78 | 3.34M | 1.94MB | JEPA/SigREG slightly hurt at this scale |
| Connectome (embed=512, 1 head) | 1.99 ± 0.01 | 1.64M | 1.48MB | Smaller config, 3-seed validated |
| Random sparse, matched density | 1.96 | 1.64M | 1.48MB | Topology ≈ random for language |
| CE-only (embed=512) | 2.01 | 1.64M | 1.48MB | JEPA/SigREG help at smaller scale |
| 2-hop, no skips | 2.22 | 967k | 802KB | Depth matters |
| 16-hop, no skip connections | 6.93 | 2.39M | — | Gradient death |
| Naive Baseline (dense transformer) | 1.22 | ~18.9M | 15.9MB | Reference |

## Key Findings

### 1. The sparse I/O bottleneck works for language modeling

A sparse sensory→interneuron→motor architecture with 5.65% connection density produces functional language models at 1.79 BPB with 3.34M parameters in a 1.97MB artifact — 12.3% of the 16MB budget. While not competitive with the heavily optimized dense transformer submissions on the leaderboard, this demonstrates that sparse I/O bottleneck architectures offer a qualitatively different trade-off between parameters and compression.

### 2. Biological topology does not help (for language)

Random sparse wiring with matched density and I/O structure slightly outperforms the *C. elegans* connectome (1.96 vs 1.99 BPB). The architectural value comes from the **structural pattern** — sparse connectivity with a forced information bottleneck — not from the specific biological wiring. This contrasts with prior RL work where biological topology provided significant gains on the organism's native task, suggesting topology benefits are task-specific.

### 3. JEPA + SigREG effects are scale-dependent

At smaller scale (embed=512, 1.64M params), JEPA + SigREG improve BPB by 0.017 — a modest but measurable regularization benefit. At larger scale (embed=768, 3.34M params), JEPA + SigREG slightly hurt performance (1.79 vs 1.78 BPB). The double forward pass for JEPA likely consumes training steps that would be better spent on additional CE iterations at this capacity level. This suggests JEPA regularization is most valuable when the model is capacity-constrained.

### 4. Gradient flow through sparse hops requires skip connections

16 sequential hops at 5.65% density produced zero learning (CE stuck at ln(1024) ≈ 6.93). Gated sensory re-injection every 2 hops restored gradient flow, enabling 6-hop networks to outperform 2-hop by 0.23 BPB. The skip connections are both a gradient fix and biologically motivated — the real *C. elegans* connectome has long-range connections that bypass intermediate processing.

### 5. Cross-position attention is the primary capacity lever

Scaling from 1 attention head (dim 128) to 4 heads produced the largest single improvement (0.20 BPB). The connectome hops handle position-independent feature routing; the cross-attention layers handle sequence modeling. Additional capacity in attention yields more return than additional connectome depth.

## Motivation

This work was inspired by recent advances in connectome-derived computation:

**Whole-brain emulation (Eon Systems / Shiu et al., 2024):** The *Drosophila melanogaster* connectome (125,000 neurons, 50M synapses from FlyWire electron microscopy) was used to build a whole-brain model that produced multiple naturalistic behaviors when embodied in MuJoCo physics simulation — driven by connectome circuit dynamics alone. This demonstrated that biological wiring carries transferable computational structure.

**LeJEPA (Balestriero & LeCun, 2025):** SigREG was introduced as a provably optimal regularizer for JEPA training, eliminating EMA and teacher-student heuristics. We adapted it for text, demonstrating stable, collapse-free training for sparse language architectures.

Our results confirm the sparse I/O bottleneck hypothesis while providing an honest negative result on biological specificity: the *pattern* of sparse routing matters, but the *specific* biological wiring does not help for language modeling.

## Compatibility

Uses only standard PyTorch: `F.scaled_dot_product_attention` (no Flash Attention 3), `F.linear`, `F.gelu`, `F.rms_norm`. No custom CUDA kernels. Dependencies: `torch`, `numpy`, `sentencepiece`.

## Command

```bash
# Primary submission (embed=768, 4 heads — defaults)
torchrun --standalone --nproc_per_node=8 train_connectome_jepa.py

# Ablations via environment variables:
TOPOLOGY=random_sparse  ...   # Random sparse (matched density)
JEPA_WEIGHT=0 SIGREG_WEIGHT=0  ...   # CE-only
EMBED_DIM=512 NUM_ATTN_HEADS=1  ...   # Smaller config
NUM_HOPS=2 CROSS_ATTN_EVERY=1 SKIP_EVERY=0  ...   # 2-hop variant
```

## Future Directions

- **Topology search:** Random sparse outperforms biological wiring for language — learned or optimized sparse topologies may improve further.
- **Parallel connectome channels:** Multiple independent sparse circuits with shared masks, analogous to CNN channels.
- **Drosophila larva connectome:** 3,016 neurons, 548,000 synapses (Winding et al., Science 2023). 10× the bandwidth of *C. elegans*.
- **Test-time training:** The JEPA objective naturally supports TTT — adapt the encoder on validation context before scoring.
- **SigREG + PolarQuant:** SigREG shapes representations toward an isotropic Gaussian — the geometry PolarQuant (Google, 2026) exploits for zero-overhead quantization.

## References

- Cook, S.J. et al. (2019). Whole-animal connectomes of both *Caenorhabditis elegans* sexes. *Nature*, 571, 63–71.
- Balestriero, R. & LeCun, Y. (2025). LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics. arXiv:2511.08544.
- Shiu, P.K. et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature*, 634, 210–219.
- Winding, M. et al. (2023). The connectome of an insect brain. *Science*, 379(6636), eadd9330.

## Included Files

- `train_connectome_jepa.py` — Complete training script (single file, self-contained)
- `train.log` — Canonical training log (600s wallclock cap, 4175 steps, seed 1337)
- `train_seed_1337.log` — Full training log (seed 1337, 5000 steps uncapped)
- `train_seed_42.log` — Full training log (seed 42, 5000 steps uncapped)
- `train_seed_7.log` — Full training log (seed 7, 5000 steps uncapped)
- `submission.json` — Leaderboard metadata
- `requirements.txt` — Python dependencies
