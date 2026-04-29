# SpinQuant V1 × LQER Asymmetric: Ablation Study

**Track:** track_non_record_16mb | **Seed:** 42 | **val_bpb:** 1.06182 | **Artifact:** 16,956,923 bytes *(oversize — non-record submission)*

---

## What This PR Demonstrates

This submission grafts SpinQuant V1 (Hadamard pre-rotation of all weight matrices before GPTQ) onto the PR #1851 base stack. The primary research question: **does SpinQuant's spectrum-flattening compose cleanly with LQER's low-rank error correction, and does it reduce GPTQ quantization damage?**

**Answer: yes. SpinQuant reduces GPTQ damage by 30% compared to the base stack.**

---

## Pipeline Diagnostics

| Stage | This PR | PR #1851 (base) |
|---|---|---|
| Pre-quant BPB | 1.06822 | 1.06490 |
| Post-GPTQ BPB | 1.07463 | 1.07406 |
| **GPTQ damage (Δ)** | **+0.00640** | **+0.00916** |
| Post-3-phase-TTT BPB | **1.06182** | **1.06128** |
| Artifact bytes | 16,956,923 | 15,952,086 |

SpinQuant's Hadamard rotation flattens weight outliers before quantization, reducing GPTQ damage by **0.00276 BPB** (30% improvement). The final gap to PR #1851 is only **0.00054 BPB** — within single-seed variance.

---

## SpinQuant V1 Implementation

- `_hadamard_rotation(dim, seed, tag)`: deterministic QR-orthogonalized Hadamard matrix, seeded — **zero serialized bytes overhead**
- Applied at 4 sites per layer: `attn_in`, `attn_proj_in`, `mlp_in`, `mlp_proj_in`
- Baked at serialize time: `W_rot = W @ R`. Rotation regenerated from seed at eval — no extra storage
- `CastedLinear._sq_active` flag: `False` during training (zero overhead, Dynamo constant-folds all rotation branches), `True` after deserialize
- LoRA paths (`_block_with_lora`, `_parallel_block_with_lora`) correctly stay in unrotated basis — LoRA adders applied to unrotated `n`, base projections use rotated weights
- LQER runs on rotated weights: `SVD(E_rot)` where `E_rot = W_rot - Wq_rot`. Algebraically valid; rank-4 correction in rotated space is equivalent

---

## Artifact Size Issue

Artifact is **16,956,923 bytes** (956,923 bytes over the 16MB cap).

Brotli compression is less efficient on Hadamard-rotated tensors. Rotation spreads weight entropy more uniformly across all matrix elements, reducing Brotli's ability to exploit local correlations and repetitions. Result: ~1MB compression penalty vs unrotated weights.

Potential remedies for a follow-up: `EMBED_BITS=7` (~524KB saving) + reduced `lqer_rank`. Not applied here as further quantization would degrade the score given the already-tight margin.

---

## Training Config

```
Hardware:        8xH100 80GB SXM
PyTorch:         2.9.1+cu128
Steps:           4881 (stopped at 600s wall clock)
Seed:            42
SPINQUANT_ENABLED=1  SPINQUANT_SEED=20260416
CASEOPS_ENABLED=1    SPARSE_ATTN_GATE_ENABLED=1
SMEAR_GATE_ENABLED=1 LQER_ENABLED=1
LQER_ASYM_ENABLED=1  MIN_LR=0.1
PHASED_TTT_NUM_PHASES=3
```

---

## Attribution

- PR #1851 base (SmearGate BOS fix + LQER Asym + Phased TTT): @aquariouseworkman
- PR #1787 base (CaseOps + SparseAttnGate + PolarNS + MIN_LR + FusedCE): @nprime06
- CaseOps tokenizer: @romeerp (PR #1729)
- SmearGate + LQER Asymmetric: @dexhunter (PR #1797)
- SmearGate BOS audit: @cocohearts (PR #1797 audit)
- Phased TTT framework: @abaybektursun (PR #549)
- GPTQ + SD clip: @clarkkev (PR #1394)
- SpinQuant V1: @X-Abhishek-X (PR #1695)
