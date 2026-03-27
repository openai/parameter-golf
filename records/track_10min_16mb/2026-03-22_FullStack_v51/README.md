## Record: v6.0 Moonshot — Dual TTT + Full Architecture Stack

**Target: <1.03 val_bpb** (stretch: <0.99) | 8xH100 SXM, 600s train + 600s eval

### Novel Contributions

1. **Two independent TTT methods** (user chooses via config):
   - **In-Place TTT** (ICLR 2026 Oral): Updates MLP output projections per-document using NTP loss with apply-then-update ordering. Targets completely different parameters than LoRA TTT.
   - **Per-document LoRA TTT** (PR #548): Rank-8 LoRA on Q/V/LM head with surprise-gated training (Titans-inspired — only top-K% highest-loss tokens get gradient updates).

2. **Full GPTQ** (Hessian-aware quantization): 256-sample calibration, per-layer Hessian H=X^TX, column-wise int6 with Cholesky error compensation. 31% quantization gap reduction over naive int6.

3. **LeakyReLU(0.5)^2 activation**: Drop-in replacement for relu^2 that preserves gradients through negative activations. -0.0015 bpb proven by 4+ teams.

4. **Eval-only XSA** on all 11 layers: Exclusive Self-Attention removes self-position contribution during eval, forcing context-only prediction. Training proceeds without XSA to avoid regression.

### Architecture (from PR #486 base)

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- 3x MLP LeakyReLU(0.5)^2 + SmearGate + BigramHash(4096) + TrigramHash(4096)
- Value Residual (ResFormer) across all layers
- GradQuant: gradient-guided adaptive Int5/6/7
- Partial RoPE (16/64 dims), LN Scale (1/sqrt(layer+1))
- EMA (decay=0.997), OrthoInit
- Full GPTQ + zstd-22

### TTT Configuration (LoRA mode)

```bash
# LoRA TTT (default: INPLACE_TTT_ENABLED=0)
TTT_LORA_LR=0.01          # LoRA optimizer LR
TTT_LORA_RANK=8            # LoRA rank
TTT_EPOCHS=20              # Epochs per document
TTT_BATCH_SEQS=32          # Documents per GPU batch
TTT_SURPRISE_TOPK=0.5      # Train on top 50% highest-loss tokens
```

### TTT Configuration (In-Place mode)

```bash
# In-Place TTT (INPLACE_TTT_ENABLED=1)
INPLACE_TTT_LR=0.001       # MLP proj update LR
INPLACE_TTT_CHUNK=256       # Chunk size for apply-then-update
```

### Run Commands

```bash
# LoRA TTT (proven, batched)
INPLACE_TTT_ENABLED=0 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py

# In-Place TTT (novel, per-document MLP adaptation)
INPLACE_TTT_ENABLED=1 SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Experiment Plan

1. Reproduce PR #486 baseline (pre-TTT ~1.11, post-LoRA-TTT ~1.09)
2. Compare LoRA TTT vs In-Place TTT on our architecture
3. Tune surprise-gating threshold: {25%, 50%, 75%}
4. Tune In-Place TTT LR: {0.0005, 0.001, 0.002}
5. 3-seed validation with best config

### Results

_Pending — needs 8xH100 validation runs._

### Provenance

- Architecture: PR #486 (ndokutovich)
- LoRA TTT: PR #548 (LoquiAuris)
- In-Place TTT: "In-Place Test-Time Training" (ICLR 2026 Oral, Feng et al.)
- Surprise gating: Inspired by "Titans: Learning to Memorize at Test Time" (NeurIPS 2025)
- LeakyReLU^2: PR #493 (parinzee)
- Full GPTQ: PR #535 (raahilshah)
- XSA: PR #503 (EthanYangTW)
