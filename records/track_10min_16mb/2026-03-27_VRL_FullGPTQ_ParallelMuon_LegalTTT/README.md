# 11L VRL + Full GPTQ + BigramHash3072 + Parallel Muon + Legal SGD TTT

**val_bpb = 1.1882** (seed 1337, preliminary — see note below) | 18.8MB (exceeds 16MB limit) | 8×H100 SXM

> **Status: non-record preliminary run.** This run was executed without `torch.compile` support due to an overlay filesystem constraint on the Vast.ai instance (`/dev/shm` noexec, overlay at 100%). Only 2002 training steps were completed at 300ms/step vs the SOTA's 7185 steps at 83ms/step. The artifact is 18.8MB (over the 16MB limit). A proper rerun with `torch.compile` is pending.

## Results

| Seed | steps | step_avg | Pre-TTT bpb | Post-TTT bpb | Artifact |
|------|-------|----------|-------------|--------------|----------|
| 1337 | 2002 | 300ms | 1.2115 | **1.1882** | 18,816,038 bytes |

## Architecture

Forked from [PR #549](https://github.com/openai/parameter-golf/pull/549) (abaybektursun, 1.1194 SOTA).

### Changes from PR #549

**(1) VRL — Value Residual Learning** ([arxiv:2410.17897](https://arxiv.org/abs/2410.17897))

Layer 0's V projection output is stored and blended into subsequent layers via a learned per-layer sigmoid gate (`vr_lambda`). This allows later layers to access "raw" value representations, similar to residual connections but in the value space.

```python
if value_residual and self.layer_idx > 0:
    lam = torch.sigmoid(self.vr_lambda)
    v = (1 - lam) * v + lam * v0_expanded
```

**(2) Full GPTQ — Hessian-aware int6 quantization**

Replaces GPTQ-lite clip-percentile search with full Hessian Cholesky error compensation:
- Collect `X^T X` Hessians from 256 calibration batches
- All-reduce across 8 ranks
- Column-wise quantization with error propagation via Cholesky decomposition of `H^{-1}`

**(3) BigramHash 1536 → 3072**

Doubles bigram hash buckets. Per PR #549 ablation, this gives free -0.0009 bpb.

**(4) Tight SWA preferred over EMA**

When SWA snapshots exist, use Tight SWA average rather than EMA:
```python
elif swa_state is not None and swa_count > 0:
    avg_state = {name: (t / swa_count) for name, t in swa_state.items()}
    base_model.load_state_dict(avg_state)
```

### Unchanged from PR #549

- 11L, 512d, 8H/4KV, LeakyReLU(0.5)² MLP 3×
- BigramHash(3072→128→512), XSA last 4 layers, Partial RoPE (16/64), LN Scale, VE128 (layers 9,10)
- EMA(0.997), GPTQ-lite int6 + lzma, Late QAT@0.15
- Parallel Muon (Parameter Banking), Legal score-first SGD TTT (all blocks, 3 epochs, cosine LR)
- WARMDOWN_ITERS=700 (adjusted for uncompiled run), MAX_WALLCLOCK_SECONDS=600

## Reproduction

```bash
export NO_COMPILE=1  # remove for environments with torch.compile support
export TMPDIR=/dev/shm/tmp  # or any writable temp dir
export NUM_LAYERS=11
export XSA_LAST_N=4
export EMA_ENABLED=1 EMA_DECAY=0.997
export SWA_ENABLED=1 SWA_EVERY=50
export VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10
export TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9
export VALUE_RESIDUAL=1
export BIGRAM_VOCAB_SIZE=3072
export FULL_GPTQ=1 FULL_GPTQ_CALIB_BATCHES=256
export MUON_WD=0.04 ADAM_WD=0.04
export MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3500  # use 700 if NO_COMPILE
export ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600
export EVAL_STRIDE=64 SEED=1337

torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-27_VRL_FullGPTQ_ParallelMuon_LegalTTT/train_gpt.py
```

## Credits

- **Base architecture**: [PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun
- **Parallel Muon / Parameter Banking**: [PR #399](https://github.com/openai/parameter-golf/pull/399) by @abaybektursun
- **Legal TTT framework**: [PR #461](https://github.com/openai/parameter-golf/pull/461) by @Christopher-Lee-McClendon
- **LeakyReLU²**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **VRL**: [arxiv:2410.17897](https://arxiv.org/abs/2410.17897) by Qin et al.
- **Full GPTQ**: [GPTQ paper](https://arxiv.org/abs/2210.17323) by Frantar et al.
