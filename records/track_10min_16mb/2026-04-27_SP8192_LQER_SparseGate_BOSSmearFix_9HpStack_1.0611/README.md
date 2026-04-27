# Record candidate: 11L XSA + LQER + SparseAttnGate + SmearGate (BOS-fixed) + PolarNS Muon + 9-hparam stack

**val_bpb: 1.06108** (3-seed mean, std 0.00090) | **~15.9 MB** | 8×H100 SXM, 600s wallclock | TTT eval

**Improvement over current leaderboard (1.0810 BPB):** **−0.01992 BPB / −0.04359 nats**

## Results

| Seed | Steps | ms/step | Pre-quant val_bpb | Post-quant val_bpb | **Post-TTT val_bpb** | Artifact |
|------|-------|---------|-------------------|--------------------|----------------------|----------|
| 42   | 4,945 | 121.3   | 1.06396           | 1.07254            | **1.05989**          | 15,897,259 |
| 0    | 4,932 | 121.7   | 1.06425           | 1.07407            | **1.06125**          | 15,900,947 |
| 1234 | 4,917 | 122.0   | 1.06438           | 1.07478            | **1.06209**          | 15,907,550 |
| **Mean** |       |         |                   |                    | **1.06108**          | 15,901,919 |

3-seed std: 0.00090 BPB / 0.00198 nats. Each individual seed beats the 1.0810 leaderboard by ≥0.0185 BPB / ≥0.0405 nats.

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4× (2048) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Fused MLP kernel | LeakyReLU-square Triton | [#1626](https://github.com/openai/parameter-golf/pull/1626) |
| Attention | Standard FA3, GQA 2:1 | Baseline |
| XSA | All 11 layers (`xsa_last_n=11`) | [#478](https://github.com/openai/parameter-golf/pull/478) |
| RoPE | Partial (16/64 dims) + YaRN | [#315](https://github.com/openai/parameter-golf/pull/315) |
| LN Scale | 1/√(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| QK Gain init | 5.0 (per-head learned) | [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| U-Net skips | Encoder-decoder skip connections + skip gates | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Parallel decoder | 2-lane parallel from layer 8+, lane mix learned | [#1626](https://github.com/openai/parameter-golf/pull/1626) |
| Depth recurrence | Loop layers 3-5 ×2 once `frac >= 0.35` | [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| Logit softcap | 30 | [#1019](https://github.com/openai/parameter-golf/pull/1019) |
| **Sparse attention gate** | Narrow head-output gate, gate_window=12 | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| **SmearGate** (BOS-fixed) | Position-mixing gate with `not_bos` mask | [#1667](https://github.com/openai/parameter-golf/pull/1667) + this work (BOS leak fix) |
| **Polar-Express Newton-Schulz** | Muon, 5 steps, per-iter minimax tuples | [#1344](https://github.com/openai/parameter-golf/pull/1344) → [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| **MIN_LR floor** | 0.10 (warmdown LR floor) | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| **Fused softcapped CE Triton kernel** | Single-pass training-only | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| **LQER asymmetric int4** | Rank-4 quant-error correction on top-3 tensors | [#1530](https://github.com/openai/parameter-golf/pull/1530) → [#1797](https://github.com/openai/parameter-golf/pull/1797) |
| **Per-group compression** | lrzip zpaq + simsort on hot tensors + brotli rest | [#1626](https://github.com/openai/parameter-golf/pull/1626) (pergroup port) |
| Quantization | GPTQ int6 + int7 embed + int8-per-row attn-gate | [#1583](https://github.com/openai/parameter-golf/pull/1583) lineage |
| TTT | Phased TTT eval, 3 phases, prefix=2500 docs, LoRA rank=80 | [#1667](https://github.com/openai/parameter-golf/pull/1667) → [#1729](https://github.com/openai/parameter-golf/pull/1729) |
| Tokenizer | sp8192 lossless caps caseops v1 reserved | [#1729](https://github.com/openai/parameter-golf/pull/1729) |

## SmearGate cross-document leak fix

SmearGate (introduced in PR #1667 and present in PR #1797) applied a per-token forward-1 smear:

```python
x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1]], dim=1)
```

In a packed validation stream, this leaks the last token of doc N into the BOS embedding of doc N+1, contaminating the first real-token context. The fix masks the previous-token term wherever the current token is BOS:

```python
not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)
```

Applied symmetrically in `_forward_hidden` and `forward_ttt` so training and TTT eval are leak-free.

## Hyperparameter stack

9 overrides validated by greedy forward-selection on 8×H100 real fixed-step (`ITERATIONS=4950, MAX_WALLCLOCK_SECONDS=0`), seed 42:

| hparam | value | default | rationale |
|---|---|---|---|
| MLP_CLIP_SIGMAS | 11.5 | 12.0 | tighter MLP clip → smaller GPTQ quant noise |
| EMBED_CLIP_SIGMAS | 14.0 | 15.0 | tighter embed clip → smaller embed quant noise |
| WARMDOWN_FRAC | 0.85 | 0.75 | longer high-LR phase, shorter warmdown |
| BETA2 | 0.99 | 0.95 | larger Adam beta2, more stable gradient ema |
| TTT_BETA2 | 0.99 | 0.999 | smaller TTT-Adam beta2, faster TTT adaptation |
| TTT_WEIGHT_DECAY | 0.5 | 1.0 | weaker TTT weight decay during phased eval |
| TTT_LORA_RANK | 80 | 96 | smaller TTT LoRA — counter-intuitive but consistent across all 3 seeds |
| SPARSE_ATTN_GATE_SCALE | 0.5 | 1.0 | softer head-output gate |
| PHASED_TTT_PREFIX_DOCS | 2500 | 2000 | longer per-phase TTT prefix; uses 455-509s of 600s eval budget |

## Training & evaluation

| | |
|---|---|
| Training | 4920±15 steps in 600s on 8×H100 SXM (~121 ms/step), warmup=20, warmdown_frac=0.85, MIN_LR=0.10, MATRIX_LR=0.026, GRAD_CLIP_NORM=0.3 |
| Optimizer | Polar-Express Muon (5 steps) on matrix params; Adam (β1=0.9, β2=0.99) on tied embeddings (lr=0.03) and scalars (lr=0.02) |
| EMA | decay=0.9965 |
| Quantization | GPTQ int6 (matrix) + int7 (tied embed) with LQER asym int4 rank-4 correction on top-3 tensors |
| Compression | per-group lrzip zpaq + simsort on hot tensors + brotli on remainder + brotli code wrapper |
| TTT | Phased TTT, 3 phases × 833-doc cumulative prefix, prefix=2500 total, LoRA rank=80 on Q/K/V/O/MLP |
| Eval time | 455-509s of 600s budget (median 470s) |

## Requirements

See `requirements.txt`. FlashAttention 3 must be installed separately:

```bash
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

PyTorch 2.9.1+cu128, CUDA 12.8, 8×H100 80GB SXM. lrzip system binary required (`apt-get install lrzip`).

## Files

- `train_gpt.py` — full training script (~3,750 lines). Configurable via env vars; defaults reproduce this submission.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — full per-seed run logs.
- `submission.json` — structured metadata (val_bpb, std, per-seed, comparison).
- `requirements.txt` — minimal Python deps (FA3 + lrzip noted separately).
- `lossless_caps.py` — bijective lowercase + private-use-area sentinel pre-encoding (caseops infrastructure, ~28 KB).
- `prepare_caseops_data.py` — CaseOps-tokenized FineWeb shard prep + per-token byte sidecar (~7 KB).
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — SentencePiece model used by CaseOps (~367 KB).

## Reproducing

```bash
DATA_DIR=./data \
VOCAB_SIZE=8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
CASEOPS_ENABLED=1 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
SPARSE_ATTN_GATE_SCALE=0.5 \
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

Implementation lineage stacks decisions from a long sequence of community PRs. Most directly:

- [PR #1797](https://github.com/openai/parameter-golf/pull/1797) by @dexhunter — sparse attention head-output gate, MIN_LR floor, fused softcapped CE Triton kernel, Polar-Express Newton-Schulz tuples, LQER asymmetric int4. This submission's architecture is closest to PR #1797's, with the SmearGate cross-doc fix applied and 9 hparam values re-tuned via multi-seed greedy forward-selection on 8×H100 real.
- [PR #1787](https://github.com/openai/parameter-golf/pull/1787) by @nprime06 — Polar Express NS, MIN_LR=0.10, sparse attention gate, fused softcapped CE.
- [PR #1729](https://github.com/openai/parameter-golf/pull/1729) by @romeerp — sp8192 lossless caps caseops v1 reserved tokenizer + tapered weight decay infra.
- [PR #1667](https://github.com/openai/parameter-golf/pull/1667) by @MarioPaerle / @classiclarryd — AttnOutGate / GatedAttn (G1) / SmearGate.
- [PR #1626](https://github.com/openai/parameter-golf/pull/1626) — fused LeakyReLU-square MLP Triton kernel, parallel decoder, per-group compression port.
- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) — LQER asymmetric quant-error correction.
- [PR #1344](https://github.com/openai/parameter-golf/pull/1344) — Polar-Express Newton-Schulz coefficients.
- [PR #1019](https://github.com/openai/parameter-golf/pull/1019) — depth recurrence, partial RoPE, QK-Gain init, logit softcap.
- [PR #478](https://github.com/openai/parameter-golf/pull/478) by @gowtham0992 — XSA-all.
- [PR #289](https://github.com/openai/parameter-golf/pull/289) — U-Net skip connections.
