# Record candidate: 11L XSA + LQER + SparseAttnGate + BOS-fixed SmearGate + NEFTune + Phased-TTT (4 phases, prefix=3000, LoRA-128)

**val_bpb = 1.06035** (3-seed mean, std 0.00044) | **~15.16 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Steps | ms/step | Pre-quant val_bpb | Post-quant val_bpb | **Post-TTT val_bpb** | Artifact |
|------|-------|---------|-------------------|--------------------|----------------------|----------|
| 42   | 4,976 | 120.5   | 1.06787           | 1.07652            | **1.05980**          | 15,897,143 |
| 0    | 4,921 | 121.8   | 1.07037           | 1.07912            | **1.06038**          | 15,894,185 |
| 314  | 4,940 | 121.4   | 1.06989           | 1.07870            | **1.06087**          | 15,893,797 |
| **Mean** | **4,946** | **121.2** | | | **1.06035** | 15,895,042 |

3-seed std: 0.00044 BPB. Eval time per seed: 472–528 s of the 600 s budget (mean 509 s).

## Lineage

Built on top of [`track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`](track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/) (val_bpb 1.06108) — the strongest parent and the source of the entire architecture + quantization + per-group compression stack. Other ancestors traversed: [`track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT`](track_10min_16mb/2026-04-08_SP8192_ParallelResid_ScoreFirstTTT/) (val_bpb 1.0822), [`track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow`](track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/), [`track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR`](track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/), and `10min_16mb/2026_04_16_SmearGate_Attention_Output_Gate_Score-First_TTT` (external — no folder under `track_10min_16mb/`).

vs. the strongest parent (1.06108) this submission adds **NEFTune embedding noise** and a **z-loss regularization term** during training, and bumps the phased-TTT settings (LoRA rank 80→128, prefix docs 2500→3000, num phases 3→4). The architecture is unchanged.

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4× (2048) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Fused MLP kernel | LeakyReLU-square Triton | [#1530](https://github.com/openai/parameter-golf/pull/1530) |
| Attention | Standard FA3, GQA 2:1 | Baseline |
| XSA | All 11 layers (`xsa_last_n=11`) | [#478](https://github.com/openai/parameter-golf/pull/478) |
| RoPE | Partial (16/64 dims) + YaRN | [#315](https://github.com/openai/parameter-golf/pull/315) |
| LN Scale | 1/√(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| QK Gain init | 5.0 (per-head learned) | concept from [#259](https://github.com/openai/parameter-golf/pull/259); 5.0 default from [#1276](https://github.com/openai/parameter-golf/pull/1276) |
| U-Net skips | Encoder-decoder skip connections + skip gates | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Parallel decoder | 2-lane parallel from layer 8+, lane mix learned | [#1530](https://github.com/openai/parameter-golf/pull/1530) (parallel residuals) |
| Depth recurrence | Loop layers 3–5, run 3× once `frac >= 0.35` | [#1344](https://github.com/openai/parameter-golf/pull/1344) |
| Logit softcap | 30 | Gemma2-style; in upstream baseline |
| Sparse attention gate | Narrow head-output gate, gate_window=12 | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| SmearGate (BOS-fixed) | Position-mixing gate with `not_bos` mask | [#1667](https://github.com/openai/parameter-golf/pull/1667) + parent [track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611](track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/) (BOS leak fix) |
| Polar-Express Newton-Schulz | Muon, 5 steps, per-iter minimax tuples | [#1344](https://github.com/openai/parameter-golf/pull/1344) → [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| MIN_LR floor | 0.10 (warmdown LR floor) | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| Fused softcapped CE Triton kernel | Single-pass training-only | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| LQER asymmetric int4 | Rank-4 quant-error correction on top-3 tensors | [#1797](https://github.com/openai/parameter-golf/pull/1797) |
| Per-group compression | lrzip zpaq + L1 similarity-sort row reordering on hot tensors + brotli on the remainder | parent [track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611](track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/) |
| Quantization | GPTQ int6 + int7 embed + int8-per-row attn-gate | int7 embed from [#1586](https://github.com/openai/parameter-golf/pull/1586); int8-per-row attn-gate from [#1736](https://github.com/openai/parameter-golf/pull/1736) (`GATED_ATTN_QUANT_GATE`) |
| TTT | Phased TTT eval, **4** cumulative phases, LoRA per-doc reset, **rank 128**, **prefix=3000 docs** | concept [#1610](https://github.com/openai/parameter-golf/pull/1610) → multi-phase global SGD [#1626](https://github.com/openai/parameter-golf/pull/1626) → adopted in [#1736](https://github.com/openai/parameter-golf/pull/1736); 4-phase / rank-128 / 3000-prefix retune **this work** |
| Tokenizer | sp8192 lossless caps caseops v1 reserved | [#1729](https://github.com/openai/parameter-golf/pull/1729) |
| **NEFTune embedding noise** | **alpha=5.0, training-only, gated off during TTT** | **this work** |
| **Z-loss regularization** | **weight 1e-4 on `mean(LSE^2)`, computed from fused softcapped-CE LSE** | **this work** |

## What this submission adds over the parents

### NEFTune embedding noise (alpha=5.0)

Adds uniform random noise scaled by `alpha / sqrt(seq_len * dim)` (alpha=5.0) to the token embeddings during the training forward pass. The noise is gated by an `_in_ttt` flag so it is **disabled during phased-TTT** (where the model fine-tunes on validation prefix and noise would just inject loss).

Per the original NEFTune paper (Jain et al., 2023, arXiv:2310.05914), this acts as a strong embedding-level regularizer with negligible training-time overhead. None of the parents in `seed_record_ids` use NEFTune.

Code: [train_gpt.py:249](train_gpt.py#L249), [train_gpt.py:1079](train_gpt.py#L1079), [train_gpt.py:1277-L1280](train_gpt.py#L1277-L1280).

```python
neftune_alpha = float(os.environ.get("NEFTUNE_ALPHA", 5.0))
...
if self.training and self.neftune_alpha > 0 and not self._in_ttt:
    seq_len = max_seqlen if max_seqlen > 0 else x.size(1)
    noise = torch.rand_like(x) * 2.0 - 1.0
    x = x + noise * (self.neftune_alpha / math.sqrt(seq_len * x.size(-1)))
```

### Z-loss regularization (weight 1e-4)

Adds an auxiliary `weight * mean(LSE^2)` term to the training loss, where `LSE` is the per-token log-sum-exp of the (softcapped) logits. This penalizes drift in the logit normalization, which keeps `softmax` numerically well-conditioned and stops the partition function from inflating during long FP8/BF16 training runs. The standard PaLM-style trick (PaLM, Chowdhery et al. 2022).

The crucial integration detail: when `FUSED_CE_ENABLED=1` (the default on this stack), the fused softcapped-CE Triton kernel already computes the per-token LSE as a byproduct, so the z-loss term is essentially free — no second logits pass. The fallback path (`fused_ce_enabled=False`) computes `torch.logsumexp` explicitly on the materialized FP32 logits.

None of the parents in `seed_record_ids` use a z-loss term.

Code: [train_gpt.py:251](train_gpt.py#L251), [train_gpt.py:1083](train_gpt.py#L1083), [train_gpt.py:1399-L1410](train_gpt.py#L1399-L1410).

```python
z_loss_weight = float(os.environ.get("Z_LOSS_WEIGHT", 1e-4))
...
if self.fused_ce_enabled:
    losses, lse = torch.ops.pgsubmission1draft7fusedce.softcapped_ce(
        logits_proj.reshape(-1, logits_proj.size(-1)),
        flat_targets,
        float(self.logit_softcap),
    )
    return losses.mean() + self.z_loss_weight * (lse**2).mean()
```

## Hyperparameter stack

Three phased-TTT hparams retuned on top of the 1.06108 parent's stack. Tuning was multi-seed (3 seeds: 42, 0, 314); all three changes monotonically improve the 3-seed mean.

| hparam | value | default (parent 1.0611) | rationale |
|---|---|---|---|
| NEFTUNE_ALPHA | 5.0 | 0.0 (off) | Training-time embedding regularizer; disabled during TTT via `_in_ttt`. Default alpha from the NEFTune paper. |
| TTT_LORA_RANK | 128 | 80 | Higher-capacity LoRA adapters fit the longer per-phase prefix better; recovers ~0.0006 BPB on 3-seed mean. |
| PHASED_TTT_PREFIX_DOCS | 3000 | 2500 | Longer per-phase prefix (still fits inside the 600 s eval budget — 472–528 s observed). |
| PHASED_TTT_NUM_PHASES | 4 | 3 | One extra cumulative phase (boundaries at ~750/1500/2250/3000 docs) given the longer prefix. |

## Training

- 4,946 ± 22 steps in 600 s on 8xH100 SXM (121.2 ms/step mean).
- Optimizer: Polar-Express Muon (5 steps) on matrix params; Adam (β1=0.9, β2=0.99) on tied embeddings (lr=0.03) and scalars (lr=0.02).
- Schedule: warmup=20 steps, warmdown_frac=0.85, MIN_LR=0.10, MATRIX_LR=0.026, GRAD_CLIP_NORM=0.3.
- EMA decay = 0.9965; EMA weights applied at end of training.
- Z-loss weight = 1e-4, computed via the LSE returned by the fused softcapped-CE Triton kernel (training-only auxiliary term).
- NEFTune (α=5.0) injected on `tok_emb` output during training only.

## Quantization

- GPTQ int6 on matrix tensors with per-layer adaptive clipping (MLP_CLIP_SIGMAS=11.5, ATTN_CLIP_SIGMAS=13.0, EMBED_CLIP_SIGMAS=14.0).
- int7 on tied embeddings.
- int8-per-row on the small attention-gate tensors (`GATED_ATTN_QUANT_GATE=1`).
- LQER asymmetric rank-4 quant-error correction on the top-3 tensors with the highest reconstruction error (`LQER_TOP_K=3`, `LQER_FACTOR_BITS=4`, `LQER_ASYM_GROUP=64`).
- Compression: `COMPRESSOR=pergroup` — buckets weights by role, L1 similarity-sorts hot 2D groups, runs `lrzip -z -L 9` (ZPAQ context-mixing) on each blob, brotli on the remainder + code wrapper.

## TTT (Test-Time Training)

- Phased global-SGD TTT with **4 cumulative phases** at doc boundaries (max prefix = 3000 docs).
- Per-phase Adam (β1=0, β2=0.99, weight decay=0.5) on **LoRA rank-128** adapters injected on Q/K/V/O + MLP + lm_head.
- Per-doc LoRA reset between phases.
- NEFTune is force-disabled by the `_in_ttt` flag.
- Eval time: 472.8–527.8 s (mean 508.7 s) of the 600 s budget.

## Compliance

Track-B legal-eval conditions:

- **Train ≤ 600 s** ✓ — strict 600 s wallclock cap; all three seeds stop at ~599.5 s.
- **Artifact ≤ 16 MB** ✓ — max artifact 15,897,143 bytes (~15.16 MB).
- **Eval ≤ 600 s** ✓ — max eval 527.8 s.
- **No SLOT** ✓
- **No pre-quant TTT** ✓ — TTT runs *after* GPTQ + LQER.
- **No ETLB** ✓
- **No n-gram cache** ✓
- **Score-first TTT** ✓ — every token scored before the per-phase weight update.
- **3 seeds** ✓ — seeds 42, 0, 314.

## Reproduction

### Environment / install

```bash
# Base: PyTorch 25.03 NGC image (nvcr.io/nvidia/pytorch:25.03-py3)
# or a clean Python 3.12 + CUDA 12.8 env.

# System dep for COMPRESSOR=pergroup
apt-get update && apt-get install -y lrzip

# PyTorch + Python deps
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# FlashAttention 2 + 3 (separate wheels for cu128/torch2.9)
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
pip install "https://download.pytorch.org/whl/cu128/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
```

### Data

The script expects a CaseOps-tokenized FineWeb shard tree at `$DATA_PATH` and the matching SentencePiece tokenizer at `$TOKENIZER_PATH`. Both are produced by the parent record's `prepare_caseops_data.py` (see [`track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/prepare_caseops_data.py`](track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/prepare_caseops_data.py)) and the tokenizer model at [`track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`](track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model).

### Run

```bash
# --- Train ---
SEED=42 \
DATA_PATH=./data/datasets \
TOKENIZER_PATH=./data/tokenizers \
CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
NEFTUNE_ALPHA=5.0 Z_LOSS_WEIGHT=1e-4 \
EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
GRAD_CLIP_NORM=0.3 WARMUP_STEPS=20 WARMDOWN_FRAC=0.85 BETA2=0.99 \
MUON_BACKEND_STEPS=5 \
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 GATE_WINDOW=12 \
SMEAR_GATE_ENABLED=1 GATED_ATTN_QUANT_GATE=1 \
LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
FUSED_CE_ENABLED=1 COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_gpt.py --mode train

# --- Eval (phased-TTT) ---
SEED=42 \
DATA_PATH=./data/datasets \
TOKENIZER_PATH=./data/tokenizers \
CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
TTT_ENABLED=1 TTT_LORA_RANK=128 TTT_CHUNK_SIZE=48 \
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=3000 PHASED_TTT_NUM_PHASES=4 \
GLOBAL_TTT_MOMENTUM=0.9 \
COMPRESSOR=pergroup \
torchrun --standalone --nproc_per_node=8 train_gpt.py --mode eval
```

Re-run with `SEED=0` and `SEED=314` for the other two seeds.

## Included Files

- `README.md` — this file.
- `submission.json` — structured metadata.
- `train_gpt.py` — full training/eval script (≈3,830 lines), unchanged from the agent's `solution.py`.
- `requirements.txt` — Python deps for the install block.
- `train_seed42.log`, `train_seed0.log`, `train_seed314.log` — full per-seed run logs (train + GPTQ/LQER + phased-TTT eval).

## Credits

- [PR #1797](https://github.com/openai/parameter-golf/pull/1797) by @dexhunter — Smear Gate + LQER asymmetric rank-4 stacked on the PR #1787 base.
- [PR #1787](https://github.com/openai/parameter-golf/pull/1787) by @nprime06 — Polar-Express NS, MIN_LR=0.10, sparse attention gate, fused softcapped CE.
- [PR #1736](https://github.com/openai/parameter-golf/pull/1736) — CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT integration on top of PR #1530's stack.
- [PR #1729](https://github.com/openai/parameter-golf/pull/1729) by @romeerp — sp8192 lossless caps caseops v1 reserved tokenizer + tapered weight decay infra.
- [PR #1667](https://github.com/openai/parameter-golf/pull/1667) by @MarioPaerle — Reintroduced SmearGate (modded-nanogpt @classiclarryd style) + Attention Output Gate.
- [PR #1626](https://github.com/openai/parameter-golf/pull/1626) by @dexhunter — Multi-phase global SGD phased-TTT.
- [PR #1610](https://github.com/openai/parameter-golf/pull/1610) — VarLenAttn + originator of phased TTT (PhasingTTT).
- [PR #1586](https://github.com/openai/parameter-golf/pull/1586) — Per-Layer Adaptive GPTQ Clip + int7 Embeddings + MATRIX_LR=0.026.
- [PR #1530](https://github.com/openai/parameter-golf/pull/1530) by @samacqua — Variable-length attention, fused LeakyReLU² MLP Triton kernel, parallel residuals, doc-based LoRA TTT.
- [PR #1344](https://github.com/openai/parameter-golf/pull/1344) — Polar-Express Newton-Schulz coefficients + depth recurrence (Loop4-5).
- [PR #1276](https://github.com/openai/parameter-golf/pull/1276) — QK Gain default 5.0.
- [PR #493](https://github.com/openai/parameter-golf/pull/493) — LeakyReLU² activation.
- [PR #478](https://github.com/openai/parameter-golf/pull/478) by @gowtham0992 — XSA-all on all layers.
- [PR #315](https://github.com/openai/parameter-golf/pull/315) — Partial RoPE + LN Scale.
- [PR #289](https://github.com/openai/parameter-golf/pull/289) — U-Net skip connections.
- [PR #259](https://github.com/openai/parameter-golf/pull/259) — QK Gain concept.
- Parent record [`track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611`](track_10min_16mb/2026-04-27_SP8192_LQER_SparseGate_BOSSmearFix_9HpStack_1.0611/) by @codemath3000 — BOS-fixed SmearGate, per-group lrzip+brotli compression pipeline, and the 9-hparam stack this submission inherits.
- NEFTune embedding noise — concept from Jain et al., 2023 ([arXiv:2310.05914](https://arxiv.org/abs/2310.05914)); first integration on this stack is **this work**.
