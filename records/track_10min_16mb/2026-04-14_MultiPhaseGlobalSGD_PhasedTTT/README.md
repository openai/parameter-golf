# Record: VarLen Attention + Fused MLP + Multi-Phase Global SGD TTT

**val_bpb: 1.07193** (3-seed mean, std 0.00063) | **2.76890 nats** | **~15.93 MB** | 8xH100 SXM, 596s train + ~331s TTT eval

**Improvement over PR #1530** (@samacqua, 1.07336 BPP): -0.00143 BPP / -0.00370 nats

**Improvement over merged SOTA** (PR #1493, 1.0810 BPP): -0.00907 BPP / -0.02344 nats

## Results (8xH100 80GB SXM, PyTorch 2.9.1+cu128, Phased TTT)

| Seed | Steps | Pre-TTT BPB | **Post-TTT BPB** | TTT gain | TTT time | Artifact |
|------|-------|-------------|------------------|----------|----------|----------|
| 42 | 4,971 | 1.08502 | **1.07280** | -0.01222 | 329.0s | 15,932,897 |
| 0 | 4,967 | 1.08392 | **1.07134** | -0.01258 | 332.1s | 15,939,841 |
| 1234 | 4,977 | 1.08517 | **1.07164** | -0.01353 | 332.8s | 15,932,419 |
| **Mean** | | | **1.07193** | -0.01278 | | |

### Supplemental Diagnostics

| Seed | Pre-EMA BPB | Post-EMA BPB | Post-Quant BPB | Post-TTT BPB | val_loss (nats) | Code size | Total | Train time | Eval time |
|------|-------------|-------------|----------------|-------------|-----------------|-----------|-------|------------|-----------|
| 42 | 1.0733 | 1.07257 | 1.08502 | 1.07280 | 2.77116 | 122,168 | 15,932,897 | 596.1s | 329.0s |
| 0 | 1.0723 | 1.07108 | 1.08392 | 1.07134 | 2.76739 | 122,168 | 15,939,841 | 596.1s | 332.1s |
| 1234 | 1.0713 | 1.07174 | 1.08517 | 1.07164 | 2.76815 | 122,168 | 15,932,419 | 596.2s | 332.8s |

## Key Innovation: Multi-Phase Global SGD

This submission introduces **multi-phase global SGD** during phased TTT evaluation. While PR #1610 (@romeerp) introduced single-phase global SGD (score prefix docs, run one round of SGD, score suffix), we extend this to **N phases** with interleaved scoring and adaptation:

1. Split 2000 prefix docs into 3 equal chunks (~666 docs each)
2. Score chunk 1 with base model (score-before-update)
3. Run distributed SGD on scored chunk 1
4. Score chunk 2 with improved model
5. Run SGD on scored chunks 1+2
6. Score chunk 3 with further improved model
7. Run SGD on all scored prefix docs
8. Score remaining 48,000 suffix docs with fully adapted model

This progressively improves the base model through multiple adaptation rounds while maintaining strict score-before-update legality. Each phase scores new tokens BEFORE any SGD update uses them.

```python
# Key code (simplified)
for phase_idx in range(num_phases):
    boundary = boundaries[phase_idx]  # [666, 1333, 2000]
    # Score docs from previous boundary to this boundary
    for doc in docs[prev_boundary:boundary]:
        score(doc)  # score-first, no adaptation yet
    # SGD on ALL scored docs so far
    global_sgd(scored_docs[:boundary])
# Score remaining 48000 suffix docs with adapted model
for doc in suffix_docs:
    score(doc)
```

**3-phase gives -0.00081 BPP over 1-phase** (1.07190 vs 1.07271, same seed). More phases (6+) cause overfitting on small subsets.

## Changes from PR #1530 Baseline

| Change | Source | Effect |
|--------|--------|--------|
| Multi-phase global SGD (3-phase) | **Novel (this work)** | -0.0008 BPP eval-time |
| Trimmed GPTQ (reserve=4s, calib=16) | PR #1586 (@dexhunter) | -0.0013 BPP, +72 training steps |
| MATRIX_LR=0.026 | PR #1586 (@dexhunter) | -0.0003 BPP (sharp optimum) |
| Per-layer adaptive GPTQ clip (MLP=12, Attn=13, Emb=15) | PR #1586 (@dexhunter) | Better quant-vs-bytes tradeoff |
| int7 embeddings (EMBED_BITS=7) | PR #1586 (@dexhunter) | -530 KB artifact, ~0 BPP cost |
| WARMDOWN_FRAC=0.75 | PR #1560 (@dexhunter) | More warmdown iterations |
| Dead code removal | This work | -1.9 KB compressed code size |

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 heads, 4 KV heads) | Baseline |
| MLP | 4x (2048) with LeakyReLU(0.5)^2, Triton fused | PR #1530 @samacqua |
| Attention | VarLen (flash_attn_varlen_func), causal | PR #1530 @samacqua |
| Recurrence | 3-layer loop (L3-5), encoder+decoder | PR #1523 @EthanYangTW |
| Skip connections | U-Net encoder-decoder | Baseline |
| RoPE | Partial (16/64 dims) | Baseline |
| Optimizer | Muon (momentum=0.97) + AdamW | PR #1530 @samacqua |
| EMA | Decay 0.9965 | Baseline |
| Quantization | Full Hessian GPTQ int6 + int7 embeddings | PR #1530, enhanced |
| Compression | Brotli quality=11 | Baseline |
| TTT | Phased LoRA TTT with multi-phase global SGD | **This work** + PR #1530 + PR #1610 |

## Rule Compliance

- **Condition 1 (Causal):** All attention uses `causal=True`. No future token leakage.
- **Condition 2 (Normalized):** All scoring uses `F.cross_entropy` (full softmax over vocabulary).
- **Condition 3 (Score-before-update):** Prefix docs are scored BEFORE any global SGD update. Each phase scores new docs first, then runs SGD on already-scored data only.
- **Condition 4 (Single pass):** Single left-to-right pass over validation data. No rescoring.
- **No val data during training:** Training uses only fineweb train shards.
- **Full validation split:** All fineweb_val shards loaded via sorted glob.
- **Byte accounting:** Tokenizer-derived byte counts including boundary/leading-space handling.

## Requirements

Python >= 3.12 (PEP 701 f-strings). Flash Attention 3 (Hopper) required.

```bash
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install sentencepiece brotli
```

## Run Command

```bash
for seed in 42 0 1234; do
  NCCL_NET=Socket SEED=$seed \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > train_seed${seed}.log 2>&1
done
```

## Lineage

```
PR #1493 (Merged SOTA, 1.0810) by @bigbag
  -> PR #1523 (1.0778) by @EthanYangTW — triple recurrence, parameter banking
    -> PR #1530 (1.07336) by @samacqua — varlen attention, fused MLP, doc-TTT
      -> PR #1610 (1.07281) by @romeerp — phased TTT (single-phase global SGD)
        -> This work (1.07193) adds:
            +-- Multi-phase global SGD (3-phase, novel)
            +-- Trimmed GPTQ (reserve=4s, calib=16)
            +-- MATRIX_LR=0.026 (sharp optimum)
            +-- Per-layer adaptive GPTQ clip
            +-- int7 embeddings
            +-- Dead code removal
```

## Credits

- @samacqua — PR #1530 base (VarLen attention, fused MLP, doc-TTT)
- @romeerp — PR #1610 phased TTT concept (single-phase global SGD)
- @EthanYangTW — PR #1523 triple recurrence, parameter banking
- @bigbag — PR #1493 merged SOTA baseline
- @abaybektursun — PR #549 legal TTT framework

## Included Files

- `train_gpt.py` — Complete training + eval script (122,168 bytes)
- `submission.json` — Metadata
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — Full seed logs
