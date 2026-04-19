# GatedDeltaNet (FLA) + Legal Score-First TTT + Brotli-11 Compression

**val_bpb: 1.01080** (3-seed mean, std 0.00115) | **~15.53 MB** (VALID; PR #1698 is 16.5-16.6 MB → INVALID) | 8×H100 80GB SXM

## Summary

This submission is built directly on @arsenis-cmd's PR #1698 (GatedDeltaNet + Legal Score-First TTT, 1.00995 BPB). PR #1698 is currently **invalid** because all 3 of its artifacts exceed the 16,000,000-byte decimal cap (16.47–16.60 MB); it cannot be merged as a record until that is fixed.

Two changes vs PR #1698:

1. **Compression: zstandard (level 22) → brotli (quality 11)**. This is the primary fix. Brotli compresses int6-GPTQ byte streams 5-8% better than zstandard on this model (verified: same bits, same weights → 15.54 MB vs 16.60 MB on seed 42). This brings the full-quality `clip_range=31` artifact comfortably under the 16,000,000-byte cap.

2. **Optional macro-phase SGD TTT**: multi-phase consolidation layered on top of PR #1698's per-chunk SGD TTT, inspired by PR #1700's Multi-Phase Global SGD TTT. Disabled in the scored run (`TTT_MACRO_PHASES=0`) — on this base it was within noise (seed 42: -0.00999 with macro vs -0.01012 without, indistinguishable), but the infrastructure is left in place for future tuning.

No other changes: architecture (K_KVShare_Wider, 10-layer GDN, 544d, 8H, KV-share stride=2), training (7000-step budget, Muon + Adam, EMA 0.997, SWA, Late QAT), and TTT (score-first SGD lr=0.005, 3 epochs/chunk, freeze first 2 blocks) are identical to PR #1698.

## Results (8xH100 80GB SXM, torch 2.9.1+cu128)

| Seed | EMA BPB | Pre-TTT BPB | **Post-TTT BPB** | TTT Gain | Artifact |
|------|---------|-------------|------------------|----------|----------|
| 42   | 1.00257 | 1.02189     | **1.01205**      | -0.00984 | 15,543,829 B |
| 314  | 1.00033 | 1.01903     | **1.00978**      | -0.00925 | 15,527,172 B |
| 999  | 1.00146 | 1.01986     | **1.01056**      | -0.00930 | 15,524,066 B |
| **Mean** | **1.00146** | **1.02026** | **1.01080 (std 0.00115)** | **-0.00946** | 15,531,689 B |

Beats merged SOTA (1.0810, PR #1493) by **-0.07020 BPB / ~-0.04867 nats**, clearing the 0.005-nat (~0.0072 BPB) threshold by a 10x margin. Seed 314 alone (1.00978) is lower than PR #1698's entire 3-seed mean of 1.00995.

## Why this is the first valid sub-1.02 submission

PR #1698's three artifacts:
- seed 42: 16,600,916 B (600,916 over cap)
- seed 314: 16,548,775 B (548,775 over cap)
- seed 999: 16,474,250 B (474,250 over cap)

All violate the 16,000,000-byte decimal cap (Rules: "The cap is decimal 16MB, i.e. 16,000,000 total bytes, not 16 MiB / 16,777,216 bytes"). The author (@arsenis-cmd) acknowledged this in the PR comments and proposed reducing `clip_range` from 31 to 24. That fix works but introduces ~+0.015 BPB quantization penalty because more weights get clipped.

This PR takes a different fix: keep `clip_range=31` (no extra quantization penalty) and replace zstandard-22 with brotli-11 for artifact compression. Brotli saves ~6% on this byte distribution, bringing all three artifacts well under 16,000,000 bytes with zero quality loss.

## Compliance (Issue #1017 Track A)

- **Condition 1 (Causality)**: Sliding-window eval is strictly causal (same as PR #1698)
- **Condition 2 (Normalized)**: Standard softmax over full vocab
- **Condition 3 (Score-before-update)**: Each 32K-token chunk is fully scored under `torch.inference_mode()` BEFORE any SGD update (same as PR #1698)
- **Condition 4 (Single pass)**: Each token scored exactly once; no rescoring across passes

The `ttt_epochs=3` multi-epoch SGD on already-scored tokens is the same pattern used in PR #1698, PR #1700, and merged SOTA PR #1493 (`TTT_EPOCHS=3`). The interpretation of Condition 4 vs multi-epoch post-score training is pending organizer clarification — see discussion on PR #1698.

## Reproduction

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install numpy sentencepiece zstandard brotli triton==3.5.1
pip install flash-linear-attention==0.4.2 fla-core==0.4.2 transformers==5.5.4 tokenizers==0.22.2 safetensors==0.7.0

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

for seed in 42 314 999; do
  SEED=$seed \
  ARCH_MODE=K VOCAB_SIZE=8192 \
  DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  MAX_WALLCLOCK_SECONDS=600 \
  INT6_CLIP_RANGE=31 \
  COMPRESSOR=brotli \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=2 TTT_MOMENTUM=0.9 \
  TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
  TTT_MACRO_PHASES=0 \
  torchrun --standalone --nproc_per_node=8 train_gdn_7k.py
done
```

## Credits

- **@arsenis-cmd** (PR #1698) — full base: GatedDeltaNet integration, K_KVShare_Wider config, all training and score-first TTT infrastructure. This submission changes only compression + adds optional (disabled-in-scored-run) macro-phase hook.
- **@resouer** (PR #1687) — K_KVShare_Wider architecture and FLA integration, consumed by PR #1698.
- **Flash Linear Attention** by @sustcsonglin — GatedDeltaNet Triton kernel (`fla-core==0.4.2`).
- **@Christopher-Lee-McClendon** (PR #461) — legal score-first TTT framework.
- **@jorge-asenjo** (PR #1700) / **@dexhunter** (PR #1626) — Multi-Phase Global SGD TTT concept (provides the macro-phase hook design; not used in scored run).
