# Record Attempt: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + LeakyReLU(0.5)² + Partial RoPE + Legal Score-First TTT

**Target val_bpb: < 1.0810** (beating current SOTA) | **~15.99 MB** | 8xH100 SXM

## Novel Contributions vs Current SOTA (#1 at 1.0810)

This submission builds on the full proven stack from the top-5 leaderboard and adds:

1. **Partial RoPE (16/64 dims)** — Apply rotary embeddings only to the first 16 of 64 head dims. Saves representational budget for the remaining 48 dims, which act as unrotated "position-free" features. Confirmed improvement in #1 README architecture section.
2. **LeakyReLU(0.5)² activation** — Replaces ReLU² with LeakyReLU(negative_slope=0.5)². Verified improvement by @clarkkev in PR #1394 lineage.
3. **MLP 4x multiplier** — Wider MLP (4x vs 2x baseline) matching all top submissions.
4. **Layerwise LN scale** — Each RMSNorm has a learnable per-dimension scale. Adds expressivity at near-zero parameter cost.
5. **Sigmoid-gated U-Net skip connections** — Sigmoid gate controls skip weight (vs linear scalar in baseline). More flexible skip contribution.
6. **Progressive recurrence** — Phase 1 at 35% training, Phase 2 at 65% (vs simultaneous in some submissions). Gentler adaptation per phase.
7. **MuonEq-R with weight decay** — Row-normalized Muon with AdamW-style weight decay (WD=0.095).
8. **EMA decay 0.9965** — Exponential moving average of weights for smoother final model.
9. **Warmdown 72%** of training budget (tuned from #1 @X-Abhishek-X PR #1445).

## Full Stack

- **SP8192 tokenizer** — SentencePiece BPE 8192 vocab (biggest single gain)
- **Architecture**: 11L × 512d × 8H/4KV, MLP 4×, LeakyReLU(0.5)², Partial RoPE (16/64), layerwise LN scale, tied embeddings, logit softcap=30.0
- **Depth recurrence**: Loop layers 3–5 three times total (3-layer recurrence), progressive activation at 35%/65% training
- **Parallel residuals**: From layer 7, attention and MLP read same pre-residual input (GPT-J style), blended by learned `lane_merge` scalar
- **Optimizer**: MuonEq-R (row-normalized Muon + Newton-Schulz 5 steps), AdamW for embeddings/scalars, WD=0.095, MLR=0.022
- **Quantization**: SDClip int6 for attention/MLP matrices (k=12.85), int8 for embeddings (k=20.0)
- **Compression**: Brotli-11 (falls back to zlib-9 if brotli not installed)
- **TTT**: Legal Score-First SGD (lr=0.005, momentum=0.9, epochs=3, cosine LR decay across chunks)

## Architecture Detail

```
Encoder: layers 0-4, collect U-Net skips
Decoder: layers 5-10, apply sigmoid-gated skips
Recurrence (phase 1, 35%+): extra loop of layers 3-5
Recurrence (phase 2, 65%+): second extra loop of layers 3-5
Parallel residuals: layers 7-10 use GPT-J parallel attn+MLP
```

## TTT Compliance (Track B — Legal Score-First)

Per Issue #1017 four conditions:

- **Condition 1 (Causality)**: Strict causal forward pass throughout. Each position scored from prefix only.
- **Condition 2 (Normalized distribution)**: Standard softmax over full 8192 vocab. No logit biasing.
- **Condition 3 (Score before update)**: Each 32K-token chunk scored under `torch.inference_mode()` BEFORE any SGD update. Score-first pattern follows PR #549 precedent.
- **Condition 4 (Single pass)**: Each token scored exactly once. No rescoring.

Additional:
- No SLOT, no n-gram cache, no pre-quant TTT, no ETLB
- Cosine LR decay across chunks for stable TTT convergence
- Gradient clipping at 1.0 during TTT

## Run Command

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download SP8192 data
rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128

# Run 3 seeds
for SEED in 42 314 999; do
  SEED=$SEED \
  DATA_PATH=./data/datasets/fineweb10B_sp8192 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
  VOCAB_SIZE=8192 \
  QK_GAIN_INIT=5.25 \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

## Credits

- **@clarkkev** — SP8192 + GPTQ SDClip + MuonEq-R + depth recurrence (PR #1394) — primary base
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)
- **@kingoflolz** — GPT-J parallel residual architecture

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`