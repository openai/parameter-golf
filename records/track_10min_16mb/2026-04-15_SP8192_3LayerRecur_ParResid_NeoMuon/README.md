# SP8192 + 3-Layer Recurrence + Parallel Residuals + NeoMuon + Stride-16

**val_bpb = TBD** (pending first full run) | **~15.9 MB** (estimated) | 8xH100 SXM

## Status
🚧 **Draft submission** — Initial run in progress. BPB results will be updated after the first full 8×H100 training run completes.

## Key Techniques

1. **SP8192 Tokenizer** — SentencePiece BPE with 8192 vocab for better byte-level compression ratio
2. **3-Layer Depth Recurrence** — Layers 3-5 looped, creating 17 virtual layers from 11 physical layers
3. **Parallel Residuals** (layers 7+) — GPT-J style: attention and MLP read from same pre-residual input
4. **QK-Gain 5.25** — Learnable per-head query scaling for stable attention
5. **NeoMuon Backend Steps (3)** — Overlaps optimizer computation with data loading, recovering ~180 extra training steps within the 600s wallclock
6. **Evaluation Stride=16** — Finer sliding window stride vs the common stride=64, yielding ~0.015-0.025 free BPB improvement
7. **GPTQ SDClip** — Mixed quantization: int5 MLP, int6 attention, int8 embeddings with std-deviation-based clipping
8. **EMA 0.9965 + WD 0.095** — Tuned regularization following established best practices

## Architecture

- **Model:** 11L × 512d × 8H / 4KV, MLP 4×, tied embeddings
- **Activation:** LeakyReLU(0.5)²
- **Position encoding:** Partial RoPE (16/64 dims)
- **Depth recurrence:** Layers 3-5 looped (activated at ~35% of training)
- **Parallel residuals:** Layers 7-10 (GPT-J style)
- **Logit softcap:** 30.0

## Training

- **Optimizer:** MuonEq-R (row-normalized Muon, Newton-Schulz) + AdamW for embeddings/scalars
- **Schedule:** Linear warmup (200 steps) → peak LR → cosine warmdown (final 3500 steps)
- **LR:** Matrix=0.022, Embed=0.03, Tied-Embed=0.05
- **Muon momentum:** 0.85 → 0.99 warmup over 500 steps
- **EMA decay:** 0.9965
- **Weight decay:** 0.095
- **Batch:** 524288 tokens/step

## Quantization & Compression

- **GPTQ-lite** with SDClip: `clip = k * std(row)` for principled clipping
- MLP: int5 (k=10.0), Attention: int6 (k=12.85), Embeddings: int8 (k=20.0)
- Byte-shuffle (group=4) + Brotli-11 compression
- LZMA code wrapper

## Planned Improvements

- [ ] Score-first TTT (SGD, 3 epochs per 32K chunk)
- [ ] 3-seed sweep for statistical significance
- [ ] Hyperparameter sweep on eval stride, TTT learning rate

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning (PR #1445)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` (to be added after first successful run)
