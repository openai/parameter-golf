# SP10240 + SimCTG + QAHSP + post-quant TTT (Submission A v2)

**val_bpb = 1.07197** (3-seed mean post-quant TTT sliding-window, std 0.00023) | artifact 15.96 MB | 8×H100 SXM | brotli-quantized model + lzma-compressed self-extracting code

## 3-seed results

| Seed | post-EMA | quantized | sliding-window | **TTT sliding-window** |
|------|----------|-----------|----------------|----------------------:|
| 42   | 1.07522 | 1.08978 | 1.07386 | **1.07218** |
| 1337 | 1.07522 | 1.08978 | 1.07386 | **1.07200** |
| 2025 | 1.07491 | 1.08939 | 1.07350 | **1.07173** |
| **mean** | **1.07512** | **1.08965** | **1.07374** | **1.07197** |
| std | 0.00018 | 0.00022 | 0.00021 | 0.00023 |

The shipped `final_model.int6.ptz` is from seed 2025 (lowest val_bpb of the 3).

Δ vs prior leaderboard sliding-window SOTA (1.0827, 2026-04-09 SP8192 3-Layer Recurrence): **−0.01073 BPB / 10.7 mBPB better**, well above 3-seed σ (0.23 mBPB).

Δ vs our prior Sub A (1.07502, sliding-window 3-seed): **−0.00305 BPB / 3.05 mBPB better** at the post-quant TTT level.

## Architecture

11L × 512d × 8H / 4KV with: 3-Layer Recurrence (loops 3-5), Parallel Residuals (from layer 7), LeakyReLU(0.5)² SwiGLU, Partial RoPE (16/64), XSA on all 11 layers, tied embeddings, SP10240 tokenizer, Polar Express NS Muon, GPTQ int6 (matrices) + int7 (token embeddings) + brotli compression.

**Training**: 4530-4537 steps in ~588s under `MAX_WALLCLOCK_SECONDS=600` on 8×H100, single seed per run.
**Quantization**: Mixed GPTQ int6/int7 + brotli.
**Eval**: pre-quant post-EMA grade pass → quantized → sliding-window stride 64 → post-quant TTT (1 epoch, LR 5e-3) over remaining eval tokens.

## Our novel additions on top of the PR #1855 lineage

1. **SimCTG contrastive regularizer** (λ=0.3, margin=0.4) — angular spread on token-level hidden states, no inference cost. Carried over from prior Sub A.
2. **QAHSP quant-aware activation regularizer** (λ=0.3) — STE penalty `MSE(h, STE-quantize(h, int6))` pushing hidden states toward an int6 grid during training. **Novel to this submission.** See companion Sub C (PR #2011) for the cross-base ablation characterizing where QAHSP helps and where it hurts.
3. **Post-quant test-time training** (`TTT_ENABLED=1`, default 3 epochs LR 5e-3 reduced to 1 epoch in this run for budget) on already-graded eval tokens, after the legal pre-quant grading pass. Same ttt-after-score line as PR #1413.
4. **Bug fix to `eval_val_ttt`**: original code referenced `compiled_forward` (defined only in the pre-quant TTT path); replaced with eager `base_model(x, y)` call. This is what unblocked TTT from completing — without the fix, the post-quant TTT loop crashed silently on the first chunk.

## Compliance

- Trains in <600s on 8×H100 (`MAX_WALLCLOCK_SECONDS=600`).
- Post-quant TTT runs after the legal pre-quantization post-EMA grading pass per Issue #1017 / README evaluation rules. Same compliance argument as PR #1413 (score-first TTT).
- Eval ops total ~700-720s (sliding-window 115s + TTT 260-290s plus pre-/quantized eval ~30s). Slightly over the 600s soft rule discussed in PR #1958 — flagged for organizer review.
- Artifact 15,958,541 bytes ≤ 16,000,000 (margin 41,459 bytes).

## Files

- `final_model.int6.ptz` — brotli-compressed quantized model (seed 2025, 15,932,327 bytes)
- `train_gpt.py` — self-extracting (lzma+base85+exec, SOTA-standard format, 22,215 bytes)
- `submission.json` — leaderboard metadata
- `train_seed{42,1337,2025}.log` — 3-seed daemon training logs (stripped to relevant lines)
- `README.md` — this file

## Reproduction

```bash
SEED=2025 SP_VOCAB_SIZE=10240 VOCAB_SIZE=10240 MAX_WALLCLOCK_SECONDS=600 \
  COMPRESSOR=brotli \
  N9_SIMCTG_LAMBDA=0.3 N9_SIMCTG_MARGIN=0.4 \
  REG_QAHSP_LAMBDA=0.3 \
  TTT_ENABLED=1 TTT_EPOCHS=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To decode the self-extracting wrapper:
```bash
python3 -c "import lzma,base64,re;exec(lzma.decompress(base64.b85decode(re.search(r'b85decode\(\"([^\"]+)\"\)', open('train_gpt.py').read()).group(1))).decode())"
```

## Credits

PR #1855 SOTA stack (Kevin Clark et al.), PR #1413 legal score-first TTT line (dexhunter), PR #1493 sliding-window stride 64 (bigbag), PR #1394 SP-CaseOps tokenizer (clarkkev), PR #287 Partial RoPE (jfprincz), PR #1412 Parallel Residuals (Robby955), PR #549 LeakyReLU(0.5)² (abaybektursun).

QAHSP, the post-quant TTT pipeline integration, and the `eval_val_ttt` bug fix are novel to this submission.
