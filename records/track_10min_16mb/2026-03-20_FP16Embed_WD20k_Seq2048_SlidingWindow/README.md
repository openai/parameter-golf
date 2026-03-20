# 2026-03-20_FP16Embed_WD20k_Seq2048_SlidingWindow

*Non-record: Stacking proven community techniques with doc-isolated sliding window eval*

**val_bpb: 1.2045** (post int8+zlib roundtrip, doc-isolated sliding window eval)

---

## Hardware note

**This result is significantly limited by hardware speed.** Our RunPod 8×H100 SXM node ran at **70ms/step** — roughly 60% slower than the 43-44ms/step reported by the baseline and other entries. This is node-to-node variance within RunPod, not a code issue. The same configuration on a 44ms node would complete ~13,600 steps instead of our 8,528, likely scoring **~1.185-1.190**.

We include this result as-is for transparency. The official evaluation on OpenAI's standardized hardware will produce the canonical score.

---

## Techniques

All techniques are borrowed from published community entries. No novel contributions in this submission — it is a composition of proven ideas.

| Technique | Source | Purpose |
|-----------|--------|---------|
| FP16 tied embedding export | WarmdownQuantization (#65), FP16Embed (#42) | Avoid int8 compounding through input+output paths |
| MLP_HIDDEN=992 | WarmdownQuantization (#65) | Offset FP16 embed artifact cost |
| WARMDOWN_ITERS=20000 | WarmdownQuantization (#65) | Tighter weight distributions → less quant damage |
| TRAIN_SEQ_LEN=2048 | LongContextSeq2048 (#49) | Longer context during training |
| Tuned LRs (0.06/0.06/0.07) | WarmdownQuantization (#65) | Optimal for aggressive warmdown |
| GRAD_CLIP_NORM=1.0 | WarmdownQuantization (#65) | Training stability |
| MUON_BACKEND_STEPS=5 | WarmdownQuantization (#65) | Better with aggressive warmdown |
| MUON_MOMENTUM=0.99 | Int6 MLP3x STE QAT (#128) | Improved Muon convergence |
| Sliding window eval (stride=64) | SlidingWindowEval (#50) | Maximum context for scoring |
| Doc-isolated eval | LoRA TTT (#77) | No cross-document context bleed |

---

## Results

| Metric | Naive Baseline | This submission |
|--------|---------------|-----------------|
| Steps completed | ~13,800 | 8,528 (hardware-limited) |
| step_avg | 43.5ms | 70.4ms |
| Pre-quant val_bpb | 1.2172 | 1.2154 |
| **Post-quant val_bpb** | **1.2244** | **1.2045** |
| Improvement | — | -0.0199 BPB |
| Artifact bytes | 15,863,489 | 15,912,648 |
| Eval time | ~16s | 43s |

---

## Configuration

```
VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
MLP_HIDDEN=992 TIE_EMBEDDINGS=1 FP16_EMBED_EXPORT=1
TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048
ITERATIONS=20000 WARMDOWN_ITERS=20000 WARMUP_STEPS=20
MAX_WALLCLOCK_SECONDS=600
MATRIX_LR=0.06 SCALAR_LR=0.06 TIED_EMBED_LR=0.07
GRAD_CLIP_NORM=1.0 MUON_BACKEND_STEPS=5 MUON_MOMENTUM=0.99
QAT=0 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 DOC_ISOLATED_EVAL=1
```

## Command

```bash
export VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4
export MLP_HIDDEN=992 TIE_EMBEDDINGS=1 FP16_EMBED_EXPORT=1
export TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=2048
export ITERATIONS=20000 WARMDOWN_ITERS=20000 WARMUP_STEPS=20
export MAX_WALLCLOCK_SECONDS=600
export MATRIX_LR=0.06 SCALAR_LR=0.06 TIED_EMBED_LR=0.07
export GRAD_CLIP_NORM=1.0 MUON_BACKEND_STEPS=5 MUON_MOMENTUM=0.99
export QAT=0 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 DOC_ISOLATED_EVAL=1
export SEED=1337 RUN_ID=kitchen_sink_9L_seed1337
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Acknowledgments

This submission composes techniques entirely from community entries:
- @samuellarson (WarmdownQuantization #65) — warmdown, FP16 embed, tuned LRs
- @mattqlf (SlidingWindowEval #50) — sliding window evaluation
- @samacquaviva (LoRA TTT #77) — doc-isolated evaluation concept
- @spokane-way (LongContextSeq2048 #49) — seq2048 training
- @rsavitt (Int6 MLP3x STE QAT #128) — MUON_MOMENTUM=0.99
- Built with [Claude Code](https://claude.com/claude-code)

## Author

GitHub: [@mrdavtan](https://github.com/mrdavtan)
Date: 2026-03-20
