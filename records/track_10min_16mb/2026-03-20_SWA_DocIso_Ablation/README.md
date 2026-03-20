# 2026-03-20_SWA_DocIso_Ablation

*Non-record: Two clean ablations — SWA for quantization robustness and doc-isolated sliding window eval*

**Key findings:**
1. **SWA does not improve int8 quantization** under default warmdown (WD=1200)
2. **Doc-isolated eval hurts by 0.009 BPB** at stride=64 — contradicts the LoRA TTT entry's finding

---

## Motivation

Two techniques are being adopted by top competition entries without clean ablation data:

**Stochastic Weight Averaging (SWA)** — PRs #162 and #180 include SWA in their stacks. The theory: averaging weights during training finds flatter minima that survive quantization better. But neither PR isolates SWA's contribution — it's buried under 5-8 other techniques.

**Doc-isolated evaluation** — the LoRA TTT entry (#77) showed doc-isolation was worth +0.011 BPB, measured at stride=256. Several subsequent entries adopted it. But nobody has tested whether this holds at stride=64, where tokens already have 960+ tokens of context.

These three runs answer both questions with controlled experiments on fast hardware (44ms/step, 13,600+ steps).

---

## Experimental design

Three runs on the same 8×H100 node, same seed, same architecture, same hyperparameters. One variable per comparison.

| Run | SWA | Eval mode | What it tests |
|-----|-----|-----------|--------------|
| A (control) | off | flat-stream sliding (stride=64) | Baseline — reproduces SlidingWindowEval entry |
| B | **on** | flat-stream sliding (stride=64) | Does SWA improve post-quant quality? |
| C | on | **doc-isolated** sliding (stride=64) | Does doc-isolation improve BPB at stride=64? |

All runs use the baseline 9L×512d architecture with default hyperparameters (WARMDOWN_ITERS=1200, MATRIX_LR=0.04, etc). No FP16 embed, no tuned LRs, no other modifications.

---

## Results

| Metric | Run A (control) | Run B (SWA) | Run C (SWA + doc-iso) |
|--------|----------------|-------------|----------------------|
| Steps | 13,651 | 13,611 | 13,619 |
| step_avg | 43.95ms | 44.08ms | 44.06ms |
| Pre-quant val_bpb | 1.2195 | 1.2200 | 1.2196 |
| **Post-quant val_bpb** | **1.1929** | **1.1933** | **1.2015** |
| Compressed model bytes | 15,819,113 | 15,816,795 | 15,812,296 |
| Eval windows | 121,134 | 121,134 | 61,286 |
| Eval time | 74s | 73s | 38s |

---

## Finding 1: SWA does not improve int8 quantization (Run A vs Run B)

| Metric | No SWA | SWA | Delta |
|--------|--------|-----|-------|
| Post-quant val_bpb | 1.1929 | 1.1933 | **+0.0004 (no improvement)** |
| Compressed model | 15,819,113 | 15,816,795 | -2,318 bytes (negligible) |

**SWA averaged 73 weight snapshots** (sampled every 50 steps from step 10,000 onward, accumulated in float32). The post-quant score is 0.0004 BPB *worse*, within noise.

### Why SWA didn't help here

SWA finds flatter minima by averaging weights collected during training with high or cyclical learning rates. Our setup uses the default warmdown schedule (WARMDOWN_ITERS=1200), which already reduces the LR in the final ~1,200 steps. By the time SWA starts averaging (step 10,000), the model is already in a narrow basin with low LR. Averaging 73 nearly-identical snapshots from a narrow region doesn't find a meaningfully flatter minimum.

### When SWA might still help

- **With aggressive warmdown (WD=20000):** The LR stays higher for longer, producing more diverse snapshots to average. PRs #162 and #180 use WD=20000 alongside SWA — the interaction may be what helps, not SWA alone.
- **With cyclical LR during the SWA window:** The original SWA paper uses cyclical scheduling to explore diverse minima. This wasn't tested.
- **With int6 quantization:** The larger quantization gap (~0.01+ BPB vs ~0.007 for int8) gives SWA more room to help.

### Recommendation

Don't add SWA with default warmdown (WD=1200) expecting it to improve quantization. If you're using aggressive warmdown (WD=20000), SWA may help but needs its own ablation in that context.

---

## Finding 2: Doc-isolated eval hurts at stride=64 (Run B vs Run C)

| Metric | Flat-stream | Doc-isolated | Delta |
|--------|------------|--------------|-------|
| Post-quant val_bpb | 1.1933 | 1.2015 | **+0.0086 (significantly worse)** |
| Eval windows | 121,134 | 61,286 | -49% fewer windows |
| Eval time | 73s | 38s | -48% faster |

**Doc-isolated eval is 0.009 BPB worse than flat-stream at stride=64.** This directly contradicts the LoRA TTT entry (#77), which found doc-isolation was worth +0.011 BPB.

### Why the results differ

The LoRA TTT entry measured doc-isolation at **stride=256**, where each token gets ~768 tokens of context. At that stride, context quality matters — removing noisy cross-document context helps because the model is somewhat context-starved.

At **stride=64**, each token gets **960+ tokens of context**. The model has abundant context. Doc-isolation at document boundaries means tokens at the start of each document go from "960 tokens of noisy cross-doc context" to "zero context." This is a massive quality drop for those tokens, and it outweighs the benefit of cleaner context everywhere else.

### The crossover point

There is a crossover stride length where doc-isolation flips from harmful to helpful:

```
stride=64:   doc-iso HURTS  (-0.009 BPB) — context quantity dominates
stride=256:  doc-iso HELPS  (+0.011 BPB) — context quality dominates (per LoRA TTT)
```

The optimal strategy depends on your stride. At stride=64, use flat-stream. At stride=256+, use doc-isolated. The crossover is somewhere in between — testing stride=128 with and without doc-isolation would pin it down.

### Recommendation

If you're using stride=64 (the most common setting in top entries), **do not use doc-isolated eval**. It costs 0.009 BPB. Use flat-stream sliding window instead.

---

## Finding 3: Baseline reproduction confirmed (Run A)

Run A's val_bpb of **1.1929** matches the SlidingWindowEval entry (#50) at **1.1925** within noise (delta 0.0004). This confirms:
- Our modified `train_gpt.py` produces correct results
- The 8×H100 hardware ran at the expected speed (44ms/step, 13,651 steps)
- The sliding window implementation is correct

---

## Implementation notes

### SWA bug discovered and fixed

An initial SWA implementation accumulated weights in bf16 for thousands of steps, causing catastrophic precision overflow (val_bpb 2.62). The fix:
- Accumulate in **float32**, not bf16
- Sample every **50 steps**, not every step
- Cast back to model dtype when applying

This is a pitfall for anyone implementing SWA in a bf16 training pipeline. See the commit history for details.

### Doc-isolated eval implementation

Documents are identified by BOS tokens (BOS_ID=1). `_find_docs()` returns (start, length) pairs. `_build_sliding_windows()` is called per document, producing windows that never cross document boundaries. Each token is scored exactly once.

---

## Reproduction

```bash
cd /workspace
git clone https://github.com/mrdavtan/parameter-golf.git
cd parameter-golf && git checkout swa-dociso-ablation-pr
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run A: control (flat-stream, no SWA)
export RUN_ID=swa_ablation_control SEED=1337 SWA=0 DOC_ISOLATED_EVAL=0 EVAL_STRIDE=64 EVAL_BATCH_SEQS=32
export QAT=0 FP16_EMBED_EXPORT=0 SMEAR_GATE=0 BIGRAM_HASH=0 USE_ZSTD=0 CURRICULUM=0
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-20_SWA_DocIso_Ablation/train_gpt.py

# Run B: SWA on (flat-stream)
export RUN_ID=swa_ablation_swa SWA=1
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-20_SWA_DocIso_Ablation/train_gpt.py

# Run C: SWA on + doc-isolated
export RUN_ID=doc_iso_ablation DOC_ISOLATED_EVAL=1
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-20_SWA_DocIso_Ablation/train_gpt.py
```

Hardware: 8×H100 SXM (RunPod Parameter Golf template), PyTorch 2.9.1+cu128, step_avg ~44ms.

---

## Acknowledgments

- SWA investigation motivated by PRs #162 (@unnir) and #180 which include SWA in their stacks
- Doc-isolation concept from the LoRA TTT entry (#77) by @samacquaviva
- Sliding window evaluation from #50 by @mattqlf
- Built with [Claude Code](https://claude.com/claude-code)

## Author

GitHub: [@mrdavtan](https://github.com/mrdavtan)
Date: 2026-03-20
