# LongContext 4096 + Full SOTA Stack (PR #349 base)

**val_bpb: TBD** (3-seed mean, post int6+lzma, sliding window stride=64 + TTT)

## Summary

This submission takes the complete SOTA stack from PR #549 exactly as-is and raises the training
and evaluation sequence length from 2048 to **4096 tokens**. That is the only code change (two
lines). Every technique from the current record is preserved:

- 11 layers, 512-dim, 8 heads (4 KV, GQA)
- MLP 3× expansion (hidden=1536), **LeakyReLU(0.5)²** activation
- **SmearGate** (learned per-dim blending of current + previous token)
- **BigramHash(1536, dim=128)** — learned token-pair embeddings
- **XSA on last 4 layers** (Efficient Partial Exclusive Self-Attention)
- **Partial RoPE** (16 of 64 head dims), NTK-aware dynamic scaling
- **LN Scale** 1/√(layer+1) per block
- **Value Embedding** (VE128, layers 9 and 10)
- **EMA** (decay=0.997, every step) + Tight **SWA** (every 50 steps when scale<0.2)
- **Late QAT** (int6 STE on CastedLinear when LR scale < 0.15)
- **GPTQ-lite** quantization — per-row optimal clip percentile search (5 candidates)
- Int6 MLP+attention, int8 embeddings, lzma compression
- **Parameter Banking + Parallel Muon** (batched Newton-Schulz, async reduce-scatter)
- **OrthoInit + muP**-scaled output projections
- **Legal score-first TTT** (PR #461 recipe, freeze=0, 3 epochs, SGD lr=0.002)
- WD=0.04, Muon momentum=0.99, warmdown=3500 iters, grad_clip=0.3

## Why 4096 Context Helps

The 4096-context training record (1.2014 BPB, 2026-03-19) predates every technique above and was
never combined with the modern stack. This submission closes that gap.

At seq_len=4096 each token sees up to 4,095 tokens of causal context during training vs 2,047 for
the current SOTA. The model learns richer long-range co-occurrences, improving perplexity on both
short- and long-range dependencies.

**RoPE scaling**: `train_seq_len=1024` is hardcoded in `Rotary.__init__`, so the existing dynamic
NTK formula fires automatically during training (`scale=4096/1024=4`, `rope_dims=16`):

```
adjusted_base = 10000 × 4^(16/14) ≈ 48,550
```

This is the correct, principled NTK-scaled base — no manual tuning required.

**Step budget**: With `train_batch_tokens=786,432` and `seq_len=4096`, each step processes 192
sequences. FlashAttention 3 scales roughly quadratically with sequence length per sequence but we
have 2× fewer sequences, so the net step time increase is ~30–40%. Expected steps ≈ 5,500–6,000
in 600s (vs ~7,200 for the SOTA at 2048). More context per step compensates.

**TTT**: Uses `seq_len = args.train_seq_len = 4096`, giving each TTT forward pass 4096 tokens of
context for scoring. TTT training adapts on 32768-token chunks = 8 sequences each. Expected TTT
time ≈ 450–550s.

## Code Changes from PR #549 Base

```diff
-    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
-    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 2048))
+    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 4096))
+    eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 4096))
```

Everything else is unchanged from PR #549.

## Run Command

```bash
SEED=1337 bash eval/eval.sh
```

Or directly:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Results

*(Pending H100 runs — 3 seeds required)*

| Seed | Steps | Pre-TTT bpb | Post-TTT bpb | Artifact |
|------|-------|-------------|--------------|----------|
| 1337 | — | — | — | — |
| 42   | — | — | — | — |
| 2025 | — | — | — | — |
| **Mean** | | | | |
