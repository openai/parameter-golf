# Int5/Int6 + BigramHash + SmearGate + SWA + LLMAdvisor — 3-seed

## Summary

**3-Seed Mean val_bpb: 1.15865** (seeds 99, 777, 2025)
**Best single seed: 1.15323** (seed 777)
**Model size:** 15,719,916 bytes / ~15.72 MB (Int5/Int6 mixed quantization, zstd-22)
**Training:** 600s wallclock cap, up to 20,000 iterations, 8×H100 SXM

---

## Architecture

- **Backbone:** 9-layer GPT, 512d embeddings, 8 attention heads (4 KV heads)
- **Parameters:** 25,517,137
- **Vocabulary:** SP1024 (SentencePiece BPE, 1024 tokens)
- **Quantization:** Mixed Int5 (MLP weights) + Int6 (attention weights), zstd-22 compression
- **Embeddings:** BigramHash
- **Gate:** SmearGate
- **EMA:** SWA (Stochastic Weight Averaging)
- **Optimizer:** Muon (matrix params) + AdamW (scalars/embeddings)
- **Tied embeddings:** Yes

---

## Training Hyperparameters

| Param | Value | Notes |
| --- | --- | --- |
| Iterations | 20,000 | Wall-clock capped at 600s |
| Warmdown Iters | 3,000 | Cosine decay warmdown phase |
| Max Wallclock | 600s | Hard stop per seed |
| Vocab Size | 1,024 | SP1024 tokenizer |
| Embed LR | 0.030 | AdamW |
| Matrix LR | 0.020 | Muon |
| Scalar LR | 0.020 | AdamW |
| Batch | 622,592 tokens/step | seq_len=2048, 8 GPUs |

---

## Evaluation Results

| Seed | Steps | step_avg | val_bpb | Artifact size | Eval time |
| --- | --- | --- | --- | --- | --- |
| 99 | 5,716 | 104.9 ms | 1.15637 | 15,544,251 bytes | 276s |
| **777** | 6,000 | 99.9 ms | **1.15323** | 15,645,316 bytes | 277s |
| 2025 | 4,893 | 122.6 ms | 1.16636 | 15,719,916 bytes | 528s |
| **Mean** | | | **1.15865** | | |

All BPB values are `final_int8_zlib_roundtrip_exact` — quantized weights loaded from disk, full eval pass, exact precision.

---

## Constraint Compliance

| Constraint | Limit | Actual | Status |
| --- | --- | --- | --- |
| Artifact bytes | 16,777,216 | 15,719,916 | ✅ |
| Training wallclock | 600s | ≤ 599.9s | ✅ |
| Hardware | 8×H100 SXM | 8×H100 SXM 80GB | ✅ |

---

## Reproduce

```bash
for SEED in 99 777 2025; do
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-22_Int5Int6_BigramHash_SmearGate_SWA_LLMAdvisor/train_gpt.py \
    --seed $SEED \
    --iterations 20000 \
    --max_wallclock_seconds 600 \
    --warmdown_iters 3000
done
```

---

## Submission Details

- **Author:** LLMAdvisor.ai
- **GitHub ID:** harborglowvintage-oss
- **Submitted:** April 30, 2026 (deadline)
- **Target:** OpenAI Parameter Golf — track_10min_16mb
- **Base script:** `records/track_10min_16mb/2026-03-22_Int5Int6_BigramHash_SmearGate_SWA_LLMAdvisor/train_gpt.py`

---

## Files Included

- `submission.json` — Metadata, per-seed results, and configuration
- `train_gpt.py` — Training script (Mar 22 LLMAdvisor stack, unchanged)
- `llmadv_v3_s99.log` — Seed 99 full training + eval log
- `llmadv_v3_s777.log` — Seed 777 full training + eval log
- `llmadv_v3_s2025.log` — Seed 2025 full training + eval log
- `run3seeds_v3_master.log` — Orchestration log with results summary
