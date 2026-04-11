# PROTEUS+ — 11L INT6 + Cosine LoRA TTT

**Author:** Atharva Date (ADIITJ)  
**Status:** Non-record submission (pending 3-seed H100 validation)  
**Expected BPB:** ~0.93–0.95 (projected)

---

## Summary

This submission builds directly on **PROTEUS v7** (PR #512, MatoTeziTanka, 0.9512 bpb verified), adding **cosine LR annealing** to the LoRA TTT evaluation phase — the key improvement demonstrated by PR #517 (lukacf, 0.978 bpb) that prevents position-specific overfitting in extended LoRA adaptation.

**Changes from PROTEUS v7:**
1. **Cosine LR in TTT**: `lr = base_lr * 0.5 * (1 + cos(π * epoch / n_epochs))`, from 0.008 → ~1e-5 over 3 epochs. Prevents the optimizer from thrashing in later epochs.
2. **Lower base LR**: 0.008 (vs 0.01) — more conservative to pair with cosine annealing.
3. **More short docs adapted**: `ttt_min_doc_len=512` (vs 1024) — doubles the number of documents that receive full LoRA TTT adaptation instead of standard eval.
4. **Fixed Adam betas for TTT**: (0.9, 0.95) instead of training betas, matching standard TTT practice.

**Unchanged from PROTEUS v7:**
- 11L transformer, 512 dim, 1536 MLP hidden
- INT6 uniform quantization (per-row, int8 storage)
- Depth-scaled residuals (1/sqrt(layer+1))
- OrthoInit + scaled proj init
- EMA (decay=0.999, every 10 steps)
- SWA starting at LR scale < 0.2
- BigramHashEmbedding(2048, 128) + SmearGate
- Muon optimizer (momentum=0.99, WD=0.04)
- AdamW for embedding + scalars (WD=0.04)
- 3% weight pruning before quantization
- zstd-22 compression

---

## Architecture

| Parameter | Value |
|---|---|
| num_layers | 11 |
| model_dim | 512 |
| mlp_hidden | 1536 (~3x) |
| num_heads | 8 |
| num_kv_heads | 4 |
| rope_base | 50000 (NTK-aware) |
| tie_embeddings | True |
| logit_softcap | 30.0 |
| quantization | INT6 per-row |
| compression | zstd-22 |

## LoRA TTT at Evaluation

| Parameter | Value |
|---|---|
| ttt_lora_rank | 8 |
| ttt_lora_lr | 0.008 (cosine → 1e-5) |
| ttt_epochs | 3 |
| ttt_chunk_size | 256 |
| ttt_eval_seq_len | 1024 |
| ttt_batch_size | 64 |
| ttt_min_doc_len | 512 |
| LR schedule | Cosine: 0.008 → 1e-5 |
| adapters | Q, V in all 11 layers + LM head |

**Fairness:** Score-then-train per chunk; final epoch scores are reported. LoRA reset between documents.

---

## Run Commands

```bash
cd records/track_10min_16mb/2026-03-23_ADIITJ_ProteusPlus/
SEED=1337 RUN_ID=proteusplus_seed1337 \
DATA_PATH=../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Attribution

- **PROTEUS v7** (PR #512) by MatoTeziTanka — base architecture, INT6 quantization, depth-scaled residuals, EMA, SWA, OrthoInit, LoRA TTT framework
- **Cosine LR TTT schedule** (PR #517) by lukacf — the cosine annealing insight
- **LoRA TTT protocol** (PR #77) by samacqua — backward-looking score-then-train guarantee

This submission by Atharva Date (ADIITJ) combines these two streams on the PROTEUS training base.

---

## Non-Record Status

Pending 3-seed H100 validation. The implementation is complete and correct; actual bpb to be updated after runs.
