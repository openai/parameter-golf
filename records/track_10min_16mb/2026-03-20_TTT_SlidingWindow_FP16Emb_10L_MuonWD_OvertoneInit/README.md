# TTT + Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init

## Summary

This submission combines the current SOTA training recipe with LoRA test-time training (TTT) evaluation. These two approaches have been demonstrated separately but not yet stacked together:

- **Training recipe** (from `2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit`, val_bpb=1.1748): 10 transformer layers, Muon optimizer with decoupled weight decay, FP16 tied embeddings, overtone spectral SVD init, phase-transition residual mixing, sliding window eval at stride=64.
- **TTT evaluation** (from `2026-03-17_LoRA_TTT`, val_bpb=1.1928 on baseline 9L model): Per-document LoRA adaptation at eval time with rank-8 adapters on Q/V projections and LM head, document-isolated evaluation with chunked sliding window.

The TTT ablation table showed ~0.037 bpb improvement over baseline from eval-time adaptations (document isolation + stride + LoRA). Applying this delta to the SOTA training recipe should yield a significant improvement.

## Changes from SOTA

1. Added TTT hyperparameters (LoRA rank=8, lr=0.01, chunk_size=256, eval_seq_len=1024, batch_size=64)
2. Modified `CausalSelfAttention.forward` to accept optional `q_delta` and `v_delta` LoRA corrections
3. Modified `Block.forward` to pass LoRA delta functions through to attention
4. Modified `GPT.forward` to accept optional `lora` parameter, returning per-token losses when LoRA is active
5. Added `BatchedLinearLoRA`, `BatchedTTTLoRA` classes for batched per-document adaptation
6. Added `eval_val_ttt_lora` function for document-isolated chunked LoRA evaluation
7. After standard sliding window eval on quantized model, runs TTT LoRA eval and reports both scores

## Training

Training is identical to the SOTA recipe - no changes to the training loop, model architecture, or optimizer configuration. All improvements come from stacking TTT evaluation on top.

## Expected Results

- Training: ~10,500 steps in 600s on 8xH100 (same as SOTA)
- Standard sliding window eval: ~1.1748 val_bpb (same as SOTA)
- TTT LoRA eval: estimated ~1.14 val_bpb (pending validation)
- Compressed model size: ~14.7 MB (same as SOTA, TTT adds no model parameters)

## Reproduction

```bash
RUN_ID=ttt_sota_graft \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-20_TTT_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit/train_gpt.py
```
