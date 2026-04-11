# Non-record: LoRA TTT Exploration on SOTA Base

**Track: non-record (negative result, informative for community)**

## Summary

Explored combining LoRA Test-Time Training (TTT) with the current SOTA
(thwu1's Int5-MLP + BigramHash + SWA, 1.1428 BPB). TTT adapts the model
per-document during evaluation using lightweight LoRA adapters.

**Result: TTT does NOT improve SOTA.** The SOTA's seq_len=2048 combined with
SmearGate and BigramHash already captures local context more effectively than
TTT's chunked per-document adaptation.

## Approach

LoRA TTT (from samacqua's baseline submission, ~0.037 BPB improvement on naive model):
1. Split each validation document into chunks of N tokens
2. Evaluate chunk, compute loss
3. Train rank-8 LoRA adapters (Q and V projections) on that chunk
4. Evaluate next chunk with adapted model
5. Reset LoRA weights between documents

The hypothesis was that TTT and SOTA techniques are complementary:
- SOTA optimizes the model architecture and training
- TTT optimizes per-document evaluation
- Expected ~0.01-0.03 BPB additional improvement

## Experimental Results (1xH100)

### 5108-step model (partial training)

| Method | val_bpb | Notes |
|--------|---------|-------|
| Standard eval | 1.1734 | No sliding window |
| TTT (chunk=256, seq=1024) | 1.3117 | **Worse by 0.138** |
| TTT (chunk=256, seq=2048) | 1.3366 | **Worse by 0.163** |

### Why TTT Fails on SOTA

1. **Context loss**: The SOTA trains with seq_len=2048. TTT chunks documents into
   256-token segments with a 1024-token sliding window, losing long-range context
   that the model was trained to exploit.

2. **Redundant local context**: SmearGate and BigramHash already inject strong
   local/bigram signals directly into embeddings. TTT's per-document adaptation
   provides diminishing returns on top of these.

3. **Document boundary artifacts**: TTT resets LoRA weights between documents.
   For short documents (common in FineWeb), the adapter barely trains before
   being discarded.

## Code Changes from SOTA Base

Minimal modifications to train_gpt.py (4 changes to existing code + new TTT code):

1. `CausalSelfAttention.forward`: added `q_delta=None, v_delta=None` params
2. `Block.forward`: added `q_delta_fn=None, v_delta_fn=None` params
3. `GPT.forward`: added `lora=None` param, per-token loss when LoRA present
4. Final eval block: TTT path when `ttt_enabled=True`

New classes/functions:
- `BatchedLinearLoRA`: batched LoRA linear layer
- `BatchedTTTLoRA`: manages per-document LoRA state
- `eval_val_ttt_lora`: TTT evaluation loop
- Helper functions: `_find_docs`, `_compute_chunk_window`, `_accumulate_bpb`

## TTT Hyperparameters Tested

| Param | Values Tested | Default |
|-------|--------------|---------|
| ttt_lora_rank | 8 | 8 |
| ttt_lora_lr | 0.01 | 0.01 |
| ttt_chunk_size | 128, 256, 512 | 256 |
| ttt_eval_seq_len | 1024, 2048 | 1024 |

## Reproduce

```bash
# Requires pre-trained SOTA model (8xH100 training)
torchrun --standalone --nproc_per_node=1 train_gpt.py

# Eval-only with pre-trained model
MODEL_PATH=final_model.int8.ptz python eval_ttt_only.py

# Disable TTT (sliding window only)
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Takeaway

TTT is most effective on models with short context windows and weak local
embeddings. On architectures that already maximize context (seq_len=2048)
and inject strong bigram signals (SmearGate + BigramHash), TTT provides
no additional benefit. This suggests the community should focus on
architecture and training improvements rather than evaluation-time adaptation
for this competition.

Built on PR #414 by @thwu1 (Int5-MLP + BigramHash + SWA).
TTT approach from @samacqua's baseline submission.
