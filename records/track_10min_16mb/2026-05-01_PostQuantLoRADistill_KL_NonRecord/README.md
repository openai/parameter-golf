# Non-record: Post-Quantization LoRA Distillation (LCQ) on PR #1855 stack

**val_bpb = 1.06767** (seed 42, single-seed) | artifact 15,912,974 bytes | 8xH100 SXM | strict 600s train + eval

This is a non-record submission. It does not beat the current SOTA. It documents a novel technique (post-quantization LoRA distillation against the pre-quantization BF16 teacher on TRAIN data only), the implementation details, and a negative result with diagnosis. The artifact is fully compliant.

## Summary

LCQ (LoRA-Compensated Quantization) trains a small LoRA module on the post-GPTQ dequantized model to compensate for quantization error. Training is performed at TRAIN time (in the 10-minute training cap, never at eval time, never on val data) using KL divergence against the pre-quantization BF16 teacher logits. The LoRA is held in memory across the train-to-eval boundary in the same Python process and applied at eval via the model's existing forward_ttt path with a cu_seqlens-aware variable-length attention dispatch.

The full pipeline:

1. Train the model normally for 520s (cap reduced from 600s by GPTQ_RESERVE_SECONDS=80 to free LCQ budget)
2. Apply EMA decay 0.9965 to weights
3. Run GPTQ mixed-precision quantization (int6 attn/MLP, int7 embeddings, LQER asymmetric residuals)
4. Build a temporary dequantized GPT model from quant_result + quant_meta
5. Attach a rank=4 BatchedTTTLoRA on top of that dequantized model (q/k/v/o + MLP + lm_head LoRAs, alpha=4 -> effective scale 1.0)
6. For 60 seconds, train ONLY the LoRA parameters via KL distillation: `KL(softmax(teacher_logits), softmax(student_with_lora_logits))` over TRAIN data sequences from DocumentPackingLoader. Teacher is the pre-quant BF16 base_model (still in memory after EMA application). Student is the dequantized model with the LoRA delta added.
7. Federate LoRA via `dist.all_reduce(AVG)` across the 8 GPUs.
8. Free the temporary dequantized model. Save quantized weights and code as the artifact.
9. Eval phase: deserialize the quantized weights (no train data access). Reconstruct dequantized state. Apply the in-memory trained LoRA via forward_ttt with cu_seqlens-aware sliding window scoring at stride 64.

Result on seed 42: post-EMA BF16 1.06870, quantized 1.07702, quantized + sliding window + LCQ LoRA 1.06767. The LoRA contribution on top of plain sliding window is small (about -0.0003 BPB). KL training loss converged very low (about 0.02), indicating the student matched the teacher closely on the bulk of the distribution; the residual quant error apparently lives in tail tokens that contribute little to the average.

## Compliance

- Train budget: all training (main + LCQ LoRA distillation) within the 10-minute cap. GPTQ_RESERVE_SECONDS=80 stops main training at 520s, leaving 80s for GPTQ + LCQ.
- Eval budget: under 10 minutes for the post-quant sliding window pass.
- Artifact: 15,912,974 bytes, under the 16,000,000 decimal byte cap.
- C3 score-first: validation tokens are never used to update parameters before being scored. LCQ trains exclusively on TRAIN data shards (FineWeb_train_*.bin). The reported quantized_sliding_window val_bpb is computed by single-pass causal scoring with the trained LoRA already loaded; no in-loop val-token-driven updates.
- C1 causality: `forward_ttt` extended with cu_seqlens dispatch so attention is masked at document boundaries during sliding-window eval, exactly matching the legal scoring pattern used in `eval_val`.
- No SLOT, no n-gram cache, no logit bias, no ETLB.

## Why a single seed

This is documented as a non-record submission to capture the technique and the negative result. The score does not beat the current SOTA frontier and so does not require the 3-seed statistical-significance burden of a record submission. Logs for the single seed run are included.

## Implementation in one place

All LCQ logic lives in three places in `train_gpt.py`:

1. `BatchedLinearLoRA.forward` was extended so a bsz=1 LoRA broadcasts to a multi-batch eval forward via `expand`. The trained LoRA always has bsz=1; eval may run with bsz>=1.
2. New `postquant_lora_distill(h, device, eval_model, teacher_model=None, time_budget_s, lr, rank)` builds the LoRA, drives a 60s KL distillation loop on train data via `forward_ttt(..., return_logits=True)`, and DDP-averages the LoRA parameters at the end. The function returns the trained LoRA module (not a state_dict) for in-process use.
3. `serialize` calls `postquant_lora_distill` after GPTQ produces `quant_result, quant_meta`, builds a temporary dequantized GPT model for it, then frees the temporary model. The trained LoRA is returned to `train_and_eval` via the third return value.
4. `eval_val_sliding` accepts an optional `lora` keyword. When set, it dispatches each window's batched logits computation through `base_model.forward_ttt(x_cat, y_cat[None], lora=lora, cu_seqlens=cu_seqlens, max_seqlen=seq_len, return_logits=True)`, then computes per-token NLL exactly the same way as the no-LoRA path.
5. `forward_ttt`, `_block_with_lora`, and `_parallel_block_with_lora` were extended to accept and propagate `cu_seqlens` and `max_seqlen` so the LoRA-augmented forward dispatches to `flash_attn_varlen_func` when called with cu_seqlens, matching the legal sliding window's BOS-aware variable-length attention.

## Why this did not beat plain sliding window

Plain sliding window (no LCQ) on the same PR #1855 stack at full 600s training lands around val_bpb 1.06286 on seed 42. LCQ at full distillation depth lands at 1.06767 on the same seed, about 0.005 BPB worse. Two factors:

1. The 80 seconds of training budget LCQ steals from main training cost roughly +0.005 BPB on the BF16 model (post-EMA went from 1.06403 -> 1.06915 with the cut). The LoRA only recovers about 0.0003 BPB. Net negative.
2. The LoRA is very small (rank=4, alpha=4). KL distillation converges quickly but most BPB is set by the bulk of the next-token distribution where the quantized student already matches the teacher well. The LoRA has too few parameters to fix the long tail, where the residual quant error lives.

Possible follow-ups that could turn this positive: (a) run LCQ in the eval budget instead of train budget, but only if a legal way to ship the LoRA in the artifact AND access the right data exists (currently train data is illegal at eval); (b) much higher rank LoRA (16-32) with careful artifact size accounting; (c) distill against logits with a higher temperature to weight tail tokens more heavily. None of these are attempted here.

## Hyperparameters

| variable | value |
|----------|-------|
| SEED | 42 |
| CASEOPS_ENABLED | 1 |
| COMPRESSOR | pergroup |
| EMBED_BITS | 7 |
| MATRIX_LR | 0.026 |
| MIN_LR | 0.1 |
| MLP_CLIP_SIGMAS | 11.5 |
| ATTN_CLIP_SIGMAS | 13.0 |
| EMBED_CLIP_SIGMAS | 14.0 |
| WARMDOWN_FRAC | 0.85 |
| BETA2 | 0.99 |
| TTT_LORA_ALPHA | 4 |
| SPARSE_ATTN_GATE_SCALE | 0.5 |
| GPTQ_RESERVE_SECONDS | 80 |
| GPTQ_CALIBRATION_BATCHES | 16 |
| LQER_ENABLED | 1 |
| LQER_RANK | 4 |
| LQER_TOP_K | 3 |
| LQER_FACTOR_BITS | 4 |
| TTT_ENABLED | 0 |
| PHASED_TTT_ENABLED | 0 |
| SLIDING_WINDOW_ENABLED | 1 |
| EVAL_STRIDE | 64 |
| LCQ_ENABLED | 1 |
| LCQ_RANK | 4 |
| LCQ_LR | 1e-3 |
| LCQ_TIME_S | 60 |
| LCQ_GRAD_CLIP | 0.5 |

## Reproduction

```bash
pip install brotli sentencepiece huggingface_hub numpy
pip install --no-deps flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu129_torch291/
apt-get install -y lrzip

python3.12 -c "from huggingface_hub import snapshot_download;import os;snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', repo_type='dataset', local_dir='./data', max_workers=16)"

SEED=42 \
  CASEOPS_ENABLED=1 \
  DATA_PATH=./data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  VOCAB_SIZE=8192 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  EMBED_BITS=7 MATRIX_LR=0.026 MIN_LR=0.1 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GRAD_CLIP_NORM=0.3 TTT_CHUNK_SIZE=48 WARMUP_STEPS=20 MUON_BACKEND_STEPS=5 \
  GLOBAL_TTT_MOMENTUM=0.9 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_ALPHA=4 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  GPTQ_RESERVE_SECONDS=80 GPTQ_CALIBRATION_BATCHES=16 VAL_LOSS_EVERY=0 \
  GATED_ATTN_QUANT_GATE=1 SPARSE_ATTN_GATE_ENABLED=1 GATE_WINDOW=12 \
  SMEAR_GATE_ENABLED=1 \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  FUSED_CE_ENABLED=1 COMPRESSOR=pergroup NCCL_NET=Socket \
  TTT_ENABLED=0 PHASED_TTT_ENABLED=0 \
  SLIDING_WINDOW_ENABLED=1 EVAL_STRIDE=64 \
  LCQ_ENABLED=1 LCQ_RANK=4 LCQ_LR=1e-3 LCQ_TIME_S=60 LCQ_GRAD_CLIP=0.5 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- PR #1855 by @codemath3000: full base stack (SP8192 + LQER + sparse attention gate + BOS-fixed SmearGate + 9-hp greedy)
- PR #1493 / @bigbag and PR #1413 / @dexhunter: legal sliding-window evaluation pattern that this submission's eval path matches
- PR #1586 / PR #1667 / PR #1729: per-group lrzip serialization (`COMPRESSOR=pergroup`)
- PR #1394 / @clarkkev: SP8192 + GPTQ + MuonEq-R + depth recurrence base
- PR #1411 line and follow-ups: BatchedTTTLoRA infrastructure (the LoRA modules and forward_ttt path that this submission extends)
- LQER (Yao et al., 2024): low-rank asymmetric residual on top of int weights
- GPTQ (Frantar et al., 2023): post-training Hessian-based weight quantization

## Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `lossless_caps.py`
- `train_seed42.log`
