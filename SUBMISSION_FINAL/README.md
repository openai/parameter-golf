# Record candidate: PR #1797 base + Token-only n-gram tilt + AsymLogit Rescale + #2060 levers + NUM_PHASES=1 — val_bpb 1.05670 (3-seed mean)

**val_bpb: 1.05670** (3-seed mean) | **15.95 MB max** | 8xH100 SXM | 600s train / 600s eval

**Improvement over merged PR #1855 leaderboard record (1.06108 BPB):** -0.00438 BPB

## Results

| Seed | Pre-quant BPB | Quant BPB | Post-TTT BPB | TTT eval s | Artifact bytes |
|------|---------------|-----------|--------------|------------|----------------|
| 314  | 1.06086950    | 1.06924071 | **1.05664** | 522.5      | 15942188     |
| 42   | 1.06062887     | 1.06909826  | **1.05655** | 533.1      | 15948872      |
| 0    | 1.06109460      | 1.06957920   | **1.05689587** | 519.7  | 15944542       |
| **Mean** | **1.06086** | **1.06931** | **1.05670** | **525.1** | **15948872** |

3-seed std: 0.00015 BPB

All seeds under the 16,000,000-byte artifact cap and 600s eval budgets.

## What is new in this submission

This stack adds three compute-aware quality levers to the merged PR #1797 / PR #1855 lineage, all rooted in published precedent:

1. Token-only n-gram tilt (per merged PR #1514). A closed-form, prefix-conditioned probability tilt using a causal token n-gram model. The within-word and word-start channels are explicitly disabled (WITHIN_TAU=99.0, WORD_TAU=99.0, WITHIN_BOOST=0.0, WORD_BOOST=0.0, AGREE_ADD_BOOST=0.0); only the strictly-causal token channel fires.

2. AsymLogit Rescale (per open PR #1923, also used by PR #2014). Two trainable scalar parameters (softcap_pos, softcap_neg) replace the fixed logit_softcap=30.0, allowing asymmetric handling of positive vs. negative logits. Adapted globally during the TTT prefix pass; not per-doc.

3. Three hyperparameter sweep values from open PR #2060: MATRIX_LR=0.028, LQER_ASYM_GROUP=32, TTT_LORA_LR=8e-5. Pure config tuning, no new mechanisms.

4. PHASED_TTT_NUM_PHASES=1 (matches PR #2014's choice). One global TTT prefix pass instead of two. Reduces eval time without harming BPB.

## Compliance

### Artifact size

All seeds under 16,000,000-byte cap. Confirmed in logs.

### Training wallclock

All training loops complete under 600s with GPTQ_RESERVE_SECONDS=0.5 (Hessian collection deducted from training budget per Issue #1017's ruling).

### Eval wallclock — full disclosure

Total eval times include the n-gram precompute, computed INSIDE the eval timer.
- Seed 314: 522.5s (precompute ~167s + TTT eval ~355s)
- Seed 42: 533.1s
- Seed 0: 519.7 s

NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 is explicitly set in the launch command. The precompute runs after the eval timer starts, exactly matching the merged PR #1514 path. Logs explicitly show ngram_hint_precompute_outside: False and a precompute_done line (not precompute_outside_timer_done).

### Token-only n-gram causality (PR #1514 precedent)

Issue #1420 raised concerns that within-word and word-start channels can be target-dependent (use boundary_lut[tok] and starts_new_word_lut[tok] where tok is the realized target). Per PR #1514 ruling, only the strictly-causal token channel is allowed.

Our setup completely disables the problematic channels:
- WITHIN_TAU=99.0, WITHIN_BOOST=0.0
- WORD_TAU=99.0, WORD_BOOST=0.0
- AGREE_ADD_BOOST=0.0

Logs confirm activation counts:
ngram_tilt:hints total=47851520 gated=628130 token_gate=628130 within_gate=0 word_gate=0 agree2plus=0

- token_gate=628130: causal token channel fires
- within_gate=0, word_gate=0, agree2plus=0: target-dependent channels never fire

The token channel is verified strictly causal: in online_ngram_state.c, the per-position loop emits token_top_token[i] based on token_context_hash(st) reading the ring buffer for tokens [0..i-1], then absorbs tokens[i] into state via token_push(st, tok) AFTER the output. C1 property preserved.

### Score-first TTT (PR #402 ruling)

The eval_val_ttt_phased function evaluates each chunk in this order:
1. forward_ttt_train computes per_tok_loss (SCORE)
2. apply_tilt produces tilted_loss for BPB accumulation (SCORE only — tilt does not enter the gradient)
3. _accumulate_bpb records the score
4. .backward() and cur_opt.step() then update LoRA weights for FUTURE chunks

The final chunk per document is scored-only (needs_train=False); never trained on.

### AsymLogit Rescale safety

softcap_pos and softcap_neg are nn.Parameter members of the base GPT module, NOT of BatchedTTTLoRA. They are adapted by global TTT once over the prefix pass; per-doc TTT touches only the LoRA adapters. ~8 bytes total in the artifact.

### Closed-form n-gram tilt preserves probability mass

Per online_ngram_tilt.py, the tilt applies logit += boost for the gated token, then renormalizes via softmax. The sum of probabilities equals 1 by construction. This is the C2 property required by PR #1514.

### val_tokens methodology note

Our val_tokens is 47,851,520, slightly different from PR #2014's 47,853,343. The 0.0017% difference comes from PR #2014's EVAL_INCLUDE_TAIL=1 flag, which is not implemented in our base codebase. The truncation behavior matches PR #1855 (merged) and PR #1797. The discrepancy is methodology, not legality. Reviewers can re-run with their tokenizer setup if direct numerical comparison to PR #2014 is desired.

### Other compliance

- No external cache or memorization: no SLOT, persistent n-gram cache, PPM mixture, logit bias table, or validation-derived precomputation. The n-gram hint table is computed at eval time from val tokens (causal prefix only).
- No training data at eval time: GPTQ calibration uses training shards only, runs in the training phase, and GPTQ_RESERVE_SECONDS=0.5 deducts that compute from the training budget.
- CaseOps byte sidecar: original-byte BPB accounting preserved per PR #1729 / #1736 lineage.
- Full validation coverage: val_tokens equals target_tokens equals 47,851,520 in all included seeds.

## Architecture and stack

| Component | Setting |
|-----------|---------|
| Model | 11 layers, 512d, 8 query heads, 4 KV heads, MLP 4x |
| Tokenizer/data | SP8192 CaseOps lossless caps, byte sidecar accounting (PR #1729 / #1736 lineage) |
| RoPE | Partial RoPE, 16 dims |
| Recurrence | Layers 3-5 looped at frac=0.35 |
| Parallel decoder | Layer 8+ |
| XSA | All 11 layers |
| Gates | BOS-fixed SmearGate, SparseAttnGate (gate_window=12, scale=0.5) |
| Optimizer | Muon on matrix params (LR=0.028), Adam on embedding/scalars (BETA2=0.99) |
| EMA | EMA_DECAY=0.9965 |
| Quantization | GPTQ int6 matrices, int7 embeddings, LQER asymmetric rank-4 (GROUP=32, TOP_K=3) |
| GPTQ reserve | 0.5s |
| Compression | per-group |
| TTT | Quantized phased LoRA TTT (RANK=80, LR=8e-5, BETA2=0.99, WEIGHT_DECAY=2.0), score-first, K-off, 1 phase, 2500-doc prefix |
| Logit softcap | AsymLogit Rescale (softcap_pos, softcap_neg, init 30.0, global TTT) |
| Tilt | Token-only n-gram tilt (TOKEN_ORDER=16, TOKEN_THRESHOLD=0.800, TOKEN_BOOST=2.625) |

## Reproduction

Same dependencies and CaseOps tokenizer/shards as merged PR #1855.
for SEED in 314 42 0; do
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
ASYM_LOGIT_RESCALE=1 
NGRAM_TILT_ENABLED=1 
NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 
TOKEN_ORDER=16 
TOKEN_THRESHOLD=0.800 
TOKEN_BOOST=2.625 
WITHIN_TAU=99.0 WITHIN_BOOST=0.0 
WORD_TAU=99.0 WORD_BOOST=0.0 
AGREE_ADD_BOOST=0.0 
DATA_DIR=./data CASEOPS_ENABLED=1 
SMEAR_GATE_BOS_FIX=0 
TTT_LORA_EMA_DECAY=0.0 TTT_UPDATE_EVERY=1 
PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=1 
EVAL_SEQ_LEN=2560 TTT_EVAL_SEQ_LEN=2560 
COMPRESSOR=pergroup 
MATRIX_CLIP_SIGMAS=12.85 ATTN_CLIP_SIGMAS=13.0 MLP_CLIP_SIGMAS=11.5 
EMBED_BITS=7 EMBED_CLIP_SIGMAS=14.0 
MATRIX_LR=0.028 MIN_LR=0.1 WARMDOWN_FRAC=0.85 BETA2=0.99 
FUSED_CE_ENABLED=1 
SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 
SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_FACTOR_BITS=4 
LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=32 
TTT_WARM_START_A=1 TTT_LORA_RANK=80 
TTT_BETA2=0.99 TTT_WEIGHT_DECAY=2.0 
TTT_LORA_LR=8e-5 
GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 
TTT_K_LORA=0 TTT_O_LORA=1 TTT_MLP_LORA=1 
EMA_DECAY=0.9965 
SEED=$SEED 
torchrun --standalone --nproc_per_node=8 train_gpt.py
done
## Lineage and credits

This submission stacks on the public CaseOps/SP8192 record lineage:

- PR #1855 by @codemath3000 — current merged leaderboard record (1.06108 BPB), direct comparison baseline
- PR #1797 by @dexhunter — Smear Gate + LQER Asymmetric, our direct base
- PR #1787 by @nprime06 — Polar Express NS, MIN_LR, Sparse Attn Gate, Fused CE, PR #1767 TTT
- PR #1736 by @dexhunter — SP8192 CaseOps + GatedAttn + PhasedTTT
- PR #1729 by @romeerp / @dexhunter — CaseOps Tokenizer
- PR #1514 (merged) — token-only n-gram tilt legality precedent
- PR #1923 — AsymLogit Rescale technique (also used in PR #2014)
- PR #2060 by @S0urC10ud — three hyperparameter sweep values
- PR #2014 by @simonbissonnette — PHASED_TTT_NUM_PHASES=1 precedent

The new contribution is the combination of token-only n-gram tilt (PR #1514's merged path) plus AsymLogit Rescale plus three #2060 hyperparameter levers plus NUM_PHASES=1, all stacked on the merged PR #1797 base while keeping the n-gram precompute inside the eval timer per PR #1514's path.

## Included files

- train_gpt.py — full training/eval script
- train_seed314.log, train_seed42.log, train_seed0.log — full per-seed logs
- submission.json — structured metadata
- README.md — this document
- prepare_caseops_data.py — CaseOps preparation, same lineage as PR #1855
- lossless_caps.py — reversible CaseOps transform, same lineage as PR #1729 / #1855
- tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model — same as PR #1855
- online_ngram_tilt.py — n-gram tilt Python wrapper
- online_ngram_state.c — n-gram state machine (token-only path used)
