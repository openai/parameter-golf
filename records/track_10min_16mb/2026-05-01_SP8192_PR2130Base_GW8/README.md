# Record candidate: PR #1797 base + Token-only n-gram tilt + AsymLogit Rescale + #2060 levers + NUM_PHASES=1 — GATE_WINDOW=8

**Comparison baseline: PR #2130 (1.05670 BPB)** — direct stack baseline.
**Merged-leaderboard comparison: PR #1855 (1.06108 BPB).**

This submission keeps the PR #2130 architecture/training stack identical and only changes: **GATE_WINDOW=8**. Every other hyperparameter, env var, and code path is byte-for-byte the PR #2130 reproduction command.

## Full validation coverage

The 3 per-seed logs evaluate the full CaseOps validation shard target set:

| Seed | `val_tokens` | `target_tokens` |
|-----:|-------------:|----------------:|
| 314 | 47,851,520 | 47,851,520 |
| 42 | 47,851,520 | 47,851,520 |
| 0 | 47,851,520 | 47,851,520 |

Validation tokens are 47,851,520 (matches the PR #1855 / PR #1797 / PR #2130 truncation; PR #2014's 47,853,343 figure is methodology, not legality).

The tokenizer, CaseOps transform, training shards, validation shard, and byte sidecar format are the canonical HF-hosted CaseOps export (`romeerp/parameter-golf-caseops-v1`) used by the merged PR #1855 setup.

## What changed vs PR #2130

| Lever | PR #2130 | This submission | Mechanism |
|-------|----------|-----------------|-----------|
| `GATE_WINDOW` | 12 | **8** | Tightens the SparseAttnGate window from 12 to 8 positions; the gate then attends to fewer (more selective) recent tokens. |

Everything else — model architecture, optimizer, schedule, TTT, quantization, compression — is byte-for-byte identical to the PR #2130 stack.

## Architecture and training stack

| Component | Setting |
|-----------|---------|
| Model | 11 layers, 512d, 8 query heads, 4 KV heads, MLP 4x |
| Tokenizer/data | SP8192 CaseOps lossless caps, byte sidecar accounting (PR #1729 / #1736 lineage) |
| RoPE | Partial RoPE, 16 dims |
| Recurrence | Layers 3-5 looped at frac=0.35 |
| Parallel decoder | Layer 8+ |
| XSA | All 11 layers |
| Gates | BOS-fixed SmearGate, SparseAttnGate (`scale=0.5`) |
| Optimizer | Muon on matrix params (LR=0.028), Adam on embedding/scalars (BETA2=0.99) |
| EMA | EMA_DECAY=0.9965 |
| Quantization | GPTQ int6 matrices, int7 embeddings, LQER asymmetric rank-4 (GROUP=32, TOP_K=3) |
| GPTQ reserve | 0.5s |
| Compression | per-group |
| Eval context | `EVAL_SEQ_LEN=2560`, `TTT_EVAL_SEQ_LEN=2560` |
| TTT | Quantized phased LoRA TTT (RANK=80, LR=8e-5, BETA2=0.99, WEIGHT_DECAY=2.0), score-first, K-off, 1 phase, 2500-doc prefix |
| Logit softcap | AsymLogit Rescale (softcap_pos, softcap_neg, init 30.0, global TTT) |
| Tilt | Token-only n-gram tilt (TOKEN_ORDER=16, TOKEN_THRESHOLD=0.800, TOKEN_BOOST=2.625) |

## Compliance notes

- **Artifact cap:** all seeds <= 16,000,000 bytes.
- **Training wallclock:** training stops at the 600s wallclock budget (matching PR #2130).
- **Eval wallclock:** all validation passes are <= 600s.
- **Score-before-update:** `quantized_ttt_phased` scores each chunk before applying that chunk's LoRA update. Inherited unchanged from PR #2130.
- **Full validation targets:** `val_tokens == target_tokens == 47,851,520` in all included logs.
- **No validation data in training:** training uses only training shards. TTT accesses validation documents left-to-right under the score-first rule.
- **No external cache or direct memorization:** no SLOT, persistent n-gram cache, PPM mixture, logit bias table, or validation-derived precomputation.
- **Original-byte BPB:** CaseOps byte sidecar accounting is preserved.

### Token-only n-gram causality (PR #1514 precedent)

This base inherits the strictly-causal token n-gram tilt from PR #2130. Within-word and word-start channels are explicitly disabled (`WITHIN_TAU=99.0`, `WORD_TAU=99.0`, `WITHIN_BOOST=0.0`, `WORD_BOOST=0.0`, `AGREE_ADD_BOOST=0.0`), so only the strictly-causal token channel fires. The C1 causality property of `online_ngram_state.c` is preserved and is unaffected by the lever change(s) in this submission.

### AsymLogit Rescale safety

`softcap_pos` and `softcap_neg` are nn.Parameter members of the base GPT module, NOT of `BatchedTTTLoRA`. They are adapted by global TTT once over the prefix pass; per-doc TTT touches only the LoRA adapters. ~8 bytes total in the artifact. Unchanged from PR #2130.

### Closed-form n-gram tilt preserves probability mass

Per `online_ngram_tilt.py`, the tilt applies `logit += boost` for the gated token, then renormalizes via softmax. Sum of probabilities equals 1 by construction (C2 property required by PR #1514). Unchanged from PR #2130.

### Eval wallclock — full disclosure

Total eval times include the n-gram precompute, computed INSIDE the eval timer. `NGRAM_HINT_PRECOMPUTE_OUTSIDE=0` is set explicitly in the launch command. Logs show `ngram_hint_precompute_outside: False` and a `precompute_done` line.

## Reproduction

Install the dependencies in `requirements.txt`. FlashAttention 3 and the `lrzip` system binary are noted there because they require separate install paths.

This submission uses the canonical CaseOps SP8192 export hosted on Hugging Face. Logs are produced from a 50,000-document validation split with 80 training shards.

Preferred data setup (matches PR #2130):

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="romeerp/parameter-golf-caseops-v1",
    repo_type="dataset",
    local_dir="./data/datasets/fineweb10B_sp8192_caseops",
    allow_patterns=[
        "datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/*",
        "datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model",
    ],
    max_workers=8,
)
PY
```

Training command (only the lever override differs from PR #2130):

```bash
for SEED in 314 42 0; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  NCCL_NET=Socket \
  DATA_DIR=./data \
  DATA_PATH=./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 \
  VOCAB_SIZE=8192 \
  ASYM_LOGIT_RESCALE=1 \
  NGRAM_TILT_ENABLED=1 \
  NGRAM_HINT_PRECOMPUTE_OUTSIDE=0 \
  TOKEN_ORDER=16 \
  TOKEN_THRESHOLD=0.800 \
  TOKEN_BOOST=2.625 \
  WITHIN_TAU=99.0 WITHIN_BOOST=0.0 \
  WORD_TAU=99.0 WORD_BOOST=0.0 \
  AGREE_ADD_BOOST=0.0 \
  SMEAR_GATE_BOS_FIX=0 \
  TTT_LORA_EMA_DECAY=0.0 \
  TTT_UPDATE_EVERY=1 \
  PHASED_TTT_PREFIX_DOCS=2500 \
  PHASED_TTT_NUM_PHASES=1 \
  EVAL_SEQ_LEN=2560 \
  TTT_EVAL_SEQ_LEN=2560 \
  COMPRESSOR=pergroup \
  MATRIX_CLIP_SIGMAS=12.85 \
  ATTN_CLIP_SIGMAS=13.0 \
  MLP_CLIP_SIGMAS=11.5 \
  EMBED_BITS=7 \
  EMBED_CLIP_SIGMAS=14.0 \
  MATRIX_LR=0.028 \
  MIN_LR=0.1 \
  WARMDOWN_FRAC=0.85 \
  BETA2=0.99 \
  FUSED_CE_ENABLED=1 \
  SPARSE_ATTN_GATE_ENABLED=1 \
  SPARSE_ATTN_GATE_SCALE=0.5 \
  SMEAR_GATE_ENABLED=1 \
  GATE_WINDOW=8 \
  LQER_ENABLED=1 \
  LQER_RANK=4 \
  LQER_TOP_K=3 \
  LQER_FACTOR_BITS=4 \
  LQER_ASYM_ENABLED=1 \
  LQER_ASYM_GROUP=32 \
  TTT_WARM_START_A=1 \
  TTT_LORA_RANK=80 \
  TTT_BETA2=0.99 \
  TTT_WEIGHT_DECAY=2.0 \
  TTT_LORA_LR=8e-5 \
  GPTQ_RESERVE_SECONDS=0.5 \
  GPTQ_CALIBRATION_BATCHES=16 \
  TTT_K_LORA=0 \
  TTT_O_LORA=1 \
  TTT_MLP_LORA=1 \
  EMA_DECAY=0.9965 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```


## Included files

- `train_gpt.py` - full training/eval script (PR #2130's version).
- `train_seed*.log` - full per-seed logs.
- `submission.json` - structured metadata.
- `README.md` - this file.
- `requirements.txt` - Python dependencies plus notes for FA3 and `lrzip`.
- `prepare_caseops_data.py` - fallback CaseOps dataset/token/byte-sidecar preparation.
- `lossless_caps.py` - reversible CaseOps transform.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` - SentencePiece tokenizer.
- `online_ngram_tilt.py` - n-gram tilt Python wrapper.
- `online_ngram_state.c` - n-gram state machine (token-only path used).

## Lineage and credits

This submission is a single-knob (resp. two-knob) refinement on top of PR #2130, which itself stacks on the public CaseOps/SP8192 record lineage:

- PR #2130 by @codemath3000 - direct comparison baseline; token-only n-gram tilt + AsymLogit Rescale + #2060 levers + NUM_PHASES=1 stacked on PR #1797 / PR #1855 lineage.
- PR #1855 by @codemath3000 - merged leaderboard record.
- PR #1797 by @dexhunter - SmearGate and LQER asymmetric rank-4 lineage; direct base of PR #2130.
- PR #1514 (merged) - token-only n-gram tilt legality precedent.
- PR #1923 - AsymLogit Rescale technique.
- PR #2060 by @S0urC10ud - hyperparameter sweep values.
- PR #2014 by @simonbissonnette - PHASED_TTT_NUM_PHASES=1 precedent.

The new contribution here is the targeted hyperparameter change (GATE_WINDOW=8) on the PR #2130 stack, isolated as a clean ablation so reviewers can compare the lever effect against PR #2130 directly.
