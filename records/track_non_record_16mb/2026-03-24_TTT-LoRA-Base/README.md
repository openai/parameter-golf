# TTT-LoRA Base — HumanAI Convention (2026-03-24)

This submission documents a **Test-Time Training (TTT) via LoRA** approach to the Parameter Golf
challenge. The base score is reported with TTT disabled; a follow-up submission with calibrated
TTT is in preparation.

---

## Novel Contribution: Per-Document TTT LoRA

The primary innovation is test-time adaptation of the model to each validation document using
rank-128 LoRA adapters injected into all attention projections and MLP layers. During evaluation,
each document receives its own fresh set of LoRA adapters trained with Adam on the preceding
chunks of that document before predicting the next chunk.

This is distinct from fine-tuning: adapters are reset to zero between documents and never shared
across documents, making it a purely test-time, per-document operation. The 10-minute evaluation
budget is used for this adaptation loop.

**Current status:** Local smoke tests on a 512-dim model showed consistent improvement
(−0.136 bpb on 50-doc test). The first production run on 8×H100 revealed that LR=0.001
(tuned on a poorly-trained smoke model) is too high for a well-trained production model and
causes over-adaptation. LR calibration is in progress; this submission reports the base score
only (TTT_ENABLED=0).

---

## Architecture Changes vs Naive Baseline

- **SmearGate**: Learnable residual mixing gate in each transformer block; allows the model
  to smoothly interpolate between full-residual and full-hidden-state at each layer.
- **Orthogonal initialisation**: All matrix parameters initialised orthogonally via
  `nn.init.orthogonal_`; improves gradient flow and training stability.
- **Bigram hash embeddings**: 2048-bucket bigram hash table added to token embeddings;
  provides cheap local context without extra parameters counted against the 16MB budget.
- **GQA (Grouped-Query Attention)**: 8 query heads, 4 KV heads; reduces KV cache and
  allows higher batch throughput during TTT eval.
- **Stochastic Weight Averaging (SWA)**: Applied over the final 5065 steps with decay=0.4;
  averages 500 checkpoints spaced every ~10 steps for a smoother final model.
- **int6 + zstd22 quantisation**: QAT-aware int6 quantisation with zstd level-22 compression;
  achieves 5.14× payload compression ratio, fitting comfortably within 16MB.

---

## Configuration

- Vocab size: 1024 (SentencePiece BPE, fineweb_1024_bpe)
- Layers: 11
- Model dim: 512
- Heads: 8 query / 4 KV (GQA)
- MLP multiplier: 3×
- Tied embeddings: yes
- Bigram hash size: 2048
- Train batch tokens: 524,288 (per step, across 8 GPUs)
- Train sequence length: 1024
- Training steps completed: 7,065 / 20,000 (wallclock cap at 600s)
- SWA steps: 5,065, decay: 0.4

---

## Command

```bash
torchrun --standalone --nproc_per_node=8 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512 \
  NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
  TIE_EMBEDDINGS=1 BIGRAM_HASH_SIZE=2048 \
  SMEAR_GATE=1 ORTHO_INIT=1 \
  TRAIN_BATCH_TOKENS=524288 TRAIN_SEQ_LEN=1024 \
  MAX_WALLCLOCK_SECONDS=600 \
  MATRIX_LR=0.04 SCALAR_LR=0.04 EMBED_LR=0.05 \
  MUON_WEIGHT_DECAY=0.04 \
  SWA_STEPS=500 SWA_DECAY=0.4 \
  QAT_BITS=6 \
  VAL_LOSS_EVERY=0 SLIDING_EVAL_STRIDE=512 \
  TTT_ENABLED=0 \
  TORCH_COMPILE=1 SDP_BACKEND=flash \
  SEED=1337 \
  train_gpt.py
```

---

## Key Metrics

- Training steps: 7,065 / 20,000 (stopped at wallclock cap, 600.116s)
- SWA applied: 5,065 steps, decay=0.4
- Peak GPU memory: 13,819 MiB allocated / 14,030 MiB reserved
- Pre-quantisation val_bpb: 1.1989 (step 7065)
- Post-quantisation val_loss (exact): 2.08757157
- Post-quantisation val_bpb (exact): **1.23637747**
- Serialised model (int6+zstd22): 15,589,140 bytes (payload ratio: 5.14×)
- Code size: 80,186 bytes
- Total submission size: 15,669,326 bytes

---

## Training Volume

- Global batch size: 524,288 tokens/step
- Steps completed: 7,065
- Total tokens seen: ~3.70 billion

---

## TTT-LoRA Configuration (for follow-up submission)

When LR is calibrated, the TTT eval will use:

- LoRA rank: 128 (injected into all attention + MLP layers)
- Chunk size: 64 tokens
- Adam steps per chunk: 4
- Learning rate: TBD (0.001 too high; sweep in progress)
- Batch size: 64 documents
- Eval cap: 480 seconds (8-minute window within 10-minute evaluation budget)

---

## Included Files

- `README.md` — this file
- `submission.json` — metadata and scores
- `train_gpt.py` — complete training script with TTT-LoRA, SmearGate, OrthoInit, SWA
- `train.log` — raw output from the 8×H100 production run

---

## Credits Request

We are applying for OpenAI compute credits to complete TTT LR calibration and submit a
competitive follow-up. The TTT-LoRA approach is architecturally sound and orthogonal to
all current leaderboard entries; we expect it to be competitive once properly calibrated.
