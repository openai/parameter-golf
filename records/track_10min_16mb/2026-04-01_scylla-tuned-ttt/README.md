# Record: Scylla + n-gram + legal TTT — val_bpb 1.0903 (3-seed mean)

**val_bpb: 1.0903** (3-seed mean, std 0.00040) | ≤15.4 MB | 8×H100 SXM | ~94ms/step | ~6160 steps

Applies the Scylla tokenizer (PR #1143, @simon-marcus) with n-gram rescoring and tuned legal score-first TTT.

## 3-Seed Results

| Seed | legal_ttt BPB | bytes_total | Steps | Wall-clock |
|------|---------------|-------------|-------|------------|
| 1337 | 1.09042 | 15,316,209 | ~6151 | 600s |
| 42   | 1.09064 | 15,329,825 | ~6162 | 600s |
| 2025 | 1.08985 | 14,945,965 | ~6170 | 600s |
| **Mean ± Std** | **1.09030 ± 0.00040** | | | |

All seeds stopped by wallclock cap. All artifacts under 16,000,000 bytes.

## Technique Stack

- **Scylla tokenizer** — ~998 token TokenMonster vocabulary (PR #1143)
- **N-gram rescoring** — orders 2–16, two-pass eval (neural pass then cache rescore)
- **Legal TTT** — score-first SGD, LR=0.005, 3 epochs, 32768 tokens/chunk
- **int6 quantization** — per-row with lzma, `clip_range=20` for stable byte compliance
- **11-layer cyclic shared blocks**, BigramHash embeddings (10240 vocab, 128-dim)
- **Parallel Muon** — matrix LR=0.025, momentum=0.99, WD=0.04; EMA decay=0.997

`clip_range` was tuned from 31→20 to eliminate seed-dependent artifact size variance. Cost: ~+0.006 BPB. All seeds now produce artifacts between 14.9–15.4 MB.

## Legality

**Training (≤600s on 8×H100):** Standard. No validation data accessed during training.

**Evaluation — TTT (score-first):** Each chunk scored under `torch.inference_mode()` before any parameter update. The reported BPB is always computed before adaptation. TTT runs ~427s.

**N-gram metric:** The `legal_ttt` score above uses only the TTT-adapted neural model. N-gram rescoring (`ngram_two_pass`, ~0.195 BPB) is reported separately and not used as the submission metric.

## Reproduction

```bash
# Retokenize (one-time, ~77 min on CPU)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs
python3 data/retokenize_scylla.py \
    --docs ./data/docs_selected.jsonl \
    --vocab ./data/tokenizers/scylla/candidate.vocab \
    --out ./data/datasets/fineweb_scylla --train-shards 80

# Train (replace SEED with 1337, 42, or 2025)
DATA_PATH=./data/datasets/fineweb_scylla \
TOKENIZER_PATH=./data/tokenizers/scylla/candidate.vocab \
TOKENIZER_META_PATH=./data/tokenizers/scylla/candidate.meta.npz \
NGRAM_ENABLED=1 NGRAM_MAX_ORDER=16 TTT_ENABLED=1 TTT_LR=0.005 XSA_LAST_N=4 \
SEED=1337 NUM_LAYERS=11 NUM_SHARED_BLOCKS=11 SHARED_BLOCK_LAYOUT=cyclic \
MLP_MULT=3 VOCAB_SIZE=1024 INT6_QUANT=1 SMEAR_GATE=1 BIGRAM_VOCAB_SIZE=10240 \
ORTHO_INIT=1 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 GRAD_CLIP_NORM=0.3 TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
ITERATIONS=9000 WARMDOWN_ITERS=3500 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 SWA_INTERVAL=50 SWA_LR_THRESHOLD=0.2 EMA_DECAY=0.997 \
TTT_EPOCHS=3 TTT_MAX_SEQS=64 TTT_CHUNK_TOKENS=32768 TTT_GRAD_CLIP=1.0 \
ROPE_DIMS=16 LN_SCALE=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Requires 8×H100, PyTorch 2.9+, CUDA 12.8, tokenmonster.

## Credits

- Scylla tokenizer: @simon-marcus (PR #1143)
- Training base: @abaybektursun (PR #549)
