# WARP (Word-Aware Representation Priors) + Legal TTT + Context-Only SLOT

**val_bpb = 1.0713** | **13.65 MB** | 1xH100 SXM, 600s wallclock

## Results (1xH100 80GB SXM, PyTorch 2.10, Flash Attention 3)

| Seed | Steps | Step Avg | Pre-quant | GPTQ int6 | Sliding Win | TTT (2ep) | SLOT (8 steps) | Artifact |
|------|-------|----------|-----------|-----------|-------------|-----------|----------------|----------|
| 1337 | 1,260 | 468ms | 1.1093 | 1.1152 | 1.1030 | 1.0952 | **1.0713** | 13,653,989 |

## Key Innovation: WARP

BPE tokenization destroys word boundary information that the model must re-learn through attention. WARP restores this at three injection points with only **183,820 additional parameters (0.7%)**:

**WARP-Len** (6,657 params) -- Word length embedding at layer 0. Each token gets an embedding based on how many BPE tokens its word contains. Injected before RMSNorm.

**WARP-Pos** (1,035 params) -- Word position bias in Q and K. Learned per-layer embeddings based on within-word position (0-7), applied to both queries and keys before RoPE. Shared across all 11 layers.

**WARP-Type** (176,128 params) -- Word type logit bias at output. A classifier (512->192->64 types) produces soft type probabilities, multiplied with a learned bias matrix (64x1024) and added to logits. No auxiliary loss needed.

All modules share `compute_word_boundary_maps()` -- detects word starts from SentencePiece leading-space convention using only token IDs.

## Architecture

Built on the LeakyReLU-squared + Legal TTT + Parallel Muon stack (by @abaybektursun):

- 11L, 512d, 8H/4KV GQA, LeakyReLU(0.5) squared, MLP 3x
- BigramHash(2816), SmearGate, XSA-11, Partial RoPE
- Parallel Muon + Adam optimizer split
- **WARP-Len** + **WARP-Pos** + **WARP-Type** (this submission)
- **ValueEmbedding removed** (freed params for WARP-Type)
- **EMA disabled** (beta=0.0, prevents +0.045 degradation at ~1260 steps)
- GPTQ int6 + lzma compression
- Legal score-first TTT (2 epochs, freeze_blocks=2, Muon)
- Context-only SLOT (lr=0.005, 8 steps)

## Run Command

```bash
PYTHONUNBUFFERED=1 USE_COMPILE=1 MAX_WALLCLOCK_SECONDS=600 \
ITERATIONS=20000 SEED=1337 TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=2048 VAL_LOSS_EVERY=250 WARMUP_STEPS=20 \
WARMDOWN_ITERS=250 SWA_EVERY=50 USE_GPTQ=1 EMA_DECAY=0.0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=2 TTT_FREEZE_BLOCKS=2 \
TTT_MUON=1 SLOT_ENABLED=1 SLOT_LR=0.005 SLOT_STEPS=8 \
python -u train_gpt.py
```

## Credits

- Base architecture: @abaybektursun (LeakyReLU-squared, Parallel Muon, Parameter Banking, XSA, BigramHash)
- TTT recipe: @Christopher-Lee-McClendon (score-first protocol), adapted by @abaybektursun
- SLOT: @AnubhavBharadwaaj (context-only variant), @abaybektursun
- WARP system: This submission
- I got a lot of help from Claude throughout the development and experimentation process.
