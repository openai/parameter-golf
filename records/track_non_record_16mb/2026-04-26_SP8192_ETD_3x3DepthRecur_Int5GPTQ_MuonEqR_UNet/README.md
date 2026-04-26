# Non-Record: SP8192 + ETD Hybrid (3 enc + 3×3 think + 4 dec) + Int5 GPTQ + MuonEqR + U-Net

**val_bpb = 1.1169** (post-quant int5 + Brotli-11 roundtrip) | **15,865,354 bytes** | 8×H100 SXM, 600s train, single seed (1337)

Single-seed exploratory run. Came within ~0.022 BPB of the merged record at the time (PR #1019, 1.1147) on the first complete configuration that fit under 16 MB at int5 — without any TTT, parallel residuals, BigramHash, sigmoid skip-gating, or seed-tuning. Compute budget ran out before the next iteration, but the gap to SOTA is mostly missing tricks, not missing capacity.

## What this submission is

A hybrid **Encode–Think–Decode (ETD)** transformer:

```
input → [3 unique ENCODE blocks] → [3 shared THINK blocks] × 3 passes → [4 unique DECODE blocks] → output
                  │                                                              ▲
                  └──────────────── U-Net skip connections (×3) ─────────────────┘
```

The motivation: pure depth recurrence (Universal Transformer-style) loses the capacity that unique early/late layers provide, while pure stacking pays full parameter cost for every layer of effective depth. ETD uses unique blocks where layers do fundamentally different jobs (input mapping at the front, output projection at the back) and shared looped blocks for the iterative-refinement middle, where the same operation can usefully be re-applied.

**Effective depth = 16, unique blocks = 10**, all at the same 16 MB disk budget as a 9-layer naive baseline.

## Architecture details

| Component | Value |
|---|---|
| `d_model` | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP multiplier | 4× (relu²-style: `LeakyReLU(0.5)(x).square()`) |
| Encoder blocks (unique) | 3 |
| Think blocks (shared) | 3, looped × **3 passes** |
| Decoder blocks (unique) | 4 |
| Effective depth | 16 |
| U-Net skip connections | 3 (encoder layers → first 3 decoder layers, learned per-channel weights) |
| Pass embedding | `nn.Embedding(num_passes, d_model)` added at each think pass |
| Tokenizer | SentencePiece BPE, **vocab 8192** (fineweb10B_sp8192) |
| Positional encoding | RoPE (base 10000) |
| Norm | RMSNorm, head-wise QK norm, learnable per-head `q_gain` (init 1.5) |
| Embeddings | Tied (`tok_emb` ↔ `lm_head`) |
| Logit softcap | 30.0 |
| Per-block scalars | `attn_scale`, `mlp_scale`, `resid_mix` (learned mixing back to embed) |
| Total params | **33.05 M** |

Code: [train_gpt.py](train_gpt.py) — the model is in `class GPT` and `class Block`; the ETD forward pass is the three-section loop in [`GPT.forward`](train_gpt.py#L962).

### Why each piece is there

- **U-Net skips** ([train_gpt.py:967-984](train_gpt.py#L967-L984)) — encoder outputs are stashed as the model goes down and added back per-channel into the matching decoder layers on the way out. Lets the decoder recover input-side information that the think loop may have abstracted away.
- **Pass embedding** ([train_gpt.py:974-975](train_gpt.py#L974-L975)) — without it, all three think passes are mathematically identical operations on different states, and the model can't easily learn pass-specific behavior.
- **GQA 8/4** — saves KV-cache memory and parameters at near-zero quality cost, standard from baseline.
- **`q_gain` per head** ([train_gpt.py:828](train_gpt.py#L828)) — learnable scalar per head that scales Q before attention; stable improvement over fixed scaling.
- **`resid_mix`** ([train_gpt.py:878](train_gpt.py#L878)) — every block re-mixes the current activations with the embedding output `x0` via a per-channel learned gate, similar to the "smear gate" idea, helping early blocks not destroy embedding signal.
- **Active-pass curriculum** (`ENABLE_LOOPING_AT`, [train_gpt.py:958-960](train_gpt.py#L958-L960)) — supports starting with 1 pass and ramping to N partway through training. Disabled in this run (full 3 passes from step 0); kept in the codebase as it was useful in earlier exploration.

## Training recipe

| Setting | Value |
|---|---|
| Hardware | 8×H100 SXM |
| Wallclock cap | 600 s |
| Iterations cap | 20,000 (hit wallclock at **4,246**) |
| Batch | 786,432 tokens / step (seq_len 2048, 48 seqs/rank) |
| Optimizer (matrices) | **MuonEq-R** (Muon + per-row gradient normalization, NS5 backend), `lr=0.02`, momentum warmup 0.92→0.99 over 1500 steps, **WD=0.09** |
| Optimizer (embed/head/scalars) | AdamW, `tied_embed_lr=0.03`, `scalar_lr=0.02`, `betas=(0.9, 0.95)`, fused |
| Warmdown | Linear to 0 over final **72%** of wallclock |
| Warmup | 20 throwaway steps (params restored, optimizer state restored — the standard "warm the kernels, not the model" trick) |
| EMA decay | 0.997 (applied before quantization) |

## Quantization & compression pipeline

Three-stage post-training compression in [train_gpt.py:1367-1454](train_gpt.py#L1367-L1454):

1. **EMA swap** — load the EMA-averaged weights into the model.
2. **GPTQ** ([train_gpt.py:536-595](train_gpt.py#L536-L595)) — full-Hessian GPTQ with 64 calibration batches:
   - **Encoder / think / decoder matrices: int5** (`clip_range = 15`)
   - **Token embedding: int8** (precision matters more, large fan-in)
   - Per-row fp16 scales (percentile clip at 99.99984%)
   - Optional QuIP-style **randomized Hadamard transform** ([train_gpt.py:489-526](train_gpt.py#L489-L526)) wrapping GPTQ — implemented but disabled (`USE_RHT=0`) in this run because non-pow2 dims in the architecture meant only some tensors qualified.
3. **Byte-shuffle + Brotli-11** ([train_gpt.py:642-678](train_gpt.py#L642-L678)) — deinterleave every 2nd byte before brotli so high/low bytes of fp16 scales group together, exposing lower entropy to the entropy coder.

| Stage | Size |
|---|---|
| Raw fp32 state dict | 123,863,013 B |
| GPTQ int5/8 payload | 33,219,904 B |
| After byte-shuffle + Brotli-11 | **15,802,332 B** |
| Code | 63,022 B |
| **Total submission** | **15,865,354 B** |
| Headroom under 16 MB cap | 134,646 B (~0.84%) |

Roundtrip sanity check passes (`worst tensor=encoder_blocks.2.attn.proj.weight max_abs_err=0.1042`).

## Results

| Metric | Value |
|---|---|
| Steps completed | 4,246 / 20,000 (wallclock cap) |
| Train time | 600,113 ms (step_avg **141.34 ms**) |
| Pre-quant val_loss | 2.8664 |
| Pre-quant val_bpb (best, step 4246) | **1.1097** |
| Post-EMA + int5 GPTQ + Brotli-11 roundtrip val_loss | 2.88515687 |
| Post-quant **val_bpb** | **1.11693413** |
| Peak GPU memory | 33,721 MiB allocated / 34,508 MiB reserved |

### Validation BPB trajectory (last 1500 steps)

| Step | val_bpb |
|---:|---:|
| 2700 | 1.1775 |
| 3000 | 1.1661 |
| 3300 | 1.1519 |
| 3600 | 1.1380 |
| 3900 | 1.1236 |
| 4200 | 1.1103 |
| 4246 (final, pre-quant) | **1.1097** |
| **4246 (post-int5-GPTQ)** | **1.1169** |

Loss was still falling at ~0.005 BPB per 100 steps when wallclock hit. With another 200–300 steps it would have crossed into the 1.10x band pre-quant.

## Why this missed the record (and what would close the gap)

Two separable problems:

1. **Step throughput.** ETD with 3-pass think recurrence costs ~141 ms/step vs ~106 ms/step for the contemporary 11-layer record runs. That's ~33% fewer training steps in the same budget. The architecture is parameter-efficient but compute-hungry — the same problem all depth-recurrence approaches face.

2. **Missing techniques.** The leaderboard winners stack things this run does *not* use:
   - **Legal score-first TTT** (~0.03 BPB on its own)
   - **Parallel residuals** in late layers
   - **BigramHash** factored embedding (frees more bits for matrices)
   - **Partial RoPE** + **higher QK gain** (5.25 vs 1.5)
   - **Sigmoid-gated skip connections** with soft-round QAT
   - **Multi-seed selection**

Each is a known +0.001 to +0.030 BPB improvement. Stacking even 2–3 of them on top of this architecture would clear the 1.0810 record.

## What I'd do next given more compute budget

- **Hybrid partial recursion** — keep the encoder/decoder unique, but only loop the *last* think block (or 1 think block × 4 passes), trading depth-recurrence cost for step throughput. Likely recovers most of the 35 ms/step penalty.
- **Enable RHT in GPTQ** by padding non-pow2 dims, which would tighten int5 quantization error and potentially let some matrices drop to int4.
- **Plug in legal TTT** — straightforward port from PR #1413, accounts for the largest single delta to record.
- **Sweep `MAX_PASSES` ∈ {2, 3, 4}** with progressive recurrence (`ENABLE_LOOPING_AT=0.4`) to see if 1 pass for the warmup phase + N passes for refinement beats N passes throughout.

## Reproduction

```bash
RUN_ID=etd_3x3x4_int5_sp8192 \
DATA_PATH=./data/datasets/fineweb10B_sp8192 \
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model \
VOCAB_SIZE=8192 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
NUM_ENCODER_LAYERS=3 \
NUM_THINK_LAYERS=3 \
NUM_THINK_PASSES=3 \
NUM_DECODER_LAYERS=4 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=4 \
TIE_EMBEDDINGS=1 \
QUANT_BITS=5 \
THINK_QUANT_BITS=5 \
EMBED_BITS=8 \
GPTQ_CALIBRATION_BATCHES=64 \
EMA_DECAY=0.997 \
MUON_WEIGHT_DECAY=0.09 \
MUON_ROW_NORM=1 \
WARMDOWN_FRAC=0.72 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgements

Compute for this work and the broader exploration sweep was funded by the **OpenAI Advanced Competitor grant** (RunPod credits). Without those credits the iteration speed needed to land on the ETD shape and tune the int5 GPTQ pipeline would not have been possible — the result is non-record but the experiments behind it are the real output.

Architectural ideas borrowed from prior leaderboard work:
- Depth recurrence: PRs #1204 (@msisovic), #1331 (@dexhunter)
- MuonEq-R row-normalization: PR #1260 (@dexhunter)
- GPTQ + per-row SDClip: PR #1394 (@clarkkev)
- High weight-decay / brotli-compression synergy: PR #1218 (@clarkkev)
- Byte-shuffle pre-brotli, EMA + warmdown 0.72: assorted record submissions in `track_10min_16mb/`

## Files

```
README.md            # this file
submission.json      # leaderboard metadata
train_gpt.py         # full training + GPTQ + brotli pipeline (single file, 63 KB)
train.log            # full 8×H100 run log (600s, 4246 steps, post-quant eval)
```
