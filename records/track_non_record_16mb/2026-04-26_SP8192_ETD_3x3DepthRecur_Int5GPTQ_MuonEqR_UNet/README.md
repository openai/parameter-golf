# Non-Record: SP8192 + ETD Hybrid + Int5 GPTQ + MuonEqR + U-Net (clean architecture-only run)

**val_bpb = 1.1169** (post-quant int5 + Brotli-11 roundtrip) | **15,865,354 bytes** | 8×H100 SXM, 600s train

## The pitch

How far can you get with **only architectural changes and better quantization**, with none of the eval-time tricks the leaderboard has converged to? No TTT, no test-time SGD, no n-gram cache, no parallel residuals, no BigramHash factored embedding, no sigmoid skip-gating with QAT — none of it. Just a hybrid Encode–Think–Decode transformer, MuonEqR, EMA, and a tight int5 GPTQ + Brotli-11 pipeline.

The result reaches **1.1169 BPB** at 15.87 MB. Hyperparameters are **not optimized** and this is **not the best run** — it's the first complete configuration of the architecture that fit under 16 MB at int5. I am actively trying to secure more compute budget to do the proper sweep; the numbers below should improve substantially once that lands.

## Architecture

A hybrid **Encode–Think–Decode (ETD)** transformer:

```
input → [3 unique ENCODE blocks] → [3 shared THINK blocks] × 3 passes → [4 unique DECODE blocks] → output
                  │                                                              ▲
                  └──────────────── U-Net skip connections (×3) ─────────────────┘
```

Pure depth recurrence (Universal Transformer-style) loses the capacity that unique early/late layers provide. Pure stacking pays full parameter cost for every layer of effective depth. ETD uses **unique blocks where layers do fundamentally different jobs** (input mapping at the front, output projection at the back) and **shared looped blocks for the iterative-refinement middle**, where the same operation can usefully be re-applied.

**Effective depth = 16, unique blocks = 10** at the same 16 MB disk budget as a 9-layer naive baseline.

### Specifics

| Component | Value |
|---|---|
| `d_model` | 512 |
| Heads / KV heads | 8 / 4 (GQA) |
| MLP multiplier | 4× with **LeakyReLU(0.5)²** activation |
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
| Total params | ~33 M |

Code: [train_gpt.py](train_gpt.py) — the model is in `class GPT` and `class Block`; the ETD forward pass is the three-section loop in [`GPT.forward`](train_gpt.py#L962).

### Why each piece is there

- **U-Net skips** ([train_gpt.py:967-984](train_gpt.py#L967-L984)) — encoder outputs are stashed as the model goes down and added back per-channel into the matching decoder layers on the way out. Lets the decoder recover input-side information that the think loop may have abstracted away.
- **Pass embedding** ([train_gpt.py:974-975](train_gpt.py#L974-L975)) — without it, all three think passes are mathematically identical operations on different states, and the model can't easily learn pass-specific behavior.
- **GQA 8/4** — saves KV-cache memory and parameters at near-zero quality cost.
- **`q_gain` per head** ([train_gpt.py:828](train_gpt.py#L828)) — learnable scalar per head that scales Q before attention; stable improvement over fixed scaling.
- **`resid_mix`** ([train_gpt.py:878](train_gpt.py#L878)) — every block re-mixes the current activations with the embedding output `x0` via a per-channel learned gate, helping early blocks not destroy embedding signal.
- **Progressive recurrence** (`ENABLE_LOOPING_AT=0.35`, [train_gpt.py:958-960](train_gpt.py#L958-L960)) — start with 1 think pass, switch to full 3 passes once the model is past the early-loss collapse phase. Lets the matrices warm up cheaply before paying the recurrence cost.
- **LeakyReLU(0.5)²** ([train_gpt.py:859](train_gpt.py#L859)) — preserves negative gradient flow through the MLP while keeping the relu² inductive bias. One-line change, free improvement over plain relu².

## Training recipe

| Setting | Value |
|---|---|
| Hardware | 8×H100 SXM |
| Wallclock cap | 600 s |
| Batch | 786,432 tokens / step (seq_len 2048, 48 seqs/rank) |
| Optimizer (matrices) | **MuonEq-R** (Muon + per-row gradient normalization, NS5 backend), `lr=0.02`, momentum warmup 0.92→0.99 over 1500 steps, **WD=0.09** |
| Optimizer (embed/head/scalars) | AdamW, `tied_embed_lr=0.03`, `scalar_lr=0.02`, `betas=(0.9, 0.95)`, fused, embed WD 0.085, adam WD 0.02 |
| Warmdown | Linear to 0 over final **40%** of wallclock |
| Warmup | 20 throwaway steps (params restored, optimizer state restored — warm the kernels, not the model) |
| Grad clip | 0.3 |
| EMA decay | 0.997 (applied before quantization) |

## Quantization & compression pipeline

This is where most of the architectural budget gets cashed in. Three-stage post-training compression in [train_gpt.py:1367-1454](train_gpt.py#L1367-L1454):

1. **EMA swap** — load the EMA-averaged weights into the model.
2. **Full-Hessian GPTQ with SDClip** ([train_gpt.py:536-595](train_gpt.py#L536-L595)) — 64 calibration batches:
   - **Encoder / think / decoder matrices: int5** (`clip_range = 15`)
   - **Token embedding: int8** (precision matters more, large fan-in)
   - SDClip per-row clipping at `k=6·std(row)` for matrices, `k=20` for embeddings
   - Per-row fp16 scales
3. **QuIP-style Randomized Hadamard Transform** ([train_gpt.py:489-526](train_gpt.py#L489-L526)) — wraps GPTQ; rotates weights into an incoherent basis so outliers flatten before quantization, with the Hessian conjugated to stay consistent. Tightens int5 quantization error meaningfully on the matrices that have power-of-two dims.
4. **Byte-shuffle + Brotli-11** ([train_gpt.py:642-678](train_gpt.py#L642-L678)) — deinterleave every 2nd byte before brotli so high/low bytes of fp16 scales group together, exposing lower entropy to the entropy coder.

| Stage | Size |
|---|---|
| Raw fp32 state dict | 123,863,013 B |
| GPTQ int5/8 payload | 33,219,904 B |
| After byte-shuffle + Brotli-11 | **15,802,332 B** |
| Code | 63,022 B |
| **Total submission** | **15,865,354 B** |
| Headroom under 16 MB cap | 134,646 B (~0.84%) |

## Results

| Metric | Value |
|---|---|
| Steps completed | 4,246 (wallclock cap) |
| Train time | 600,113 ms (step_avg **141.34 ms**) |
| Pre-quant val_bpb (best) | **1.1097** |
| Post-EMA + int5 GPTQ + Brotli-11 roundtrip val_loss | 2.88515687 |
| Post-quant **val_bpb** | **1.11693413** |
| Peak GPU memory | 33,721 MiB allocated / 34,508 MiB reserved |

### Validation BPB trajectory (last ~1500 steps)

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

Loss was still falling at ~0.005 BPB per 100 steps when wallclock hit. Even another 200–300 steps would have crossed cleanly into the 1.10x band pre-quant.

## Why this missed the record (and what would close the gap)

This run is intentionally clean — architecture and quantization only. The leaderboard winners stack things this submission does *not* use:

- Legal score-first TTT (~0.03 BPB on its own)
- Parallel residuals in late layers
- BigramHash factored embedding (frees more bits for matrices)
- Partial RoPE + higher QK gain (5.25 vs 1.5)
- Sigmoid-gated skip connections with soft-round QAT

Each is a known +0.001 to +0.030 BPB improvement. Stacking even 2–3 of them on top of this architecture would clear the current record. **That's a deliberate choice, not an oversight** — the goal here was to see how much can come from architecture and quantization alone.

The other limitation is **step throughput**: ETD with 3-pass think recurrence costs ~141 ms/step vs ~106 ms/step for the 11-layer record runs. That's ~33% fewer training steps in the same budget — the architecture is parameter-efficient but compute-hungry, the standard depth-recurrence tradeoff.

## What's next

- **Hyperparameter sweep.** Almost nothing here is tuned. Encoder/decoder layer split, think-pass count, MLP multiplier, MuonEq-R LR/WD, embedding LR, EMA decay, RHT seed strategy — the local optima are unexplored. Most likely-impactful: tightening the int5 clip-sigma schedule and a proper progressive-recurrence frac sweep.
- **More architecture research.** The ETD shape itself has a lot of unexplored levers: asymmetric think-block widths, partial recursion (loop only the *last* think block at higher pass count), think-block-specific attention configs, encoder/decoder asymmetry beyond just layer count.
- **Push the int5 budget further.** Int5 is so compact (5 bits × ~33M params ≈ 21 MB pre-compression, which Brotli takes to ~16 MB) that there's substantial room for improvement here: padding non-pow2 dims so RHT covers more tensors, per-tensor bit-allocation based on Hessian-diagonal sensitivity, mixed int4/int5 with the saved bytes spent on more think layers or wider embeddings.

All of the above need compute. **I am actively trying to secure more budget to do this properly.**

## Reproduction

### Setup (one-time)

The submission depends on `brotli` for the int5 compression pipeline, and on the **SP8192 tokenizer + pre-tokenized FineWeb** hosted by @clarkkev on HuggingFace (the official `data/cached_challenge_fineweb.py` ships sp1024/sp4096 by default, not sp8192):

```bash
pip install brotli

rm -f data/manifest.json
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

The `rm` is required because the download script caches a manifest from the default repo, and a stale one won't include sp8192.

### Run

```bash
export RUN_ID=etd_33x34_mlpx4x4_int5_SD6_pr40
export NUM_ENCODER_LAYERS=3
export NUM_THINK_LAYERS=3
export NUM_THINK_PASSES=3
export NUM_DECODER_LAYERS=4
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export THINK_MLP_MULT=4
export THINK_KV_HEADS=0
export THINK_NUM_HEADS=0
export MAX_WALLCLOCK_SECONDS=600
export TIE_EMBEDDINGS=1
export SEED=1337
export VAL_LOSS_EVERY=100
export TRAIN_LOG_EVERY=25
export QUANT_BITS=5
export THINK_QUANT_BITS=5
export EMBED_BITS=8
export GPTQ_CALIBRATION_BATCHES=64
export USE_RHT=1
export EMA_DECAY=0.997
export ENABLE_LOOPING_AT=0.35
export MUON_WEIGHT_DECAY=0.09
export MUON_ROW_NORM=1
export GRAD_CLIP_NORM=0.3
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export VOCAB_SIZE=8192
export DATA_PATH=./data/datasets/fineweb10B_sp8192
export TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe.model
export EMBED_WEIGHT_DECAY=0.085
export ADAM_WEIGHT_DECAY=0.02
export EMBED_CLIP_SIGMAS=20
export INT8_CLIP_PERCENTILE=99.99984
export MATRIX_CLIP_SIGMAS=6
export WARMDOWN_FRAC=0.4

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Acknowledgements

Compute for this work and the broader exploration sweep was funded by the **OpenAI Advanced Competitor grant** (RunPod credits). The iteration speed required to land on the ETD shape, get the int5 GPTQ + RHT pipeline tight, and tune the compression pipeline would not have been possible without those credits.

Components borrowed from prior leaderboard work:
- **SP8192 tokenizer + pre-tokenized FineWeb dataset**: [PR #1394](https://github.com/openai/parameter-golf/pull/1394) by @clarkkev — first introduced the 8192-vocab SentencePiece tokenizer and made the pre-tokenized shards publicly available via HuggingFace ([kevclark/parameter-golf](https://huggingface.co/datasets/kevclark/parameter-golf)). This entire submission is downstream of that data being available.
- **LeakyReLU(0.5)² activation**: [PR #493](https://github.com/openai/parameter-golf/pull/493) by @parinzee, [PR #518](https://github.com/openai/parameter-golf/pull/518) by @sofiabod
- **MuonEq-R row-normalization**: PR #1260 (@dexhunter)
- **GPTQ + per-row SDClip**: PR #1394 (@clarkkev)
- **High weight-decay / brotli-compression synergy**: PR #1218 (@clarkkev)
- **QuIP-style Randomized Hadamard Transform**: from the QuIP / QuIP# line of work, adapted into the GPTQ pipeline here
- Byte-shuffle pre-brotli, EMA + warmdown: assorted record submissions in `track_10min_16mb/`

## Files

```
README.md            # this file
submission.json      # leaderboard metadata
train_gpt.py         # full training + GPTQ + brotli pipeline (single file, 63 KB)
train.log            # 8×H100 run log (600s, post-quant eval)
```
