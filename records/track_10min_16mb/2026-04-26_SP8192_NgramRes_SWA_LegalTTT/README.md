# Record: NgramRes + Sliding-Window Attention + Legal Score-First TTT

**val_bpb = 1.08034** (3-seed mean, std 0.00034) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results


| Seed | Sliding BPP | **TTT BPP** | Code stub | Model | **Total** |
|------|-------------|-------------|----------:|----------:|----------:|
| 42   | 1.08173     | **1.08039** | 19,940    | 15,966,406 | **15,986,346** |
| 314  | 1.08211     | **1.08064** | 19,940    | 15,967,359 | **15,987,299** |
| 999  | 1.08137     | **1.07997** | 19,940    | 15,971,159 | **15,991,099** |
| **Mean** | **1.08174** | **1.08034** | **19,940** | **15,968,308** | **15,988,248** |
| **Std**  | **0.00037** | **0.00034** | —      | 2,515      | 2,515          |

Apr-9 record reference (`track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT`): val_bpb = 1.0810.


## Key Techniques

We decomposed the language modeling into two components: an n-gram model for local context, and a residual model for long context:

```
logits(y | x) = α(t) · ngram(x)  +  (1 − α(t)) · residual(x)
loss          = cross_entropy( logits, y )
```

The basic idea is from [[1]], i.e., spending effort on learning the knowledge that can be cheaply captured by n-gram (the local context) seems a waste.

[1]: https://aclanthology.org/2022.findings-emnlp.109

1. **Neural Ngram Model** — a small 3-gram MLP (2 layers, d_hidden = 64, d_embed = 64) reads the same input embeddings, produces logits via the tied output projection, and is mixed into the main logits with α = 0.3. To save memory, the head shares both the input embedding (`NGRAM_SHARE_EMB=1`) and a per-position pad embedding, and ties its output to `tok_emb.weight` (`NGRAM_TIE_OUTPUT=1`). Adds ~0.6 M params (~4 % of the model); int6-quantized identically to the rest of the matrices.
2. **Sliding-Window Attention on layers 0-3** — `flash_attn_3_func(window_size=(512, 0))` on the first four blocks; layers 4-10 keep full causal attention (`SWA_LAST_EARLY_LAYER=4`, `SWA_WINDOW=512`). Frees attention compute on the early layers that handle local syntax, leaving more wallclock for the rest of the stack.
4. **Legal Score-First TTT** — SGD (lr = 0.005, momentum = 0.9), 3 epochs per 32 K-token chunk, gradient clip 1.0. Score-before-update ordering preserved. Same legal framework as the Apr-9 record (PR #1493).
5. **GPTQ int6 + int8 embeddings + Brotli** — full-Hessian GPTQ with SDClip (k = 12.85 for matrices, k = 20.0 for embeddings). The NgramRes head matrices are included in the int6 quantize set. Code shipped as a 2-line LZMA + base85 stub (~20 KB). 

## Architecture

The main backbone, i.e., the residual model is from Apr-9 record (PR #1493). We only add an neural ngram head and train it jointly with the backbone. The NgramRes method is orthogonal to almost all the leaderboard records.

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps) for 2D matrices, AdamW for embeddings / scalars / NgramRes-head bias and gain terms. ~4 699-4 702 steps in 588 s on 8 × H100 SXM (`MAX_WALLCLOCK_SECONDS=600`, with 12 s reserved for GPTQ Hessian collection). Linear wallclock-driven warmdown to LR = 0 over the final 72 % of training. EMA decay 0.9965. Muon momentum warmup 0.92 → 0.99 over the first 1 500 steps. WD = 0.095, MLR = 0.022, head LR = 0.008, embed LR = 0.030.

## Quantization

Identical pipeline to the Apr-9 record (PR #1493): full-Hessian GPTQ with SDClip — `clip = k * std(row)` for principled rate-distortion. int6 for attention / MLP / NgramRes matrices, int8 for token embeddings. Byte-shuffle + Brotli-11 compression. Zero selective pruning needed; model fits natively under 16 MB on all three seeds (~15.99 MB).

## TTT (Test-Time Training)

Score-first, chunk-based SGD adaptation at eval time:

- Chunk val tokens into 32 K-token chunks.
- For each chunk: (1) score all sliding windows under `torch.no_grad()`; (2) train the model on scored chunk tokens with SGD.
- 3 epochs per chunk, momentum = 0.9, gradient clip 1.0.
- Distributed all-reduce across 8 GPUs.
- TTT eval time: ~370-410 s on the three seeds (within the 600 s eval budget).

NgramRes-head parameters are *not* frozen during TTT; the head and the main transformer adapt jointly. The mixing weight α stays fixed at 0.3 throughout TTT.

## Compliance

Per Issue #1017 (Track B — legal eval-time adaptation):

- **Condition 1 (Causality):** Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Condition 2 (Normalized distribution):** Standard softmax over the full vocab. NgramRes mixes hidden representations *before* the shared `lm_head` projection, so logits remain a single normalised softmax over the 8 192-token vocab. No n-gram cache, no logit biasing.
- **Condition 3 (Score before update):** Each chunk fully scored under `torch.no_grad()` BEFORE any SGD update. Training only on already-scored tokens.
- **Condition 4 (Single pass):** Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:

- No SLOT (standard or causal).
- No pre-quant TTT on val data (model is quantized once during training; TTT adapts at eval time only).
- No ETLB (eval-time logit bias).
- No n-gram cache or logit tilt.
- All artifacts under 16 000 000 bytes on all 3 seeds (max: 15,991,099 — seed 999).
- Training under 600 s on all 3 seeds (~588 s actual; cap is 600 s with 12 s reserved for GPTQ).
- Eval (sliding + TTT) under 600 s on all 3 seeds (slowest: seed 42 at ~473 s = 91 s sliding + 382 s TTT).

## Reproduction

This branch is self-contained the same way the Apr-9 record is: the `train_gpt.py` at the repo root **is** the LZMA + base85-encoded byte-counted artifact, and `torchrun` executes it directly. Decoding happens at runtime via the 2-line stub's `exec(lzma.decompress(...))`. The readable source is in `train_gpt.readable.py` (AST-equal).

### Setup (skip if already prepared)

> The Python env and `data/datasets/fineweb10B_sp8192/` are **shared across all sync-pg-ngramres-lc branches** on the standard H100 host. If `python3 -c "import flash_attn_interface, sentencepiece, torch"` succeeds and `data/datasets/fineweb10B_sp8192/fineweb_val_*.bin` exists, jump straight to "Per-seed launch" below.

Only on a fresh host:

```bash
pip install -r requirements.txt
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
./run.sh data         # one-time FineWeb + sp8192 download (~15 GB / ~25 min, idempotent)
```

### Per-seed launch — runs the LZMA stub directly:

```bash
SEED=42 \
  RUN_ID=n4_swa_ttt_seed42 \
  MODEL_PATH=final_n4_swa_ttt_seed42.pt \
  QUANTIZED_MODEL_PATH=final_n4_swa_ttt_seed42.int6.ptz \
  ROPE_BASE=10000 SWA_LAST_EARLY_LAYER=4 \
  NGRAM_ENABLED=1 ALPHA=0.3 NGRAM_N=3 NGRAM_NUM_LAYERS=2 \
  NGRAM_D_EMBED=64 NGRAM_D_HIDDEN=64 NGRAM_LR=0.02 \
  NGRAM_SHARE_EMB=1 NGRAM_SHARE_PROJ=1 NGRAM_TIE_OUTPUT=1 \
  TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  TTT_MOMENTUM=0.9 TTT_CHUNK_TOKENS=32768 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Repeat with `SEED=314` and `SEED=999` for the other two seeds. 

## Credits

- **@ghrua** (this build) — NgramRes head design, sliding-window attention on layers 0-3,  stack assembly + 3-seed run.
- **@bigbag** — The residual model is based on PR #1493:
  - **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394).
  - **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413).
  - **@abaybektursun** — Score-first TTT framework (PR #549).
  - **@Robby955** — Parallel residuals on SP8192 (PR #1412).
  - **@msisovic** — Parallel residuals concept (PR #1204).
  - **@X-Abhishek-X** — Hyperparameter tuning: WD = 0.095, MLR = 0.022, EMA = 0.9965 (PR #1445, #1471).

## Acknowledgements

Submitted to OpenAI's Model Craft Challenge. The NgramRes design draws on the long literature of n-gram-augmented neural language models; this build's specific contribution is mixing the n-gram contribution into the residual stream's logit space (via a tied output projection) rather than as a logit-bias term, which preserves Track-B compliance and quantizes well under the existing GPTQ pipeline.

## Included Files

- `README.md` 
- `submission.json` 
- `train_gpt.py` — LZMA + base85-encoded 2-line stub. The byte-counted artifact form; identical scheme to the Apr-9 record. `torchrun` executes this directly.
- `train_gpt.readable.py` — `ast.unparse(ast.parse(...))` expansion of `train_gpt.py` for human review (byte-identical AST to the encoded form).
- `requirements.txt`, `data/cached_challenge_fineweb.py` — env + dataset preparation.