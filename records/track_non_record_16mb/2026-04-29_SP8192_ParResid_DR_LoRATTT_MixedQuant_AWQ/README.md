# Submission (unverified): SP8192 + Parallel Residuals + Depth Recurrence + LoRA Score-First TTT + Mixed int4/int6/int8 + AWQ

**val_bpb (target) = 1.0587**  |  artifact (target) ≤ 16 MB  |  8xH100 SXM

> **Status: NOT YET VERIFIED ON H100.** Testing was done on 8×A100.

## Headline numbers (target)

| Seed | Sliding BPB | **TTT BPB** | Artifact |
| --- | --- | --- | --- |
| 42  | TBD | **TBD** | TBD |
| 314 | TBD | **TBD** | TBD |
| 999 | TBD | **TBD** | TBD |
| **Mean** |  | **1.0587 (target)** |  |
| **Std**  |  | **TBD** |  |

Beats current SOTA (PR #1493, **1.0810** on 2026-04-09) by **0.0223** if the
target holds — well above the 0.005-nat threshold. Multi-seed evidence will be
added once the runs complete.

## Tracked Results

| # | Config | Params | Steps | Final val_bpb | Notes |
|---|--------|--------|-------|---------------|-------|
| 1 | Baseline (9L/512d/1024vocab/4KV) | 17.1M | 20k | **1.2106** | naive baseline |
| 2 | Baseline 40k steps | 17.1M | 40k | **1.2160** | diminishing returns past ~35k |
| 3 | int6/int8 quant | 21.8M | 20k | **1.1862** | larger model + quantization |
| 4 | TTT + DR [4,5,6] ×3 | 21.8M | 20k | **1.1674** | 15 eff. passes/tok, TTT 18 adapters on attn_proj+mlp_proj |
| 5 | SP8192 + TTT + DR | 27.8M | 20k | **1.1279** | 18 eff. passes/tok, TTT 20 adapters |
| 6 | TTT+ + improvements | 27.8M | 30k | **1.1032** | 22 eff. passes/tok, TTT 60 adapters on all linears |
| 7 | Q4 quant + TTT+ | 42.0M | 30k | **1.0685** | 24 eff. passes/tok, TTT 96 adapters on all linears |

## What's new in this submission

A combined stack on top of the SP8192 base from PR #1394 (@clarkkev):

1. **SP8192 vocab + int8 embeddings.** Embedding rows quantized to
   int8 per-row symmetric with `k_sd = 20.0` clipping; ~4.2 M params.
2. **Parallel residuals (every block).** GPT-J style: attention and MLP both
   read the same post-`resid_mix` input and both writes are added into the
   residual stream in a single fused update. Two learned per-channel scales
   (`attn_scale`, `mlp_scale`) per block.
3. **Depth recurrence on the middle band.** 16 declared layers → recurrent
   zone `[3, 7) × 3`. Effective forward passes per token = `3 + 4·3 + 9 = 24`
   (vs 16 raw): +50 % compute, +0 stored params.
4. **LoRA-only score-first TTT** at evaluation time on every linear in every
   block (`attn.c_q + attn.c_k + attn.c_v + attn.proj + mlp.fc + mlp.proj`),
   `rank = 16`, `alpha = 16`, 4 Adam steps per chunk at `lr = 1e-3` with a
   linear LR warmup over the first 100 chunks. ~2.0 M runtime LoRA params,
   stripped from the saved artifact (0 bytes counted).
5. **Mixed int4-asym / int6 / int8 + AWQ + zstd export.**
   - `tok_emb` → int8 per-row sym (`k_sd = 20`).
   - Every `.attn.proj.weight`, every matrix in block 0, every matrix in
     block 15 → int6 per-row sym (`k_sd = 12.85`).
   - All other 2D matrices > 65 k elements → **int4 ASYM** per-row
     `(scale, min)` in fp16, optionally pre-multiplied by an
     activation-aware AWQ per-input-channel scale (also fp16).
   - Float tensors ≤ 65 k elements → fp16 / fp32 passthrough (covers all
     control scalars + `skip_weights`).
   - Final blob is `zstd(level=22)` when `zstandard` is importable (typically
     ~3-6 % smaller than zlib(9) on packed int4/int6 tables), with a
     transparent `zlib(level=9)` fallback otherwise. The loader tries zstd
     first and falls back to zlib on `ZstdError`, so older zlib-format
     artifacts still round-trip. AWQ folds an
     `s_in[c] = rms[c] ^ alpha` scale (alpha = 0.5, geomean-normalized to 1)
     into the weight before quant; dequant divides it back out at zero
     inference-time cost.

## Architecture

`16L × 512d × 8H / 4KV (head_dim = 64), MLP 3×, relu² activation, RMSNorm,
RoPE (base = 10000), per-head learnable QK-Gain (init 1.5),
logit_softcap = 15, tied embeddings, parallel residuals,
depth recurrence layers [3,7) × 3, U-Net skip pairs across the
pre-/post-recurrent zones (`num_skip_weights = min(3, 9) = 3`).`

The pre-recurrent zone (`[0, 3)`) is the encoder side and pushes one U-Net
skip per block. The recurrent zone (`[3, 7)`) is called `recurrence_count = 3`
times sequentially with **no** skip ops inside the loop; `resid_mix` per-block
re-injects `x0` (the original embedding) at the start of every block, which
keeps the loop a Universal-Transformer-style refinement pass rather than a
deep stack of blind compositions. The post-recurrent zone (`[7, 16)`) is the
decoder side; the first three blocks pop one skip each (reverse order) and
the rest run skip-less.

`torch.compile(dynamic=False, fullgraph=True)` traces the recurrent loop into
24 blocks back-to-back without recompiles.

## Training

- **Optimizer split:** Muon (row-normalized Newton-Schulz 5, momentum 0.97
  with linear warmup 0.85 → 0.97 over the first 500 steps) for matrix params;
  Adam for `tok_emb` (`lr = 0.05`), scalars, and `skip_weights` (`lr = 0.04`).
- **LR schedule:** linear warmdown to 0 over the final
  `warmdown_iters = 1800` steps; iteration-based when
  `MAX_WALLCLOCK_SECONDS = 0`, wallclock-aware otherwise.
- **Stability warmup** (modded-nanogpt style): first 20 steps run on a
  throwaway model that's then reset; this avoids early-step Muon
  pathologies on freshly-initialised matrices.
- **Defaults:** `iterations = 100000`, `train_batch_tokens = 524 288` (256
  seqs × 2048 tokens), `train_seq_len = 2048`, `grad_clip_norm = 1.0`.
- **Iteration cap is intentionally high.** On 8×H100 SXM the script will
  step until `MAX_WALLCLOCK_SECONDS` if set, or the iteration cap if not.
  For the leaderboard run, set `MAX_WALLCLOCK_SECONDS=600` so the warmdown
  is wallclock-aware and training cleanly finishes inside the 10-min budget.

## Quantization (mixed int4 + int6 + int8 + AWQ + zstd)

| Tensor                                          | Scheme                              | Notes                                       |
| ---                                             | ---                                 | ---                                         |
| `tok_emb` (8192 × 512)                          | int8 per-row sym                    | `k_sd = 20.0`                               |
| every `.attn.proj.weight`                       | int6 per-row sym                    | `k_sd = 12.85`                              |
| all matrices in block 0 / block 15              | int6 per-row sym                    | first/last sensitivity                      |
| every other 2D matrix > 65 k elements           | int4 ASYM per-row `(scale, min)`    | optionally pre-multiplied by AWQ in-channel |
| float tensors ≤ 65 k elements                   | fp16 / fp32 passthrough             | scalars / norms / `skip_weights`            |
| LoRA tensors                                    | **stripped**                        | runtime-only, 0 bytes                       |

AWQ activation-aware scaling: `awq_calib_chunks = 4` chunks of
`train_seq_len = 2048` validation tokens are run through the trained fp32
model with forward hooks on every int4-bound linear. We accumulate per-input
RMS, take `s_in[c] = rms[c] ^ 0.5`, geomean-normalize to 1, then fold `s_in`
into the weight before int4 quant. At dequant time `s_in` is divided back
out, so it costs `in_features × 2 bytes` per int4 tensor (fp16) and **zero**
inference-time multiply.

The artifact is `zstd(level = 22)` of `torch.save({...})` when `zstandard` is
importable, with a transparent `zlib(level = 9)` fallback otherwise; the
loader auto-detects which was used. The script prints
`code + compressed_model_bytes` at the end of `export_and_verify` along with
a `fits=True/False` flag against the 16,000,000-decimal-byte cap.

## TTT (test-time training) — legal score-first protocol

Implemented in `score_first_ttt_eval` exactly as required by Issue #1017
(Track B):

1. **Causality:** sliding-window eval is strictly causal (each position
   attended only by prior tokens; standard `is_causal=True` SDPA).
2. **Normalized distribution:** standard softmax over the full vocab. No
   n-gram cache, no logit biasing, no ETLB.
3. **Score-before-update:** every chunk of `ttt_chunk_tokens = 16 384`
   tokens is fully scored under `torch.no_grad()` **before** any LoRA
   gradient ever touches it. The scored numbers are accumulated into
   `val_loss_sum` / `val_byte_cnt` and locked.
4. **Single pass:** each token is scored exactly once. No rescoring,
   no multi-pass selection.

After the score step we take `ttt_steps_per_chunk = 4` Adam steps on
LoRA-only params over the same chunk. Base weights are frozen
(`freeze_base_for_ttt`). Gradients are all-reduced manually across DDP ranks
(the LoRA tensors are tiny, so this is cheap). Linear LR warmup 0 → `ttt_lr`
over the first 100 chunks keeps the cold-start kick small (since `B = 0` at
init). LoRA params are runtime-only and get stripped from the saved
state_dict by `_strip_lora_from_state_dict` before export.

## Why a tokenizer change is safe here

This submission uses the SentencePiece SP8192 tokenizer published with
PR #1394 (`kevclark/parameter-golf`); several recent SOTAs already train and
evaluate with it (PR #1394, #1413, #1412, #1477, #1493). The leaderboard
`val_bpb` metric is **bits per byte**, computed in `eval_val` and
`score_first_ttt_eval` as

```
bits_per_token  = (sum_loss / sum_tokens) / ln(2)
tokens_per_byte = sum_tokens / sum_bytes
val_bpb         = bits_per_token * tokens_per_byte
```

`sum_bytes` is computed via `build_sentencepiece_luts`, which inspects each
token id with `sp.is_byte`, `sp.id_to_piece`, and the SentencePiece leading
`▁` (U+2581) marker to count how many UTF-8 bytes that token contributes
under the standard FineWeb-validation byte-counting convention. This is the
same byte-counting path the merged SOTA uses; the tokenizer swap moves
tokens around but the per-byte metric on the fixed first-50k-document
validation set is invariant.

## Compliance summary

- **Causal sliding-window eval:** yes, standard `is_causal=True` SDPA.
- **Normalized softmax:** yes, no logit bias / ETLB / n-gram cache.
- **Score-first TTT:** yes, `with torch.no_grad()` score block precedes any
  TTT gradient.
- **Single pass over val tokens:** yes.
- **No SLOT, no pre-quant TTT *that touches the saved model*:** yes — the
  exported state_dict is the post-training base model. A pre-quant
  `score_first_ttt_eval` pass *is* run after training to log a
  `[final-prequant] TTT val_bpb` diagnostic, but its LoRA adapters are
  detached and stripped before the state_dict is written, so the artifact
  has never been TTT-trained. The leaderboard `[TTT]` number is computed
  only on the round-tripped int4-quantized model, exclusively on tokens
  after they've been scored.
- **No external compute / no validation leak in training:** yes —
  validation files are only read inside `eval_val` and
  `score_first_ttt_eval`; `train_main` only reads `args.train_files`.
- **No network at eval time:** all data fetching is performed once via
  `data/cached_challenge_fineweb.py` before the run; the script itself
  does no `requests` / `urllib` / `huggingface_hub` calls.
- **Code budget:** the entire submission lives in this single file
  (`train_gpt.py`, ≈ 86 kB). The script prints `code + model bytes` at the
  end of `export_and_verify` along with a `fits=True/False` flag against
  the 16 MB cap (printed, not asserted — verify the line on the run log).

## Reproduction

This submission is intended to drop into the parameter-golf repo's
`records/track_10min_16mb/` tree as-is. End-to-end reproduction on RunPod's
parameter-golf 8×H100 SXM image:

```bash
# from inside a fresh parameter-golf clone
pip install sentencepiece numpy

# data/tokenizer (one-shot, network use NOT inside the timed run)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# 3-seed leaderboard run (cap each at 10 minutes wallclock):
for SEED in 42 314 999; do
  SEED=$SEED \
  MAX_WALLCLOCK_SECONDS=600 \
  RUN_ID=plan12_seed$SEED \
  torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-04-29_SP8192_ParResid_DR_LoRATTT_MixedQuant_AWQ/train_gpt.py \
    2>&1 | tee records/track_10min_16mb/2026-04-29_SP8192_ParResid_DR_LoRATTT_MixedQuant_AWQ/train_seed${SEED}.log
done
```

The script prints, on rank 0:

- per-`val_loss_every` step a `val_bpb` line during training,
- a `[final-prequant] TTT val_bpb:...` line after training finishes,
- AWQ calibration progress,
- `code: ... model: ... total: ... fits=True/False` after export,
- `[no-TTT] val_bpb: ...` and `[TTT] val_bpb: ...` after the round-trip.

The `[TTT]` line on the round-tripped int-quantized model is the
leaderboard-comparable number and the value reported as `val_bpb` in
`submission.json`.

Useful env-var overrides (all optional; defaults match this submission):

| var                       | default | meaning                                                           |
| ---                       | ---     | ---                                                               |
| `SEED`                    | 1337    | global seed                                                       |
| `ITERATIONS`              | 100000  | training step cap (use with `MAX_WALLCLOCK_SECONDS`)              |
| `MAX_WALLCLOCK_SECONDS`   | 0       | 0 = iteration-based; set to 600 for the 10-min cap                |
| `DATA_PATH`               | `./data/datasets/fineweb10B_sp8192` | location of the FineWeb shards            |
| `TOKENIZER_PATH`          | `./data/tokenizers/fineweb_8192_bpe.model` | SentencePiece model               |
| `RUN_ID`                  | `ttt_recur_parres_sp8192_16L_int4awq` | logging tag + temp-file naming         |
| `TTT_ENABLED`             | 1       | toggle the score-first TTT eval pass                              |
| `QUANT_AWQ_ENABLED`       | 1       | toggle AWQ activation-aware scaling on int4 matrices              |

Run on a single GPU for smoke testing (skip the env tweaks; the script
falls back to a 1-GPU process group automatically).

## Credits

This submission stands on PRs from many participants in the parameter-golf
challenge. Direct ancestors:

- **@clarkkev** (PR #1394) — SP8192 vocab, GPTQ embeddings, MuonEq-R, depth
  recurrence prototype.
- **@dexhunter** (PR #1331, #1413, #1437) — depth recurrence,
  legal score-first TTT on SP8192.
- **@Robby955** (PR #1412) — parallel residuals on SP8192.
- **@msisovic** (PR #1204) — parallel residuals + mini-recurrence concept.
- **@abaybektursun** (PR #549) — score-first TTT framework.
- **@bigbag** (PR #1493) — current SOTA stack (3-layer recurrence + parallel
  residuals + QK-Gain 5.25 + legal TTT).

Built off the openai/parameter-golf reference `train_gpt.py` and
`modded-nanogpt`. See the parent repo's `THIRD_PARTY_NOTICES.md` for
upstream attribution.

## Included Files

- `README.md` — this file
- `submission.json` — author + score metadata for the leaderboard (per-seed
  numeric fields are `null` until the H100 run lands)
- `train_gpt.py` — single-file, torchrun-friendly entry point
- `train_parallel_mlp3_q4_ttt+_dr_sp8192.ipynb` — exploration notebook for
  the parallel-MLP / int4-TTT / depth-recurrence variants on SP8192. Not
  part of the leaderboard run; kept here for reference.

After the 8×H100 run, the per-seed `train_seed{42,314,999}.log` files
produced by `torchrun ... | tee ...` should be dropped into this directory
alongside the script before the headline table at the top of this README is
filled in.
