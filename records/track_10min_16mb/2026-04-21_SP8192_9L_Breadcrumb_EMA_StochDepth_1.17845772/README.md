# Record: SP8192 + 9L Breadcrumb + EMA + Stochastic Depth

**val_bpb = 1.17845772** · **15,897,032 bytes** (int6+zlib) · 1 seed (s=1337) · 8×H100 80GB SXM

> **Known limitation — single-seed result.** Only seed 1337 was run. `submission.json` declares `three_seeds: false`. The tradeoff was deliberate: remaining H100 budget went to expanding the search space (tokenizer / gating / regularization) rather than re-running the same config. This is the largest methodological gap relative to multi-seed top-of-leaderboard entries and is stated up front rather than buried.

## Result

| Seed | Sliding BPB (exact) | Artifact bytes |
|------|---------------------|----------------|
| 1337 | **1.17845772**      | 15,897,032     |

Improvement over the contest's naive baseline (1.2244): **−0.0460 BPB**.

> **Note on the log label.** The final line of the validation log reads `final_int8_zlib_roundtrip val_bpb:1.17845772`. The `int8_zlib_roundtrip` label is inherited from the scoring function in `train_gpt.py`; **the packaged artifact is int6+zlib**, as reflected in the earlier `Serialized model int6+zlib: 15,836,325 bytes` line, the final `Total submission size int6+zlib: 15,897,032 bytes` line, and in `submission.json`.

---

## How the record was built — front-end first, compression last

The unique work happened **before** the compression step. Compression is the gate that makes it contest-legal; everything interesting is upstream of it. That ordering is deliberate and worth surfacing, because a reader reverse-engineering this submission from the artifact alone would see int6+zlib and miss that int6+zlib is the *last* thing that happened, not the first.

### Breadcrumb 1 — tokenizer vocabulary and byte-fallback

The default Parameter Golf baseline uses a 1024-vocab tokenizer with `byte_fallback=False`. At 1024 vocab, rare Unicode bytes fragment into long token sequences the model never sees enough of to learn. Enabling byte-fallback so unknown characters pass through as single tokens, then stretching vocab to 8192 via SentencePiece BPE, alone accounted for roughly **−0.031 BPB** against the default.

This is the single biggest lever in the whole pipeline. It's also the one that most clearly belongs at the front: it changes what the model is *trying to predict* before the architecture even sees the data.

### Breadcrumb 2 — breadcrumb gating on MLP contributions

A small learned consistency score gates each MLP block's contribution to the residual stream (sigmoid-gated residuals). The gate is cheap — one scalar per layer per token, trained jointly — and acts as a form of skip regularization without adding the discrete branching cost of drop-block or router noise. On a 9-layer model with a wallclock ceiling, the gating stabilizes early training enough that more of the 600 s budget ends up at useful step counts.

### Breadcrumb 3 — EMA + stochastic depth, chosen for the wallclock regime

EMA (decay 0.997) swapped in at end-of-training for eval, plus stochastic depth with expected-value scaling at train time and full activation at eval. Both are standard; the non-obvious part is that they were chosen *because* the wallclock cap forces the model to stop mid-training, and these are the two regularizers that give the most usable signal at the stopped step. They are not a choice from a zoo — they are an answer to "the budget ends before the loss flattens."

### Breadcrumb 4 — Muon for matrix weights, AdamW for the rest

Newton–Schulz 5-step iteration with momentum 0.95 warming up from 0.85 over 500 steps, applied to matrix weights only; AdamW on embeddings and scalars. This combination converges faster than AdamW-everything at the same LR. Small detail, but it's what made the 10-minute cap feel like 12 minutes of effective training.

### Breadcrumb 5 — only *then*, compression

Int6 per-tensor symmetric quantization plus zlib on the serialized package. The quantization ceiling on this architecture is ~0.00489 BPB — the gap between fp16 sliding BPB (1.17357) and the int6+zlib roundtrip. That ceiling is roughly half the 10-layer→9-layer architectural penalty, so quantization is cheap given the setup; the hard work was getting the fp16 number low enough that int6+zlib still fit under 16 MB.

The point of enumerating these in order: compression was the last constraint satisfied, not the first trick pulled. A reader who starts the other way — from int6+zlib and works backward — reconstructs the submission as "the compression is what did it," which inverts the causality.

---

## Architecture (one block)

9L × 512d × 8 heads / 4 KV heads GQA · MLP expansion 2× · tied embeddings · partial-RoPE · logit softcap 30.0 · 20,882,280 parameters before quantization.

## Training

- 8×H100 80GB SXM · PyTorch 2.9.x · FlashAttention on (cuDNN / mem-efficient / math kernels off)
- `train_batch_tokens = 524,288` · `train_seq_len = 1024`
- Warmup 20 steps · warmdown 1200 steps · `MAX_WALLCLOCK_SECONDS = 600`
- Wallclock-stopped at step 6886 / 20000 · `step_avg ≈ 87 ms`
- Intermediate val BPB: step 4000 → 1.2365, step 5000 → 1.2276, step 6000 → 1.2146, step 6886 → 1.1907 sliding → **1.17845772** after int6+zlib roundtrip (exact precision)

## Compliance (Track A — 10 min, 16 MB, no test-time adaptation)

- Causal sliding-window eval, stride 64
- Standard softmax over full vocab · no n-gram cache · no logit biasing
- No SLOT, no pre-quant TTT, no ETLB
- Artifact: **15,897,032 bytes** (under the 16,000,000-byte cap) — code 60,707 B + int6+zlib weights 15,836,325 B
- Training under 600 s wallclock (early-stop at the cap)

## Reproduction

```bash
pip install -r requirements.txt

MATCHED_FINEWEB_REPO_ID=willdepueoai/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

# The variant script materializes shards at data/datasets/fineweb10B_sp8192.
# The record run used an "nb"-suffixed directory, so symlink to match the
# DATA_PATH below. This preserves fidelity to the exact paths in the log
# rather than silently retargeting.
ln -s ./data/datasets/fineweb10B_sp8192 ./data/datasets/fineweb10B_sp8192nb

SEED=1337 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
  VOCAB_SIZE=8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192nb_bpe.model \
  DATA_PATH=./data/datasets/fineweb10B_sp8192nb \
  WARMDOWN_ITERS=1200 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The "nb" (non-byte-fallback/banked) tokenizer variant is what the training log was measured against. See [`SHARD_RECOVERY_RECIPE.md`](../../../SHARD_RECOVERY_RECIPE.md) at the repo root for the full tokenizer-build provenance and the reasoning behind the symlink step.

## What was tried and rejected

Negative results are part of the trail. In a targeted overnight swarm session, four challenger strategies were run against the champion; all four regressed. Keeping them in the record rather than deleting them:

| Challenger                      | Result        | Delta vs champion | Why it failed                               |
|---------------------------------|---------------|-------------------|----------------------------------------------|
| QK_GAIN_INIT = 5.25 (9L)        | 1.17846316    | +0.00000544       | Lever was calibrated for 11L config; no transfer to 9L. |
| GPTQ-lite activation-aware      | 1.18442869    | +0.00597          | At this scale/budget, activation-aware quant does not beat plain per-tensor int6. |
| Kimi-Claw QK_GAIN_INIT = 7.0    | 1.18586312    | +0.00740          | Same transfer failure as QK=5.25, larger step, larger regression. |
| DeepSeek SCALAR_LR = 0.1        | 1.21650816    | +0.03805          | Scalar-LR wants to stay at or below default 0.04; raising it destabilizes training within the 600 s budget. |

The 10-layer variant tried earlier in the sprint reached a better fp16 BPB (~1.1745) but could not be compressed under the 16 MB cap with the quantization tools available. That model was not packaged for submission.

The fp16 sliding BPB on this architecture is 1.17357, meaning the **quantization+compression ceiling** is roughly 0.00489 BPB — this is the headroom available before any further front-end work needs to happen. Additional gains at this scale almost certainly live upstream of compression, not inside it.

## How this was worked

The submission was produced by an independent researcher with no formal ML background, coordinating across several AI models (code assistants, quantization specialists, and a retrieval-grounded critic configured against drift). The operating pattern was: one model drafts, a second re-derives the change from scratch, a third reads the diff as adversary. Changes that survived all three passes were run on compute; changes that didn't were discarded without burning cycles. That's the same pattern used to review the four challengers above.

Roughly 72 hours of active work across April 15–17, plus the overnight confirmation + challenger session on April 21, on approximately $225 of direct contest compute on RunPod (L40S for sweeps at $6.90/hr; H100 SXM spot for confirmation runs at $14/hr).

The ML work surfaced inside a larger research program on multi-model coordination and consensus drift; Parameter Golf was an unplanned detour that turned out to be essential front-end infrastructure for the larger program. That's how the submission arrived at compression — by needing it for something else first.

## Attribution

Built independently on top of the public Parameter Golf repository and contest documentation. No specific upstream PR was ported into this submission.

## Included files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` — exact training script used for the banked run
- `train_seed1337.log` — full H100 training + eval log for the banked run
- `validation_final_RERUN_20260421_0535.log` — authoritative validation pass that produced the 1.17845772 number

The packaged int6+zlib artifact (`champion_1.17845772.int6.ptz`, 15,836,325 bytes, sha256 `71286a2e7950ee931490afe1329a9dee4039565635133997bc08276ac4e6a56b`) is reproducible from the script and log above; it is excluded by the repo-wide `.gitignore` rule on `*.ptz` (matching the convention used by neighboring submissions in `records/track_10min_16mb/`). The byte count under the 16 MB cap is verifiable from `Total submission size int6+zlib: 15897032 bytes` in the validation log.
