# Record: SP8192 + Strict Full-Val Byte PPM Mixture

**val_bpb = 1.0049** (3-seed mean, std 0.0007) | **~15.995 MB** | 8xH100 SXM

This submission starts from the merged 2026-04-09 SP8192 + 3-layer recurrence + parallel residuals + QK-gain 5.25 base stack, removes eval-time TTT from the packed script, and adds a strict full-validation byte-level PPM mixture in `eval_val_sliding`.

## 3-Seed Results

| Seed | Post-EMA BPB | **PPM BPB** | Artifact |
|------|-------------:|------------:|---------:|
| 42   | 1.0871 | **1.0049** | 15,997,433 |
| 7    | 1.0875 | **1.0057** | 15,995,226 |
| 1337 | 1.0863 | **1.0043** | 15,993,603 |
| **Mean** | **1.0869** | **1.0049** | **15,995,421** |
| **Std** | | **0.0007** | |

Compared to the 2026-04-09 base record's legal TTT mean of 1.0810 BPB, this strict PPM mixture improves by **0.0761 BPB**. The plain NN scores reported by the scorer (`nn_token_bpb`) remain around 1.0795-1.0812; the gain is from the online byte PPM mixture.

## Key Techniques

1. **SP8192 base stack** - inherits the merged SP8192 + GPTQ SDClip + 3-layer recurrence + parallel residuals + QK-gain 5.25 architecture and training recipe.
2. **Strict full-val byte PPM** - reconstructs the byte stream from already-scored target tokens and SentencePiece byte LUTs, then scores every byte with a prefix-only PPM model.
3. **Prefix-only binary gate** - chooses the NN/PPM mixture lambda from context confidence before observing the current byte, avoiding target-conditioned gate selection.
4. **Score-before-update byte order** - every byte is scored from previous bytes only, then inserted into the PPM tables for future bytes.
5. **Native C scorer** - runtime-compiled open-addressed context tables, rolling context keys, inline byte counts, fixed order-0 counts, cached integer logs, and precomputed lambda logs.
6. **Compact sliding collection** - per-rank raw token/NLL files in `/tmp`; rank 0 gathers and runs the strict sequential PPM scorer. No full-length GPU position buffers or large NCCL all-reduces.
7. **Eval-time controls for budget** - `SKIP_QUANTIZED_EVAL=1`, `SLIDING_BATCH_SEQS=32`, and `PPM_LOG_CACHE_SIZE=1048576` keep full eval under 600s.

## Architecture And Training

The neural base is unchanged from the 2026-04-09 SP8192 record: 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, partial RoPE, layerwise LN scale, tied embeddings, logit softcap=30, depth recurrence over layers 3-5, and parallel residuals from layer 7.

Training uses the inherited MuonEq-R/AdamW recipe, EMA 0.9965, WD 0.095, matrix LR 0.022, warmdown 0.72, and a 588s effective train cap (`MAX_WALLCLOCK_SECONDS=600`, `GPTQ_RESERVE_SECONDS=12`).

## Quantization

Full-Hessian GPTQ with SDClip is unchanged from the base stack: int6 attention/MLP matrices, int8 token embeddings, float16 scalar/gating parameters, byte-shuffle, and Brotli-11 compression. The packed script is trimmed to 21.4KB by removing TTT and the Python PPM reference from the final artifact.

## Strict PPM Evaluation

For each scored target token, the scorer spreads the NN token NLL uniformly over that token's emitted bytes. If a SentencePiece leading-space marker should contribute an actual space byte, that byte is scored first. For every byte:

1. Build context keys from previous bytes only.
2. Score the byte with PPM-D style escape probabilities.
3. Compute context confidence as `max_count / (total + unique)` at the deepest available prefix context.
4. Use `lambda_lo` if confidence is high, otherwise `lambda_hi`.
5. Mix normalized NN byte probability and PPM byte probability.
6. Update byte counts after scoring.

Default parameters used for the record logs:

| Env var | Value |
|---|---:|
| `PPM_ORDER` | `4` |
| `PPM_LAMBDA_HI` | `0.9` |
| `PPM_LAMBDA_LO` | `0.05` |
| `PPM_CONF_THRESHOLD` | `0.9` |
| `PPM_LOG_CACHE_SIZE` | `1048576` |
| `SKIP_QUANTIZED_EVAL` | `1` |
| `SLIDING_BATCH_SEQS` | `32` |

## Compliance

Per Issue #1017-style eval-time constraints:

- **Causality:** The neural model is evaluated by causal sliding windows. The byte PPM table only contains previous bytes at the time each byte is scored.
- **Score before update:** PPM counts are updated only after the current byte's mixed log-probability is recorded.
- **Full validation:** Formal logs use all 40,540,160 scored target tokens / 151,078,222 bytes. Debug subsets are non-scoring.
- **Single scoring path:** The returned `quantized_sliding_window` BPB is the PPM mixture score for the full stream; there is no post-hoc best-of selection.
- **No SLOT.**
- **No TTT in the packed artifact.**
- **No pre-quant validation adaptation.**
- **No ETLB/logit bias.**
- **No n-gram cache or precomputed validation cache.**
- **Artifact under 16,000,000 bytes on all three seeds.**
- **Training under 600s on all three seeds.**
- **PPM eval under 600s on all three seeds.**

Review note: this is a byte-level online mixture scoring object rather than a pure token-level NN score. The logs report `nn_token_bpb`, `nn_byte_bpb`, `ppm_only`, and `mix_bpb` for auditability.

## Reproduction

```bash
python3 -m pip install brotli sentencepiece
# install the same flash-attention package used by the base SP8192 records if missing
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

RUN_ID=strict_ppm_trim_seed42_8gpu_order4_b32 \
SEED=42 \
PPM_ENABLED=1 \
PPM_NATIVE_ENABLED=1 \
PPM_ORDER=4 \
PPM_LAMBDA_HI=0.9 \
PPM_LAMBDA_LO=0.05 \
PPM_CONF_THRESHOLD=0.9 \
PPM_LOG_CACHE_SIZE=1048576 \
SKIP_QUANTIZED_EVAL=1 \
SLIDING_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-26_SP8192_StrictFullValPPM/train_gpt.py
```

Change `SEED` and `RUN_ID` for seeds 7 and 1337.

## Credits

- Base stack: merged 2026-04-09 SP8192 + 3-layer recurrence + parallel residuals + QK-gain 5.25 + legal TTT record and credited lineage.
- PPM idea lineage: PR #1835 / #1795 discussion. This version changes the implementation to strict full-val scoring, prefix-only gating, native sequential scoring, and no subset claim.

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed7.log`
- `train_seed1337.log`
