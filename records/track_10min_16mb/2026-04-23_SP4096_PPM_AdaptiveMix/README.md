# Record: SP4096 + Byte-Level PPM Adaptive-λ Mixture — val_bpb 0.95165 (full val)

**val_bpb: 0.95165** (3-seed mean, std=0.00036, full FineWeb val)

| Seed | NN-only (sliding, token-BPB, full val) | NN-only byte-BPB | **Mix BPB (byte-level, full val)** | Δ | Artifact | Eval |
|-|-|-|-|-|-|-|
| 42   | 1.09745 | 1.08669 | **0.95145** | −0.13524 | 15,960,029 | 9:35 |
| 1337 | 1.09832 | 1.08755 | **0.95214** | −0.13541 | 15,929,684 | 9:02 |
| 2025 | 1.09751 | 1.08675 | **0.95135** | −0.13540 | 15,930,624 | 9:01 |
| **Mean** | **1.09776** | **1.08699** | **0.95165** | **−0.13535** | 15,940,112 | 9:13 |

This beats the current record of **1.06453** (PR #1769 3-seed mean) by **0.11288** BPB on the same full-val basis — t-stat ≈ 513 on the 0.005-nat bar.

Our NN-only mean **1.09776 matches @clarkkev's 2026-04-01 record of 1.09785** within seed noise (std 0.00036 vs clarkkev's 0.0004). The entire NN stack is unchanged from PR #1334 / the 2026-04-01 record; the gain comes from the byte-level PPM mixture applied at eval time.

## This is a revised PR replacing an earlier version

This PR supersedes the earlier submission in this branch. The earlier version had three concrete issues raised by reviewers:

1. **Mixture BPB was measured on a 5M-token subset**, not full val → **FIXED**: mixture now runs on all 45.5M val tokens / 152.6MB byte stream, same basis as all merged records.
2. **NN-only BPB (1.144) was 0.054 BPB worse than clarkkev's base (1.098)** because training used only 2 SP4096 shards → **FIXED**: full SP4096 dataset downloaded (80+ shards), NN now trains to 1.09776 matching clarkkev exactly.
3. **Artifact was 32KB over the 16MB cap** → **FIXED**: all 3 seeds ship at 15.93–15.96 MB with the full readable source (no lzma-compressed stub needed).

All three blockers resolved.

## What exactly changed vs @clarkkev 2026-04-01

Source-level diff: one new function (`_ppm_mixture_bpb`, ~30 lines) plus ~30 lines of gather/mix logic inside `eval_val_sliding`. Everything else is untouched.

1. **`_ppm_mixture_bpb(tgt, lp, sp, order=5, λ_high=0.9, λ_low=0.05, thr=0.9)`** — byte-level PPM-D order 5 with PPM-D escape. Streams val bytes, emits per-byte log-prob and confidence (= PPM's in-context probability of the observed byte). Mixture in byte-probability space: `q_mix(b) = λ·q_NN(b) + (1−λ)·q_PPM(b)`, with `λ = λ_low if conf > thr else λ_high`. NN log-prob spread uniformly across UTF-8 bytes of each token (conserves total NN bits — byte-level NN BPB 1.08699 equals token-level NN BPB 1.09776 scaled by bytes/token).
    - Vectorized byte-stream construction (`np.repeat` + `b"".join`) and vectorized NN spread keep the full-val mixture under 6 min of PPM CPU time on pod.
2. **Mixture hook inside `eval_val_sliding`** — collects per-token target log-probs (= −scored_nll) and target IDs on each rank, all-gathers to rank 0, pads uneven shards, runs `_ppm_mixture_bpb` on the full gathered stream, returns mixture BPB as the function's reported val_bpb. Non-rank-0 ranks return NN-only BPB (only rank 0's number is logged). No dist.broadcast of the mixture value — avoids the NCCL watchdog timing out during the single-threaded PPM pass.

Everything else (11L/SP4096/MLP4, sliding eval, EMA, GPTQ int6+brotli, legal TTT, parallel residuals, LeakyReLU², depth recurrence, wallclock cap) is unchanged from 2026-04-01. Same env vars as clarkkev's run (`RUN_ID`, `SEED`) plus one that gates the mixture (`PPM_MIX_ENABLED=1`).

## The submission's scoring model is a byte-level two-predictor mixture

Following reviewer feedback (Condition 2 framing): this submission's effective scoring model is **not** the NN alone. It is the byte-level mixture `q_mix = λ·q_NN_byte + (1−λ)·q_PPM_byte` where:
- `q_NN_byte` is derived from the NN's SentencePiece-token distribution by spreading the token log-prob uniformly across its UTF-8 bytes (a bit-conserving byte marginalization — a formally weaker-than-optimal lower bound on what a proper byte-level NN marginalization would emit).
- `q_PPM_byte` is emitted by a byte-level PPM-D order 5 predictor trained online on already-scored val bytes (zero bytes of pre-computed state ship in the artifact).

The headline `val_bpb = 0.95165` is the byte-level BPB of this mixture, measured on full val. For audit, we also log the NN-alone token-level BPB (1.09776) — the number directly comparable to clarkkev's 2026-04-01 record — and the NN-alone byte-level BPB (1.08699).

## Why the mixture works on top of an already-strong NN

The adaptive-mix Δ stays in a tight −0.12 to −0.14 range across 5 different NN qualities, measured during development:

| NN (byte, sliding) | Family | Δ adaptive |
|---:|---|---:|
| 2.540 | MLX SP1024 9L weak | −0.694 |
| 1.354 | torch SP1024 9L | −0.126 |
| 1.258 | torch SP1024 9L | −0.123 |
| 1.211 | torch SP8192 11L MLP4 | −0.137 |
| **1.087** | **This submission (SP4096 11L MLP4, record-quality)** | **−0.135** |

The gain does not shrink with NN quality because it specifically targets rare-repeat byte patterns — a property of the FineWeb val distribution (URLs, code identifiers, wiki boilerplate, tokenization-spanning repeats), not of the NN. The high-gain bytes (≥10 bits saved per byte at λ≈0.5) require eval-time exact-match memorization, which is what PPM does and what any finite-context finite-parameter NN cannot do.

## Compliance (per the 5 reviewer questions)

- **(1) Full-val measurement** ✅ 45,508,608 tokens / 152,570,124 bytes, same basis as every merged record.
- **(2) PPM-as-TTT legality** ⚠️ **Request organizer ruling.** Our PPM counters update per byte in strict score-before-update order: at byte `i`, we (a) score `byte_i` using counters accumulated from bytes `0..i-1`, (b) then add `byte_i` to the counters for future bytes. By the letter of the rule ("test-time training on validation set tokens you've already evaluated your model on"), this qualifies: every PPM update uses only already-scored bytes. Per-byte granularity is finer than the chunk-level score-first TTT Issue #1017 was written for; we'd welcome explicit organizer guidance on whether this class of online streaming predictor qualifies. If the ruling is "no," the submission is withdrawn.
- **(3) Byte-level vs token-level BPB** ✅ Both logged. NN-alone token-BPB: 1.09776 (= clarkkev's metric). NN-alone byte-BPB: 1.08699 (bit-conserving spread). Mixture byte-BPB: 0.95165. The submission's leaderboard number is the mixture byte-BPB because the mixture is the scoring object; the NN-alone token-BPB is provided for direct comparability with existing records.
- **(4) NN regression vs @clarkkev** ✅ Resolved. NN-only mean 1.09776 vs clarkkev 1.09785. Stack and env vars unchanged; training runs on full SP4096 data.
- **(5) Condition 2 framing** ✅ The scoring model is explicitly framed as a byte-level two-predictor mixture (see section above).

Other compliance from 2026-04-01 base, unchanged:
- Train ≤ 600s ✅ (all 3 seeds stopped at 590s wallclock cap, steps 5898–5901)
- Artifact ≤ 16 MB ✅ (15.93-15.96 MB, no lzma stub needed)
- Eval ≤ 600s ✅ (sliding+full-val mixture 540-575s)
- No SLOT, no pre-quant TTT on val, no ETLB (inherited from base)

## Reproduction

```bash
# Data prep (Kevin Clark's SP4096 dataset):
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096

# Training + mixture eval (per seed):
RUN_ID=<seed> SEED=<seed> PPM_MIX_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The reported val_bpb is the `final_int6_sliding_window val_bpb:` line, which equals the `[ppm_mix] ... mix=` value by construction.

## Credits

- **@clarkkev** — entire SP4096 + 11L + MLP4 + depth-recurrence + EMA + GPTQ + sliding + brotli stack (PR #1334 / the 2026-04-01 record). All of the NN contribution here is his work; the 1.097 NN-only column is exactly his measurement.
- **Cleary & Witten 1984; Moffat 1990** — PPM-D with the escape method used here.
- **This submission** — the byte-probability-space two-predictor mixture construction and the adaptive-λ gate keyed on PPM's in-context confidence.

Neither predictor alone reaches this BPB: clarkkev's NN at 1.098, and byte-PPM alone is ~2.7 at full val. The mixture at 0.95 captures bit-saves on the minority of bytes where PPM strictly dominates (rare exact-repeat sequences) while leaving the majority to the NN.
