# Record: SP4096 + Byte-Level PPM Adaptive-λ Mixture (strict-legal gate) — val_bpb 1.01252

**val_bpb: 1.01252** (3-seed mean, std=0.00044, full FineWeb val, **strict-legal outcome-independent gate**)

| Seed | NN_full (sliding, token-BPB, full val) | Mix BPB (byte-level, full val) | Δ | Artifact | Eval |
|-|-|-|-|-|-|
| 42   | 1.09740 | **1.01228** | −0.07436 | 15,953,442 | 521s |
| 1337 | 1.09823 | **1.01303** | −0.07443 | 15,921,608 | 506s |
| 2025 | 1.09728 | **1.01226** | −0.07426 | 15,924,697 | 485s |
| **Mean** | **1.09764** | **1.01252** | **−0.07435** | 15,933,249 | 504s |

Beats current record **1.06453** (PR #1769) by **0.05201** BPB — t-stat ≈ 107 on the 0.005-nat bar.

Our NN-only mean **1.09764 matches @clarkkev's 2026-04-01 record of 1.09785** within seed noise. The entire NN stack is unchanged from PR #1334 / the 2026-04-01 record; the gain comes from a byte-level PPM adaptive-λ mixture applied at eval time.

## This PR supersedes an earlier (now-invalidated) attempt

The earlier version of this submission (on branch `record-sp4096-ppm-adaptive-mix`, PR #1795 at commit `07d20c3`, claiming val_bpb 0.95165) used a **target-conditioned gate** — `cf[i] = P_PPM(observed_byte)` — which made the reported score depend on the realized byte value. This was correctly flagged by @nprime06 in the PR comments as not a valid scoring rule, and that number is retracted.

The revised gate in this version is **strictly a function of the prefix and PPM state, frozen before the observed byte is looked up**. See the next section.

## The mixture, and why the gate is now outcome-independent

The scoring model is a byte-level two-predictor mixture:

`q_mix(b) = λ·q_NN_byte(b) + (1−λ)·q_PPM_byte(b)`

where:

- **`q_NN_byte`** — NN's SentencePiece-token distribution, spread uniformly across UTF-8 bytes of each token. Conserves total NN bits (byte-BPB of NN alone equals token-BPB scaled by bytes/token).
- **`q_PPM_byte`** — byte-level PPM-D order 4 predictor. Builds its suffix-count table online from val bytes the NN has already graded in the same sliding pass. Zero precomputed state ships in the 16MB artifact.
- **`λ`** (the gate) — adaptive: `λ = 0.05 if cf > 0.9 else 0.9`, where `cf = max_count / total` at the **deepest context with any data**, computed from the PPM state and the prefix **before any lookup of the observed byte**.

The key code:

```python
cf_mx = 0; cf_tot = 256; cf_seen = False
for o in range(lim, -1, -1):
    k = h[-o:] if o else b""        # context key: prefix only
    e = tabs[o].get(k)               # lookup: prefix only
    if e is None: continue
    if not cf_seen:                  # first context found = deepest with data
        cf_mx = e[1]                 # max_count, frozen HERE
        cf_tot = e[0]                # total, frozen HERE
        cf_seen = True               # — BEFORE any d.get(x) below
    tot = e[0]; d = e[2]
    c = d.get(x, 0)                  # now uses x — but cf already frozen
    if c > 0:
        pf = esc * (2*c - 1) / (2*tot); break
    esc *= len(d) / (2*tot)
cf[i] = (cf_mx / cf_tot) if cf_seen else 1/256
```

**Formal property:** for any two possible next-bytes `x_a`, `x_b` at the same position (same prefix `h`, same PPM state `tabs`), `cf[i]` is bitwise identical between the two cases. Therefore `λ[i] = np.where(cf > T, L_, H)` is identical. Only `q_NN(x)` and `q_PPM(x)` depend on `x` — which is correct for predictor scores.

This answers @nprime06's specific concern on PR #1795 mechanically, not rhetorically.

## What changed vs @clarkkev 2026-04-01

Source-level diff: one new function (`_ppm_mixture_bpb`, ~55 lines including the strict-legal gate tracking) plus ~30 lines of gather/mix logic inside `eval_val_sliding`. Everything else is unchanged from the 2026-04-01 record:

- 11 layers, SP4096, MLP mult 4, depth recurrence, sliding-window eval, EMA, GPTQ int6 + brotli, LeakyReLU², parallel residuals, legal TTT framework
- Same env vars (`RUN_ID`, `SEED`), plus one gating the mixture (`PPM_MIX_ENABLED=1`)
- Same wallclock cap, same train pipeline, same GPTQ calibration

## Compliance

- **Train under 600s** ✅ all 3 seeds stopped at 590s wallclock cap (steps 5898–5901)
- **Artifact under 16 MB** ✅ 15.92–15.95 MB natively (no lzma-compressed stub needed)
- **Eval under 600s** ✅ all 3 seeds 485–521s (using PPM order 4 — order 5 was 15s over cap due to max_count tracking overhead; benchmarking showed order 4 only 0.02 BPB worse in mix)
- **No SLOT, no pre-quant TTT on val, no ETLB** ✅ inherited from base, unchanged
- **3 seeds, p ≪ 1e-10 on the 0.005-nat bar** ✅ (t-stat ≈ 107)
- **`no_ngram_cache: false`** — byte-level online PPM predictor built from empty counters during sliding eval. **Per-byte semantics: score byte_i using counters from bytes 0..i-1 (score-before-update), then add byte_i to counters for future bytes.** All PPM state is constructed from val tokens the NN has already graded, consistent with the rule text "test-time training on validation set tokens you've already evaluated your model on". **Organizer ruling explicitly requested** (see @nprime06 and @dexhunter review comments on PR #1795) on whether this class of online streaming predictor qualifies as legal score-first TTT — if ruled no, submission withdrawn.

## Reviewer concerns from PR #1795 (status)

| # | Concern | Status |
|---|---|---|
| 1 | Full-val measurement (not 5M subset) | ✅ RESOLVED — 45.5M tokens / 152.6 MB bytes |
| 2 | PPM-as-TTT class legality | ⚠️ Organizer ruling requested (category question) |
| 3 | Byte-level vs token-level BPB | ✅ BOTH logged (NN_token=1.098, NN_byte=1.087, mix=1.013) |
| 4 | NN regression vs clarkkev | ✅ RESOLVED — 1.0976 mean matches 1.0978 |
| 5 | Condition 2 framing (scoring model is a mixture) | ✅ Explicit in README above |
| **@nprime06**: target-conditioned gate | ✅ RESOLVED — strict-legal outcome-independent gate, see code |

## Reproduction

```bash
# Data prep (Kevin Clark's SP4096 dataset):
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp4096

# Training + mixture eval (per seed):
RUN_ID=<seed> SEED=<seed> PPM_MIX_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The reported val_bpb is the `final_int6_sliding_window val_bpb:` line, which equals the `[ppm_mix] ... mix=` value by construction.

## Credits

- **@clarkkev** — entire 2026-04-01 SP4096 + 11L + MLP4 + depth-recurrence + EMA + GPTQ + sliding + brotli stack (PR #1334, #1419, #1445). All of the NN contribution (1.098 BPB) is his work.
- **Cleary & Witten 1984; Moffat 1990** — PPM-D.
- **This submission** — the byte-probability-space two-predictor mixture construction with an **outcome-independent** adaptive-λ gate keyed on PPM's state-only max-count ratio.

Neither predictor alone reaches this BPB: clarkkev's NN is at 1.098, byte-PPM alone ≈2.5 on full val. The mixture at 1.013 captures the bits PPM strictly wins on (rare exact-repeat bytes — URLs, code identifiers, cross-doc duplicates) while leaving the rest to the NN. The −0.074 Δ is smaller than the retracted illegal-gate claim (−0.135) but is **mechanically defensible**: no function of the observed byte enters the gate.
