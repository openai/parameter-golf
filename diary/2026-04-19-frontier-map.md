# Frontier Map — as of 2026-04-19

Snapshot of the Record-track frontier on `openai/parameter-golf`. All PRs below are **Record** track (not Non-record). Scope: last ~2 weeks of activity.

**Legend:**
- ✅ MERGED — accepted, in main
- 🟢 CLEAN — open, no compliance concerns raised
- 🟡 DISPUTED — open, pending maintainer ruling
- 🔴 BROKEN — self-withdrawn, bug confirmed, or canonical bpb > SOTA
- ⛔ BANNED — uses explicitly ruled-illegal mechanism

---

## Timeline view (dated)

### 2026-04-09 — ✅ merged SOTA 1.0810
- **#1493** bigbag — SP8192 + 3L Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT — **1.0810** ✅

### 2026-04-10 to 04-11 — first post-SOTA challengers
- **#1523** superseded (closed) — Triple Recurrence + Banking + Fused MLP — 1.0778
- **#1529** msisovic — Improved Parallel Residuals — 1.0758 🟢
- **#1530** samacqua — VarLen attn + Fused MLP + doc-indep TTT — **1.07336** 🟢 ← **FOUNDATION A**

### 2026-04-12 to 04-14 — multi-phase TTT emerges
- **#1557** ndokutovich — N-gram Tilt + Hessian SDClip — 1.0773 🟡 (self-flagged AT-RISK)
- **#1578** mikeapedia — Lossy Casefold Tokenizer — 1.0668 🟡 (tokenizer dispute, lossy variant)
- **#1610** romeerp — VarLenAttn + PhasingTTT — 1.0728 🟢
- **#1626** samacqua — VarLen + Fused MLP + Multi-Phase Global SGD TTT — **1.07193** 🟢 ← **FOUNDATION B**

### 2026-04-16 — gates
- **#1667** MarioPaerle — SmearGate + AttnOutGate — 1.07139 🟢 (untested, 0 audits)

### 2026-04-17
- **#1687** resouer — K_KVShare_Wider FLA 🔴 CLOSED (parent of GDN cluster, same byte bug)
- **#1693** dexhunter — Casefold V4 + AttnOutGate + MP Global SGD TTT — 1.05733 🟡 (tokenizer dispute)
- **#1695** X-Abhishek-X — SpinQuant V1 + MP-SGD-TTT — 1.0759 🟢
- **#1696** kings-crown — Block Attn Residuals + Tuned Legal TTT — 1.1224 🟢 (above SOTA)
- **#1698** arsenis-cmd — GatedDeltaNet + Legal Score-First TTT — 1.00995 🔴 (byte bug + artifact oversize)
- **#1700** jorge-asenjo — SP8192 + MP-SGD + Phased TTT — 1.07219 🟢
- **#1705** genji0306 — K_KVShare_Wider FLA 🔴 CLOSED (GDN family)

### 2026-04-18
- **#1711** aamodbhatt — GatedDeltaNet + Score-First TTT + Brotli — 1.00980 🔴 CLOSED (byte bug, self-withdrawn)
- **#1712** aamodbhatt — GatedDeltaNet + Brotli (No TTT) — 1.01902 🔴 CLOSED (byte bug, self-withdrawn)
- **#1715** G3sparky — QK-Gain 5.5 — 1.0810 🟢 (ties SOTA, no beat)
- **#1716** himanshudongre — SP8192 + BigramHash d=32 + Path A v3 — 1.07882 🟡
- **#1722** deborahnelson — Trinity SLOT v3 + Pre-Quant TTT — 0.65802 ⛔ (SLOT + pre-quant stack)
- **#1723** SlavH — Nairi 9L 512D vocab1024 — 0.5384 🔴 (invalid eval, target clamping)
- **#1727** yahya010 — MP-SGD TTT 4 phases + QK-Gain 5.25 — **1.07217** 🟢 (config diff on #1700)

### 2026-04-19 — big day
- **#1729** romeerp — Lossless CaseOps + Tapered WD — **1.06780** 🟡 (tokenizer dispute, cleanest variant)
- **#1731** Victory963 — Hadamard+AWQ+Layerwise 🔴 CLOSED (replaced by #1732)
- **#1732** Victory963 — Hadamard+AWQ+Layerwise+Hessian — 1.0785 🟢 (untested, kitchen sink)
- **#1734** yahya010 — GatedDeltaNet + Legal TTT + Brotli-11 — 1.01080 🔴 CLOSED (byte bug, self-withdrawn)
- **#1735** AjAnubolu — SP8192 + Parallel Pre-Quant TTT — **1.0429** 🟡 (pre-quant TTT dispute)
- **#1736** dexhunter — SP8192 + CaseOps + GatedAttn + QuantGate + Loop45 + PhasedTTT — **1.06549** 🟡 (tokenizer dispute)
- **#1738** alertcat — #1735 + CaseOps V15 — **1.03540** 🟡🟡 (tokenizer + pre-quant TTT, double dispute)

### 2026-04-19 (today)
- See above — big day: #1729, #1732, #1735, #1736, #1738 (5 frontier PRs in one day)

---

## Family map (by mechanism)

### 🟢 CLEAN frontier chain — everything merges-ready

```
                #1493 ✅ 1.0810 (merged SOTA)
                       │
         ┌─────────────┴─────────────┐
         │                           │
     FOUNDATION A                FOUNDATION B
     #1530 🟢 1.07336            #1626 🟢 1.07193
     VarLen + doc-indep TTT       VarLen + MP Global SGD TTT
         │                           │
         ├─ #1610 🟢 1.0728          ├─ #1700 🟢 1.07219
         │  (+ phased global SGD)    │  (+ SP8192 + 3 phases)
         │                           │   │
         │                           │   └─ #1727 🟢 1.07217
         │                           │      (phases=4, config-only)
         │                           │
         │                           └─ (#1670 Casefold → #1693)
         │
         └─ #1667 🟢 1.07139 (SmearGate + AttnOutGate, standalone, no audits)

    Independent quantization branch:
    #1529/#1445 → #1695 🟢 1.0759 (SpinQuant V1 — Hadamard rotation before GPTQ)
```

**Clean floor today: ~1.071.** Achievable with no disputed rulings.

### 🟡 DISPUTED — tokenizer family (waiting on Issue #1604)

**The axis of the debate is lossy vs. lossless**, not "any normalization vs. none." Two sub-families:

**Lossy casefold sub-family** — irreversible `.lower()` after NFKC; decoded token IDs cannot reproduce original capitalization:
```
#1578 mikeapedia 04-13  Lossy Casefold             1.0668  🟡 likely ILLEGAL
#1670 (parent, Casefold V4 base)                           🟡 likely ILLEGAL
#1693 dexhunter  04-17  Casefold V4 + gates        1.0573  🟡 likely ILLEGAL (inherits lossy)
```

**Lossless CaseOps sub-family** — bijective transform with `TITLE`/`ALLCAPS`/`CAPNEXT`/`ESC` operator tokens; `decode(encode(s)) == s`; BPB denominator against original UTF-8 via byte sidecar:
```
#1729 romeerp    04-19  Lossless CaseOps + WD      1.0678  🟡 likely LEGAL
#1736 dexhunter  04-19  Lossless CaseOps + gates   1.0655  🟡 likely LEGAL
```

**Issue #1604 thread analysis:**

| Commenter | Date | Position | Targets |
|---|---|---|---|
| SPThole | 04-14 | concern about generative usability under case-fold | (general) |
| sharpobject | 04-14 | **"exact bytes of validation documents must be reproducible by decoding the token IDs"** — this is the clean principle | (proposed standard) |
| tejasnaladala | 04-16 | formal argument: lossy casefold breaks BPB semantics because `byte_count` is no longer of original bytes | #1670, #1585, #1578 (all lossy) |
| mikeapedia | 04-16 | counter: NFKC is already lossy, every submission counts post-NFKC bytes | (defense of lossy) |
| andrewbaggio1 | 04-17 | "casefold should be illegal, opens door to removing spaces" | (against lossy) |

**Crucial asymmetry:** every argument against specifically targets lossy casefold. Nobody in the thread has challenged CaseOps (lossless + original-UTF-8 byte counting) on its bijective property. The sharpobject principle ("exact bytes reproducible from token IDs") is the likely synthesis — CaseOps satisfies it, casefold doesn't.

**Expected ruling:**

| Family | Expected ruling | Survivors |
|---|---|---|
| Lossy casefold | illegal | — (kills #1578, #1670, #1693) |
| Lossless CaseOps | legal | #1729 (1.0678), #1736 (1.0655) |

**If this plays out:** floor drops to ~1.065 (#1736) with high confidence. Dexhunter's architectural add-ons (GatedAttn, QuantGate, Loop45) are preserved.

### 🟡 DISPUTED — pre-quant TTT (waiting on Issue #1017 Condition 3) — **~85-90% dead on physics**

Uses 21 epochs of AdamW on val data *before* artifact freeze.

```
#1735 AjAnubolu  04-19  SP8192 + Parallel Pre-Quant TTT      1.0429
#1738 alertcat   04-19  #1735 + CaseOps V15 (inherits both)  1.0354  🟡🟡 double dispute
```

**The physics argument (stronger than the textual one):**

From bigbag's own Issue #1017: **"Corpus-level TTT has a ceiling of approximately 0.0003 bits."** (Verified — direct quote from the issue body.)

#1735 claims **−0.038 bpb** from pre-quant TTT alone. That's **~100× the ceiling** the merged-SOTA author put on paper. The FineWeb train/val splits are random samples from the same source distribution with "negligible divergence across every measure" — there is no distributional ground for TTT to make up.

Either:
(a) bigbag's ceiling analysis is wrong — unlikely; he's the authority and has measured it empirically
(b) #1735 is extracting signal from somewhere it shouldn't — i.e. the val stream itself

"Frozen before eval" is a textual defense. The ceiling is a physical one. **Physical usually wins rule-interpretation disputes.**

**Estimated probability of ruling against: 85–90%.** Regardless of whether the textual ruling on Condition 3 lands strict or permissive, a headline number 100× over the stated corpus-level ceiling will not be allowed to sit as SOTA.

**Legal-regardless artifact:** the **8-GPU federated-averaging** systems trick itself is a pure parallelization idea — it speeds up any multi-epoch TTT loop without affecting legality. Worth extracting even after this PR family dies.

### 🔴 BROKEN — GatedDeltaNet (GDN) cluster

All four share the same `build_sentencepiece_luts` byte-counting bug that inflates denominator by ~17.46% → real canonical bpb is ~1.19, **worse than SOTA**.

```
#1687 resouer       K_KVShare_Wider FLA                    🔴 CLOSED (parent arch)
#1698 arsenis-cmd   GDN + Legal Score-First TTT   1.00995  🔴 OPEN, byte bug + artifact oversize
#1705 genji0306     K_KVShare_Wider FLA           1.0339   🔴 CLOSED
#1711 aamodbhatt    GDN + Score-First TTT         1.00980  🔴 CLOSED by author
#1712 aamodbhatt    GDN + Brotli (No TTT)         1.01902  🔴 CLOSED by author
#1734 yahya010      GDN + Legal TTT + Brotli-11   1.01080  🔴 CLOSED by author ("canonical would not beat SOTA")
```

**Authors of 3 of 4 PRs closed their own submissions** after byte-bug audit.

**Residual interest:** the K_KVShare_Wider architecture itself (10L 544d, KV-share stride=2) might be real but we'd need a clean re-implementation with canonical byte accounting to know. Low priority.

### ⛔ BANNED — Trinity / SLOT / N-gram family

Stacking banned mechanisms. Ignore.

```
#1246 deborahnelson  Trinity v7+skip                 0.22311  ⛔ (N-gram Order-22 + SLOT + Pre-Quant)
#1722 deborahnelson  Trinity SLOT v3 + Pre-Quant    0.65802  ⛔
#1557 ndokutovich    N-gram Tilt + SDClip           1.0773   🟡 (self-flagged AT-RISK, N-gram family)
```

Also-banned mechanisms ruled out earlier:
- N-gram caches with target-in-key (PR #779 ruling, 2026-03-27)
- SLOT per-window eval-time AdamW (PR #1376 ruling)

### 🔴 OTHER INVALID

```
#1723 SlavH   Nairi 9L 512D vocab1024   0.5384  🔴 (model vocab 1024 vs tokenizer 8192 → target clamping, not a valid BPB)
```

---

## At-a-glance rankings (filter by status)

| Rank | Status | bpb | PR | What |
|---|---|---|---|---|
| 1 | 🟡🟡 | 1.0354 | #1738 | CaseOps + pre-quant TTT (double dispute) |
| 2 | 🟡 | 1.0429 | #1735 | pre-quant TTT (disputed) |
| 3 | 🟡 | 1.0573 | #1693 | Casefold + gates (tokenizer disputed) |
| 4 | 🟡 | 1.0655 | #1736 | CaseOps + gates (tokenizer disputed) |
| 5 | 🟡 | 1.0668 | #1578 | lossy Casefold (tokenizer disputed) |
| 6 | 🟡 | 1.0678 | #1729 | lossless CaseOps (tokenizer disputed, cleanest) |
| 7 | 🟢 | 1.0714 | #1667 | SmearGate + AttnOutGate (untested) |
| 8 | 🟢 | 1.0719 | #1626 | VarLen + MP-SGD TTT ← **FOUNDATION** |
| 9 | 🟢 | 1.0722 | #1727 | phases=4 on #1700 |
| 10 | 🟢 | 1.0722 | #1700 | SP8192 + MP-SGD Phased TTT |
| 11 | 🟢 | 1.0728 | #1610 | VarLen + PhasingTTT |
| 12 | 🟢 | 1.0734 | #1530 | VarLen + doc-indep TTT ← **FOUNDATION** |
| 13 | 🟢 | 1.0759 | #1695 | SpinQuant V1 (Hadamard) + MP-SGD |
| 14 | 🟢 | 1.0785 | #1732 | Hadamard + AWQ kitchen sink |
| — | ✅ | 1.0810 | #1493 | **merged SOTA** |

## Key rulings to watch

| Issue | Affects | Expected outcome | Stake |
|---|---|---|---|
| **#1604 tokenizer** | lossy casefold: #1578/#1670/#1693 | illegal (high confidence) | no frontier loss — lossless CaseOps survives |
| **#1604 tokenizer** | lossless CaseOps: #1729, #1736 | legal (high confidence) | floor drops to 1.0655 |
| **#1017 C3 pre-quant TTT** | #1735, #1738 | illegal (85-90%, ceiling-based) | no real floor change — these were never real |

**Expected map post-rulings:**
- True legal floor: **~1.0655** (#1736) via lossless CaseOps stack
- Uncontested-legal floor if we're conservative: ~1.0714 (#1667) or ~1.072 (#1530/#1626 foundations)
- The −0.035 "pre-quant TTT" tier is fiction; treat it as non-existent

## Pace
- **Clean frontier drift:** 1.0810 → 1.0714 in 10 days (−0.0087 bpb / ~−0.0009/day)
- **Disputed frontier (if ruled legal):** 1.0810 → 1.0354 in 10 days (−0.046 bpb)
- **New PRs/day** on record track: ~2–3 (accelerating — 5 frontier PRs on 04-19 alone)
- **Deadline: 2026-04-30.** 11 days left.
