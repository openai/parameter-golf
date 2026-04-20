# Frontier Scan — 2026-04-19

Scanning all Record-track PRs on `openai/parameter-golf` updated in the last 3 days (04-17 → 04-19) to decide whether to rebase spec-000 baseline.

**Current merged SOTA:** 1.0810 (PR #1493, 2026-04-09, 11 days stale).

## Candidate list (27 PRs)

Sorted by claimed bpb. Legend: ★ = open, ✗ = closed, ? = state unclear.

| bpb | PR | State | Author | Title |
|---|---|---|---|---|
| 0.22311 | #1246 | ★ | deborahnelson | Trinity v7+skip |
| 0.5384 | #1723 | ★ | SlavH | Nairi 9L 512D vocab1024 |
| 0.65802 | #1722 | ★ | deborahnelson | Trinity SLOT v3 + Pre-Quant TTT |
| 1.00980 | #1711 | ✗ | aamodbhatt | GatedDeltaNet FLA + Score-First TTT + Brotli |
| 1.00995 | #1698 | ★ | arsenis-cmd | GatedDeltaNet (FLA) + Legal Score-First TTT |
| 1.01080 | #1734 | ✗ | yahya010 | GatedDeltaNet + Legal TTT + Brotli-11 |
| 1.01902 | #1712 | ✗ | aamodbhatt | GatedDeltaNet FLA + Brotli (No TTT) |
| 1.03540 | #1738 | ★ | alertcat | PR#1735 + CaseOps V15 |
| 1.0339 | #1705 | ✗ | genji0306 | K_KVShare_Wider FLA |
| 1.04090 | #1687 | ✗ | resouer | K_KVShare_Wider full-recipe FLA |
| 1.0429 | #1735 | ★ | AjAnubolu | SP8192 + Parallel Pre-Quant TTT |
| 1.05733 | #1693 | ★ | dexhunter | Casefold V4 + AttnOutGate + Multi-Phase Global SGD TTT |
| 1.06549 | #1736 | ★ | dexhunter | SP8192 + CaseOps + GatedAttn + QuantGate + Loop45 + PhasedTTT |
| 1.0668 | #1578 | ★ | mikeapedia | Custom Casefold Tokenizer (older) |
| 1.0678 | #1729 | ★ | romeerp | CaseOps Tokenizer + Tapered WD |
| 1.07139 | #1667 | ★ | MarioPaerle | SmearGate + Attn Output Gate + Legal TTT |
| 1.07217 | #1727 | ★ | yahya010 | SP8192 MP-SGD TTT 4 phases + QK-Gain 5.25 |
| 1.0759 | #1695 | ★ | X-Abhishek-X | Stage 3 + SpinQuant V1 + MP-SGD-TTT |
| 1.07882 | #1716 | ★ | himanshudongre | SP8192 + BigramHash d=32 + Path A v3 |
| 1.0810 | #1715 | ★ | G3sparky | QK-Gain 5.5 |
| 1.1161 | #1494 | ★ | G3sparky | XSA-all + GPTQ + FA3 dtype fix |
| 1.12242 | #1696 | ★ | kings-crown | Block Attn Residuals + Tuned Legal TTT |

## Per-PR Notes

Notes added below as I work through each one.

---

### GDN cluster (#1698, #1711, #1712, #1734) — **BROKEN, not a frontier**

All four PRs use Gated DeltaNet (linear attention via the FLA library) on top of @resouer's `K_KVShare_Wider` architecture (PR #1687, also closed). Pre-TTT ~1.019, post-TTT ~1.010. Claimed to beat SOTA by 0.07 bpb.

**The bug (byte-counting, reported by dexhunter, empirically verified by yahya010):**

`build_sentencepiece_luts` pre-credits +1 byte to every `▁`-prefixed token:
```python
base_bytes[i] = len(piece[1:].encode("utf-8")) + 1
```
Then the eval accumulator adds `+1` *again* for the same leading-space condition:
```python
tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev])
```

**Effect:** leading-space byte is double-counted → byte denominator is inflated by ~17.46% vs canonical `sp.decode_ids(...).encode('utf-8')` → reported bpb is `1/1.177` of truth.

**Canonical-corrected numbers (yahya010's 1M-token empirical check):**
- #1711 claimed 1.00980 → canonical ~**1.188**
- #1712 claimed 1.01902 → canonical ~**1.199**
- #1734 claimed 1.01080 → canonical ~**1.189**
- #1698 claimed 1.00995 → canonical ~**1.188**

All worse than the 1.0810 merged SOTA.

**Closures:**
- #1711, #1712 closed by author aamodbhatt ("same byte-counting bug")
- #1734 closed by author yahya010 with explicit "canonical would not beat the merged-SOTA threshold"
- #1687 (the parent K_KVShare_Wider arch) closed
- #1698 (arsenis-cmd) still open but has **two** blocking issues: artifact oversize (16.47–16.60 MB vs 16.00 MB cap) AND the byte bug; author hasn't responded to the byte bug yet

**Interesting artifact:** the K_KVShare_Wider arch itself (10L 544d 8h, KV-share stride=2) may still be a real idea, but we can't tell because every reported number in the cluster used the buggy LUT. Needs clean re-implementation to validate.

**Canonical reference LUT:** merged PR #1019 — `base_bytes[i] = len(piece.encode("utf-8"))` (no pre-credit).

**Conclusion:** GDN is not the frontier. Skip.

---

### #1735 AjAnubolu — SP8192 + Parallel Pre-Quant TTT (1.0429, 3-seed, open) — **DISPUTED**

**Claim:** −0.038 BPB vs merged SOTA. Two innovations:
1. **8-GPU parallel pre-quant AdamW TTT** via federated averaging — each rank runs 1/8 of val chunks, `all_reduce(AVG)` syncs weights after every epoch. 21 epochs in 377s.
2. **Epoch-level cosine LR** (prior TTT was per-chunk cosine that reset each epoch). Ablation: 9-epoch per-chunk 1.0663 → 9-epoch epoch-level 1.0558 → 21-epoch epoch-level 1.0327.

**The dispute (dexhunter comment):** Pre-quant TTT runs 21 full AdamW epochs over the **full val stream** with `loss.backward()` + `optimizer.step()` before BPB is scored on **those same tokens**. This is adapt-then-score on val, which Issue #1017 Condition 3 prohibits. README forbids training-time use of val data. Merged score-first TTT precedent (PR #549) keeps adapter updates strictly *after* score.

**Author's defense (alertcat, via #1738):** "The model weights are frozen AFTER TTT ends and BEFORE GPTQ quantization; the final artifact is a fixed predictor — once the int6 model is serialized, no further adaptation occurs during eval." Argues pre-quant TTT is training-phase, not eval-phase.

**Maintainer ruling:** NOT YET ISSUED. This is the single biggest open interpretive question. If ruled illegal, the entire #1735 → #1738 branch collapses.

**Interesting even if ruled illegal:** The 8-GPU federated-averaging parallelization trick itself is a legitimate systems improvement — it would work for *any* TTT flavor that involves multiple epochs over independent chunks. Worth extracting for our own TTT.

---

### #1738 alertcat — PR #1735 + CaseOps V15 (1.0354, 3-seed, open) — **DISPUTED (inherits #1735)**

**Claim:** Stacks CaseOps tokenizer (PR #1729) on top of #1735's pre-quant TTT stack. Non-trivial integration work (added byte sidecar plumbing to three eval functions). Artifact 15.996 MB — 3.8 KB under cap.

**Novelty:** Zero — it's literally "#1735 + #1729 together for the first time."

**Risk:** Inherits the pre-quant TTT dispute from #1735. Also inherits whatever tokenizer ruling applies to CaseOps (parent #1729).

**Note:** Borderline statistical significance — 0.0003 BPB above the 0.0072 record threshold. Author acknowledges. 3-seed std 0.00057.

---

### #1693 dexhunter — Casefold V4 + AttnOutGate + Multi-Phase Global SGD TTT (1.05733, 3-seed, open) — **DISPUTED (tokenizer)**

**Claim:** −0.0237 BPB vs casefold leader #1585, 3.4x past record threshold. Three components:
1. **Attention Output Gate** (from #1667 @MarioPaerle) — per-head multiplicative gate on attention out, zero-init (identity at start), 1,056 params total.
2. **SmearGate** — input-dependent per-channel residual mixer (13 params), backward-looking.
3. **Casefold V4 + Multi-Phase Global SGD TTT** (inherited from his #1670).

**Compliance status:** Casefold legality is "pending organizer review at Issue #1604." Multi-Phase Global SGD TTT isn't flagged as broken here but relates to the phased-TTT family.

**Interesting technical detail:** AttnOutGate implementation is designed to compose with `fullgraph=True` torch.compile via explicit `.contiguous()` barriers — implies past frontier PRs were hitting compile-graph breaks that this fixes. Useful engineering pattern regardless of whether we adopt the gate.

---

### #1729 romeerp — CaseOps Tokenizer + Tapered WD (1.0678, 3-seed, open) — **CLEAN except tokenizer dispute**

**Claim:** −0.013 BPB over merged SOTA. Two ingredients:
1. **Lossless CaseOps tokenizer** (`lossless_caps_caseops_v1`) — reversible transform: factorizes text into lowercase stream + `TITLE`/`ALLCAPS`/`CAPNEXT`/`ESC` side-channel operators. Original text reconstructed exactly by replaying operators. **Validation BPB charged against exact original UTF-8 bytes via byte sidecars** — this is the "honest" variant that should sidestep the lossy-casefold dispute on #1578.
2. **Mild tapered weight decay** — `WD_TAPER_START_FRAC=0.70`, `WD_TAPER_FINAL_MULT=0.50`. Full Muon WD early, half WD late.

**Compliance:** Uses legal multi-phase TTT from PR #1626. Score-first. Zero comments/flags as of scan.

**Dataset:** Hosted publicly at `romeerp/parameter-golf-caseops-v1`.

**This is the cleanest "sub-1.07 without disputed tokenizer/TTT" submission in the scan.** Tokenizer dispute is narrowly about whether adding `\uE001`-`\uE003` private-use operator tokens to the vocab counts as "changing the val text" — but byte accounting is against original UTF-8 so the BPB denominator is honest. Much stronger compliance story than #1578 (lossy lowercase).

---

### #1727 yahya010 — MP-SGD TTT 4 phases + QK-Gain 5.25 (1.07217, 3-seed, open) — **CLEAN**

`train_gpt.py` **byte-identical to PR #1700** (jorge-asenjo). Only two env-var changes: `PHASED_TTT_NUM_PHASES=4` (up from 3) and `QK_GAIN_INIT=5.25` (matching merged SOTA). Adds ~25–70s eval time for one extra phase, fits in 600s budget. Net −0.00883 vs merged SOTA. Zero review comments (filed 04-18, 2 days ago). Lowest-risk beat-SOTA PR in the whole scan — it's literally a config diff on an accepted base. Useful as evidence that adding phases compounds monotonically, but the actual value is the base (#1700 → #1626 → #1530).

---

### #1695 X-Abhishek-X — Stage 3 + SpinQuant V1 + MP-SGD-TTT (1.07590, 3-seed, open) — **CLEAN**

−0.005 BPB vs #1529 baseline. Two components:
1. **SpinQuant V1** — Hadamard random rotation of weight matrices before INT6 GPTQ. Spreads outliers uniformly, reduces quantization error without changing float predictions. `R` stored as non-parameter buffer, never touched by optimizer. Published technique from Meta AI 2024 so the legal question is settled.
2. **MP-SGD-TTT** — uses the #1626 TTT foundation, so legality story inherits that chain.

Banked architecture porting (`qo_bank`, `kv_bank`, etc.) required per-slot rotation at bake time — non-trivial integration work but isolated to quantization path. Zero review comments.

**Relevance to us:** SpinQuant is an orthogonal quantization win that composes with anything else on the frontier. If we rebase, this is a cheap +0.005 to stack on.

---

### #1732 Victory963 — Hadamard + AWQ + Layer-wise Precision + Hessian SDClip (1.0785, 3-seed, open) — **CLEAN (untested)**

Kitchen-sink quantization stack: Hadamard rotation + AWQ + mixed Int8/Int6/Int4 by layer sensitivity + Fisher-information calibration + 3L depth recurrence + parallel residuals + QK-Gain 5.25 + Legal TTT. Claims +0.0025 over merged SOTA (thin margin). Zero comments, no audit.

**Concerns:** 8 simultaneous ingredients makes attribution impossible. Without ablation it's unclear which component does the work. Low priority for us to copy wholesale.

---

### Banned/broken cluster

**#1722 Trinity SLOT v3 + Pre-Quant TTT (0.65802)** — uses SLOT (banned per #1336/#1376) and Pre-Quant TTT (disputed). Self-quotes a community review as "LOOKS CLEAN" but that quote is from an earlier cleaner Trinity variant, not this one. The Trinity framework family has history of stacking banned patterns.

**#1723 Nairi 9L 512D vocab1024 (0.5384)** — KenMalloy comment: "The submitted score is invalid because the evaluation changes the targets via clamping. A valid BPB would require rerunning with a tokenizer/model vocabulary that can represent the validation bytes without collapsing tokens." Model vocab 1024, tokenizer vocab 8192 → target clamping drops information from denominator. DQ.

**#1246 Trinity v7+skip (0.22311)** — Comments reveal the Trinity family uses "N-gram Order-22 + Per-Sample SLOT + Pre-Quant TTT" all stacked. Every banned pattern in one PR. The 0.22–0.37 bpb claims are fantasy.

**#1557 ndokutovich — N-gram Tilt (1.07730)** — self-flagged AT-RISK on community OLYMPUS tracker under "N-gram cache (03-27)" category. Author is pre-defending. Status: contested.

---

### Supporting foundation PRs (for reference)

**#1626 samacqua — VarLen + Fused MLP + Multi-Phase Global SGD TTT (1.07193)** — created 04-14, open, clean. Parent of #1700/#1727/#1729/#1693 multi-phase chain. Sister PR to #1530 (different TTT flavor).

**#1700 jorge-asenjo — SP8192 + MP-SGD + Phased TTT (1.07219)** — creates the "3-phase" baseline that #1727 bumps to 4 phases.

**#1667 MarioPaerle — SmearGate + AttnOutGate (1.07139)** — **zero audit comments** (4 days old). Simple gates with 1,069 total added params, zero-init. Low risk but unverified.

**#1715 G3sparky — QK-Gain 5.5 (1.0810)** — ties SOTA, no beat. Not a frontier move.

---

## Synthesis

### Legal uncontested frontier (ignoring tokenizer / pre-quant disputes)

```
1.07139  #1667  SmearGate + AttnOutGate [untested, clean]
1.07193  #1626  VarLen + FusedMLP + MP-SGD TTT [foundation]
1.07217  #1727  #1700 + phases=4 [hyperparam, clean]
1.07219  #1700  SP8192 + MP-SGD Phased TTT [clean]
1.0728   #1610  VarLen + PhasingTTT [clean]
1.07336  #1530  VarLen + FusedMLP + doc-indep TTT [foundation]
1.07590  #1695  SpinQuant V1 + MP-SGD TTT [clean, novel quantization]
1.0785   #1732  Hadamard+AWQ kitchen-sink [clean, untested]
```

**Floor without tokenizer/pre-quant disputes: ~1.071.** About −0.010 below merged SOTA.

### Disputed-but-real frontier (if maintainer blesses CaseOps + pre-quant TTT)

```
1.03540  #1738  #1735 + CaseOps V15 [disputed x2: pre-quant TTT + tokenizer]
1.0429   #1735  Parallel Pre-Quant TTT [disputed: pre-quant TTT]
1.05733  #1693  Casefold V4 + AttnOutGate + MP-SGD TTT [disputed: tokenizer]
1.06549  #1736  SP8192 + CaseOps + gates + PhasedTTT [disputed: tokenizer]
1.0668   #1578  Lossy Casefold [disputed: tokenizer lossiness]
1.0678   #1729  CaseOps + Tapered WD [disputed: tokenizer, cleanest of the bunch]
```

If both rulings land **against** the challengers → legal floor stays ~1.071.
If **CaseOps lossless is ruled legal** but pre-quant TTT is not → floor moves to ~1.055 (#1693 or #1729 + SmearGate compounding).
If **both** are ruled legal → floor moves to ~1.035 (#1738 territory).

### Dependency DAG of the clean frontier

```
#1493 (merged SOTA 1.0810)
  └─ #1529 Improved parallel residuals
  └─ #1530 VarLen + fused MLP + doc-indep TTT  (foundation A)
       ├─ #1610 + phased global SGD
       └─ #1736 + CaseOps + gates  (tokenizer-disputed)
  └─ #1626 VarLen + fused MLP + MP global SGD  (foundation B)
       ├─ #1700 + SP8192 + 3 phases
       │    └─ #1727 phases=4 + QK-gain 5.25
       ├─ #1670 + Casefold V4
       │    └─ #1693 + AttnOutGate + SmearGate  (tokenizer-disputed)
       └─ #1729 + CaseOps + tapered WD  (tokenizer-disputed, cleanest)
  └─ #1529/#1445 quant lineage
       └─ #1695 + SpinQuant V1 + MP-SGD
  └─ #1667 SmearGate + AttnOutGate (standalone)
```

**Two parallel foundations:** #1530 (doc-indep TTT) and #1626 (multi-phase global SGD TTT). Sister PRs, same author (samacqua), 3 days apart. Both open, both unmerged. Everything else stacks on one or the other.

### Active maintainer rulings pending
1. **Issue #1604** — tokenizer legality (CaseOps lossless, Casefold lossy). Affects 6 PRs claiming 1.035–1.068.
2. **Issue #1017 Condition 3** — does pre-quant TTT on val data (before artifact freeze) count as "training on val"? Affects 2 PRs claiming 1.035–1.043.

### Rate of change

Over 4 days (04-17 → 04-20), counting only legitimate/clean PRs:
- **Uncontested frontier drift:** 1.0810 (merged) → 1.0714 (#1667 open). **−0.0096 bpb in 4 days.**
- **If tokenizer blessed:** floor drops another ~0.015 (to #1693/#1729 territory).
- **If both blessed:** floor drops further to #1738 territory (~1.035), but this requires two favorable rulings.

### Go/no-go for us

**If we want to legally compete:** rebase onto #1530 or #1626 foundation. Net delta vs our spec-000 (1.0810 regime) would be roughly −0.008 just from rebasing. Then +0.005 from SpinQuant if we adopt it. Then maybe +0.005 from SmearGate+AttnOutGate. That gets us to ~1.060.

**If tokenizer rules in our favor later:** lossless CaseOps (#1729 approach) is a clean +0.004 that composes on top.

**10 days left (deadline 04-30).** Frontier drift rate of ~0.0025/day on the clean side means the legal floor by 04-30 could be ~1.060. To land on the leaderboard we need at least one novel idea beyond rebase. Pure rebase lands us mid-pack.


