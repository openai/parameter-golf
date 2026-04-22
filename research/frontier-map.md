# Frontier dependency map — snapshot 2026-04-22

Record-track PRs on `openai/parameter-golf`, clustered by code lineage (not by
date). Built from PR bodies' explicit "builds on" / "starts from" references.
Numbers are 3-seed mean `val_bpb` as claimed in each PR.

## TL;DR

- **Only #1493 is merged.** It defines official SOTA = 1.0810.
- Unmerged submissions have pushed to **1.0284** publicly (PR #1758, but prequant-TTT-disputed/likely-illegal).
- Best **clean** submission: **#1756** (romeerp) @ **1.06505** — below our baseline.
- Best **tokenizer-disputed/likely-legal**: **#1769** (dexhunter) @ **1.06453** — now below our baseline 1.06549. Lever: single env-var GPTQ σ-clip retune.
- **#1771** (bigbag) @ **1.06513** also below baseline via depth curriculum + LoRA-TTT warm-start-A synthesis.
- Several sub-1.02 GatedDeltaNet PRs exist but are either closed or disputed on
  legality / artifact size. Treat those as "contested frontier," not baseline.
- The community has split into **at least four code trunks**, each with its own
  descendants. Tokenizer and TTT innovations are the orthogonal levers that
  jump across trunks.

## Trunks

### Trunk A — #1523 stack (classic depth-recurrence + parallel residuals)

```
#1493  MERGED 1.0810  bigbag   [official SOTA]
 │
 ├─ #1552  open        Tanush1912    RecurLoRA v2 — off #1493 directly
 │                                   ↳ borrowed by #1530
 ├─ #1770  open 1.0796  liujshi   +V-Gate (per-head gates V-proj input + head output scale)
 │
 └─ #1523  CLOSED 1.0778  EthanYangTW [superseded trunk — code survives]
     │    param banking, fused MLP, Muon 0.97, triple recurrence, bigram hash
     │
     ├─ #1529  open 1.0758  msisovic   richer parallel residuals (split-lane)
     │    └─ #1578  open 1.0668  mikeapedia   +casefold tokenizer [DISPUTED: lossy]
     │
     ├─ #1530  open 1.0734  samacqua   varlen attn + fused MLP + doc-indep LoRA TTT
     │    │   (also pulls orthogonally from #1552)
     │    │
     │    ├─ #1610  open 1.0728  romeerp    +phased global-SGD eval pass
     │    │   └─ #1767  open 1.07209  renqianluo  +LoRA warm-start A + alpha/rank scaling + WD=1.0 (non-CaseOps stack)
     │    ├─ #1586  open 1.0749  dexhunter  +per-layer GPTQ clip, int7 emb, MLR=0.026
     │    │   └─ #1626  open 1.0719  dexhunter  +multi-phase global-SGD TTT
     │    │        └─ #1729  open 1.0678  romeerp  +CaseOps tokenizer + late WD taper
     │    └─ #1736  open 1.0655  dexhunter  +CaseOps (from #1729) + attn out-gate  [DISPUTED: tokenizer]
     │         ├─ #1755  open 1.07462  OE-GOD   #1493+CaseOps compose (above our baseline)
     │         ├─ #1756  open 1.06505  romeerp  Recurrence depth curriculum [1→3→4], eval depth=4 ← BELOW BASELINE
     │         ├─ #1766  open pending  tashapais  +Recur-Alpha (learned carry scalar per looped block)  [DISPUTED: tokenizer]
     │         ├─ #1769  open 1.06453  dexhunter  +MLPClip12 (GPTQ MLP σ-clip 10→12) ← BELOW BASELINE  [DISPUTED: tokenizer]
     │         └─ #1771  open 1.06513  bigbag   +RecurDepthCurr + LoRA-TTT warm-start-A (synth #1756+#1767) ← BELOW BASELINE  [DISPUTED: tokenizer]
     │
     └─ #1667  open 1.0714  MarioPaerle  SmearGate + attn output gate
                                         (refs #1493, #1586; base unclear)
```

### Trunk B — Multi-Phase Global SGD TTT (jorge-asenjo)

```
#1700  open 1.07219  jorge-asenjo   Multi-Phase Global SGD + Phased TTT
 └─ #1727  open 1.07217  yahya010   +QK_GAIN=5.25, +4th SGD phase
```

The multi-phase SGD idea propagated into Trunk A via dexhunter's #1626; #1700
is the original submission of this TTT regime.

### Trunk C — Parallel Pre-Quant TTT (AjAnubolu)

```
#1735  open 1.0429  AjAnubolu
     SP8192 + 3-layer depth recurrence + parallel residuals + QK-Gain 5.25
     + 8-GPU Parallel Pre-Quant AdamW TTT (21 epochs, epoch-level cosine LR)
     Fixed predictor — no eval-time adaptation
 │
 └─ #1738  open 1.03540  alertcat   +CaseOps Tokenizer V15  [current public frontier]
       └─ #1758  open 1.02840  kilojoules  PreQuant TTT retune (LR→1e-3, unfreeze-all)  [DISPUTED: pre-quant TTT]
```

This is a *different* TTT paradigm from Trunks A/B: AdamW on 8 GPUs in parallel
before quantization, many epochs, fixed predictor at eval. Does not cite #1523.

### Trunk D — GatedDeltaNet / FLA (contested)

```
#1687  CLOSED 1.04090  resouer      K_KVShare_Wider full-recipe FLA
 └─ #1698  open   1.00995  arsenis-cmd  GatedDeltaNet + Legal Score-First TTT
     │                                  [artifact size exceeds 16MB cap]
     └─ #1734 CLOSED 1.01080  yahya010  GatedDeltaNet + Brotli-11 (size-fixed)
                                         [closed same-day — unclear why]
```

Linear-attention / Mamba-family architectures. Claims sub-1.02 numbers but
every entry has a merge blocker. Needs deeper legality/size diligence before
we take the numbers at face value.

## Orthogonal lever classes

These are roughly additive across trunks:

| Lever | Representative PRs | Approx Δ bpb |
|---|---|---|
| Tokenizer — lossy casefold | #1578 | ~–0.01 (disputed) |
| Tokenizer — lossless CaseOps | #1729, #1736, #1738 | ~–0.0075 |
| TTT — phased global SGD | #1610, #1626, #1700, #1727 | ~–0.001 to –0.005 |
| TTT — parallel pre-quant AdamW | #1735 | ~–0.02 (new paradigm) |
| Quant — per-layer GPTQ clip + int7 emb | #1586 | ~–0.006 |
| Gates — SmearGate / attn out-gate | #1667, #1736 | ~–0.002 |
| Arch — richer parallel residuals | #1529 | ~–0.002 |
| Arch — GatedDeltaNet | #1698 trunk | large but contested |

## Active composer-authors

- **dexhunter** — #1586 → #1626 → #1736, also authored CaseOps; ports across trunks
- **romeerp** — #1610, #1729; bridges SGD-TTT and tokenizer levers
- **yahya010** — #1727, #1734; pushes hyperparam + GatedDeltaNet combos
- **AjAnubolu** — #1735; owns the parallel pre-quant TTT paradigm
- **alertcat** — #1738; first to graft CaseOps onto Trunk C, currently on top

## Obvious gaps nobody has posted yet

1. **dexhunter quant/TTT stack (#1626/#1586) + #1736 tokenizer/gate stack** —
   same author, not yet combined.
2. **Trunk C base (#1735) + Trunk A gates/quant (#1586, #1667)** — #1738 only
   composed CaseOps onto #1735; more levers should stack.
3. **Trunk B Multi-Phase TTT (#1727) + Trunk C parallel pre-quant (#1735)** —
   two different TTT regimes; whether they cooperate or conflict is unknown.

## Caveats

- "Δ" numbers are PR-body claims; std across seeds varies 0.0002–0.0015, and
  Trunks C/D have higher std than Trunk A.
- "Closed" ≠ "invalid." #1523 is closed-superseded; #1734 may be
  closed-withdrawn; #1687/#1698/#1731 are closed/blocked on merge eligibility.
  Status needs re-checking before building on any closed PR.
- This map is built from PR body text only; branch-level git ancestry may
  differ. Worth a manual check before we pin a spec to any of these.
