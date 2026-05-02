# CaseOps records — family tree with leak/clean annotations

**Updated 2026-05-02 with strict re-audit applied** (see `verdicts.md` for criteria).

Legend: `[C]` = CLEAN (val docs not in train), `[L]` = LEAK (val docs in train), `[?]` = AMBIGUOUS (cannot resolve from PR artifacts alone).

## Tree 1 — Merged trunk (linear ancestry)

```
#1493 [pre-CaseOps boundary, clean by lineage]
   ↓
#1626 [pre-CaseOps boundary, VarLen, clean by lineage]
   ↓                                                    ← BOUNDARY: pre-CaseOps to CaseOps
#1729 [C]  @romeerp        bpb=1.0678  (Apr 18)
   │       — first CaseOps record; cached_challenge_fineweb.py from romeerp/parameter-golf-caseops-v1
   ↓                                                    ←== LEAK INTRODUCED HERE ==
#1736 [L]  @dexhunter      bpb=1.06549 (Apr 19)
   │       — first prepare_caseops_data.py default; train docs 10k+, val docs 0–49,999
   │       — OUR CURRENT RESEARCH BASELINE
   ↓
#1769 [L]  @dexhunter      bpb=1.06453 (Apr 22)
   │       — +MLPClip12; same prep
   ↓
#1787 [L]  @nprime06       bpb=1.06335 (Apr 23)
   │       — +Polar Express NS, MIN_LR, SparseAttnGate, FusedCE; same prep
   ↓
   ├──→ #1797 [L]  @dexhunter  bpb=1.06157  (Apr 25)  — +SmearGate +LQER int4
   │       │
   │       ↓                                            ←== LEAK FIXED HERE ==
   │    #1851 [C]  @aquariouseworkman  bpb=1.06128  (Apr 27)
   │       │       — +SmearGate BOS-fix; SWITCHED to /dev/shm/pgolf_data (HF subset, 39 shards)
   │       │       — current merged-leaderboard SOTA leader
   │       │
   │       ├──→ #1855 [L]  @codemath3000  bpb=1.06108  (Apr 27)
   │       │       — 9-hparam stack; LEAK RE-INTRODUCED — author rebuilt locally with default --val-docs
   │       │       — DATASET_AUDIT.md (PR #2018) verified --val-docs=10000 byte-for-byte
   │       │
   │       └──→ #1868 [C]  @Christopher-Lee-McClendon  bpb=1.06141  (Apr 29)
   │               — 3-seed reproduction of #1851; STAYED on HF dataset
   │               — LATEST clean merged record
```

## Tree 2 — Unmerged frontier branches off #1855

#1855 became the dominant fork point for the unmerged frontier. Most descendants inherited the leaky local prep workflow.

```
#1855 [L] @codemath3000 bpb=1.06108
   │
   ├──→ #1908 [C]  @romeerp  bpb=1.06081  — README explicit HF source; +AWQ-lite GPTQ
   │
   ├──→ #1923 [L]  @jorge-asenjo  bpb=1.05971  — +AsymLogit +AWQ-lite; ORIGINAL val=9.66M (default --val-docs=10000), val-only re-pulled from HF after corruption; train still doc 10k+ → leak
   │
   ├──→ #1945 [C] ← *flipped from [L] in re-audit*  @alertcat  bpb=1.05943
   │       │       — finalize_v18.sh has `snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', local_dir='/workspace/caseops_data')`
   │       │       — README's prepare_caseops_data.py "Data setup" is stale — actual run used HF
   │       │       — IF this is correct, #1945 at 1.05943 is a clean-frontier candidate
   │       │
   │       ├──→ #1953 [?] ← *downgraded from [L] in re-audit*  @andrewbaggio1  bpb=1.05855
   │       │       │       — V21 + TTT tweaks. PR ships only train_gpt.py + logs. No prep evidence.
   │       │       │       — Path matches HF target. Parent #1945 confirmed HF. **Lean CLEAN.**
   │       │       │
   │       ├──→ #1967 [L]  @ndokutovich  bpb=1.05851  — V21 + LeakyReLU 0.3 + N-gram Tilt
   │       │       │       — setup.sh invokes prepare_caseops_data.py default; ALSO has within/word boundary_lut C1 leak
   │       │       │
   │       │       └──→ #2018 [L]  Simon Marcus  bpb=1.04722  (Apr 30)
   │       │              │       — multi-parent (#1945, #1967, #1953, #1855); +Gated XSA, LQER top-1, AsymLogit, n-gram tilt
   │       │              │       — DATASET_AUDIT.md is gold-standard leak documentation
   │       │              │       — note: parent #1945 is CLEAN but #2018 audit explicitly proves LEAK construction
   │       │              │
   │       │              ├──→ #2118 [L]  @aquariouseworkman  bpb=1.04350  (May 1)
   │       │              │              — CURRENT FRONTIER (claimed); submission.json: "--val-docs=10000 train shards + 50k val eval"
   │       │              │              — same author who shipped clean #1851 a week earlier
   │       │              │
   │       │              └──→ #2041 [?] ← *downgraded from [L] in re-audit*  @jorge-asenjo  bpb=1.05692
   │       │                              — No prep invocation in PR; double-nested path, ambiguous
   │       │
   │       └──→ #2014 [L]  @simonbissonnette  bpb=1.05759
   │               │       — "uses same shards as PR #1855"; /dev/shm/pgolf_caseops_data_80_l17_final
   │               │
   │               └──→ #2078 [L]  @hi-aduek  bpb=1.05804  — #2014 reproduction
   │
   ├──→ #2007 [L]  @Elubrazione  bpb=1.05899  — LongCtx + NoQV; triple nesting + ships prep
   │       │
   │       └──→ #2060 [L]  @S0urC10ud  bpb=1.05792  — 5-knob retune
   │               │
   │               └──→ #2100 [L]  @someone114514  bpb=1.05807  — LongCtx + No-QV + Prefix3500
   │
   ├──→ #2019 [C]  @aquariouseworkman  bpb=1.05847  — README explicit: snapshot_download from HF
   │
   ├──→ #2031 [C]  @deborahnelson8788726  bpb=1.05985  — README explicit: 39 train shards from HF
   │
   ├──→ #2068 [C]  @jayaram1125  bpb=1.06172  (parent #1797)  — cached_challenge_fineweb.py from HF
   │
   ├──→ #2071 [L]  @jamesEmerson112  bpb=1.0066 (claimed)  (parent #1851)
   │       — SEPARATE LEAK: symlink-leak (audit-flagged); SP8192 path symlinked to CaseOps shards
   │
   ├──→ #2075 [?] ← *downgraded from [L] in re-audit*  @deusexnatura  — PairGeom-V; ships prep but no explicit invocation
   │
   ├──→ #2101 [L]  @OnlyJundong  bpb=1.05845  — AWQ-lite + AsymLogit + GradCentral; ships prep
   │       │
   │       └──→ #2117 [L]  @JulianTang2027  — 3-seed reproduction of #2101
   │
   ├──→ #2109 [L]  @izlley  bpb=1.05917  — MP3 marker-pair fusion (CUSTOM dataset variant); val_tokens=36.56M
   │
   ├──→ #2121 [L]  @Kbediako  bpb=1.06099  — StageB v2; ships prep
   │
   ├──→ #2123 [L]  @vaibhavmishra1  bpb=1.05933  — closed; superseded by #2124
   │
   └──→ #2124 [L]  @vaibhavmishra1  bpb=1.05933  — resubmission of #2123
```

## Tree 3 — Out-of-CaseOps-scope (in date window but different lineage)

```
#1493 [pre-CaseOps boundary]
   ↓
#2027 [C]  @H1cSuNtDr4C0n3S  bpb=1.08064  (Apr 30)
       — SP8192 QRescue + JEPA-Lite; non-CaseOps SP8192 lineage; clean by lineage

(separately:)
#1915 [not in working set; bulk-classified clean in state.json]
   ↓
#2050 [INHERIT]  @AidenGeunGeun  bpb=1.06083  (Apr 30)
       — eval-only on frozen #1915 quantized artifacts; data verdict depends on #1915
```

## Tree 4 — Symlink leak branch (separate mechanism)

```
#1851 [C]
   ↓
#2071 [L]  @jamesEmerson112  bpb=1.0066 (claimed)
       — caseops_enabled=False but pod data paths symlinked to CaseOps-tokenized shards
       — README admits: "active via symlinked data"
       — NOT the val10k-train leak; orthogonal mechanism
```

## Where leak transitions occur

| Edge | Author of child | Action |
|---|---|---|
| #1729 [C] → #1736 [L] | @dexhunter | **LEAK INTRODUCED**: first use of `prepare_caseops_data.py` default `--val-docs=10000`, started the leaky CaseOps trunk |
| #1797 [L] → #1851 [C] | @aquariouseworkman | **LEAK FIXED**: switched to `/dev/shm/pgolf_data` (39-shard HF subset); first clean record post-#1736 |
| #1851 [C] → #1855 [L] | @codemath3000 | **LEAK RE-INTRODUCED**: rebuilt locally with `prepare_caseops_data.py` default, despite parent being clean |
| #1851 [C] → #1868 [C] | @Christopher-Lee-McClendon | (clean stays clean) — used HF dataset same as parent |
| #1855 [L] → #1908 [C] | @romeerp | **LEAK FIXED**: README explicit HF source |
| #1855 [L] → #1923 [L] | @jorge-asenjo | (leak stays leak) — only val-side fix, train kept default-prep |
| #1855 [L] → #2019 [C] | @aquariouseworkman | **LEAK FIXED**: snapshot_download from HF |
| #1855 [L] → #2031 [C] | @deborahnelson8788726 | **LEAK FIXED**: HF first-39 explicit |
| #1855 [L] → #2068 [C] | @jayaram1125 | **LEAK FIXED**: cached_challenge_fineweb.py from HF |
| #2018 [L] → #2118 [L] | @aquariouseworkman | **REGRESSION**: same author who fixed leak in #1851 now ships leaky #2118; submission.json admits |

## Author behaviors

| Author | Records | Shipped status |
|---|---|---|
| @romeerp | #1729 [C], #1908 [C] | Always clean |
| @dexhunter | #1736 [L], #1769 [L], #1797 [L] | Always leaky (started the leak) |
| @nprime06 | #1787 [L] | Leaky |
| @aquariouseworkman | #1851 [C], #2019 [C], #2118 [L] | Mostly clean; regressed on #2118 |
| @codemath3000 | #1855 [L] | Leaky (re-introduced after #1851 fixed it) |
| @Christopher-Lee-McClendon | #1868 [C] | Clean |
| @jorge-asenjo | #1923 [L], #2041 [L] | Leaky |
| @jamesEmerson112 | #2071 [L] (symlink) | Different leak mechanism |
| @alertcat | #1945 [L] | Leaky |
| @andrewbaggio1 | #1953 [L] | Leaky |
| @ndokutovich | #1967 [L] | Leaky |
| Simon Marcus | #2018 [L] | Leaky (with audit doc) |
| @deborahnelson8788726 | #2031 [C] | Clean (HF) |
| @jayaram1125 | #2068 [C] | Clean (HF) |
| @vaibhavmishra1 | #2123 [L], #2124 [L] | Leaky |

## Key takeaways

1. **The clean trunk is short**: pre-CaseOps → #1729 → (#1851 → #1868). Three actual record submissions in the post-leak-introduction era.
2. **The leaky trunk is long**: #1736 → #1855 → V21 (#1945) → #1967/#1953 → #2018 → #2118, with many sibling forks.
3. **Same authors switch verdicts across PRs**: @aquariouseworkman shipped clean #1851 / #2019 and leaky #2118 within a week.
4. **Once a fork "fixes" the leak by going HF, it stays clean** (e.g., #1908, #2019, #2031, #2068 all sit downstream of leaky #1855 but went HF).
5. **Conversely, "fixing" doesn't propagate**: #1851's HF switch didn't stop #1855 from re-introducing the leak using a sibling local prep.
