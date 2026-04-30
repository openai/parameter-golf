# Parameter Golf PR Archive — Master Index

*Built 2026-04-11 from 65 PRs spanning the ~1,426 total PR space.*
*Each PR directory contains the README(s), train_gpt.py (where applicable), submission.json, run scripts, and a `raw_excerpt.md` with verbatim author claims. Key parent→child DAG edges have `.diff` files in `diffs/`.*

**Total archive: 5.7 MB, 65 PRs, 28 diffs.**

---

## How to read this archive

- **Nodes** are PR directories (`PR-NNNN/`) containing author-original files and our `raw_excerpt.md` (objective facts only).
- **Edges** (where meaningful) live in `diffs/PR-parent_to_PR-child.diff` — a unified diff between the two train_gpt.py files. Edge sizes below indicate how much changed.
- **A few PRs store their train_gpt.py as a compressed LZMA+base85 blob** (e.g., PR-1296, PR-1331). Their diffs look like one-line changes because the entire source is packed into a single string. To read those files you need to decompress them first (`python -c "import lzma, base64; print(lzma.decompress(base64.b85decode(...)).decode())"`).
- **7 PRs have no train_gpt.py** (analysis-only): #1162, #1222, #1271, #1272, #1281, #1341, #1385. Their content lives in markdown files and analytical scripts.

---

## The DAG at a glance

```
                BASELINE (1.2244)
                      │
                      ▼
        PR-65 ── SmearGate + OrthoInit + Int6 QAT + BigramHash (1.1556)
                      │
                      ▼
        PR-162 ── WD tuned + SWA (1.1458)
                      │
                      ▼
        PR-198 ── 11L + WD=0.04 + SWA (1.1318, jfprincz)
                      │
                      ▼
        PR-287 ── EMA replaces SWA + XSA4 (1.1271)
                      │
                      ▼
        PR-315 ── Partial RoPE + LN Scale (1.1248)
                      │
                      ▼
        PR-414 ── GPTQ-lite + EMA + warmdown3500 (1.1228)
                      │
                      │ ← PR-461 (Legal Score-First TTT recipe)
                      │ ← PR-493 (LeakyReLU², ext)
                      │ ← PR-399 (Parallel Muon + Parameter Banking)
                      ▼
        PR-549 ── LeakyReLU² + Legal TTT + Parallel Muon (1.1194)
                      │
                      │ ← PR-478 (XSA-all)
                      │ ← PR-535 (Full Hessian GPTQ)
                      ▼
        MERGED SOTA (PR-1019, records/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072)
        1.1147 BPB — "the merged frontier is STALE"
                      │
                      │
           ┌──────────┼─────────────────────────┬────────────┐
           ▼          ▼                         ▼            ▼
       PR-1218    PR-1204                   PR-1217        PR-1344
       SP4096 +   Mini Depth Recurrence      MuonEqR       Polar Express NS
       MLP4x +    + Parallel Residuals       origin        (4 minimax steps)
       WD=0.085   (origin PR)
           │          │
           │          ├─────────┬─────────────┐
           │          ▼         ▼             ▼
           │      PR-1285   PR-1296       PR-1331
           │      MuonEqR+  SP4096+        3-layer recurrence
           │      depth    depth           (7-line edit on 1296)
           │      rec      rec +
           │      all-int6 parallel
           │          │    residuals
           │          ▼
           │      PR-1260
           │      Mixed int5/int6
           │
           ▼
       PR-1394 ── SP8192 + GPTQ EMBEDDINGS + SDClip + Loop45x2 (1.08563, 5-seed)
         │       "THE unlock: embedding quantization lets SP8192 fit"
         │
         ├─────────┬─────────┬─────────┬─────────┬─────────┐
         ▼         ▼         ▼         ▼         ▼         ▼
      PR-1408   PR-1420   PR-1413   PR-1399   PR-1400   PR-1426
      dTTT +    Triple    SP8192+   PreQuant  Hadamard  Int4-packed
      BH3072    loop +    Legal     TTT+ETLB  rotation  + 13L +
      1.0800    n-gram    TTT       (CONT.)   pre-GPTQ  PreQuant TTT
      (#1)      tilt      1.0828    1.0898    1.1035    (PENDING)
                1.0801                                  ← your int4 bet
                (#2)
```

---

## DAG edges (computed diffs)

Reading the diff sizes tells you how much of a "step" each PR was. Small diffs (< 200 lines) = focused changes; large diffs (> 1000 lines) = major rewrites or parent+child directory differences in a monolithic file.

| Edge | Diff lines | Meaning |
|------|-----------|---------|
| PR-65 → PR-162 | 1,376 | First major WD/SWA tuning on top of foundational stack |
| PR-162 → PR-198 | 1,306 | 11L + SWA full integration (nearly a rewrite) |
| PR-198 → PR-287 | **128** | Focused change: SWA → EMA + XSA4 on last 4 layers |
| PR-287 → PR-315 | **154** | Partial RoPE + LN Scale (zero-param wins) |
| PR-315 → PR-414 | 1,375 | GPTQ-lite introduction + warmdown extension |
| PR-414 → PR-549 | 1,148 | LeakyReLU² + Legal TTT + Parallel Muon bundle |
| PR-549 → PR-703 | 671 | Full GPTQ + MiLe loss + 8-bit Muon momentum |
| PR-549 → PR-1218 | 3,034 | Big jump: SP4096 + MLP4x + WD=0.085 |
| PR-1218 → PR-1394 | 1,179 | SP8192 + GPTQ embeddings + SDClip + depth rec |
| PR-1394 → PR-1408 | 3,548 | dTTT + BigramHash3072 + extensive refactor → frontier #1 |
| PR-1394 → PR-1413 | 1,413 | SP8192 + legal TTT + QK-gain 5.0 → frontier #3 |
| **PR-1394 → PR-1420** | **204** | **Triple loop + n-gram tilt — shockingly focused for frontier #2** |
| **PR-1394 → PR-1415** | **94** | **Pure ETLB addition — one-file surgical change** |
| PR-1394 → PR-1399 | 2,141 | Pre-Quant TTT + ETLB (contested, leaning illegal) |
| PR-1394 → PR-1400 | 3,448 | Hadamard rotation pre-GPTQ (under compliance review) |
| PR-1394 → PR-1426 | 3,325 | Int4-packed MLP + 13L (the int4 bet — results pending) |
| PR-1218 → PR-1204 | 2,511 | Mini depth recurrence + parallel residuals origin |
| PR-1204 → PR-1285 | 1,092 | MuonEq-R + all-int6 on top of depth rec |
| PR-1285 → PR-1260 | 416 | Mixed int5/int6 precision |
| PR-1204 → PR-1296 | 886 | SP4096 + depth rec refinement (aryanbhosale) |
| **PR-1296 → PR-1331** | **7** | **3-layer recurrence — literally a hyperparameter flip (compressed blob)** |
| PR-1218 → PR-1217 | 1,634 | MuonEq-R origin (bigbag) |
| PR-1218 → PR-1344 | **0** | Polar Express NS didn't touch train_gpt.py at that base — path mismatch (edge exists in README only) |
| PR-1218 → PR-1395 | 1,154 | Linear LR warmdown pre-GPTQ |
| PR-535 → PR-692 | 1,185 | CROWN-Q quant-variance penalty on top of Full GPTQ |
| PR-692 → PR-1156 | 2,781 | EGGROLL post-GPTQ bin refinement |
| PR-1218 → PR-1291 | 161 | SLOT integration on clean base (SLOT is dead post-#1240) |
| PR-1217 → PR-1418 | 2,208 | MuonEq-R + Parallel Muon = 40% regression (100+ negatives) |

**Edge takeaways:**
- **Small diffs at the frontier are the most informative.** `#1394 → #1415` at 94 lines is pure ETLB addition. `#1394 → #1420` at 204 lines is the triple-loop + n-gram tilt. You can read these diffs in 5 minutes and see exactly what the frontier additions are.
- **Small diffs also hide bigger things when files are compressed.** `#1296 → #1331` is 7 lines because both files are LZMA+base85 blobs; the real change is buried in the compressed string.
- **Big diffs (>2000 lines) usually mean full rewrites or refactors**, not a small technique addition. Read the README first before the diff.

---

## PR directory — ordered by relevance

### Tier 1: Quantization work (your primary focus)

| PR | Author | Claim | What it is | Legality | Why it matters for you |
|----|--------|-------|-----------|----------|------------------------|
| **#1426** | aravhawk | BPB pending | **True Int4 bit-packing + 13L + PreQuant TTT** | Mixed (int4 OK, PreQuant TTT contested) | **THE reference implementation for your bet.** First attempt at int4 packed storage. Also uses PreQuant TTT which you should probably NOT inherit. Study the `pack_int4` function and how they handle 13L layer budgets. |
| **#1394** | clarkkev | 1.08563 (5-seed) | **SP8192 + GPTQ embeddings + SDClip + depth rec** | Clean | The current quantization state-of-the-art in a clean submission. SDClip replaces reconstruction-error clip search with `c = k·σ(row)`. Read this carefully — it's what you'll be extending. |
| **#535** | raahilshah | 1.1204 | **Full Hessian GPTQ (origin)** | Clean | The core GPTQ machinery every frontier PR builds on. Reference implementation for Cholesky-based error compensation. |
| **#692** | — | 1.1186 | **CROWN-Q (quant-variance penalty during warmdown)** | Clean (closed for doc issues only) | **Revivable.** Trains weights to be quantization-tolerant via variance penalty. Closed for documentation reasons, not technical ones. A natural complement to int4 QAT. |
| **#1400** | tmancino | 1.10352 | **Hadamard rotation pre-GPTQ (68× lower MSE)** | Compliance review | Rotates weights before quantization to decorrelate column structure. Has stronger theoretical motivation than SDClip. Compliance status uncertain — worth watching. |
| **#1156** | koltondrake | 1.1161 | **EGGROLL: post-GPTQ int6 bin refinement** | Clean | Zeroth-order refinement of quantization bins after GPTQ. Not SGD — coordinate descent on discrete grid. Legal post-training operation. |
| **#1251** | Ibarra | 1.1349 | **Online Hessian GPTQ (negative)** | Clean but regressive | **Read the commit message to understand why it failed.** 17ms/step overhead for too little gain. A cautionary tale about adding compute to the training loop for GPTQ improvements. |
| **#414** | signalrush | 1.1228 | **GPTQ-lite origin** | Clean | Historical baseline. Diagonal-Hessian GPTQ — superseded by full Hessian (#535), but useful for understanding progression. |
| **#1395** | dttdrv | 1.0924 | **Linear LR warmdown pre-GPTQ (-61% quant gap)** | Clean | **Free quantization win.** Single-line schedule change reduces quant gap from 0.038 to 0.014. Should be in your stack. |

### Tier 2: Frontier (clean submissions pushing sub-1.09 BPB)

| PR | Author | Claim | Technique | Status |
|----|--------|-------|-----------|--------|
| **#1408** | aamodbhatt | **1.0800** (3-seed) | dTTT + BigramHash3072 + SP8192 stack | Frontier #1 — "cleanish" (dTTT novelty) |
| **#1420** | abaybektursun | **1.08014** (3-seed) | Triple loop + n-gram tilt | Frontier #2 — clean |
| **#1413** | dexhunter | **1.08279** (3-seed) | SP8192 + Legal TTT + QK-gain 5.0 | Frontier #3 — cleanest of top 3 |
| **#1296** | aryanbhosale | **1.0897** (3-seed) | SP4096 + depth rec + parallel residuals + MuonEq-R | Clean |
| **#1331** | dexhunter | **1.0900** (3-seed) | 3-layer depth recurrence (hyperparameter twist on #1296) | Clean |
| **#1285** | dexhunter | **1.0912** (3-seed) | MuonEq-R + depth rec + all-int6 | Clean |
| **#1344** | Omrigotlieb | **1.0923** (3-seed) | Polar Express NS (4 minimax steps) | Clean |
| **#1260** | dexhunter | **1.0929** (3-seed) | MuonEq-R + depth rec + mixed int5/int6 | Clean |

### Tier 3: Core technique sources

| PR | Author | What it added | Why it matters |
|----|--------|--------------|----------------|
| **#65** | aquariouseworkman | Int6 STE QAT + SmearGate + OrthoInit + MLP3x + BigramHash foundational bundle | The foundational Day 2 stack everything builds on |
| **#162** | raahilshah | Weight decay + SWA tuning on top of PR-65 | Established WD as a quantization-friendliness lever |
| **#198** | jfprincz | 11L + WD=0.04 + SWA (1.1318) | First 11-layer stack — depth over width win |
| **#287** | jfprincz | EMA replaces SWA + XSA on last 4 layers | EMA became standard; XSA started here |
| **#315** | jfprincz | Partial RoPE (16/64 dims) + LN Scale (1/sqrt(l+1)) | Zero-parameter wins that still carry to the frontier |
| **#374** | unnir | VE128 shared value embedding | ~-0.001 BPB, carried through all subsequent records |
| **#478** | gowtham0992 | XSA-all 11 layers (vs last 4) | Proved -0.006 BPB from covering all layers |
| **#399** | abaybektursun | Parallel Muon + Parameter Banking (15× faster NS) | +190 training steps — the biggest "free compute" win |
| **#461** | Christopher-Lee-McClendon | Legal Score-First TTT recipe | The recipe that kept TTT alive after issue #402 |
| **#549** | abaybektursun | LeakyReLU² + Legal TTT + Parallel Muon bundle | 1.1194 — the stack that became the merged SOTA via PR-1019 |
| **#1204** | Marko Sisovic | Depth recurrence + parallel residuals origin | **THE frontier breakthrough.** Everything post-April builds on this. |
| **#1217** | bigbag | MuonEq-R (row-normalized Muon) origin | Free -0.001 to -0.002 BPB, standard in frontier |
| **#1218** | clarkkev | SP4096 + MLP4x + WD=0.085 stack | The base that #1394 (and everything past it) built on |
| **#1344** | Omrigotlieb | Polar Express NS (4 minimax Newton-Schulz steps) | +180 training steps vs standard 5-step NS |
| **#1125** | pranjal.jain | QK-Gain=4.0 (45-experiment ablation on 1×5090) | Methodology: how to pin down a hyperparameter rigorously |

### Tier 4: Training / optimizer experiments

| PR | Author | BPB | Claim | Status |
|----|--------|-----|-------|--------|
| **#510** | SelfAnush | 1.1989 (1 seed) | MUD optimizer (triangular Gram, arxiv:2603.17970) | Non-record; TRSM 4.5× slower than Muon on H100 |
| **#530** | j420 | 1.4963 (1×H100) | MATRIX_LR sweep, found 0.03 > 0.02 for -0.059 BPB | Non-record; proved MATRIX_LR=0.03 |
| **#703** | — | 1.1171 (3-seed) | Full GPTQ + MiLe loss + Cache+Backout + 8-bit Muon momentum | **Unexplored.** 8-bit Muon is a real memory saver (62%) |
| **#873** | — | 1.0467 (1 seed, post-quant) | E2E TTT MAML-style meta-learning | **Not reproducible?** The 1.0467 was single-seed; MAML ≠ competitive at multi-seed |
| **#917** | TheDryhtscipe | — (4060 Ti, 6L/256d) | ConsensusWindowBypass replaces SmearGate | Small scale only — unknown if it scales |
| **#1130** | Gusanidas | 1.1140 (12 seeds) | KitchenSinkV2 — residual lambdas + 12-seed validation | **Methodology gold.** The 12-seed rigor is an example to follow |

### Tier 5: Contested / eval-time tricks (mostly avoid)

| PR | Author | BPB | Trick | Status |
|----|--------|-----|-------|--------|
| **#1291** | dentity007 | 1.0925 | Vocab4096 + MLP4x + SLOT | **SLOT — contested.** Killed by #1240. |
| **#1240** | Baggio | 1.1064 (record attempt) + causal analysis (non-record) | **SLOT flip test — THE killshot.** Contains `prove_slot_causal_violation.py` | The most important compliance artifact in PG history. Study the flip test methodology. |
| **#1399** | AnubhavBharadwaaj | 1.0898 | Pre-Quant TTT + ETLB + all-int6 | **Contested, leaning illegal.** This is the "training in the post-training window" hack. |
| **#1415** | bigbag | 1.0913 | SP4096 + 3-layer rec + ETLB | ETLB contested |
| **#873** | — | 1.0467 | E2E TTT MAML | Single-seed, probably not causal |
| **#1222** | — | — (non-record) | TTT E2E FOMAML analysis | **Crucial:** this PR *showed* that TTT benefits are massively overstated. Read as a warning, not an implementation. |
| **#1341** | himanshudongre | — | TTT+GPTQ incompatibility analysis | Proved that post-GPTQ TTT is neutral or negative |

### Tier 6: Negative results (do not repeat these)

| PR | Finding | Magnitude |
|----|---------|-----------|
| **#363** | SmearGate hurts depth recurrence | Confirmed conflict |
| **#989** | SWA sabotages QAT in a 2×2 factorial ablation | -3.64 mBPB |
| **#1147** | Mathematical proof that hashed n-gram caches are invalid (partition function ~1000) | Mass disqualifications of PRs #659-#887 |
| **#1162** | Step-1000 BPB correlates 0.86 with final | **Meta-finding:** enables rapid hyperparameter search |
| **#1218→#1418** | MuonEq-R + Parallel Muon = 40% regression | Do not combine |
| **#1222** | TTT benefits massively overstated — just compensating for missing context | Historical recalibration |
| **#1251** | Online Hessian GPTQ net negative due to 17ms/step overhead | Cautionary — don't add training-loop GPTQ |
| **#1271** | Scylla tokenizer: 93% of claimed gap was byte accounting error | Cautionary on tokenizer BPB calculations |
| **#1272** | Comprehensive negative results: most "strong model" augmentations don't help | Broad sanity check |
| **#1293** | Universal Transformer + ACT: compute overhead eats training budget (7,392 vs 13,780 steps) | Adaptive compute doesn't work at this scale |
| **#1330** | JEPA collapses in causal LMs | Objective mismatch |
| **#1341** | TTT+GPTQ fundamentally incompatible | Full GPTQ already captures what TTT provides |
| **#1418** | 100+ experiments compilation: MuonEq-R+ParallelMuon, Hadamard marginal, much more | The reference for "what not to try" |

### Tier 7: Novel architectures (mostly failed)

| PR | Author | BPB | Architecture | Verdict |
|----|--------|-----|--------------|---------|
| **#148** | Ivan Verbovoy | 1.2196 | Cross-repeat skip connections (early depth rec exploration) | Superseded by #1204 parallel residuals |
| **#352** | Austin Tarango | 1.1659 | Memory Tokens (64 learnable global vectors) | **-0.014 BPB reported but never adopted.** Possibly revivable. |
| **#660** | HugoOchoaLP | 1.1826 (over budget) | Soft MoE with dense gating | Gating overhead, but cleaner than sparse MoE |
| **#812** | andrewmouldon | 1.2236 | BankLinear shared QKV with depth-aware mix coefficients | Interesting param-sharing idea, never pushed to frontier |
| **#1268** | samquiring | 1.1875 | Mamba3 hybrid + parallel associative scan | Best SSM attempt — still off frontier |
| **#1281** | rlphlrnrys | 3.1728 (non-record) | MatrixGPT O(n) recurrent | Interesting form factor, not competitive |
| **#1293** | 5en5e1 | 1.2409 | Universal Transformer + ACT | Adaptive compute overhead — see negatives tier |
| **#1302** | vlivashkin | 1.1078 | Split-LR + online n-gram agreement | Clean novel hybrid, just off frontier |
| **#1316** | AR6420 | 1.1310 (1 seed) | MLP Megakernel + attention fusion (custom Triton) | One of the rare custom-Triton PRs |
| **#1327** | mrbese | 1.1276 (1 seed) | BESE 287-vocab tokenizer | Extreme small vocab, not worth it |
| **#1330** | luciobaiocchi | — | JEPA v2 multi-step | Collapsed — see negatives tier |
| **#1403** | Rhoahndur | 1.3485 | Masked Diffusion LM | Wrong objective for causal LM scoring |

### Tier 8: Bandit track (n-gram as model)

These are a separate category — they're not transformers, they're lookup tables. Don't compete with them directly.

| PR | Author | BPB | Approach | Legal? |
|----|--------|-----|----------|--------|
| **#1379** | Lucas Ercolano | 0.4162 | Mixed quant + causal n-gram mixer | Legitimate bandit frontier |
| **#1105** | abaybektursun | 1.0962 | FusedMLP + Brotli + memmap + causal n-gram tilt | Clean, transformer-based but with n-gram tilt |
| **#1302** | vlivashkin | 1.1078 | Split-LR + online n-gram agreement | Clean |
| **#1147** | Robby Sneiderman | — (analytical) | Mathematical proof that hashed n-gram caches are invalid | The authoritative compliance document |

### Tier 9: Unexplored opportunities

| PR | What's there | Why it's worth revisiting |
|----|-------------|--------------------------|
| **#703** | 8-bit Muon momentum (62% memory reduction) + MiLe loss | Never adopted in main lineage |
| **#352** | Memory Tokens (-0.014 BPB) | Author reports gain but never integrated |
| **#1385** | Compressor-Aware Training (differentiable LZ77 proxies) | Novel technique, never at frontier |
| **#692** | CROWN-Q (quant variance penalty) | Closed for docs, technique itself is valid |
| **#1400** | Hadamard rotation pre-GPTQ | Under compliance review |

---

## What's in each PR directory

Every `PR-NNNN/` contains at minimum:
- `raw_excerpt.md` — objective facts (author, BPB, seeds, run command, verbatim README claims)
- The author's own files, flattened from their source paths. Filenames like `records__track_10min_16mb__<dir>__README.md` preserve the origin directory in the filename.

Most PRs also have:
- `records__..__train_gpt.py` — the main training script (may be LZMA+base85 compressed for some authors)
- `records__..__submission.json` — machine-readable metadata and claimed scores
- `records__..__README.md` — author's own description
- Occasionally: `run_*.sh`, `RESULTS.md`, `APPROACH.md`, analysis scripts

**Analysis-only PRs (no train_gpt.py):**
- **#1162** — Abay Bektursun's step-1000 correlation meta-analysis (lives in `meta_analysis/README.md`)
- **#1222** — TTT overstated analysis (has `train_e2e_proper.py`, `train_ttt_e2e.py`)
- **#1271** — Scylla tokenizer audit (has `retokenize_proper.py`)
- **#1272** — Comprehensive negative results (has `ngram_test.py`, `online_logit_bias.py`, `retokenize_proper.py`)
- **#1281** — MatrixGPT (has `train_matrix.py` instead)
- **#1341** — TTT+GPTQ incompatibility analysis (has `clark_ttt_eval.py`, `sgd_ttt_eval.py`)
- **#1385** — Compressor-Aware Training (has `train_golf.py`)

---

## Reconciled authorship corrections

A few of the author attributions in my earlier meta-analysis were wrong. The archive's `submission.json` files are authoritative:

- **#1408** is by **aamodbhatt**, not abaybektursun (both work together on abaybektursun's team, but this record is aamodbhatt's)
- **#1413** is by **dexhunter**, not clarkkev (though it builds directly on clarkkev's #1394)
- **#1420** has dual BPB numbers: **1.08014** (submission.json) vs **1.08309** (5-seed mean in README) — the 5-seed number is more trustworthy
- **#198** is jfprincz's 1.1318 record, not the EfficientPartialXSA record I originally thought. The real PR#198 was already updated in our earlier search results.

---

## Strategic recommendations based on this archive

For your quantization-focused submission, the highest-value reading order is:

1. **Read first:** `PR-1394/` (the SDClip + GPTQ embeddings stack). This is the baseline you'll be extending.
2. **Then:** `PR-1426/` (the int4 attempt). This is the only reference implementation of int4 packed storage in the competition. Study the `pack_int4` function closely.
3. **Then:** `PR-692/` (CROWN-Q). A natural complement to int4 that nobody else has combined with frontier depth recurrence.
4. **Then:** `PR-1156/` (EGGROLL). Legal post-training refinement.
5. **Then:** `PR-1400/` (Hadamard rotation). Theoretical motivation you can layer in front of GPTQ.
6. **Avoid:** PR-1399, PR-1415 — the ETLB / pre-quant TTT contested stuff. Don't build on them.
7. **Read once, never build on:** PR-1240 (SLOT killshot) and PR-1222 (TTT overstated). Important precedents but dead ends.

The diffs to actually read carefully (in order of insight-per-line):

- `diffs/PR-1394_to_PR-1415.diff` (94 lines) — pure ETLB addition, surgical
- `diffs/PR-1394_to_PR-1420.diff` (204 lines) — triple loop + n-gram tilt, surgical
- `diffs/PR-198_to_PR-287.diff` (128 lines) — SWA → EMA transition
- `diffs/PR-287_to_PR-315.diff` (154 lines) — Partial RoPE + LN Scale zero-param wins
- `diffs/PR-1218_to_PR-1291.diff` (161 lines) — SLOT integration (to see what SLOT code LOOKS like for avoidance purposes)

Everything else is a larger rewrite where reading the README is more efficient than reading the diff.

---

*Archive built 2026-04-11. Authoritative sources: `raw_excerpt.md` per PR (objective), `diffs/*.diff` (actual code changes), `../META_ANALYSIS.md` (broader interpretation).*
