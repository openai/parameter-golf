# Parameter Golf Submission Analysis Summary

**Generated**: 2026-04-14  
**Coverage**: 24 local submissions + 10 GitHub PRs = 34 total

## Files Created

### Analysis Documents

1. **complete-techniques-mapping.md** (THIS IS YOUR MAIN REFERENCE)
   - Comprehensive mapping of all 34 submissions
   - Techniques ranked by adoption (96% → 4%)
   - Local submission rankings by BPB
   - Technique evolution timeline
   - Top 10 GitHub PR innovations

2. **local-techniques-analysis.md**
   - Deep analysis of 24 local submissions (21 record + 3 non-record)
   - 37 unique techniques detected
   - Technique-to-submission mapping
   - Chronological record submission table

3. **top-prs-analysis.md**
   - Top 10 GitHub PRs analyzed
   - 20 unique techniques found across PRs
   - JEPA, GDN, U-Net, Diffusion, LoRA, Depth Recurrence

4. **leaderboard-guide.md**
   - GitHub leaderboard map (60+ URLs)
   - Current SOTA (1.1147 BPB)
   - Landmark submissions
   - Leaderboard statistics

### Python Scripts

- `analyze_techniques.py` — Analyzes 24 local train_gpt.py files
- `analyze_top_prs.py` — Extracts techniques from 10 GitHub PRs

---

## Key Findings

### Must-Have Techniques (96% adoption in local)

- **Muon Optimizer** — Baseline for all submissions
- **Int6 Quantization (QAT)** — Core compression
- **GQA** — Attention efficiency
- **EMA + Cosine Warmdown** — Regularization

### High-Impact Techniques (30-70% adoption)

- **Sliding Window Eval** — Free +0.019 BPB
- **SWA** — Ensemble averaging (+0.001 BPB)
- **BigramHash** — Co-occurrence patterns (+0.004 BPB)
- **XSA** — Cross-position mixing (+0.003 BPB)

### SOTA Synthesis (1.1147 BPB)

```
Base (Muon + Int6 + GQA) 
  + Cosine warmdown
  + Sliding window eval (+0.019)
  + EMA + SWA
  + BigramHash 3072×112 (+0.004)
  + XSA on all layers (+0.003)
  + Full GPTQ (Hessian quant) (+0.005)
  + LZMA compression
= 1.1147 BPB (SOTA)
```

### Experimental Techniques (GitHub PRs)

- **JEPA** — Different pre-training objective
- **GDN** — Alternative architecture (doesn't compress well)
- **U-Net** — Encoder-decoder (novel but not competitive)
- **Masked Diffusion** — Alternative generation (slower)
- **LoRA** — Low-rank fine-tuning (works with QAT)
- **Depth Recurrence** — Shared blocks (massive param reduction)

---

## What You Have

✅ **24 complete local submissions** with full code:
- `records/track_10min_16mb/` (21 record submissions)
  - Each has: `train_gpt.py`, `README.md`, `submission.json`, 3 seed logs
- `records/track_non_record_16mb/` (3 non-record submissions)

✅ **Top 10 GitHub PRs** scraped:
- Full markdown content in `.firecrawl/pr-*.md`
- Technique analysis extracted

✅ **Comprehensive mapping**:
- See `complete-techniques-mapping.md` for full reference

---

## Next Steps

1. **Start with SOTA submission** (2026-03-25):
   - Read: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md`
   - Study: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`
   - Test: Run with phase 1 baseline on your setup

2. **Review technique evolution**:
   - Start from 03-17 baseline
   - See what changed at each date step in 03-20, 03-22, 03-25

3. **Pick attack vector**:
   - Replicate SOTA locally first (validation)
   - Then pick from tier 3/4 for novel contribution (depth recurrence? diffusion?)

---

## Document Navigation

```
docs/
├── complete-techniques-mapping.md         ← START HERE (full reference)
├── local-techniques-analysis.md           ← Local submission details
├── top-prs-analysis.md                    ← GitHub PR innovations
├── leaderboard-guide.md                   ← Leaderboard URLs + history
├── techniques-analysis.md                 ← BPB impact analysis
├── winning-techniques.md                  ← Tier strategy
├── roadmap.md                             ← Submission phases
├── knowledge-required.md                  ← 9 domains to learn
├── config.py                              ← Your tunable variables
└── ANALYSIS_SUMMARY.md                    ← This file

records/
├── track_10min_16mb/                      ← 21 ranked submissions (CODE + LOGS)
└── track_non_record_16mb/                 ← 3 non-record experiments (CODE + LOGS)

.firecrawl/
├── pr-*.md                                ← 10 GitHub PRs (full content)
└── leaderboard-urls.json                  ← 60 leaderboard URLs
```

---

**Status**: Analysis complete. Ready for experimentation!
