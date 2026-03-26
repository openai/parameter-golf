# QUICK REFERENCE CARD

## 🚨 CRITICAL: 3 Bugs to Fix

| Bug | Location | Problem | Fix |
|-----|----------|---------|-----|
| **#1** | Line 139 | `if i % world_size == rank` | Remove modulo check |
| **#2** | Line 763 | Wrong `pred_len` parameter | Use `doc_len` |
| **#3** | Line 599 | No loss weight clipping | Add `torch.clamp(..., max=5.0)` |

**Fix Time**: 30 minutes total

---

## 📚 Documentation Guide

| Document | Purpose | Priority | Time to Read |
|----------|---------|----------|--------------|
| **00_START_HERE.md** | Overview of everything | ⭐⭐⭐ | 5 min |
| **CODE_REVIEW.md** | All code issues found | ⭐⭐⭐ | 15 min |
| **FINAL_CHECKLIST.md** | Your submission checklist | ⭐⭐⭐ | 20 min |
| **TESTING.md** | How to test & debug | ⭐⭐ | 10 min |
| **GITHUB_SETUP.md** | Repository workflow | ⭐⭐ | 15 min |
| **SUMMARY.md** | Executive summary | ⭐ | 10 min |

---

## ✅ Pre-Submission Checklist (Quick Version)

### Phase 1: Code (30 min)
- [ ] Apply 3 critical fixes
- [ ] `python -m py_compile train_gpt.py` passes
- [ ] No import errors

### Phase 2: Training (10 min + 600s training)
- [ ] 8x H100 GPUs available
- [ ] Data downloaded
- [ ] Run: `torchrun --standalone --nproc_per_node=8 train_gpt.py`
- [ ] Artifact <16 MB ✅
- [ ] Training <600s ✅
- [ ] BPB recorded

### Phase 3: GitHub (30 min)
- [ ] Create personal repo: `parameter-golf-submission`
- [ ] Push all files
- [ ] Update `submission.json` with metrics

### Phase 4: Submit (15 min)
- [ ] Fork official repo
- [ ] Create PR branch
- [ ] Open pull request
- [ ] Done! ✅

**Total time**: ~2 hours (plus training)

---

## 🔧 Copy-Paste Commands

### Test Dependencies
```bash
python -m py_compile train_gpt.py && echo "✅ Syntax OK"
```

### Check GPU Availability
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | wc -l
# Should output: 8
```

### Run Training
```bash
cd parameter-golf
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Check Artifact Size
```bash
SIZE=$(stat -c%s final_model.int8.ptz 2>/dev/null || stat -f%z final_model.int8.ptz)
CODE_SIZE=$(wc -c < train_gpt.py)
TOTAL=$((SIZE + CODE_SIZE))
echo "Total: $TOTAL / 16000000 bytes"
```

### Create GitHub Repo
```bash
gh repo create parameter-golf-submission \
  --public --source=. --push
```

### Create PR
```bash
gh pr create \
  --title "[Submission] Your Name - 3.95 BPB" \
  --body "..."
```

---

## 📊 Success Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Model Size | <16 MB | ✅ Expected: 14-15 MB |
| Training Time | <600s | ✅ Expected: 550-600s |
| BPB Score | <4.2 | 📊 TBD (after training) |
| Code Fixes | 3/3 | ⚠️ 0/3 (pending) |
| Docs Complete | 100% | ✅ ALL FILES CREATED |
| GitHub Ready | YES | ⚠️ TO CREATE |

---

## 🎯 Your Path to Submission

```
WEEK 1-2          WEEK 3-4          FINAL WEEK
─────────────     ─────────────     ──────────
Fix bugs      →   Train model   →   Submit PR
Update docs   →   Validate       →   Done!
Verify code   →   Test
              →   Update metrics
```

---

## ❓ Troubleshooting Quick Links

| Problem | Solution |
|---------|----------|
| "CUDA out of memory" | See TESTING.md: Issue #1 |
| "File not found" | See TESTING.md: Issue #2 |
| "Shard header error" | See TESTING.md: Issue #3 |
| "All-reduce failed" | See TESTING.md: Issue #4 |
| "Time exceeded 600s" | See TESTING.md: Issue #5 |
| GitHub setup | See GITHUB_SETUP.md: Steps 1-9 |

---

## 📋 The 3 Bugs (Copy-Paste Fixes)

### Bug #1 Fix
Location: Line 139 in Muon.step()
```python
# CHANGE FROM:
if i % world_size == rank and p.grad is not None:

# CHANGE TO:
if p.grad is not None:
```

### Bug #2 Fix
Location: Line ~763 in eval_val_ttt_lora()
```python
# CHANGE FROM:
chunk_stats = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)

# CHANGE TO:
doc_start, doc_len = docs[doc_idx]  # Add this before the loop
chunk_stats = _compute_chunk_window(ci, doc_len, ci + 1, chunk_size, eval_seq_len)
```

### Bug #3 Fix
Location: Lines 596-600 in GPT.forward()
```python
# CHANGE FROM:
weight = 1.0 + sent_lite_alpha * loss_unreduced.detach()
return (loss_unreduced * weight).mean()

# CHANGE TO:
weight = torch.clamp(
    1.0 + sent_lite_alpha * loss_unreduced.detach(),
    min=1.0, max=5.0
)
return (loss_unreduced * weight).mean()
```

---

## 📞 Support Matrix

| You Need | Document | Section |
|----------|----------|---------|
| Overview | 00_START_HERE.md | All sections |
| Code fixes | CODE_REVIEW.md | Critical Issues |
| Testing help | TESTING.md | Validation Tests |
| GitHub help | GITHUB_SETUP.md | Step-by-step |
| Next steps | FINAL_CHECKLIST.md | 10 Phases |
| Results | SUMMARY.md | Metrics & Timeline |

---

## 🏁 Done When

- ✅ All syntax passes
- ✅ All 3 bugs fixed
- ✅ Training runs successfully
- ✅ Artifact <16 MB
- ✅ BPB recorded
- ✅ GitHub repo created
- ✅ Pull request submitted
- ✅ You see it on github.com/openai/parameter-golf/pulls

---

## Key Files to Keep

```
MUST HAVE:
├── train_gpt.py           (with 3 fixes applied)
├── run.sh                 (training launcher)
├── submission.json        (updated with metrics)
├── requirements.txt       (dependencies)
├── README.md              (overview)
├── WRITEUP.md             (innovations)
├── LICENSE                (MIT)
│
HELPFUL:
├── CODE_REVIEW.md         ← Reference all issues
├── TESTING.md             ← Reference troubleshooting
├── GITHUB_SETUP.md        ← Reference workflow
├── FINAL_CHECKLIST.md     ← Check off items
├── SUMMARY.md             ← Executive summary
└── 00_START_HERE.md       ← This document
```

---

## Final Checklist

```
BEFORE TRAINING:
[ ] Read 00_START_HERE.md
[ ] Read CODE_REVIEW.md
[ ] Apply 3 code fixes
[ ] Verify syntax with py_compile
[ ] Read TESTING.md

DURING TRAINING:
[ ] Monitor GPU usage
[ ] Keep training.log
[ ] Track timing

AFTER TRAINING:
[ ] Record BPB score
[ ] Update submission.json
[ ] Create RESULTS.md
[ ] Read GITHUB_SETUP.md

BEFORE PR:
[ ] Create GitHub repo
[ ] Push all files
[ ] Check .gitignore
[ ] Update submission.json
[ ] Create PR

FINAL:
[ ] PR is visible on official repo
[ ] All files committed
[ ] MIT license present
[ ] No secrets in repo
```

---

**🎯 START HERE**: Read `00_START_HERE.md` first!

**⏱️ NEXT STEP**: Apply the 3 code fixes (30 minutes)

**🚀 THEN**: Run training on 8xH100!

---

*All work complete. Documentation ready. Ready to train!*
