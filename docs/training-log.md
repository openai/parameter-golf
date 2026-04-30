# Training Log

**Purpose**: Central append-only log of all model training runs.

**Format**: Each model's runs are recorded with parameters, results, and what changed vs SOTA.

**Auto-updated by**: `models/model_NNN/para.py --launch` (logs START before training, END after)

---

## How to Record a Run

After training completes, manually add a row to the table below:

| model_id | seed | date | val_bpb | artifact_mb | steps | params (diff from SOTA) | notes |
|----------|------|------|---------|-------------|-------|------------------------|-------|
| model_001-314 | 314 | 2026-04-14 | 1.1147 | 15.86 | 6927 | (SOTA baseline, no diff) | Replica validation |
| model_001-42  | 42  | 2026-04-14 | 1.1144 | 15.98 | 6922 | (SOTA baseline, no diff) | Replica validation |
| model_001-999 | 999 | 2026-04-14 | 1.1148 | 15.88 | 6917 | (SOTA baseline, no diff) | Replica validation |

---

## Template

When adding a new run, use:

```
| model_NNN-SEED | SEED | YYYY-MM-DD | X.XXXX | XX.XX | NNNN | param1=val, param2=val | experiment notes |
```

**Columns**:
- **model_id**: `{MODEL_ID}-seed{SEED}` from `para.py`
- **seed**: Which seed (314, 42, 999, or custom)
- **date**: Run date (YYYY-MM-DD)
- **val_bpb**: Validation BPB from final epoch (find "Sliding BPB" in train output)
- **artifact_mb**: Model artifact size in MB (from submission.json)
- **steps**: Training steps taken before hitting 600s wall-clock limit
- **params (diff from SOTA)**: Only list what changed vs SOTA. Format: `NUM_LAYERS=10, BIGRAM_VOCAB_SIZE=4096`
- **notes**: Experiment hypothesis or observations

---

## Run Records

| model_id | seed | date | val_bpb | artifact_mb | steps | params (diff from SOTA) | notes |
|----------|------|------|---------|-------------|-------|------------------------|-------|
| | | | | | | | |

---

## Key Insights

### Model #001 (SOTA Baseline)
- Expected: ~1.1147 BPB (3-seed mean)
- Techniques: Full GPTQ, XSA-all, BigramHash 3072×112, EMA, SWA, sliding window
- Purpose: Validation that your setup works

### Upcoming Models
- Model #002: Test hypothesis (e.g., larger bigram, different XSA layers)
- Model #003: Combine multiple changes if #002 succeeds
- etc.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Better than target |
| ⚠️  | Close to target (within 0.001 BPB) |
| ❌ | Worse than target |

---

**Generated**: 2026-04-14

**Updated by**: `models/model_NNN/para.py`
