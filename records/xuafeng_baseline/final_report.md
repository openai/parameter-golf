# Parameter Golf - Final Experiment Report

## Summary of All Runs

### Run 1: Baseline (1x H100)
| Metric | Value |
|--------|-------|
| try_id | `baseline_sp1024_try1` |
| GPU | 1x H100 SXM |
| val_bpb | **1.3274** |
| Steps | 1404 / 20000 (wallclock cap) |
| Artifact | 13.8 MB |

### Run 2: QAT + Trigram (8x H100, seed 42)
| Metric | Value |
|--------|-------|
| try_id | `qat_int5_trigram_seed42` |
| GPU | 8x H100 SXM |
| val_bpb | **1.14423** |
| Steps | 6614 |
| Artifact | 16.2 MB (**over limit**) |

### Run 3: QAT only (8x H100, seed 1337)
| Metric | Value |
|--------|-------|
| try_id | `qat_int5_notrigram_seed1337` |
| GPU | 8x H100 SXM |
| val_bpb | **1.14476** |
| Steps | 6649 |
| Artifact | 15.8 MB |

### Run 4: QAT only (8x H100, seed 2024) — INCOMPLETE
Pod terminated at step 4500. Last val_bpb: 1.2166.

### Run 5: TTT on #1 (8x H100) — NOT RUN
Pod terminated before training started. Insufficient balance for 8x H100 ($4 remaining, need ~$7).

---

## Leaderboard Comparison

| Rank | Entry | val_bpb | Author |
|------|-------|---------|--------|
| 1 | 10L Int5-MLP + BigramHash | **1.14276** | thwu1 |
| — | Our QAT (seed 42 + trigram) | 1.14423 | xuafeng |
| — | Our QAT (seed 1337) | 1.14476 | xuafeng |
| 2 | Int6 MLP3x + SmearGate | 1.14582 | Raahil Shah |
| 3 | 11L MLP3x + Int6 QAT | 1.15020 | aruniyer |

**Our QAT runs rank between #1 and #2** but did not beat #1.

---

## Ideas Tested and Results

### 1. QAT + Int5/Int6 — DISPROVEN
- **Hypothesis**: QAT would reduce quantization penalty by 40-60%
- **Result**: QAT added ~0.002 BPB penalty vs #1
- **Why**: Post-training quantization + SWA acts as beneficial regularization. QAT removes this benefit.

### 2. TrigramHash — NEGLIGIBLE
- **Hypothesis**: 3-token patterns would add ~0.001 BPB improvement
- **Result**: ~0.0005 BPB (within noise), pushed artifact over 16MB
- **Why**: BigramHash(10240) already captures most local patterns

### 3. TTT (Test-Time Training) — IMPLEMENTED, NOT TESTED
- **Implementation**: Complete in `train_gpt_ttt.py`
- **Approach**: LoRA rank-8 adapters on Q/V projections + LM head, trained per-document during eval
- **Rationale**: #1 uses only 170s of 600s eval budget; TTT uses the remaining 430s
- **Expected**: -0.003 to -0.010 BPB improvement based on TTT entry #9's results
- **Status**: Ready to run when budget is available (~$7 needed for one 8x H100 run)

---

## Cost Breakdown

| Pod | Config | Duration | Cost |
|-----|--------|----------|------|
| parameter-golf-h100 | 1x H100 | ~28 min | ~$1.26 |
| pgolf-experiments | 1x H100 | ~5 min | ~$0.22 |
| pgolf-8xh100-qat | 8x H100 | ~50 min | ~$17.93 |
| pgolf-ttt (terminated) | 8x H100 | ~5 min | ~$1.79 |
| pgolf-ttt-final (terminated) | 8x H100 | ~2 min | ~$0.72 |
| **Total spent** | | | **~$21.92** |
| **Remaining balance** | | | **$4.04** |

---

## Files Produced

| File | Description |
|------|-------------|
| `parameter_golf_report.md` | Initial baseline training report |
| `parameter_golf_process.md` | End-to-end RunPod process guide |
| `parameter_golf_next_ideas.md` | Original ideas (pre-experiment) |
| `parameter_golf_experiment_report.md` | QAT experiment results |
| `parameter_golf_final_report.md` | This report |
| `records/xuafeng_qat_int5_trigram/train_gpt.py` | QAT + Trigram modified script |
| `records/xuafeng_qat_int5_trigram/train_gpt_ttt.py` | TTT implementation (ready to test) |

---

## Next Steps (When Budget Available)

1. **Top up RunPod balance** to at least $10
2. **Run TTT experiment**: `train_gpt_ttt.py` on 8x H100 with seeds 42, 1337, 2024
3. If TTT beats #1: prepare submission with README, logs, submission.json
4. If TTT doesn't beat #1: the leaderboard is likely saturated at this architecture scale

### To run TTT:
```bash
# On 8x H100 pod:
cd /workspace/parameter-golf
# Upload train_gpt_ttt.py as train_gpt.py
SEED=42 RUN_ID=ttt_seed42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```
