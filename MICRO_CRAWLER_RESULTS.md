# Micro Crawler H100 Experiment Results
**2026-03-24 | 8xH100 SXM | 600s wallclock | seed 1337**

## Architecture
4 flat blocks (unique) + 2 crawler blocks x 2 loops (shared, orthogonal positions)
= 8 effective depth, 6 stored blocks, dim=640, 10H/5KV GQA, MLP 4x, TrigramHash

## Results

| Run | Description | Sliding BPB | Post-EMA | Quant Gap | Steps | ms/step | Artifact | Quant Method |
|-----|-------------|-------------|----------|-----------|-------|---------|----------|-------------|
| **Run 1** | Baseline (broken lr_mul, no gate) | **1.1377** | 1.1513 | 0.0097 | 7,694 | 78 | 16.86MB | per-row int6 |
| Run 1.5 | lr_mul fix + recursive cadence 2/4/6 | 1.1384 | 1.1520 | 0.0097 | 7,313 | 82 | 16.33MB | per-row int6 |
| Run 3 | Self-ref gate (C-only) + GPTQ | 1.1415 | 1.1575 | 0.0072 | 7,150 | 84 | 16.33MB | GPTQ Hessian |
| **Run 6** | **PD gate (EMA) + GPTQ** | **1.1375** | **1.1535** | **0.0075** | **7,076** | 85 | 16.65MB | GPTQ Hessian |

## Baselines

| Model | Sliding BPB | Quant Gap | Steps | Artifact |
|-------|-------------|-----------|-------|----------|
| Frugendorff Squared 6x2 | 1.1478 | 0.0146 | 4,390 | 15.15MB |
| GS v7 11L (legal TTT) | 1.1206 | 0.0058 | 6,990 | 15.56MB |
| XSA-11 GPTQ b64/pd002 | 1.1208 | ~0.006 | ~7,000 | 15.56MB |

## Key Findings

### Architecture
- Micro crawler beats Frugendorff by 0.010 BPB (1.1375 vs 1.1478)
- 4 unique flat blocks train cleanly — no gradient conflict
- Only 2 shared blocks → minimal quant compounding (gap 0.0075 vs 0.0146)
- ~78-85ms/step → 7,000+ steps vs Frugendorff's 4,390

### lr_mul
- Broken LR (QAT at step 2) self-corrects by step ~400 as step_ms averages down
- Fix made no measurable difference — run 1 and run 1.5 within noise

### Cadence
- Recursive cadence (2→4→6) had no effect vs broken cadence stuck at 6
- With only 2/6 blocks sharing, gradient conflict is mild — cadence unnecessary for vanilla training
- BUT: PD gate and cadence are coupled — PD needs frequent C steps for fresh consensus

### Deliberation Gate
- Gate on C-steps only (run 3): HURT pre-quant by 0.006 BPB — not enough training signal
- PD gate on all steps (run 6): neutral pre-quant (-0.002), GPTQ recovered it
- PD was 0.007 ahead mid-training (steps 5000-7000) but post-processing (EMA/distill) didn't capture lead
- Detached EMA consensus goes stale with tapered cadence

### GPTQ
- Hessian-aware GPTQ drops quant gap from 0.0097 → 0.0072-0.0075
- Crawler blocks get naturally blended Hessians from both firings during calibration
- 37/37 layers calibrated via GPTQ (0 naive fallback)

## Pending Experiments

| Run | Description | Hypothesis |
|-----|-------------|-----------|
| Run 7 | No gate + GPTQ only | Safe play: run1 pre-quant + GPTQ gap → ~1.135 |
| Run 8 | Bidirectional PD (learned ref) + fixed cadence 2 + GPTQ | Gradient flows both ways, EMA stays fresh → PD actually helps |
| Run 4 | Self-ref gate + dim=720 | Wider model, more gate signal |

## File Inventory

| File | Status |
|------|--------|
| train_gpt_micro_crawler_h100_run1_1.1377.py | FROZEN — never modify |
| run_micro_crawler_h100_run1_1.1377.sh | FROZEN — never modify |
| train_gpt_micro_crawler_h100_run2.py | GPTQ + trigram 2048 |
| train_gpt_micro_crawler_h100_run3_selfref.py | Self-ref gate (C-only) + GPTQ |
| train_gpt_micro_crawler_h100_run4_selfref_d720.py | Run3 at dim=720 |
| train_gpt_micro_crawler_h100_run5_persistent_delib.py | PD (detached EMA) + GPTQ |
| train_gpt_micro_crawler_h100_run6_best_plus_delib.py | Run1 base + PD + GPTQ |
| train_gpt_micro_crawler_h100_run7_gptq_only.py | Run1 base + GPTQ only |
| train_gpt_micro_crawler_h100_run8_pd_fixed_cadence.py | Bidirectional PD + fixed cadence 2 + GPTQ |
