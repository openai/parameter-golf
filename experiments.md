# Experiment Log

All experiments conducted March 22-24, 2026 for the Parameter Golf competition.

## Results Summary

### Step 4: Architecture Exploration (March 22)

- **run1** — 11L+VR+GA+3%prune, 1xH100 — BPB 1.4795, 16.32MB — OVER SIZE
- **run2** — 11L+VR+GA+8%prune, 1xH100 — BPB 1.4823, 16.51MB — OVER SIZE (worse)
- **run3** — 11L+VR+noGA+3%prune, RTX4500 — BPB 2.4917, 16.71MB — OVER SIZE
- **run4** — 10L+VR+noGA+3%prune, RTX4500 — BPB 2.3241, 15.00MB — FITS
- **run5** — 10L+VR+GA+3%prune, RTX4500 — 15.09MB — FITS
- **run7** — 10L+VR+GA+SWA+XSA4+LateQAT, 8xH100 — **BPB 1.1492, 15.38MB — BEST (step4)**
- **run8** — 10L+VR+GA+TTT, RTX4500 — 14.93MB — TTT OOM on 32GB
- **run9** — 10L+TTT, 1xH100 — 15.17MB — TTT OOM (compiled model)
- **run10** — TTT fix test, RTX4500 — 15.20MB — Still OOM on 32GB
- **run12** — TTT fix verified, 1xH100 — 15.16MB — TTT OOM fix works

**Key findings:** 11L doesn't fit in 16MB without GPTQ. 10L with VR+GA is the sweet spot. TTT needs H100 80GB.

### Step 5: Screening Experiments (March 23)

- **S1** — Baseline 10L, RTX4500 — BPB 1.4266, 15.17MB — Reference (1 shard)
- **S3** — BigramHash 10240, RTX4500 — BPB 1.4201, 15.82MB — Slight improvement
- **S4** — Bigram 10240 dim64, RTX4500 — BPB 1.4574, 15.49MB — Worse than S3
- **S5** — 11L, RTX4500 — BPB 1.4960, 16.82MB — OVER SIZE
- **S7** — 640d 8L, RTX4500 — BPB 1.5021, 18.70MB — OVER SIZE
- **H1** — Baseline, 1xH100 — BPB 1.5980, 15.13MB — Reference
- **H2** — Bigram 10240, 1xH100 — BPB 1.5637, 16.22MB — OVER SIZE
- **H3** — EMA, 1xH100 — BPB 1.5429, 15.11MB — Better than baseline
- **H4** — EMA+Bigram10240, 1xH100 — BPB 1.5249, 16.05MB — OVER SIZE
- **H5** — EMA+TTT, 1xH100 — 15.29MB — TTT incomplete
- **F1** — EMA, 8xH100 — BPB 1.3453, 15.43MB — EMA baseline
- **F2** — EMA+VR+GA+XSA4+QAT, 8xH100 — BPB 1.3357, 15.70MB — Full stack + EMA
- **F3** — F2 + TTT SGD 10ep, 8xH100 — BPB 1.4523, 15.59MB — TTT WORSE (rank sync bug!)

**Key findings:** TTT with SGD made BPB worse due to missing rank sync bug. EMA worse than SWA. BigramHash 10240 pushes over size limit.

### Step 6: TTT Rewrite + Competitive Submission (March 23)

- **E1** — TTT 10ep AdamW Cosine, 8xH100 — BPB 1.1579, 15.60MB, eval 379s — OK (TTT barely helps)
- **E2** — TTT 30ep AdamW Cosine, 8xH100 — BPB 1.1239, 15.59MB, eval 743s — OVER eval budget
- **E3** — TTT 20ep + LeakyReLU, 8xH100 — BPB 1.1389, 15.49MB, eval 561s — OK
- **E4** — Muon TTT 20ep + LeakyReLU, 8xH100 — BPB 1.2206, 15.29MB, eval 570s — Muon TTT worse
- **E5** — TTT 22ep + LeakyReLU, 8xH100 — **BPB 1.1354, 15.35MB, eval 603s — BEST rule-compliant**

**Key findings:**
- Rewrote TTT: batch 32 seqs/GPU (was 1 seq x 256 chunks = 500x slower)
- Per-step grad sync via all_reduce (was post-TTT param sync = broken)
- LR 0.0005 (was 0.008 = 16x too high), WD 0.0 (was 0.01)
- Per-layer LR: 3x proj, 0.5x fc (matches competition PRs #481/#518)
- Per-step cosine decay (not per-epoch)
- Muon TTT failed: loss stayed high (2.01 vs AdamW's 1.90)

### Step 7: Push to Sub-1.10 (March 24)

- **E6** — 11L+GPTQ (broken path), RTX4500 — 16.49MB — GPTQ 0 layers (path bug)
- **E6b** — 11L+GPTQ (path fixed), RTX4500 — Pod preempted
- **E7** — 11L+GPTQ+TrigramHash+TTT30, 8xH100 — **BPB 1.1261, 17.08MB, eval 775s — Best BPB but OVER size+time**

**Key findings:**
- GPTQ calibration had two bugs: doubled data path + dtype mismatch (uint16 tokens + float/bf16 matmul)
- Both fixed: autocast during calibration + correct data path
- GPTQ successfully calibrated 78 layers on 11L model
- 11L + GPTQ still 17.08MB (1.08MB over) — needs int5-all or better compression
- TrigramHash working: reuses BigramHash embedding table, zero extra parameters
- TTT 30ep with batch=64: 562s TTT + 213s eval = 775s (over 600s eval budget)

---

## Bug Fixes Found During Experiments

- **TTT rank sync missing** — TTT made BPB worse on multi-GPU — Fixed in Step 6 (E1)
- **TTT chunk-based too slow** — 360s/epoch with all params — Fixed in Step 6 (rewrote to batched)
- **TTT LR too high (0.008)** — Poor convergence — Fixed in Step 6 (changed to 0.0005)
- **TTT WD=0.01** — Unnecessary regularization — Fixed in Step 6 (changed to 0.0)
- **GPTQ data path doubled** — 0 layers calibrated — Fixed in Step 7
- **GPTQ dtype mismatch** — Crash during calibration — Fixed in Step 7
- **GPTQ no autocast** — Float/BFloat16 mismatch in hooks — Fixed in Step 7

## Cost Tracking

- **Mar 22** — 8xH100 spot + RTX4500, ~3h, ~$45 — run1-run12
- **Mar 23** — RTX4500 + 1xH100 + 8xH100, ~4h, ~$30 — S1-S7, H1-H5, F1-F3
- **Mar 23** — 8xH100 spot (2 preempted), ~1.5h, ~$25 — E1-E5 (preemptions wasted ~$8)
- **Mar 23** — 8xH100 on-demand x 2, ~0.7h, ~$30 — E4+E5 parallel
- **Mar 24** — RTX4500 (preempted) + 1xH100 + 8xH100 on-demand, ~1h, ~$25 — E6, E6b, E7
- **Total** — ~10h, ~$155, 31 experiments