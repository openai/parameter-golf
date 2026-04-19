# Next Session

## Phase

**Updated 2026-04-19.** Session 3 closed with a clean negative result on the posterior corrector. PR4 non-record evidence package submitted upstream at `records/track_non_record_16mb/2026-04-19_pr1610_reproduction_corrector_negative/`. Branch `submission/pr1610-corrector` pushed. Awaiting upstream reviewer signal.

## Immediate next action

Read only:
1. `AGENTS.md`
2. `docs/campaign/AGENT_SYNC.md`
3. `CLAUDE.md`
4. `docs/codex-memory/decisions.md` (section "Non-record PR4 submitted...")

**Primary task for next session: do nothing on the PR for 48 hours unless a reviewer comments.** Per discipline note from Session 3 review: self-comments from the submitter within 48 hours of opening signal to reviewers that the PR didn't do its own work. The submission stands on its own.

**Secondary task (unblocked by PR status): Fallback Cascade Level 1A on preserved Gate A seed-0 checkpoint.** Export-only quantization refinement (`#1586`-style `clip_sigmas` tuning + int7 embeddings), 1-2 requant-only runs, $6-12, 1-day time-box. Kill criterion: < 0.001 BPB gain or artifact exceeds cap.

Preconditions for Level 1A:
- Pod `utwe9wnuze72ds` must be terminated (idle from Session 3).
- New pod required (Level 1A uses different entry path than Gate A).
- Preserved checkpoint available at `amay01/parameter-golf-pr1610-reproduction-artifacts/runs/runs_20260418_2204.tar.gz` (MD5 `caf8adf63d8c80965f6671beba95d7aa`).

**If a reviewer comments on PR4:**
- Triage within one session: factual correction → amend commit; methodological critique → respond with evidence, not defense; rejection → close cleanly and record the feedback in `decisions.md`.

Subsequent sessions after Level 1A:
- If Level 1A shows ≥ 0.002 BPB gain: consider Level 1B (MATRIX_LR tuning, requires retraining).
- If Level 1A shows < 0.001 BPB gain: park the record-hunt, treat PR4 as the terminal submission for this competition cycle.

## What happened in Session 05c-plus / 05f (MEASURED)

8xH100 result:
- sliding s64 val_bpb: `1.12557920` (anchor delta: **-0.00347**, positive)
- pre_quant EMA: `1.14186715`
- int6 roundtrip: `1.14933197`
- step_avg: `100.39 ms` (+9.02ms vs anchor, **regressed**)
- steps: `5977` (587 fewer than anchor due to throughput)
- artifact: `15,589,271` bytes

Quality-positive but throughput regressed materially. Not a seed-validation branch.

05f 8xH100 follow-up:
- sliding s64 val_bpb: `1.12660664` (**worse** than 05c-plus by `+0.00103`)
- pre_quant EMA: `1.14190308`
- int6 roundtrip: `1.15026661`
- step_avg: `100.51 ms` (no throughput recovery)
- artifact: `15,630,854` bytes (+41,583 vs 05c-plus)

05g 8xH100 follow-up:
- sliding s64 val_bpb: `1.12584234` (**worse** than 05c-plus by `+0.00026`)
- pre_quant EMA: `1.14203044`
- int6 roundtrip: `1.14963535`
- step_avg: `98.67 ms` (modest recovery)
- artifact: `16,475,467` bytes (**over the cap** on the old export path)

Conclusion: 05f and 05g are both negative follow-ups. Do not continue the local tweak line.

## Current diagnostic workflow

Artifacts:
- `diagnostics/2026-03-31_05c_plus/final_model.pt`
- `diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- `diagnostics/2026-03-31_05c_plus/train.log`
- `diagnostics/2026-03-31_05c_plus/diagnostics_float.txt`
- `diagnostics/2026-03-31_05c_plus/diagnostics_int6.txt`

Utility:
- `scripts/diagnostics/diagnose_weights.py`
- `scripts/diagnostics/compress_probe.py`

Approaches:
- single-checkpoint weight statistics:
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt`
- float-vs-int6 comparison on the same checkpoint:
  - `python scripts/diagnostics/diagnose_weights.py final_model.pt final_model.int6.ptz`
- export-path feasibility:
  - `python scripts/diagnostics/compress_probe.py diagnostics/2026-03-31_05c_plus/final_model.int6.ptz`
- interpret these reports together with the measured 05c-plus / 05f / 05g logs before choosing the next larger fork

Scope:
- useful for weight norms, outliers, sparsity, SmearGate / VE / Bigram scale inspection, and float-vs-int6 damage proxies
- not sufficient for activation-level claims by itself

## Files to read first

1. `docs/campaign/AGENT_SYNC.md`
2. `CLAUDE.md`
3. `scripts/diagnostics/compress_probe.py`
4. `diagnostics/README.md`
5. `records/track_non_record_16mb/2026-03-30_training_bundle_plus/train_gpt.py`

## Next session — PR #1610 RunPod execution

Use the committed RunPod pipeline on branch `submission/pr1610-corrector`
at SHA `876bb3603eaeb9213d23e555645b49ed30d66738`.

Required precondition:
- `00_verify_pod.sh` must confirm the warmup-fix ancestor
  `a33191f572430566b88c4d61badb0369e1e6f9a3` is present in history.

Session 3 operator flow:
1. Launch pod per `scripts/runpod_pipeline/pod_launch.md`
2. Start automated stages 0–3:
   - `bash scripts/runpod_pipeline/run_all.sh`
3. Review fork decision:
   - `bash scripts/runpod_pipeline/04_decide_and_proceed.sh`
4. Choose one manual Stage 4 path:
   - primary: `BEST_ALPHA=<x> BEST_ORDERS='<y>' bash scripts/runpod_pipeline/04a_gate_b.sh`
   - fallback: `bash scripts/runpod_pipeline/04b_fallback_level1a.sh`
5. Preserve artifacts before teardown:
   - `UPLOAD_TARGET="hf:<repo>:<path>" bash scripts/runpod_pipeline/05_preserve_artifacts.sh`
   - or `UPLOAD_TARGET="rsync:<user@host>:<path>" bash scripts/runpod_pipeline/05_preserve_artifacts.sh`

Do not use `s3://` for Stage 5; S3 support was removed from the script and docs.
