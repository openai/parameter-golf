# Proxy Results

Prep-budget record campaign tracker for `2026-04-05_RecordCampaign_ARSelfGen_XSAall_Bigram3072`.

Promotion rule:
- Promote an ablation only if it improves the stock proxy by at least `0.001 BPB` while preserving artifact headroom and sane eval behavior.
- If no ablation clearly wins, keep the final `8xH100` campaign on the stock stack.

Current strongest saved `8xH100`-backed result in this file:

- `record_seed314_stage1_jaksencharles` + `record_seed314_stage2_jaksencharles`
- seed `314`
- final sliding BPB: `1.13071788`
- final artifact bytes: `15,651,808`
- interpretation: serious submission baseline and funded rerun launchpad, not a
  live-frontier record claim

| Variant | Seed | GPU | App | Diagnostic BPB | Roundtrip BPB | Sliding BPB | Artifact Bytes | Terminal Stage | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| smoke | 314 | H100:1 | `ap-vj3cEfg1a2K9SyGOa9isDj` | `3.6030` | `3.61914005` | `n/a` | `4793127` | `quantized_roundtrip_eval complete` | `Smoke path passed after fixing hidden stride=64 fallback; exits without submission sliding eval by design.` |
| stock | 314 | H100:1 | `ap-Bkk6nnKsUsB5aHUQv6Rex8` | `1.5195` | `1.96724228` | `1.95393670` | `7414667` | `submission_sliding_eval complete` | `Baseline proxy for promotion decisions. Full end-to-end path succeeded with stock SOTA-style settings on the cloned record stack.` |
| warmdown4500 | 314 | H100:1 | `ap-yopxFTaYPOkG1DoKAvZ3Hx` | `1.5427` | `2.07158499` | `stopped` | `7235707` | `submission_sliding_eval stopped` | `Rejected early to protect prep budget after worse-than-stock diagnostic and materially worse roundtrip score. Not promoted.` |
| arcalib96 | 314 | H100:1 | `ap-cqCWsDTEop12ztjy82FcMd` | `1.5175` | `stopped` | `stopped` | `n/a` | `gptq_collect_hessians complete; stopped` | `Closer to stock in training/diagnostic, but became unobservable or stalled immediately after Hessian collection. Not promoted over a clean stock baseline.` |
| final_8xh100_seed314_aborted | 314 | H100:8 | `ap-ZkPmJtc3RXKPf2zmJKOK61` | `1.1486` | `aborted` | `aborted` | `aborted` | `gptq_generate_autoregressive_data in progress; user-stopped` | `First real 8xH100 record attempt after launcher-path fix. Healthy distributed bring-up (`world_size:8`, `~120ms/step`), reached `step:4947` at the wallclock cap with `val_bpb:1.1491`, then was manually stopped during GPTQ calibration before any final quantized/sliding score was produced.` |
| smoke_itssdivo | 314 | H100:1 | `ap-d14nh9AKs2ztLkdav26CrB` | `3.6030` | `3.61895439` | `n/a` | `4792583` | `quantized_roundtrip_eval complete` | `New-account reconnect proof on workspace itssdivo. Verified recreated volume, exact tokenizer path, all shard paths, and full record smoke runtime on the new account.` |
| final_8xh100_seed314_timeout_itssdivo | 314 | H100:8 | `ap-TbzUK6YffGKQzTkgbdYrHF` | `1.1514` | `timeout` | `timeout` | `timeout` | `gptq_collect_hessians complete; launcher timeout` | `Resumed record attempt on the new account. Healthy distributed bring-up (`world_size:8`) and sustained `~127-129ms/step`, reached `step:4636 val_bpb:1.1520`, completed AR self-generation and full Hessian collection, then timed out before selective prune and final quantized/sliding evaluation.` |
| nproc1_bug_deviousdivo | 314 | H100:8 | `ap-JOYdywYe33OVQEF8XsHRDR` | `n/a` | `stopped` | `stopped` | `n/a` | `user-stopped immediately` | `Entrypoint default nproc_per_node=1 was passed to H100:8 run, causing world_size=1 (single-GPU training). Stopped immediately after detection. No useful training data produced.` |
| final_8xh100_seed314_deviousdivo | 314 | H100:8 | `ap-COAxpQkmlKWoS7r577M4yJ` | `1.1509` | `budget_exhausted` | `budget_exhausted` | `budget_exhausted` | `gptq_generate_autoregressive_data in progress; budget_exhausted` | `Fixed nproc bug (--nproc-per-node 8). Healthy distributed bring-up (world_size:8), step_avg:~125ms, torch291-cu128+FA3. Training completed: step:4707 val_bpb:1.1515, post-EMA diagnostic val_bpb:1.1509. GPTQ AR data gen ~50% done when ~$23 credit exhausted (~$60+/hr H100:8 burn rate — entire GPTQ pipeline holds H100:8, not just training). No final sliding BPB. All 3 H100:8 attempts failed to complete GPTQ within budget/timeout.` |
| twostage_smoke_stage1_deviousdivo | 314 | H100:1 | `ap-srnbIIK55ws056w6sBydIM` | `3.6033` | `n/a` | `n/a` | `106289590` | `skip_quantize complete; checkpoint_saved` | `Two-stage pipeline Stage 1 validation. SKIP_QUANTIZE=1 path works: checkpoint saved to volume at /data/parameter-golf/checkpoints/twostage_smoke_validate/final_model.pt. exit_code=0.` |
| twostage_smoke_stage2_deviousdivo | 314 | H100:1 | `ap-1DjdmQfkCtP4PHChwkjOZZ` | `n/a` | `3.61907703` | `n/a (EVAL_STRIDE=0)` | `4793500` | `quantized_roundtrip_eval complete` | `Two-stage pipeline Stage 2 validation. run_gptq.py loaded checkpoint from volume, ran full GPTQ pipeline (AR gen→Hessian→prune→roundtrip eval). Roundtrip BPB matches smoke baseline exactly. exit_code=0. Two-stage flow proven end-to-end.` |
| record_seed314_stage1_jaksencharles | 314 | H100:8 | `ap-hFJWBguKINPxP9pR3MVRpd` | `1.1501` | `n/a` | `n/a` | `n/a` | `skip_quantize complete; checkpoint_saved` | `Competition shot Stage 1 (jaksencharles account). world_size:8, step_avg:~124ms, torch291-cu128+FA3. Training completed: step:4783 val_bpb:1.1506, post-EMA diagnostic:1.1501. Checkpoint saved to /data/parameter-golf/checkpoints/record_seed314/final_model.pt. exit_code=0.` |
| record_seed314_stage2_jaksencharles | 314 | H100:1 | `ap-zDhg2OWmFK8Y7Phu7tJzxt` | `n/a` | `1.15442828` | `1.13071788` | `15651808` | `submission_sliding_eval complete` | `Competition shot Stage 2 (jaksencharles account). GPTQ pipeline on H100:1: AR gen 64 seqs in 241s → Hessian 68 layers → no pruning needed (14.93MB < 15.9MB target) → roundtrip 1.1544 → sliding 1.1307. Artifact 15.65MB (under 16MB). exit_code=0. Two-stage pipeline validated end-to-end. RESULT: 1.1307 does NOT beat merged SOTA (1.1147). Need frontier stack to go lower.` |
