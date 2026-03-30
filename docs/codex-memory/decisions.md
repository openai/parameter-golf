# Locked Decisions

## Strategy

- Do not lead with the RFN thesis idea.
- Build a strong non-TTT anchor first.
- Treat TTT as a later integration, only after the anchor is stronger.
- Treat RFN or attribution-graph work as a sidecar probe, not the main competition bet.

## Competition phase

- The root `8xH100` baseline is now the fixed reference point.
- The next `8xH100` runs must be actual model changes.
- Session 03 pre-TTT anchor work is complete at `val_bpb=1.12904446` (sliding s64) on Pegasus `8xH100`.
- Session 04 targeted delta sweep is closed.
- Session 05 mainline is now **GPTQ correctness first**, not throughput audit first.

## Session 03 decisions

- Session 03 anchor uses SDPA not FA3. The donor record used `flash_attn_3_func`, but the anchor port kept `scaled_dot_product_attention` to avoid introducing an untested kernel dependency in the first anchor run. This is a deliberate conservatism, not an oversight.
- NTK RoPE with `train_seq_len=1024` confirmed as deliberate. The anchor sets `rope_train_seq_len=1024` for NTK-aware scaling even though `TRAIN_SEQ_LEN=2048`. This is intentional and matches the donor record behavior.
- Throughput is a plausible bottleneck, but not the only remaining gap. Session 03 finished at `91.37 ms/step`, but the pre-quant to roundtrip gap (`1.14472403 -> 1.15247273`) means export-side work still deserves isolated measurement.
- NGC container + fscratch confirmed as optimized Pegasus path. The NGC 26.03 container with `/fscratch` for data staging avoids `/netscratch` I/O bottlenecks and resolves OOM issues from container-level overhead.

## Hardware

- Pegasus `8xH100` is now the primary execution base.
- Launch Pegasus multi-GPU work with Slurm-native `srun`, not `torchrun --standalone`.
- Force `--nodes=1` on challenge-shaped `8xH100` runs.
- RunPod stays reserved for final validation or granted credits.

## Workflow

- Keep competitive experiments in self-contained folders under `records/track_non_record_16mb/YYYY-MM-DD_<tag>/`
- Do not modify existing public record folders
- Document every run with manifests and experiment summaries
- Prefer additive, well-understood public techniques over speculative novelty
- Keep Session 04 deliberately narrow: one isolated delta per run, no stacked backend/export/model bundles
- Do not use `| tail -1` on Pegasus training jobs.
- Use `PYTHONUNBUFFERED=1` or `python -u` for Pegasus logs.
- For competition ports, use source priority: PR code first, local repo second, papers/web third.
- When a post-training export path breaks, debug it on the same checkpoint before spending more H100 time on retraining.

## Session 04 decisions

- GPTQ-lite percentile clip search rejected — marginal BPB regression + artifact cap violation. Export gap is not caused by clip suboptimality. Sliding s64 val_bpb `1.12941356` vs anchor `1.12904446` (+0.00036910), artifact `16219752` bytes exceeds `16000000` cap. Anchor int6+zstd with fixed row-max remains the viable export path.
- LeakyReLU^2 classified as neutral/tie — sliding s64 val_bpb `1.12904123` vs anchor `1.12904446` (-0.00000323), effectively zero. Pre-quant and roundtrip both slightly better. Artifact `168356` bytes smaller. But step time `+0.72 ms` slower, costing `53` steps. Not a standalone graduating delta. Keep as a possible stack component for artifact headroom or when combined with a throughput-positive change. Measured anchor comparison used `enable_math_sdp(True)` — isolation preserved correctly.
- Session 04 ends at `1 failed + 1 neutral`. Do not force a Delta 3 by default.

## Session 05 decisions

- TTT is parked as an execution target until Phase 1 (FA3) and Phase 2 (Full Hessian GPTQ) are measured.
- The legality audit remains useful background, but current execution focus is stronger pre-TTT base + throughput.
- FA3 is back in scope as a deliberate Session 05 throughput investigation, not as an anchor bring-up risk.
- The current saved-container FA3 runtime is rejected as a throughput path. It is slower and worse than the Session 03 anchor.
- Any further FA3 work is gated on vendor-tuned NGC runtime compatibility.
- Throughput-first is no longer the main strategic frame. Current frontier evidence says quality-first improvements matter more than raw ms/step.

### Session 05 audit decisions (2026-03-29)

1. **2026-03-22 record is the primary first-wave porting reference** — it uses the same CastedLinear/DDP/standard-Muon architecture as our anchor. Use it for FA3, VE128, SWA, warmdown 3500, and Late QAT.
2. **2026-03-23 #1 record is the TTT reference** — use it for score-first TTT protocol porting only.
3. **FA3 is the first implementation target** — leading hypothesis for largest throughput contribution, architecturally independent of Parameter Banking.
4. **Parameter Banking and Parallel Muon are second-wave** — 2026-03-22 achieves 1.1233 without them.
5. **LeakyReLU² re-test is gated on FA3** — the throughput-coupling hypothesis (Session 04 Delta 2 neutrality caused by +0.72ms eating training steps) must be tested, not assumed.
6. **Lane A (isolated deltas) is the default** — switch to Lane B (bundled reproduction) only if time pressure or slow progress demands it.
7. **Score-first TTT appears compliant** — matches PR #461 public precedent; torch.inference_mode() guards provide hard scoring-phase statefulness guarantee. Not a formal ruling.
8. **FA3 microbenchmark is sufficient to justify FW-1** — direct FA3 on `25.02` + wheel beat SDPA flash by `11.44x` in the isolated attention benchmark. This is kernel-only evidence, not an end-to-end training claim.
9. **Container split is now explicit** — keep NGC `26.03` as the standard stable path, but use the saved Pegasus `25.02` FA3 container for the explicit FA3 experiment path.
10. **No ad hoc FA3 job installs once the container exists** — build `/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh` once, then reuse it.
11. **`--no-deps` is rejected for FA3 on stock 25.02** — the import fails against the bundled PyTorch ABI (`aoti_torch_abi_version` missing).
12. **Keep the exact FA3 wheel filename** — shortened wheel names break pip compatibility parsing.
13. **Current saved-container FA3 path is a clean negative result** — `92.67 ms/step` and sliding s64 `1.12958984` are both worse than the anchor (`91.37 ms`, `1.12904446`). Do not rerun it as-is.
14. **The packaging problem is part of the research problem** — the microbenchmark win did not survive replacing the vendor-tuned NGC torch stack with the pip-installed generic stack.

## Session 05b decisions

- Full Hessian GPTQ selected as Phase 2 implementation target based on competitive analysis: all 4 top PRs (#634, #1019, #1060, #1072) use the same core GPTQ algorithm with identical hyperparameters (block_size=128, percdamp=0.01, actorder=True).
- Post-training calibration (not online accumulation) chosen as the Hessian collection method — simpler, proven in PRs #634 and #1060.
- 128 calibration sequences from training data (not validation) — matches prompt budget, avoids leakage.
- `clip_percentiles=[1.0]` only was a deliberate conservative start — but PR analysis revealed working PRs (#634, #1060) actually run FULL GPTQ loop 5× with `[0.9990, 0.9995, 0.9999, 0.99999, 1.0]` and keep best MSE. This is NOT the same as GPTQ-lite's percentile search (which changed scales without error compensation). Multi-percentile search with full GPTQ should be added after fixing the core bug.
- Working PRs use symmetric clamping `[-31, 31]` in export, not `[-32, 31]`. This alignment is now landed locally in the repaired Session 05b code.
- Export path restructured for rank-0-only GPTQ: Hessian collection + quantization + file write inside `if master_process:`, barrier, then all ranks read file for eval. Avoids undefined `hessians` on non-master ranks.
- **1xH100 smoke test revealed correctness bug**: roundtrip gap 0.212 BPB (27x worse than anchor). GPTQ pipeline mechanics work but quantized weights reconstruct very poorly. Must debug before 8xH100 run.
- Standard NGC 26.03 container used (no FA3 dependency) — confirmed correct, no container issues.
- **Strategic pivot: quality > throughput** — PR #1089 (1.1086 BPB leader) uses NO FA3, runs at 93ms/step (slower than our anchor), wins purely on model quality innovations (Turbo-Muon, EngramLite, mixed-precision GPTQ, brotli+byte-shuffle, 3.5x MLP). Throughput is nice-to-have, not the priority.
- **PR #1089 is the new frontier reference** (was PR #1060 at 1.1122). Update gap estimates accordingly.
- **Top-down entry rule** for leaderboard means threshold-crossing, not rank-climbing. Must beat current #1 to enter.
- The first Session 05b smoke is a **clean GPTQ correctness failure**, not a meaningful quality result.
- The `1xH100` smoke's training-side numbers are not comparable to the `8xH100` anchor because `WORLD_SIZE` changes `grad_accum_steps`.
- Missing multi-percentile search and symmetric clamp are confirmed divergences from working PRs, but are **not yet proven** to be the sole root cause of the catastrophic roundtrip gap.
- The safest current diagnosis is that the local GPTQ quantizer drifted too far from the known-good PR implementation.
- No more `8xH100` GPTQ runs until the export path passes a same-checkpoint A/B sanity check and a corrected `1xH100` smoke.
- PR-code diff isolated one concrete loop bug: the local within-block residual update used `W_block[:, j + 1:]`, while PRs `#634`, `#1019`, and `#1060` use `W_block[:, j:]`.
- A PR-grounded GPTQ repair is now landed locally:
  - within-block residual update matches the PR loop
  - 5-percentile reconstruction search is in place
  - symmetric `[-31, 31]` clamp is in place
  - `_classify_param` now targets only block `attn` / `mlp` weights, excluding `bigram.proj`
  - export writes `gptq_layer_diagnostics.json` with legacy-rowmax vs percentile-naive vs GPTQ per-layer MSE
- Because no saved checkpoint exists in the repo and this local shell lacks `torch`, the repair is currently code-reviewed and syntax-checked only. The next gate remains same-checkpoint export-only verification.
- The first server replay after that repair still showed `worse_than_legacy_rowmax=66` and `worse_than_percentile_naive=66`.
- That makes the remaining bug look systematic, not like a small number of bad layers.
- Same-checkpoint replay ablations are now measured:
  - `replay_ref`: `1.82064983 -> 2.15605819`, gap `+0.33540836`
  - `replay_noact`: `1.82064982 -> 2.21586588`, gap `+0.39521606`
  - `replay_noact_full`: `1.82064982 -> 2.21590301`, gap `+0.39525319`
- `actorder=False` made the result worse, so `actorder` is not the root cause.
- `block_size=1536` was effectively identical to `block_size=128` once `actorder=False`, so block partitioning is not the root cause either.
- The next debug target should move upstream to Hessian construction / interpretation and away from more inner-loop or block-size tuning.
- Export-only replay mode is now part of the Session 05b toolchain so future ablations can load `final_model.pt` directly and vary:
  - `GPTQ_ACTORDER`
  - `GPTQ_BLOCK_SIZE`
  - `GPTQ_CALIBRATION_SAMPLES`
  - `EXPORT_TAG`
- A debug-only replay switch now exists:
  - `EXPORT_SKIP_SLIDING_EVAL=1`
  - it only applies to export-only replay runs
  - it skips the slow submission-style sliding-window eval after `final_int6_roundtrip_exact`
- The public merged record bar moved on 2026-03-30: PR `#1019` is now the official merged #1 at `1.1147` BPB, so any official entry now needs to beat that line by at least `0.005` nats with `p < 0.01`.
- The 2026-03-30 Hessian-path patch (`9cea7e9`) was a real test, not a fix:
  - `replay_ref_hfix`: `1.82064877 -> 2.15770170`, gap `+0.33705293`
  - `gptq_diag`: still `66/66` worse than both naive baselines
  - conclusion: forward-hook + average+damp alignment is insufficient on top of the current local rewrite
- Do not spend more time on small Hessian nudges. The next implementation step should be a more faithful transplant of one complete working PR Hessian/quantization slice.

## Session 05c-plus decisions (2026-03-30)

1. **05c-plus replaces the original 05c bundle.** The original 05c prompt (`docs/campaign/prompts/session_05c_training_bundle.md`) specified XSA-all + VE128 + SWA + warmdown3500 and excluded LeakyReLU². The actual 05c-plus bundle is XSA-all + VE128 + warmdown3500 + LeakyReLU² and excludes SWA. Rationale:
   - SWA dropped: both PR #1019 and #634 collect SWA snapshots but only apply EMA at export — SWA is dead code in both references. Including it would be cargo cult.
   - LeakyReLU² added: aligns with PR #1019 architecture and is the leading hypothesis for why GPTQ fails on the current anchor (relu creates sparse Hessians, leaky_relu does not). Including it in the training bundle enables future GPTQ replay without retraining.
2. **Base is Session 03 anchor, not Session 05b.** The 05c prompt allowed stacking on 05b if GPTQ graduated. GPTQ is parked after 7 ablations, so the base is the clean anchor (`records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py`). This keeps the diff narrow and avoids dragging GPTQ debug complexity into a training run.
3. **GPTQ replay requires a separate merge step.** The 05c-plus training script adds VE parameters and changes the MLP activation. The parked Session 05b GPTQ script has the old architecture. A 05c-plus checkpoint cannot be replayed through the current 05b export path without porting VE128 + LeakyReLU² into the GPTQ script first. This is Phase 2 work, gated on 05c-plus training results.
4. **VE proj uses orthogonal init (not zero init).** The reference implementation's `nn.init.zeros_` in the VE constructor is dead code — overwritten by `_init_weights` with orthogonal + proj scaling. Our implementation matches: no `_zero_init` flag on VE proj, so `_init_weights` applies `orthogonal_(gain=1.0) * (1/sqrt(22))`. VE output is still small at init due to the learnable `scale=0.1`.

## Hard gates

- No more infrastructure-only baseline reruns unless variance evidence is specifically needed
- The old TTT gate is now cleared because the pre-TTT anchor is in place, but TTT still requires an explicit legality / portability audit before implementation
- No RFN continuation unless it clearly helps a controlled test
- Do not combine throughput, pre-TTT, and TTT changes in one run before the Session 05 audit identifies the portable pieces

## Memory design

- shared memory in repo: `docs/campaign/AGENT_SYNC.md`
- repo-side Codex mirror: `docs/codex-memory/`
- private Codex mirror: `~/.codex/memories/parameter-golf/`
