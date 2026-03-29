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
- Session 05 throughput + pre-TTT + TTT audit is the new mainline.

## Session 03 decisions

- Session 03 anchor uses SDPA not FA3. The donor record used `flash_attn_3_func`, but the anchor port kept `scaled_dot_product_attention` to avoid introducing an untested kernel dependency in the first anchor run. This is a deliberate conservatism, not an oversight.
- NTK RoPE with `train_seq_len=1024` confirmed as deliberate. The anchor sets `rope_train_seq_len=1024` for NTK-aware scaling even though `TRAIN_SEQ_LEN=2048`. This is intentional and matches the donor record behavior.
- Throughput is a plausible bottleneck, but not the only remaining gap. Session 03 finished at `91.37 ms/step`, but the pre-quant to roundtrip gap (`1.14472403 -> 1.15247273`) means export-side work still deserves isolated measurement.
- NGC container + fscratch confirmed as optimized Pegasus path. The NGC 26.03 container with `/fscratch` for data staging avoids `/netscratch` I/O bottlenecks and resolves OOM issues from container-level overhead.

## Hardware

- Pegasus `8xH100` is now the primary execution base.
- Launch Pegasus multi-GPU work with Slurm-native `srun`, not `torchrun --standalone`.
- RunPod stays reserved for final validation or granted credits.

## Workflow

- Keep competitive experiments in self-contained folders under `records/track_non_record_16mb/YYYY-MM-DD_<tag>/`
- Do not modify existing public record folders
- Document every run with manifests and experiment summaries
- Prefer additive, well-understood public techniques over speculative novelty
- Keep Session 04 deliberately narrow: one isolated delta per run, no stacked backend/export/model bundles

## Session 04 decisions

- GPTQ-lite percentile clip search rejected — marginal BPB regression + artifact cap violation. Export gap is not caused by clip suboptimality. Sliding s64 val_bpb `1.12941356` vs anchor `1.12904446` (+0.00036910), artifact `16219752` bytes exceeds `16000000` cap. Anchor int6+zstd with fixed row-max remains the viable export path.
- LeakyReLU^2 classified as neutral/tie — sliding s64 val_bpb `1.12904123` vs anchor `1.12904446` (-0.00000323), effectively zero. Pre-quant and roundtrip both slightly better. Artifact `168356` bytes smaller. But step time `+0.72 ms` slower, costing `53` steps. Not a standalone graduating delta. Keep as a possible stack component for artifact headroom or when combined with a throughput-positive change. Measured anchor comparison used `enable_math_sdp(True)` — isolation preserved correctly.
- Session 04 ends at `1 failed + 1 neutral`. Do not force a Delta 3 by default.

## Session 05 decisions

- TTT is now back in scope because the pre-TTT anchor exists and Session 04 has finished.
- TTT should be treated as necessary but not sufficient from the current anchor; the local `1.1194` record has pre-TTT base `1.1218`, so stronger pre-TTT work is still required.
- The next phase should separate:
  - throughput audit
  - pre-TTT stack-gap audit
  - TTT correctness / portability audit
- FA3 is back in scope as a deliberate Session 05 throughput investigation, not as an anchor bring-up risk.

## Hard gates

- No more infrastructure-only baseline reruns unless variance evidence is specifically needed
- The old TTT gate is now cleared because the pre-TTT anchor is in place, but TTT still requires an explicit legality / portability audit before implementation
- No RFN continuation unless it clearly helps a controlled test
- Do not combine throughput, pre-TTT, and TTT changes in one run before the Session 05 audit identifies the portable pieces

## Memory design

- shared memory in repo: `docs/campaign/AGENT_SYNC.md`
- repo-side Codex mirror: `docs/codex-memory/`
- private Codex mirror: `~/.codex/memories/parameter-golf/`
