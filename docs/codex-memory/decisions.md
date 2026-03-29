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
- Session 04 targeted delta sweep is the current mainline.

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

## Hard gates

- No more infrastructure-only baseline reruns unless variance evidence is specifically needed
- No TTT implementation before the pre-TTT anchor is in place
- No RFN continuation unless it clearly helps a controlled test
- Session 04 deltas must be measured in isolation before combining

## Memory design

- shared memory in repo: `docs/campaign/AGENT_SYNC.md`
- repo-side Codex mirror: `docs/codex-memory/`
- private Codex mirror: `~/.codex/memories/parameter-golf/`
