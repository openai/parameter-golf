# Codex Bootstrap

If this is a fresh Codex session, read these files in order:

1. `@docs/codex-memory/project-state.md`
2. `@docs/codex-memory/decisions.md`
3. `@docs/codex-memory/next-session.md`
4. `@docs/campaign/AGENT_SYNC.md`
5. `@docs/campaign/artifacts/2026-03-28_a100_evidence_summary.md`
6. `@docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md`
7. `@docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
8. `@docs/campaign/sessions/03_pre_ttt_anchor_port.md`

Then proceed with the next pending action.

## Current status

- Session 03 pre-TTT anchor is complete
- Session 03 sliding s64 val_bpb: `1.12904446` on `8xH100 SXM5`
- int6 roundtrip val_bpb: `1.15247273`, artifact `15751324` bytes
- throughput bottleneck identified: SDPA vs FA3 (`91.37 ms/step` vs target)
- NGC container + fscratch confirmed as optimized Pegasus path
- competition phase continues with Session 04 isolated deltas

## One-line resume prompt

Use this in a fresh Codex chat:

```text
Read @docs/codex-memory/BOOTSTRAP.md first. Session 03 anchor is complete as of March 28, 2026. 8xH100 sliding s64 val_bpb 1.12904446, int6 roundtrip val_bpb 1.15247273, step_avg 91.37 ms, artifact 15751324 bytes (headroom 248676). Throughput is the primary bottleneck: SDPA not FA3. NGC container + fscratch is the confirmed Pegasus path. Begin Session 04: FA3 integration, GPTQ-lite, LeakyReLU^2 as isolated deltas.
```
