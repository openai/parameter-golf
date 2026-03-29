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
- Session 04 Delta 1 (GPTQ-lite clip search) is COMPLETE — FAILED (worse BPB + artifact cap violation)
- Session 04 Delta 2 (LeakyReLU^2) is the next immediate action
- H100 node allocated for ~22 more hours

## One-line resume prompt

Use this in a fresh Codex chat:

```text
Read @docs/codex-memory/BOOTSTRAP.md first. Session 04 Delta 1 (GPTQ-lite) FAILED as of March 28, 2026: sliding s64 1.12941356 (worse), artifact 16219752 (over cap). Anchor remains 1.12904446 sliding s64, 1.15247273 roundtrip, 91.37 ms/step, 15751324 bytes. H100 node allocated ~22 more hours. Begin Delta 2: LeakyReLU^2 as isolated delta on Session 03 anchor.
```
