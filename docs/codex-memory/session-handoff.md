# Session Handoff

Date: 2026-03-28

## Current Truths

- Session 03 pre-TTT anchor port is complete.
- Sliding s64 val_bpb: `1.12904446` on `8xH100 SXM5`, `serv-3342`.
- Pre-quant EMA val_bpb: `1.14472403`.
- Int6 roundtrip val_bpb: `1.15247273`.
- Steps: `6564`, step_avg: `91.37 ms`.
- Artifact: `15751324` bytes (model `15692752` + code `58572`).
- Throughput is the primary bottleneck (SDPA vs FA3), not model fidelity.
- NGC 26.03 container + fscratch is the confirmed optimized Pegasus path.
- Session 04 Delta 1 (GPTQ-lite clip search) is COMPLETE — FAILED.
- Session 04 Delta 2 (LeakyReLU^2) is the next immediate action.
- H100 node is allocated for ~22 more hours.

## What Was Done In Session 03

- Ported the clean pre-TTT anchor (2026-03-21 style) into a self-contained script on top of root `train_gpt.py`.
- Features ported: SmearGate + BigramHash, XSA on last 4 layers, partial RoPE 16/64, layerwise LN scale, EMA, Muon/Adam weight decay, mixed int6 export + zstd, stride-64 sliding eval.
- Ran the anchor on `8xH100 SXM5` (`serv-3342`) under NGC 26.03 container.
- Measured all three eval metrics (sliding s64, pre-quant EMA, int6 roundtrip).
- Confirmed artifact fits under the 16MB cap with `248676` bytes headroom.

## What Was Learned

### rope_train_seq_len bug
- The anchor sets `rope_train_seq_len=1024` for NTK-aware scaling even though `TRAIN_SEQ_LEN=2048`. This is deliberate and matches the donor record behavior. It is not a bug but could appear as one to a fresh reader.

### Container OOM
- Initial attempts to run inside containers hit OOM due to container-level memory overhead on top of GPU allocation. Resolved by using the NGC 26.03 container with proper resource requests.

### fscratch setup
- `/netscratch` I/O can bottleneck data loading. Using `/fscratch` for data staging avoids this. The path must be set up per-job since `/fscratch` is ephemeral.

### SDPA throughput gap
- Session 03 anchor achieves `91.37 ms/step` with SDPA versus root baseline's `51.66 ms/step`. The anchor has more compute per step (more layers, XSA, SmearGate), but the gap is also partly due to using SDPA instead of FA3. The donor record used `flash_attn_3_func`.

## What Was Done In Session 04 Delta 1

- Ran GPTQ-lite percentile clip search as an isolated delta on top of the Session 03 anchor.
- Single change: replaced fixed row-max int6 quantization with GPTQ-lite 5-percentile MSE clip search.
- Training was identical to the anchor.

## Delta 1 Results (FAILED)

- Sliding s64 val_bpb: `1.12941356` (WORSE than anchor `1.12904446` by `+0.00036910`)
- Roundtrip val_bpb: `1.15277272` (WORSE than anchor `1.15247273` by `+0.00029999`)
- Pre-quant EMA val_bpb: `1.14520403` (effectively identical to anchor `1.14472403`)
- Artifact size: `16219752` bytes — OVER the `16000000` byte cap (anchor was `15751324`)
- Steps: `6565`, step_avg: `91.37 ms` (identical to anchor as expected)

## What Was Learned From Delta 1

- GPTQ-lite clip search hurts zstd compressibility more than it helps quantization quality.
- The export gap between pre-quant EMA and roundtrip is not caused by clip suboptimality.
- Anchor int6+zstd with fixed row-max remains the viable export path.
- The artifact size increase (`+468428` bytes) pushes over the 16MB cap, making this path non-viable even if BPB were neutral.

## Locked Scope For Remaining Session 04 Deltas

### Delta 2: LeakyReLU^2 activation (NEXT IMMEDIATE ACTION)
- Replace relu^2 with LeakyReLU^2
- Measure val_bpb impact
- H100 node allocated for ~22 more hours

### Delta 3: one small schedule or token-path tweak
- Pending Delta 2 result

### Discipline
- Each delta is a separate run with one change
- Compare against Session 03 anchor as the fixed reference
- Only combine after each is measured in isolation

## Source Of Truth Files

- `docs/campaign/AGENT_SYNC.md`
- `docs/campaign/artifacts/03a_pre_ttt_anchor_diff_analysis.md`
- `docs/campaign/artifacts/03b_root_train_gpt_port_gap_audit.md`
- `docs/codex-memory/project-state.md`
- `docs/codex-memory/next-session.md`
- `docs/codex-memory/decisions.md`
