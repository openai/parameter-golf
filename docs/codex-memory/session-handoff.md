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
- Session 04 isolated deltas are the next mainline work.

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

## Locked Scope For Session 04

### Delta 1: FA3 integration (highest priority)
- Replace SDPA with `flash_attn_3_func`
- Target: significant step_avg reduction from `91.37 ms`
- This is the single highest-leverage change

### Delta 2: GPTQ-lite compression
- Add GPTQ-lite quantization to the export path
- Measure roundtrip val_bpb and artifact size impact

### Delta 3: LeakyReLU^2 activation
- Replace relu^2 with LeakyReLU^2
- Measure val_bpb impact

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
