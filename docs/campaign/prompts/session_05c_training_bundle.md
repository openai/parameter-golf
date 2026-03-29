# Session 05c: Training Stack Bundle Prompt

Paste everything below the line into a fresh Claude Code session. Run this AFTER Session 05b (GPTQ) results are in.

---

Session 05c: Bundle four training-side improvements as a single delta on the Session 03 anchor (or on the 05b GPTQ result if it graduated).

## Entry point

Read in this exact order before doing anything else:

1. `AGENTS.md` — shared entry point, current working mode
2. `docs/campaign/AGENT_SYNC.md` — mutable source of truth (check if 05b GPTQ graduated)
3. `CLAUDE.md` — standing rules and operational constraints
4. `docs/campaign/PEGASUS_H100_RUNBOOK.md` — container paths, allocation shapes, env vars

Then read for implementation context:

5. `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py` — anchor code
6. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` — #1 record (reference for all four features)
7. `docs/campaign/artifacts/05_ttt_correctness_audit.md` — feature comparison table
8. If 05b graduated: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/` — the GPTQ delta (use as base instead of anchor)

## What this bundle contains

Four training-side changes that are orthogonal to each other and to GPTQ:

### 1. XSA 4 → 11 layers (trivial)
- **Current**: `xsa_last_n=4` — XSA on layers 7-10
- **Change**: `xsa_last_n=11` — XSA on all layers
- **Expected gain**: `-0.002 to -0.003 BPB` (PR #1060 uses XSA-all)
- **Implementation**: Single constant change. Find `xsa_last_n` or the loop that sets `use_xsa = True` and extend it to all layers.
- **Risk**: Very low. XSA is already working on 4 layers.

### 2. Value Embedding VE128, layers 9-10 (low effort)
- **Current**: Not present
- **Change**: Add shared 128-dim embedding that reinjects token identity into attention values on layers 9-10
- **Expected gain**: `-0.001 to -0.002 BPB`
- **Implementation**:
  - Add `nn.Embedding(vocab_size, 128)` shared across layers
  - Add `nn.Linear(128, kv_dim, bias=False)` per VE-enabled layer
  - Add learnable per-layer scale parameter
  - In forward: project embedding, scale, add to v before attention
- **Reference**: Search #1 record for `value_embed` or `ve_` or `VE`
- **Artifact impact**: +~100KB (fits within headroom)
- **Risk**: Low. Standard additive feature.

### 3. Tight SWA every 50 steps (low effort)
- **Current**: Disabled (`swa_enabled=False` or not present)
- **Change**: Accumulate model snapshots every 50 steps when `lr_scale < 0.2` (during warmdown), average with EMA for final export
- **Expected gain**: `-0.001 to -0.002 BPB`
- **Implementation**:
  - Add SWA state accumulator (CPU-side, running mean of model weights)
  - Start collecting when warmdown enters final phase
  - Average SWA buffer with EMA before quantization export
- **Reference**: Search #1 record for `swa` or `SWA` or `stochastic_weight`
- **Artifact impact**: None (CPU state only, not serialized)
- **Risk**: Low. Pure weight averaging.

### 4. Warmdown 3500 (trivial)
- **Current**: `warmdown_iters=3000`
- **Change**: `warmdown_iters=3500`
- **Expected gain**: `-0.0002 BPB` + enables more SWA snapshots
- **Implementation**: Single constant change.
- **Risk**: None.

### NOT in this bundle
- **LeakyReLU²**: Measured as neutral/slightly slower in isolation (Session 04 Delta 2). Revisit only if the stronger stack is working.
- **Coprime-stride loader**: High potential but changes data sampling semantics. Keep as next bundle or bisect candidate.
- **FA3**: Container ABI issue unresolved. Parked.
- **Parameter Banking / Parallel Muon**: Full architecture rewrite. Deferred.
- **TTT**: Parked per strategy.

## Base for this delta

**If 05b GPTQ graduated**: Stack on top of the GPTQ delta. This means the combined result tests training improvements + better quantization.

**If 05b GPTQ failed or is pending**: Stack directly on the anchor. GPTQ can be added later independently since it only touches the export path.

Either way, this bundle touches ONLY training-side code. The quantization/export path stays as-is (either anchor's naive int6 or 05b's GPTQ).

## Container and runtime

Standard NGC 26.03 container (same as anchor, same as GPTQ delta).

```bash
# 8xH100 full run
srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null
    python -u records/track_non_record_16mb/2026-03-29_training_bundle/train_gpt.py
  '
```

## Success criteria

- Sliding s64 `val_bpb` < `1.1260` (anchor is `1.1290`, target ~`-0.003` from the bundle)
- Pre-quant EMA `val_bpb` < `1.1420` (anchor is `1.14472`, training should improve)
- `step_avg` within ±3ms of anchor (`91.37 ms`) — XSA-all adds ~2ms, VE128 adds ~1ms, SWA is free
- Artifact ≤ `16,000,000` bytes
- Steps ≥ `6400` (slightly fewer due to step_avg increase is acceptable)

## Implementation order within the bundle

Do all four in one script, but apply them in this order for clean diff review:

1. Warmdown 3500 (one constant)
2. XSA 4→11 (one constant or loop change)
3. VE128 (new module + forward path modification)
4. Tight SWA (new accumulator + export path modification)

## Tools and skills

- **`/research-engineer`** skill — for structured implementation
- **`deepwiki` MCP** — query `openai/parameter-golf` for VE and SWA implementations
- **`context7` MCP** — for PyTorch APIs only
- **`claude-mem` MCP** — for past decisions
- **`gh` CLI** — for fetching reference implementations
- **Parallel Agent subagents** — for reading multiple reference files simultaneously

## Git conventions

- `research(protocol): Session 05c — training bundle (XSA-all + VE128 + SWA + warmdown3500)`
- `research(results): Session 05c — [RESULT] training bundle 8xH100`
- Stage specific files only. Do NOT stage unrelated files.

## Pegasus conventions

- NEVER `torchrun`. Always `srun`.
- Always `--nodes=1` for 8xH100.
- Always `PYTHONUNBUFFERED=1`.
- Always `--gpu-bind=none`.
- Container: NGC 26.03 (standard path).

## Documentation conventions

Update in this order after results:

1. `records/track_non_record_16mb/2026-03-29_training_bundle/README.md`
2. `records/track_non_record_16mb/2026-03-29_training_bundle/submission.json`
3. `docs/campaign/AGENT_SYNC.md`
4. `docs/codex-memory/project-state.md`
5. `docs/codex-memory/next-session.md`
6. `docs/codex-memory/decisions.md`
7. Claude memory files

## If the bundle regresses

Bisect by reverting one change at a time in reverse order:
1. Remove SWA → re-run
2. Remove VE128 → re-run
3. Revert XSA to 4 → re-run
4. Revert warmdown to 3000 → confirms anchor parity

Each bisect step is a single constant or module removal. This is why these four were chosen — they're independently revertible.
