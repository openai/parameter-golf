# Session 05b: Full Hessian GPTQ Implementation Prompt

Paste everything below the line into a fresh Claude Code **planning mode** session.

---

Session 05b: Plan and implement Full Hessian GPTQ as an isolated delta on the Session 03 anchor.

## Entry point

Read in this exact order before doing anything else:

1. `AGENTS.md` — shared entry point, current working mode
2. `docs/campaign/AGENT_SYNC.md` — mutable source of truth for objectives, results, next steps
3. `CLAUDE.md` — standing rules and operational constraints (including Pegasus rules)
4. `docs/campaign/PEGASUS_H100_RUNBOOK.md` — container paths, allocation shapes, env vars

Then read these for implementation context:

5. `records/track_non_record_16mb/2026-03-28_pre_ttt_anchor/train_gpt.py` — our anchor (focus on quantization code: search for `quantize_int6`, `mixed_quantize_int6`, `dequantize_mixed_int6`, and the export section near the end)
6. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` — GPTQ-lite reference (lines 885-940, compare to anchor's quantization)
7. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` — #1 record quantization (lines 1232-1370, more complex due to parameter banking)
8. `records/track_non_record_16mb/2026-03-29_fa3_port/README.md` — FA3 port results (context: FA3 failed to improve step_avg due to pip torch downgrade, so we are pivoting to quantization)

## Background and motivation

### Why Full Hessian GPTQ?

The competitive analysis (2026-03-29) identified Full Hessian GPTQ as the single highest-impact optimization available:

- **PR #1060** (openai/parameter-golf): `1.1122` BPB (3-seed mean), no TTT. Key changes over merged #1: coprime-stride loader + **Full Hessian GPTQ** + XSA-all. Claims `-0.005 to -0.007 BPB` from GPTQ alone.
- **PR #1072** (openai/parameter-golf): `1.117` BPB (1-seed). Uses **Online Hessian GPTQ** — accumulates H=X^TX during training.
- Both beat the merged #1 (`1.1194`) **without TTT**.

Our anchor uses naive per-row int6 quantization (`amax` scaling). The GPTQ-lite clip search (Session 04 Delta 1) was tried and **failed** — it exceeded the 16MB artifact cap without improving BPB.

Full Hessian GPTQ is fundamentally different from GPTQ-lite clip search:
- GPTQ-lite: tries 5 clip percentiles, picks best MSE → still row-max scaling
- Full Hessian GPTQ: uses the Hessian (H = X^T X) to weight quantization errors by their downstream impact, applies Cholesky-based error compensation column by column

### Current quantization in the anchor

The anchor's export path (search for `mixed_quantize_int6` in the anchor):
1. Collects the EMA state dict
2. For each weight tensor classified as "mlp" or "attn": `quantize_int6_per_row(t)` — clamps to [-31, 31] with per-row `amax` scaling
3. For other tensors: `quantize_float_tensor(t)` — int8 with global amax
4. Compresses with zstd level 22
5. Artifact must be ≤ 16,000,000 bytes

### What Full Hessian GPTQ changes

Replace `quantize_int6_per_row` with a GPTQ algorithm that:
1. **Collects Hessian**: During or after training, compute H = X^T X for each linear layer (X = input activations)
2. **Cholesky decomposition**: H_inv via Cholesky for numerically stable column-by-column quantization
3. **Sequential quantization**: For each column, quantize the weight, compute the error, compensate remaining columns using H_inv
4. **Per-row scaling**: Still use int6 range [-31, 31] with per-row scales, but the quantization order and error compensation minimize actual loss degradation

The key references for the algorithm:
- Original GPTQ paper: Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2023)
- PR #634 on openai/parameter-golf (referenced by PR #1060 as "Full Hessian GPTQ")
- PR #1019 on openai/parameter-golf (also referenced)

## Current fixed facts

- Session 03 anchor sliding s64: `1.12904446`, step_avg: `91.37 ms`, steps: `6564`
- Session 03 anchor roundtrip: `1.15247273`
- Session 03 anchor pre-quant EMA: `1.14472403`
- Session 03 anchor artifact: `15,751,324` bytes (headroom: `248,676` bytes)
- Session 04 Delta 1 GPTQ-lite clip search: **FAILED** (artifact `16,219,752` — over cap)
- Session 05 FA3 port `8xH100`: `step_avg 92.67ms`, roundtrip `1.15296` — **SLOWER than anchor** (pip torch downgrade killed gains)
- The quantization gap between anchor roundtrip (`1.15247`) and pre-quant EMA (`1.14472`) is `0.00775 BPB` — this is the export-side loss that GPTQ should reduce
- PR #1060 claims Full Hessian GPTQ gives `-0.005 to -0.007 BPB` improvement over naive quantization

## Task

### Planning phase (this session)

1. **Research the GPTQ algorithm** — use sources in this strict priority order:
   1. **PR diffs from openai/parameter-golf** — PR #634 and #1019 are the direct references. Fetch with: `gh pr view 634 --repo openai/parameter-golf --json files,body` and `gh pr view 1019 --repo openai/parameter-golf --json files,body`. Read the actual quantization functions.
   2. **Local repo code** — `records/track_10min_16mb/` contains GPTQ-lite implementations to understand the existing infrastructure.
   3. **GPTQ paper** — Frantar et al. (2023), Algorithm 1. Use `web search` to fetch the pseudocode if not familiar.
   4. **`context7` MCP** — for PyTorch API details only (`torch.linalg.cholesky`, `torch.linalg.solve_triangular`), NOT as a primary GPTQ source.
   5. **`deepwiki` MCP** — for broader repo questions if the PR diffs are unclear.

2. **Understand the Hessian collection options**:
   - **Post-training calibration** (preferred): After training completes, run a forward pass over calibration data, collect H = X^T X per linear layer via hooks.
   - **Online accumulation** (PR #1072 style): Accumulate H during training every N steps. More complex — defer unless post-training approach hits a wall.
   - **Default to post-training calibration** for this first implementation.

3. **Design the implementation plan**:
   - Where does Hessian collection hook into the training loop?
   - How does the GPTQ quantization function replace `quantize_int6_per_row`?
   - What's the memory overhead of storing H per layer? (H is `[in_features, in_features]` — for dim=512 that's 512×512×4 = 1MB per layer, manageable)
   - Does the Cholesky decomposition need any numerical stability tricks (damping diagonal)?
   - How does this interact with the existing zstd compression?
   - Will the artifact still fit under 16MB?

4. **Lock the calibration budget** — the plan must commit to concrete numbers:
   - Calibration happens **after training**, not during. It uses the final EMA model.
   - Calibration set: a fixed subset of training data (NOT the validation set — that would be leakage).
   - Calibration samples: **128 sequences** (256 max). More is diminishing returns per the GPTQ paper.
   - Calibration sequence length: `2048` (match training).
   - Wall-clock budget for calibration + GPTQ: **target ≤ 30 seconds** on 1xH100. The full 600s training budget is already consumed by training. GPTQ runs post-training on CPU or single GPU using the saved EMA weights. Anything > 60 seconds requires explicit justification and a faster fallback proposal.

5. **Produce a ranked implementation plan** saved to `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md`

### Implementation constraints

- Work in a new directory: `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/`
- Copy the anchor's `train_gpt.py` as the starting point
- The delta must be **isolated**: only change the quantization/export path. Do not change the model architecture, training dynamics, hyperparameters, or attention mechanism.
- **Preserve the exact serialized artifact format:**
  - Same int6 packed weights (int8 storage, [-31, 31] range)
  - Same per-row fp16 scales
  - Same `mixed_quantize_int6` → zstd compression pipeline
  - Same `dequantize_mixed_int6` roundtrip path (eval must work unchanged)
  - Only the **weight value selection and error compensation logic** inside `quantize_int6_per_row` changes
- The anchor's `quantize_int6_per_row` function is the surgical target. `mixed_quantize_int6` calls it per tensor — the wrapper stays, the per-row logic gets replaced with GPTQ column-by-column quantization.
- Everything upstream (training, EMA, warmdown) stays identical.
- Artifact must remain ≤ 16,000,000 bytes.

## Container and runtime

This delta runs on the **standard NGC 26.03 container** — no FA3 dependency, no pip torch replacement.

```bash
# Standard smoke (1xH100)
srun -p H100 --ntasks=1 --gpus-per-task=1 --cpus-per-task=6 \
  --mem=64G --time=00:15:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export PYTHONUNBUFFERED=1
    pip install --no-cache-dir sentencepiece zstandard &&
    python -u records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py
  '

# Full 8xH100
srun -K -p H100 --nodes=1 --ntasks=8 --gpus-per-task=1 --gpu-bind=none --cpus-per-task=6 \
  --mem=200G --time=00:20:00 \
  --container-image=/enroot/nvcr.io_nvidia_pytorch_26.03-py3.sqsh \
  --container-mounts="$(pwd)":"$(pwd)" --container-workdir="$(pwd)" \
  bash -c '
    export LOCAL_RANK=$SLURM_LOCALID RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS
    export PYTHONUNBUFFERED=1
    export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1
    export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=bond,eth NCCL_P2P_LEVEL=NVL
    pip install --no-cache-dir sentencepiece zstandard &&
    python -u records/track_non_record_16mb/2026-03-29_full_hessian_gptq/train_gpt.py
  '
```

Note: NGC 26.03 has `sentencepiece` and `zstandard` preinstalled in some builds. The `pip install` is a safety net — it completes instantly if already present.

## Success criteria

- Roundtrip `val_bpb` < `1.1525` (anchor's `1.15247` — any improvement counts)
- Sliding s64 `val_bpb` < `1.1290` (anchor's `1.12904`)
- Artifact ≤ `16,000,000` bytes (CRITICAL — GPTQ-lite failed here)
- `step_avg` within ±1ms of anchor (`91.37 ms`) — quantization is post-training, should not affect training speed
- Pre-quant EMA `val_bpb` should remain **statistically unchanged** from anchor (`1.14472`), since training code is intended to be identical. Small seed/nondeterminism variation is acceptable; a shift >0.001 indicates an unintended training-side change.

The key metric is the **roundtrip gap**: anchor loses `0.00775 BPB` from pre-quant to roundtrip. If Full Hessian GPTQ cuts this to `0.003-0.004`, that's `0.004 BPB` free improvement.

## Tools and skills

Use these if available (prefer but do not require):

- **`/research-engineer`** skill — for structured implementation with scientific rigor
- **`deepwiki` MCP** — query `openai/parameter-golf` for GPTQ implementations in PRs #634, #1019, #1060
- **`context7` MCP** — look up PyTorch quantization APIs, torch.linalg.cholesky, GPTQ algorithm docs
- **`claude-mem` MCP** — search past observations/decisions about quantization experiments
- **Code navigation MCPs (serena)** — for symbol-level exploration of quantization functions
- **Parallel Agent subagents** — for researching PR implementations and the GPTQ paper simultaneously
- **`gh` CLI** — for fetching PR code: `gh pr view <number> --repo openai/parameter-golf --json body,files`

If any MCP or skill is not available, fall back to Grep, Glob, Read, WebSearch, and WebFetch tools.

## Git conventions

Follow the established commit message prefixes:
- `research(protocol):` — before running an experiment (implementation commit)
- `research(results):` — after a run, with measured results
- `docs:` or `docs(campaign):` — documentation-only changes

Rules:
- Never use `--no-verify` or `--no-gpg-sign`
- Stage specific files, never `git add -A` or `git add .`
- Do NOT stage unrelated modified files (check `git status` first)
- Do NOT stage: `.serena/`, `docs/*.pdf`, `docs/*.txt`, unrelated READMEs

## Pegasus conventions

- **NEVER use `torchrun --standalone`** — use Slurm-native `srun`
- **Always `--nodes=1`** for 8xH100 runs
- **Always `PYTHONUNBUFFERED=1`** — never hide output with `| tail -1`
- **Always `--gpu-bind=none`** for NCCL peer-to-peer
- Container: **NGC 26.03** for this delta (no FA3 dependency)
- The saved FA3 container (`/netscratch/$USER/containers/pytorch_25.02_fa3.sqsh`) is NOT needed here
- Data: prefer `/fscratch` if available, fallback to `/netscratch`
- Sync: `git pull` on Pegasus before running

## Documentation conventions

After completing meaningful work, update in this order:

### Record folder
1. `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/README.md`
2. `records/track_non_record_16mb/2026-03-29_full_hessian_gptq/submission.json`

### Shared docs (canonical truth)
3. `docs/campaign/AGENT_SYNC.md` — add measured results, update next steps
4. `docs/campaign/artifacts/05b_full_hessian_gptq_plan.md` — the planning artifact from this session

### Codex memory (project-persistent, repo-committed)
5. `docs/codex-memory/project-state.md` — update after session completion
6. `docs/codex-memory/next-session.md` — update with next immediate action
7. `docs/codex-memory/decisions.md` — record GPTQ decision and result classification

### Claude memory (cross-conversation persistent)
8. `~/.claude/projects/-home-amay-Work-parameter-golf/memory/project_parameter_golf.md`
9. `~/.claude/projects/-home-amay-Work-parameter-golf/memory/MEMORY.md`

### Record folder schema
- Each experiment: `records/track_non_record_16mb/YYYY-MM-DD_<tag>/`
- Never mutate the anchor folder
- Contents: `train_gpt.py`, `README.md`, `submission.json`, `requirements.txt`

## Key research questions for the planning phase

1. What is the exact GPTQ algorithm (column-by-column with Cholesky H_inv)?
2. How do PR #634 and #1019 implement Hessian collection — post-training calibration or online?
3. What calibration data should be used? (Likely: a subset of training data, or the validation set)
4. How many calibration samples are needed? (Typical: 128-256 sequences)
5. What is the damping factor for Cholesky stability? (Typical: 0.01 * mean diagonal)
6. Does GPTQ change artifact compressibility? (GPTQ-lite hurt zstd — Full Hessian might too or might not)
7. Can we reuse the existing `mixed_quantize_int6` infrastructure and just replace the per-row quantization?
8. What's the wall-clock cost of GPTQ? (Needs to fit within the 600s training budget, or run post-training within eval budget)

## Lessons from Session 05a (FA3 port)

Apply these lessons from the FA3 port experience:

- **Do not assume microbenchmarks translate to end-to-end gains.** Always measure full training.
- **Container compatibility matters.** This delta uses standard NGC 26.03 — no pip torch replacement risk.
- **Test the artifact size early.** GPTQ-lite failed because it hurt zstd compressibility. Check artifact size before running full experiments.
- **Isolated deltas only.** Do not bundle quantization changes with model architecture changes.
- **Output buffering kills debugging.** Always use `PYTHONUNBUFFERED=1`.
