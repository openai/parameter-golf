# Spec 008 — PR #1736 reproduction (new baseline)

**Slug:** `1736-reproduction`
**Created:** 2026-04-20
**Links to idea:** `research/ideas/1736-improvement.md`

## Hypothesis

We can reproduce dexhunter's unmerged PR #1736 (val_bpb 1.06549, 3-seed mean, std 0.00070) on our pod fleet, and from that point forward use its 3-seed mean as our new local baseline. This replaces spec-000's merged-SOTA #1493 baseline (1.0810) as the comparator for specs 009+.

## Baseline

Comparison reference for this spec is #1736's **seed-42** number from their `submission.json`: **val_bpb = 1.06610**.

(For reference: their full 3-seed set was 42=1.06610, 0=1.06473, 1234=1.06563, mean=1.06549±0.00070. We only reproduce seed 42; per-seed comparison is apples-to-apples and sufficient for screening.)

Our spec-000 number (1.0810, merged-SOTA replica) remains on the books only as a legacy reference for backward-compat sanity reruns.

## Expected Δ

Not a delta experiment — a baseline migration. Success criterion is *reproducing* the number within noise, not beating it.

## Accept criteria

### Phase 1 — data prep
- `prepare_caseops_data.py` completes without error.
- Output directory contains `fineweb_train_*.bin`, `fineweb_val_*.bin`, and `fineweb_val_bytes_*.bin` in the nested path expected by `train_gpt.py`.
- Byte sidecars' summed byte count matches the original FineWeb-10B validation corpus byte count (quick sanity check — single Python script).

### Phase 2 — 2×H100 smoke
- Stack imports cleanly (flash-attn-3 loads, no CUDA version errors).
- Training runs 50–100 steps with no NaN, finite first-step loss.
- Step time within 2× of expected for 2×H100 (i.e., not catastrophically slow).

### Phase 3 — 8×H100 single-seed official
- Seed 42 completes without NaN / divergence.
- Artifact < 16,000,000 bytes (decimal cap, per #1736's submission).
- Within 600 s train + 600 s eval budget.
- **Primary accept:** val_bpb within **±0.003 of 1.06610** (#1736's seed-42 number).
- **Pre-GPTQ checkpoint saved** to `runs/008-1736-reproduction/seed_42/pre_gptq.pt` (FP weights, right before GPTQ quantization runs). This is the hotstart input for specs 009+ quant experiments.

## Config diff

No config diff relative to our baseline — #1736 is a fully self-contained submission with its own `train_gpt.py`. Launch is their script, their env-var block, unmodified.

**Env block (all phases, verbatim from #1736 README):**

```
NCCL_NET=Socket
DATA_DIR=./data
CASEOPS_ENABLED=1
PHASED_TTT_ENABLED=1
PHASED_TTT_PREFIX_DOCS=2000
PHASED_TTT_NUM_PHASES=3
MLP_CLIP_SIGMAS=12.0
ATTN_CLIP_SIGMAS=13.0
EMBED_BITS=7
EMBED_CLIP_SIGMAS=15.0
MATRIX_LR=0.026
GPTQ_RESERVE_SECONDS=4
GPTQ_CALIBRATION_BATCHES=16
GATED_ATTN_ENABLED=1
GATED_ATTN_INIT_STD=0.005
GATED_ATTN_QUANT_GATE=1
SEED=<42|0|1234>
```

**Phase 2 smoke only:** override `ITERATIONS` to ~50–100 and disable eval (mechanism: env var or command-line flag per `train_gpt.py`'s interface — execution to determine on pod).

## Code changes

- **Branch:** `research` (no separate `exp/<slug>` branch — this spec is the baseline migration itself, so the code lands directly on `research`).
- **Pinned commit (theirs):** `e100586d60b2a228c66f0e04b35654160b657c21` on `dexhunter/caseops-gatedattn-quantgate-1.06549` (PR #1736 head).
- **Diff:** bulk import of #1736's submission directory at

  `records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT/`

  (9 files, ~6856 lines). No modifications to `train_gpt_sota.py` or other existing repo files. Our training script stays untouched — this spec uses **their** `train_gpt.py` from the submission directory.

- **Pinned commit (ours):** `154c9b85736fcddf49db4d54ecb21e87bd1406af` on `research` (fast-forward import of #1736's submission dir, one commit past prior `research` HEAD).

## Hardware ladder

- [x] **Phase 1** — CPU (data prep). Any tiny pod, or run on a GPU pod during setup.
- [x] **Phase 2** — 2×H100 smoke (~10 min, ~$0.50). Skipped only if execution has very high confidence in the integration.
- [x] **Phase 3** — 8×H100 single-seed official (~30 min, ~$10).

## Seed plan

**Single seed: 42.** Apples-to-apples against #1736's seed-42 number (1.06610). Multi-seed confirmation is deferred to a final leaderboard spec if/when a composition looks submission-ready. Screening single-seed-vs-single-seed is consistent with our step-matched-comparison convention and saves ~$20 per quant experiment downstream.

## Inputs

- **Raw FineWeb-10B docs** — existing dataset on persistent volume. (Execution to confirm volume — likely already present from prior specs; if not, pull via `data/download_hf_docs_and_tokenize.py`.)
- **Pre-trained SP tokenizer** — bundled in #1736's submission dir: `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` (366 KB, 8192 vocab with reserved CaseOps operator slots).
- **CaseOps-transformed data** — produced by `prepare_caseops_data.py` in Phase 1; persisted to volume for reuse by specs 009+.
- **No hotstart checkpoint.** Full-from-scratch training.

## Execution protocol

### Phase 1 — data prep

On a CPU or idle GPU pod:

```bash
cd /workspace/parameter-golf
cd records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

python3 prepare_caseops_data.py \
  --docs <path_to_fineweb10B_raw_docs.jsonl> \
  --out /workspace/data/datasets/fineweb10B_sp8192_caseops/datasets \
  --sp ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Output path must match what `train_gpt.py` expects (nested `datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/`).

### Phase 2 — 2×H100 smoke

Single seed (42), short run:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

# Same env-var block as phase 3 below, plus:
ITERATIONS=100 \
DISABLE_EVAL=1 \
torchrun --standalone --nproc_per_node=2 train_gpt.py
```

(Exact smoke-override mechanism TBD per `train_gpt.py`'s interface; execution to confirm on pod.)

If smoke passes → proceed to Phase 3. If smoke fails → halt, post diagnosis, flag research.

### Phase 3 — 8×H100 single-seed official (seed 42)

**Before launching:** patch `train_gpt.py` to save the pre-GPTQ FP checkpoint. Find the code path right before the GPTQ quantization call and inject:

```python
if int(os.environ.get("SAVE_PRE_GPTQ", "0")) and rank == 0:
    torch.save(model.state_dict(), os.environ["PRE_GPTQ_CKPT_PATH"])
```

Gated on an env var so the change is invisible when not requested (keeps the reproduction as clean as possible). One-line equivalent acceptable.

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT

mkdir -p /workspace/runs/008-1736-reproduction/seed_42

NCCL_NET=Socket DATA_DIR=./data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
SAVE_PRE_GPTQ=1 \
PRE_GPTQ_CKPT_PATH=/workspace/runs/008-1736-reproduction/seed_42/pre_gptq.pt \
SEED=42 \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /workspace/runs/008-1736-reproduction/seed_42/train.log 2>&1
```

### Kill protocol

- After Phase 3 completes (or on any stop-early trigger): `runpodctl pod stop $POD_ID`.
- Between phases: keep pod warm; the full sequence fits in a single session.

## Checkpoints to emit

**Exactly one:** `runs/008-1736-reproduction/seed_42/pre_gptq.pt` — FP16/FP32 weights saved right before GPTQ quantization runs.

Rationale: the entire quant-family spec chain (009 SpinQuant, plus any future per-group-bit / AR-selfgen / AWQ experiments) can hotstart off this single checkpoint because SpinQuant and its siblings are post-training transforms. Per-experiment cost drops from ~$10 retrain to ~$1–2 rotate-and-requant. The one-line injection into `train_gpt.py` is gated on an env var so the reproduction itself is unaffected.

No intermediate / phase-boundary checkpoints. No post-GPTQ checkpoints (the training log + `final.json` carry the info we'd want).

## Stop-early criteria

- Import / CUDA failure on smoke → halt, flag.
- NaN in train_loss at any step → halt, mark failed.
- Step time > 2× expected → halt, investigate.
- Artifact > 16 MB → halt, flag (our build mismatches #1736's compression).
- Seed-42 val_bpb > 0.003 off 1.06610 → halt, flag research; decide whether to retry or fall back to #1626 clean-foundation baseline before spawning downstream quant specs.

## Cost estimate

| Item | Cost |
|---|---|
| Phase 1 (data prep, CPU-idle GPU) | ~$1–2 |
| Phase 2 (2×H100 smoke, 10 min) | ~$0.50 |
| Phase 3 (8×H100, single seed, ~30 min) | ~$10 |
| Buffer for debug | ~$5 |
| **Total** | **~$17** |

Downstream quant experiments (spec 009+) hotstart off the checkpoint from this run, so each costs ~$1–2 instead of ~$10.

## Extra artifacts

- `runs/008-1736-reproduction/seed_42/train.log` — full training stdout/stderr
- `runs/008-1736-reproduction/seed_42/pre_gptq.pt` — pre-GPTQ FP checkpoint (hotstart input for spec 009+)
- `runs/008-1736-reproduction/seed_42/artifact.ptz` (or whatever `train_gpt.py` writes) — the submission artifact
- `runs/008-1736-reproduction/smoke/train.log` — Phase 2 smoke log
- `runs/008-1736-reproduction/final.json` — seed-42 summary (bpb, artifact size, wall times, step at which pre_gptq.pt was saved)
- `runs/008-1736-reproduction/notes.md` — execution narrative + any deviations from this spec

## Open questions for interview

1. **Data source** — is FineWeb-10B raw (`docs_selected.jsonl`) already on the persistent volume from prior specs, or do we pull it via `data/download_hf_docs_and_tokenize.py`? Which volume (NA-1 vs JP) does the pod region pin?
2. **HF shortcut** — is `romeerp/parameter-golf-caseops-v1` on HuggingFace byte-compatible with what `prepare_caseops_data.py` produces? If yes, Phase 1 can be replaced with a ~20 GB download + schema check (saves ~2 hours). Quick way to test: download one val shard + its byte sidecar from HF, compare byte-for-byte against local prep output on a sample doc.
3. **flash-attn-3 install** — is the wheel at `https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/` still reachable from the pod's region? Preflight step per #1736 README: `pip install flash_attn_3 --no-deps --find-links <wheel-url>`. If unreachable, fallback?
4. **Smoke override mechanism** — does `train_gpt.py` accept an `ITERATIONS` / `MAX_STEPS` env var, or a `DISABLE_EVAL` flag? Execution should grep the script for the iteration count constant and pick the cleanest override path. If no clean override exists, we can instead just run the full training and abort after ~2 min of logging — the smoke goal is the first 50 steps of log output.
5. **Pre-GPTQ hook location** — execution should grep `train_gpt.py` for where GPTQ is invoked (likely a function call on the FP model) and inject the `torch.save(...)` one line before, gated on `SAVE_PRE_GPTQ`. Verify the saved state_dict loads correctly before declaring Phase 3 a pass (simple `torch.load(...)` check on the pod).

## What this spec does NOT do

- Does not attempt to beat 1.06610. Success is *reproduce*.
- Does not run 3 seeds — seed 42 only. Multi-seed confirmation is deferred to a potential final leaderboard spec.
- Does not save intermediate / phase-boundary checkpoints. One pre-GPTQ checkpoint only.
- Does not run a full 2×H100 mini (40 min, ~$3) — only a ~10 min smoke.
- Does not modify `train_gpt.py` beyond the env-var-gated pre-GPTQ checkpoint save.
- Does not test other unmerged PRs (#1735, #1738, #1667, #1729, #1695) as alternative bases. Those belong in specs 009+.
