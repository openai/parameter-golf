# Spec 028 — Post-train clip=12 + TTT on 008 model (baseline diagnostic)

**Slug:** `post-train-clip12-ttt`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `research` (commit `6456188` — spinquant_hotstart.py)
**Links to:** `research/ideas/match-1769-baseline.md`

## Hypothesis

Spec 009 ran GPTQ+TTT on the 008 pre_gptq.pt but likely used MLP_CLIP_SIGMAS=10.0
(execution deviated from spec; result 1.0801 post-GPTQ matches clip=10 behavior, not
clip=12). Running the same pipeline explicitly with clip=12 should reproduce #1769's
post-GPTQ (~1.076) and post-TTT (~1.064) on our hardware.

This is a **pure diagnostic** — no training, no code changes, no experimental levers.
The sole question: does our pipeline match #1769 on identical inputs?

## Baseline

| run | post-GPTQ | post-TTT | clip |
|---|---|---|---|
| spec 009 baseline | 1.0801 | 1.0673 | 10.0 (suspected) |
| #1769 (dexhunter) | ~1.076 | 1.06453 | 12.0 |

## Expected result

post-TTT ~**1.064–1.065** if our pipeline matches #1769.
post-TTT ~**1.067** if there is a systematic GPTQ gap independent of clip setting.

## Accept criteria

| post-TTT bpb | interpretation | action |
|---|---|---|
| ≤ 1.065 | Pipeline matches #1769. Cross-layer carry hurts GPTQ. | Close 026 arc. Stack 027 levers on clean base. |
| 1.065–1.067 | Marginal gap. Possibly 008's 38 fewer steps vs #1769. | Run one more seed to confirm. |
| > 1.067 | Systematic GPTQ gap regardless of clip. | Diff our GPTQ code vs #1769 before any more spend. |

## Config diff vs spec 009

| | spec 009 | spec 028 |
|---|---|---|
| MLP_CLIP_SIGMAS | 10.0 (suspected, not set) | **12.0 (explicit, verified)** |
| ARTIFACT_DIR | 009-spinquant-hotstart/baseline | 028-post-train-clip12-ttt |

No other changes. Same checkpoint, same TTT settings, same script.

## Hardware ladder

**8×H100 JP only.** No training — GPTQ (~60s) + TTT (~400s) = ~8 min total.
No mini rung needed (no code changes, pure env var).

## Run command

```bash
pip install --break-system-packages brotli sentencepiece
python -c "import brotli, sentencepiece"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout 6456188

# Sanity verify clip is being read from env
grep "MLP_CLIP_SIGMAS\|mlp_clip_sigmas" spinquant_hotstart.py | head -5
# Must show os.environ read — if hardcoded, halt and report to research

mkdir -p /runpod/runs/028-post-train-clip12-ttt
mkdir -p /tmp/torch_inductor_cache_028

NCCL_NET=Socket DATA_DIR=/runpod/data \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_028 \
ARTIFACT_DIR=/runpod/runs/028-post-train-clip12-ttt \
SPINQUANT_MODE=baseline \
SPINQUANT_SEED=42 \
HOTSTART_FP_CKPT=/runpod/runs/008-1736-reproduction/seed_42/pre_gptq.pt \
SEED=42 \
torchrun --standalone --nproc_per_node=8 spinquant_hotstart.py \
  > /runpod/runs/028-post-train-clip12-ttt/run.log 2>&1
```

**Verify MLP_CLIP_SIGMAS=12.0 is active** — check the run.log for the GPTQ line:
`GPTQ:mlp_clip_sigmas=12.0` or similar. If absent or shows 10.0, halt — the env var
is not reaching the GPTQ code and we need to patch the script.

## Inputs

- Checkpoint: `/runpod/runs/008-1736-reproduction/seed_42/pre_gptq.pt` (on JP volume)
- No training data needed for GPTQ/TTT (uses `/runpod/data/` for TTT eval only)

## Checkpoints / artifacts

- `run.log` — GPTQ log + TTT trajectory
- `final_model.int6.ptz` — quantized artifact
- `final.json` — post-GPTQ bpb, post-TTT bpb, eval time

## Stop-early criteria

- `spinquant_hotstart.py` missing or `NotImplementedError` for baseline mode → halt
- Checkpoint not found at `/runpod/runs/008-1736-reproduction/seed_42/pre_gptq.pt` → halt
- GPTQ hangs > 5 min → halt
- TTT eval hangs > 15 min → halt

## Cost estimate

| item | cost |
|---|---|
| 8×H100 JP × ~10 min (GPTQ + TTT compile + TTT eval) | ~$4 |

## Open questions for executor interview

1. **Checkpoint present on volume?** Verify `/runpod/runs/008-1736-reproduction/seed_42/pre_gptq.pt`
   exists and is ~135 MB before provisioning the pod.

2. **MLP_CLIP_SIGMAS env reaching the script?** After the run, grep the log for the
   clip value used. If spinquant_hotstart.py has the clip hardcoded at 10.0 (not reading
   from env), patch that one line and re-run before reporting results.

3. **JP stock?** Provision with `--template-id y5cejece4j`. Do not use other templates.
