# RunPod Pipeline — PR #1610 + Posterior Corrector

**Branch**: `submission/pr1610-corrector`
**Ancestry anchor**: `a33191f572430566b88c4d61badb0369e1e6f9a3` (warmup fix — verified by `00_verify_pod.sh`)
**Session launch SHA**: `<LAUNCH_SHA>` (pinned via `EXPECTED_SHA` env var; see Block 1)
**Image**: `amay01/parameter-golf@sha256:6206b37e0f363c3886323b391e64bb0e46b1623e203b6b9e55f165fd774ea2cf`

See `pod_launch.md` for pod creation. Then run the blocks below **in order**.

---

## Block 1 — Clone repo + run automated pipeline (Stages 0–3)

Paste this as one block immediately after SSH connection:

```bash
cd /workspace
git clone \
  --branch submission/pr1610-corrector \
  https://github.com/amrayach/parameter-golf.git
cd parameter-golf
git checkout <LAUNCH_SHA>                   # pin exact launch commit (see header)
EXPECTED_SHA="<LAUNCH_SHA>" \
  bash scripts/runpod_pipeline/run_all.sh 2>&1 | tee /workspace/pipeline_run_all.log
```

**What it does**: verifies pod, downloads ~24 GB dataset, runs Gate A full train+eval
(seed 0, ~20 min), runs 3 corrector ablations (~15 min total). Stops before Stage 4.

**Kill signals baked in**:
- BPB > 1.07516564 (published 1.07216564 + 0.003) → pipeline aborts (don't pay for Stage 4)
- eval > 600s → pipeline aborts
- artifact > per-seed limit → pipeline aborts
- disk < 20 GB remaining at any stage → pipeline aborts

**Expected wall-clock**: ~55–65 min. Cost: ~$20–23.

---

## Block 2 — Review and decide

```bash
bash scripts/runpod_pipeline/04_decide_and_proceed.sh
```

Reads `runs/ablation_summary.json` and prints whether corrector passes the kill criterion.
Output will tell you exactly which Block 3 command to run.

---

## Block 3a — Gate B (primary path, corrector passed kill criterion)

Replace `<ALPHA>` and `<ORDERS>` with the values shown in Block 2 output:

```bash
BEST_ALPHA=<ALPHA> BEST_ORDERS='<ORDERS>' \
  bash scripts/runpod_pipeline/04a_gate_b.sh \
  2>&1 | tee /workspace/gate_b.log
```

Example with typical best config:
```bash
BEST_ALPHA=0.3 BEST_ORDERS='5,8,12' \
  bash scripts/runpod_pipeline/04a_gate_b.sh \
  2>&1 | tee /workspace/gate_b.log
```

**What it does**: trains seeds 1 and 2 with the best corrector config (~40 min total),
verifies 3-seed mean within 0.002 of published 1.07280628.

---

## Block 3b — Fallback (corrector killed)

```bash
bash scripts/runpod_pipeline/04b_fallback_level1a.sh \
  2>&1 | tee /workspace/fallback.log
```

**What it does**: eval-only requantization variants on seed-0 checkpoint (~10 min total).

---

## Block 4 — Preserve artifacts (ALWAYS run before pod termination)

Choose your upload target:

**Option A — HuggingFace Hub:**
```bash
UPLOAD_TARGET="hf:amrayach/parameter-golf-runs:pr1610-corrector" \
  bash scripts/runpod_pipeline/05_preserve_artifacts.sh
```

**Option B — rsync to Pegasus:**
```bash
UPLOAD_TARGET="rsync:amay@pegasus.dfki.de:/netscratch/amay/runs" \
  bash scripts/runpod_pipeline/05_preserve_artifacts.sh
```

**If no upload target yet (save locally, upload later):**
```bash
bash scripts/runpod_pipeline/05_preserve_artifacts.sh
# then manually: scp root@<pod>:/workspace/runs_*.tar.gz .
```

---

## Key files after pipeline completes

| File | Contents |
|------|----------|
| `runs/gate_a_summary.json` | Seed-0 BPB, eval time, artifact size, PASS/FAIL |
| `runs/ablation_summary.json` | All 3 ablation deltas, best config, recommended path |
| `runs/gate_b_summary.json` | 3-seed mean BPB, PASS/FAIL |
| `runs/seed0_log.txt` | Full train+eval log, seed 0 |
| `runs/seed1_log.txt` | Full train+eval log, seed 1 |
| `runs/seed2_log.txt` | Full train+eval log, seed 2 |
| `runs/ablation_*_log.txt` | Per-ablation eval logs |
| `/workspace/checkpoints/seed{0,1,2}/` | FP32 + int6 checkpoints |

---

## Quick log inspection commands

```bash
# Gate A result
grep "quantized_ttt_phased\|Total submission size\|GATE_A" runs/seed0_log.txt | tail -5

# Ablation deltas
cat runs/ablation_summary.json | python3 -c "
import json, sys
s=json.load(sys.stdin)
for l,r in sorted(s['runs'].items()):
    print(f'{l}: alpha={r[\"config\"][\"CORRECTOR_ALPHA\"]} delta={r[\"delta\"]:+.6f}')
print('Recommended:', s['recommended_path'])
"

# Gate B 3-seed summary
cat runs/gate_b_summary.json
```

---

## Emergency: disk almost full

```bash
# Check usage
df -h /workspace
du -sh /workspace/parameter-golf/data /workspace/checkpoints /workspace/parameter-golf/runs

# Safe to remove if Gate A checkpoint already persisted to /workspace/checkpoints/seed0/
rm -rf /workspace/parameter-golf/runs/seed0/final_model.pt
```

---

## Abort / re-run a single stage

Each stage is idempotent where possible. To re-run from Stage 3 only:

```bash
# Remove stale ablation results first
rm -f runs/ablation_results.json runs/ablation_summary.json
bash scripts/runpod_pipeline/03_ablations.sh
```

To re-run Gate A (forces fresh training even if checkpoint exists):
```bash
rm -f runs/seed0_log.txt /workspace/checkpoints/seed0/final_model.int6.ptz
bash scripts/runpod_pipeline/02_gate_a.sh
```
