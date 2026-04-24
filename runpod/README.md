# RunPod standardization for Parameter Golf

This folder contains the pieces that make 8×H100 pod spin-up take ~45 s instead
of the typical 8–15 min. Three files:

- `Dockerfile` — custom image, extends `runpod/pytorch:2.9.1-cu128-py311` with
  all competition deps pre-installed. Build once, push once, reuse every pod.
- `bootstrap.sh` — idempotent per-pod entry. Sets compile-cache env vars to
  the Network Volume, clones the repo if missing, downloads FineWeb SP8192 if
  missing. No-op on warm pods.
- `launch_experiment.sh` — kicks off a training+eval run. Takes submission
  folder + seed + optional `KEY=VAL` overrides. Logs go to
  `/workspace/runs/<timestamp>_<folder>_s<seed>/`.

## One-time setup (do this now)

1. **Build and push image**

   ```bash
   docker build -t $DOCKERHUB_USER/parameter-golf:2026-04-25 -f runpod/Dockerfile .
   docker push $DOCKERHUB_USER/parameter-golf:2026-04-25
   ```

2. **Create a 150 GB Network Volume** in the IN region (same region as the
   Secure 8×H100 SKUs you will use for the final 3-seed runs).

3. **Create a RunPod Template** with:
   - Container Image: `$DOCKERHUB_USER/parameter-golf:2026-04-25`
   - Container Disk: 50 GB
   - Volume Mount Path: `/workspace`
   - Volume Disk: 150 GB (attached above)
   - Container Start Command: `bash -c "cp runpod/bootstrap.sh /workspace/ 2>/dev/null; /workspace/bootstrap.sh; sleep infinity"`
   - Expose TCP Port 22 for SSH

4. **First warm-up** — deploy a 1×H100 Community pod from the template. The
   first run will download the dataset (~20 GB, 10–15 min) onto the volume.
   Tear the pod down. From now on all pods attached to this volume skip this.

## Running Day 3 sweep (the one that matters)

Spin up one 8×H100 Community pod from the template. After boot (~45 s):

```bash
ssh <pod>
cd /workspace/repo/parameter-golf
git pull --rebase
# First, reproduce the PR #1797 baseline (mixture off) as a sanity anchor.
/workspace/launch_experiment.sh records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix 42 \
    NGRAM_MIX_ENABLED=0 PREQUANT_ONLY=1
# Then eval the same trained artifact with mixture on, sweeping alpha/beta.
for alpha in 1.0 2.0 3.0; do
  for beta in -0.10 -0.25 -0.40; do
    /workspace/launch_experiment.sh records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix 42 \
        NGRAM_MIX_ENABLED=1 NGRAM_MIX_ALPHA=$alpha NGRAM_MIX_BETA=$beta \
        TTT_ENABLED=0 PREQUANT_ONLY=1
  done
done
```

Each eval is ~90 s wallclock after compile warm-up (cache hit). 9-point sweep
takes ~15 min. Cost: ~$6.

## Kill-switch budget guards

`launch_experiment.sh` only needs a seed + submission folder; CI-style driver
scripts go above it. Recommended paranoia:

- Hard-cap `MAX_WALLCLOCK_SECONDS` to 620 for any record attempt.
- After each Day 3 sweep point, parse `val_bpb` from the log; if no config
  beats base by ≥ 0.003 bpb, stop the sweep and pivot (this is the Day-3
  kill-switch referenced in the budget plan).
