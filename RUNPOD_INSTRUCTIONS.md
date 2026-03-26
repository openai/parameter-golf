# RunPod v2: Fair Comparison Instructions

## Step 1: Push your code to GitHub

```bash
cd ~/Projects/parameter-golf-bese
git add -A
git commit -m "v2: fast BPE, LeakyReLU², EMA, 11L/3x MLP defaults"
git push origin experiment-results
```

## Step 2: Create RunPod Pod

Go to https://www.runpod.io/console/pods and create:
- **GPU:** 1x H100 SXM ($2.69/hr) — use this for development/testing
- **Template:** `runpod/parameter-golf:latest`
- **Container disk:** 50GB (need room for decoded docs + BESE shards)

Expected cost: ~$2-3 for the full run (~40-60 min).

## Step 3: SSH into the pod and run

```bash
# SSH in (replace POD_ID with your actual pod ID)
ssh root@POD_ID -i ~/.runpod/ssh/RunPod-Key-Go

# On the pod:
cd /workspace

# Clone your BESE repo (branch: experiment-results)
git clone -b experiment-results https://github.com/mrbese/parameter-golf.git bese

# Download FineWeb data (if not already cached by template)
cd parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024 && cd ..

# Install sentencepiece (if not in template)
pip install sentencepiece

# Run the fair comparison
python3 bese/scripts/runpod_v2.py --num-merges 250 --num-layers 11 --model-dim 512 --mlp-mult 3
```

## What it does

1. **Decodes ALL 10 SP shards** → ~80K+ FineWeb documents (full data parity with baseline)
2. **Trains BESE+BPE** (250 merges) using the fast indexed algorithm (~30s vs 22min)
3. **Exports BESE binary shards** (matching upstream format)
4. **Runs baseline training** (SP1024, 9L/512d/2x MLP, 10 min)
5. **Runs BESE training** (BESE+BPE, 11L/512d/3x MLP, LeakyReLU², EMA, 10 min)
6. **Prints side-by-side results** (val_loss, val_bpb, model size)

## Expected timeline on 1xH100

| Step | Time |
|------|------|
| Download FineWeb data | ~5 min (if not cached) |
| Decode all 10 SP shards | ~3-5 min |
| Train BESE BPE (fast) | ~30 sec |
| Export BESE shards | ~5-10 min |
| Baseline training | 10 min |
| BESE training | 10 min |
| **Total** | **~35-45 min** |

## Troubleshooting

- **Disk full:** The pod needs ~50GB. If template only has 20GB, recreate with larger container disk.
- **OOM:** If 1xH100 OOMs on 11L/3x MLP, try `--model-dim 512 --mlp-mult 2` first.
- **SSH drops:** Use `screen` or `tmux` before running the script: `screen -S bese` then run the command. Reattach with `screen -r bese`.
