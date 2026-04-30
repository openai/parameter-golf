# Parameter Golf — Project Guide

## What This Is

This is an entry for the [OpenAI Parameter Golf challenge](https://github.com/openai/parameter-golf): train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8×H100s. Scored by validation bits-per-byte (BPB) on FineWeb — lower is better.

We use an **autoresearch loop** that runs on RunPod GPU pods: Claude Code (`claude -p`) proposes surgical edits to `train_gpt.py`, training runs, results are evaluated, and improvements are kept.

## Key Files

| File | Purpose |
|------|---------|
| `train_gpt.py` | The training script (from OpenAI). This is what gets modified by experiments. |
| `autoresearch.py` | The autoresearch loop — orchestrates propose→train→evaluate→keep/revert. |
| `program.md` | High-level research directives that guide Claude's proposals. |
| `run_lane.sh` | Launches a specific research lane (core, storage, eval_time, etc). |
| `pod.sh` | RunPod pod management: create, ssh, sync, pull, stop, destroy. |
| `EXPERIMENTS.md` | Human-readable experiment tracker and findings log. |
| `.env.local` | Contains `RUNPOD_API_KEY` and `ANTHROPIC_API_KEY` (not in git). |
| `.pod_id` | ID of the original RunPod pod (not in git). |
| `.lane_pods` | `host:port:label` entries for lane pods (not in git). |

## Architecture

### Autoresearch Loop (`autoresearch.py`)

Each experiment cycle:
1. **Propose**: `claude -p` reads `train_gpt.py`, makes a surgical edit, returns a description
2. **Train**: `torchrun --standalone --nproc_per_node=$GPUS train_gpt.py`
3. **Evaluate**: Parse `val_bpb` (post-int8-quantization) and artifact size from output
4. **Keep/Revert**: If BPB improved and under 16MB, keep. Otherwise revert to previous code.
5. **Snapshot**: Save code + rationale to `autoresearch/experiments/NNNN/`

**Pipelining**: While training experiment N, speculatively proposes experiment N+1. If N is reverted (most common), the speculative proposal is ready immediately. If N is kept, the stale proposal is discarded and re-proposed.

### Research Lanes

Multiple lanes can run in parallel on separate pods, each exploring a different axis:

| Lane | Focus | Key env vars |
|------|-------|-------------|
| `core_discovery` | Architecture, hyperparameters, training tricks | Default settings |
| `storage_discovery` | Compression, quantization, artifact size reduction | `MAX_QUANTIZATION_GAP`, `STORAGE_MAX_REGRESSION` |
| `eval_time_discovery` | Eval-time compute tricks (using the 10-min eval window) | `MAX_EVAL_TIME_MS` |
| `core_promotion` | Validate discoveries at full scale (8×GPU, 10 min) | `EXPERIMENT_SECONDS=600` |

Each lane gets its own namespace directory: `autoresearch/<namespace>/` with independent history, experiments, and best code.

Launch a lane: `BACKGROUND=1 bash ./run_lane.sh <lane_key>`

### Key Metrics

- **val_bpb**: Post-int8-quantization bits-per-byte (THE competition metric)
- **prequant_bpb**: Pre-quantization BPB (shows raw model quality)
- **quantization_gap**: val_bpb - prequant_bpb (how much compression hurts)
- **artifact_bytes**: Total submission size (must be < 16,000,000)
- **model_params**: Parameter count
- **last_step**: Training steps completed in the time budget

## RunPod Infrastructure

### Pod Management (`pod.sh`)

```bash
./pod.sh create [gpus]   # Spin up H100 pod (original pod, saved to .pod_id)
./pod.sh status          # Show pod info
./pod.sh ssh             # Connect to original pod
./pod.sh sync            # Push local files to original pod
./pod.sh pull            # Pull experiment results from ALL pods (original + lanes)
./pod.sh stop            # Pause original pod (keeps volume)
./pod.sh destroy         # Terminate original pod
```

### Current Pods

Pod details are in `.pod_id` (original) and `.lane_pods` (lanes). To check current pods:

```bash
export RUNPOD_API_KEY=$(grep RUNPOD_API_KEY .env.local | cut -d= -f2)
runpodctl pod list -o json
```

### SSH to any pod

```bash
ssh -o StrictHostKeyChecking=no root@<HOST> -p <PORT>
```

Pods land in `/workspace/`. The repo is at `/workspace/parameter-golf/`.

### Setting Up a New Pod

1. Create: `runpodctl pod create --name "name" --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 --template-id "y5cejece4j" --container-disk-in-gb 50 --volume-in-gb 50 --cloud-type SECURE --ssh -o json`
2. Get SSH info: `runpodctl pod get <pod_id> -o json` → look at `.ssh.ip` and `.ssh.port`
3. SSH in and set up:
   ```bash
   cd /workspace && git clone https://github.com/openai/parameter-golf.git && cd parameter-golf
   python3 data/cached_challenge_fineweb.py --variant sp1024
   curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs
   npm install -g @anthropic-ai/claude-code
   pip install --break-system-packages anthropic
   ```
4. Auth Claude Code: copy `/root/.claude/.credentials.json` from an existing authed pod, OR run `claude` interactively and log in with Max subscription
5. Sync local repo files: `rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" --exclude .git --exclude data/datasets --exclude data/tokenizers /path/to/parameter-golf/ root@HOST:/workspace/parameter-golf/`
6. Launch a lane: `PYTHONUNBUFFERED=1 BACKGROUND=1 bash ./run_lane.sh <lane_key>`

### Syncing Changes to Running Pods

When you modify `autoresearch.py`, `program.md`, or other files locally:

```bash
# Sync to a specific pod
rsync -avz -e "ssh -p PORT -o StrictHostKeyChecking=no" \
  autoresearch.py program.md \
  root@HOST:/workspace/parameter-golf/

# Kill and restart the lane
ssh root@HOST -p PORT "kill \$(ps aux | grep 'python3 autoresearch' | grep -v grep | awk '{print \$2}') 2>/dev/null"
ssh root@HOST -p PORT "cd /workspace/parameter-golf && PYTHONUNBUFFERED=1 BACKGROUND=1 bash ./run_lane.sh <lane_key>"
```

### Monitoring

```bash
# Tail a lane's output
ssh root@HOST -p PORT "tail -f /workspace/parameter-golf/autoresearch/<namespace>/autoresearch.out"

# Check all lanes at once
for HP in "HOST1:PORT1:lane1" "HOST2:PORT2:lane2"; do
  HOST=$(echo $HP | cut -d: -f1); PORT=$(echo $HP | cut -d: -f2); LANE=$(echo $HP | cut -d: -f3)
  echo "=== $LANE ==="; ssh -o StrictHostKeyChecking=no root@$HOST -p $PORT "tail -10 /workspace/parameter-golf/autoresearch/$LANE/autoresearch.out" 2>&1; echo ""
done
```

### Pulling Results

`./pod.sh pull` syncs from all pods (original + `.lane_pods`). It pulls:
- `autoresearch/<namespace>/experiments/NNNN/README.md` — rationale + results
- `autoresearch/<namespace>/experiments/NNNN/train_gpt.py` — code snapshot
- `autoresearch/<namespace>/history.jsonl` — machine-readable history
- `autoresearch/<namespace>/train_gpt.best.py` — current best code

Training logs (`*.log`) are excluded to keep git clean.

### Cost

- 1×H100 SXM: ~$2.69/hr (secure cloud)
- Always `./pod.sh stop` or destroy pods when not in use
- Check balance: `runpodctl user`

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `EXPERIMENT_SECONDS` | 180 | Training time budget per experiment |
| `GPUS` | 1 | Number of GPUs for training |
| `AUTORESEARCH_MODEL` | opus | Claude model for proposals |
| `CLAUDE_EFFORT` | high | Thinking effort level |
| `MAX_EXPERIMENTS` | 100 | Max experiments before stopping |
| `BASELINE_BPB` | 0 | Seed BPB for first comparison |
| `AUTORESEARCH_NAMESPACE` | "" | Subdirectory for lane isolation |
| `AUTORESEARCH_LANE` | core | Lane type (core/storage/eval_time) |
| `AUTORESEARCH_STAGE` | discovery | Stage (discovery/promotion) |
| `PYTHONUNBUFFERED` | — | Set to 1 for real-time log output |
