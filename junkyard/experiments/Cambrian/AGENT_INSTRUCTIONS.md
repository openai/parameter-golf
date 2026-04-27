# Cambrian Bio Seam Sweep — Agent Instructions

You are managing a Vast.ai GPU rental to complete the Cambrian bio seam ablation sweep.
Read ALL of this before taking any action.

---

## Your Goal

Run `experiments/Cambrian/run_bio_sweep.sh` on an 8×H200 (or 8×H100 SXM) pod and collect
the results table. The sweep tests 6 arms (pure GDN baseline + 4 individual bio seams + all).
Each arm takes ~3.5 minutes. Total ~25 minutes of compute.

---

## Step 1 — Rent a Pod

Use the Vast.ai CLI. Find a suitable instance:

```bash
vastai search offers 'num_gpus=8 gpu_name=H200 reliability>0.95 inet_down>500' -o dph_total
```

If no H200, fall back to H100 SXM4:
```bash
vastai search offers 'num_gpus=8 gpu_name=H100_SXM4 reliability>0.95 inet_down>500' -o dph_total
```

Requirements:
- **8 GPUs** (not fewer — sweep is tuned for 8)
- **H200 or H100 SXM4** (need NVLink for NCCL)
- **pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime** image
- **100+ GB disk**
- Reliability > 0.95

Create the instance:
```bash
vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
  --disk 100 \
  --ssh \
  --direct
```

Wait for it to show `running`, then get SSH command:
```bash
vastai ssh-url <INSTANCE_ID>
```

---

## Step 2 — Connect and Clone Repo

SSH in using the provided command (uses `~/.ssh/id_ed25519_apollo`):
```bash
ssh -i ~/.ssh/id_ed25519_apollo -p <PORT> root@<HOST>
```

Inside the pod:
```bash
cd /workspace
git clone https://github.com/newjordan/parameter-golf.git
cd parameter-golf
git checkout test
```

---

## Step 3 — Run Pod Setup

```bash
bash experiments/pod_setup.sh
```

This downloads datasets and tokenizers. Takes 5-10 minutes.

---

## Step 4 — Fix Environment (CRITICAL — do this EVERY time after pod_setup.sh)

The FA3 wheel installed by pod_setup.sh upgrades torch and installs a broken NCCL.
Fix it:

```bash
pip uninstall nvidia-nccl-cu13 -y
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 -q
pip install 'nvidia-nccl-cu12==2.23.4' -q
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
```

Verify:
```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.device_count(), 'GPUs')"
# Expect: 2.5.1+cu124   8 GPUs
```

**DO NOT** run `pip install triton==3.2.0` — it breaks torch.compile.

---

## Step 5 — Run the Sweep

```bash
mkdir -p logs
WALLCLOCK=180 NPROC=8 DELTA_LAYERS=2 bash experiments/Cambrian/run_bio_sweep.sh 2>&1 | tee logs/cambrian_bio_sweep_agent.log
```

The sweep runs 6 arms sequentially. Expected output per arm:
```
--- gdn_base (myelin=0 circadian=0 clonal=0 astrocyte=0) ---
...
step:47/20000 val_loss:X val_bpb:X.XXXX ...
DIAGNOSTIC post_ema val_bpb:X.XXXX
final_eval:skipped sliding/ngram by SKIP_FINAL_EVAL=1
    done -> /workspace/parameter-golf/logs/cambrian_bio_sweep_.../gdn_base.log
```

At the end, a results table prints automatically.

---

## Step 6 — Collect Results

After the sweep finishes, extract the key metrics:

```bash
LOG_DIR=$(ls -td /workspace/parameter-golf/logs/cambrian_bio_sweep_* | head -1)
for f in gdn_base gdn_myelin gdn_circadian gdn_clonal gdn_astrocyte gdn_all; do
    log="${LOG_DIR}/${f}.log"
    train_bpb=$(grep -oP 'stopping_early.*\n.*val_bpb:\K[\d.]+' "${log}" 2>/dev/null \
             || grep 'val_bpb:' "${log}" | grep 'step:' | tail -1 | grep -oP 'val_bpb:\K[\d.]+')
    ema_bpb=$(grep -oP 'DIAGNOSTIC post_ema val_bpb:\K[\d.]+' "${log}" 2>/dev/null || echo N/A)
    echo "${f}: train_bpb=${train_bpb} ema_bpb=${ema_bpb}"
done
```

Also grab the auto-generated results table at the end of the sweep log:
```bash
grep -A 20 'RESULTS (vs gdn_base)' logs/cambrian_bio_sweep_agent.log
```

---

## Step 7 — Report Results

Report these numbers back:
1. The results table (EMA_BPB and DELTA columns for all 6 arms)
2. The step-47 val_bpb for each arm (from `step:N/20000 val_bpb:X.XXXX` lines)
3. Any arms that crashed and the error

Known baseline for comparison (from prior run 2026-03-27):
- gdn_base: step-47 val_bpb = **3.3328**, EMA val_bpb = 3.7961
- gdn_myelin: step-47 val_bpb = **3.2150**, EMA val_bpb = 3.8345

---

## Step 8 — Shut Down Pod

**IMPORTANT: Destroy the pod after results are collected. Don't leave it running.**

```bash
vastai destroy instance <INSTANCE_ID>
```

---

## Troubleshooting

**NCCL error at startup**: Re-run the Step 4 env fix. Make sure LD_LIBRARY_PATH is exported.

**OOM error**: The sweep uses DELTA_LAYERS=2. If OOM occurs, try DELTA_LAYERS=1.

**Arm hangs after "Code size: X bytes"**: Kill it (`Ctrl+C`) and check that `SKIP_FINAL_EVAL=1`
is being passed. The sweep script sets this automatically — if running run.sh directly, set it manually.

**"torch._dynamo hit recompile_limit"**: These warnings are harmless. Training continues in eager
mode for the affected function. Step time may be slightly higher but results are valid.

**Port already in use / NCCL init hang between arms**: The sweep script kills lingering processes
with `pkill -f train_gpt.py` before each arm. If still stuck, `kill -9` all python processes
and restart from the failed arm by editing the NAMES array in run_bio_sweep.sh.

**Step time > 6s/step**: Something is wrong with GPU detection or NCCL. Re-run env fix.
Expected: ~3.5-4.5s/step on 8×H200.
