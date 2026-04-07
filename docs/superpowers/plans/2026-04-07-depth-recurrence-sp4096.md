# Experiment 4b: Depth Recurrence with SP4096 Tokenizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run Experiment 4's winning BigramHash + depth recurrence config with SP4096 tokenizer (4096 BPE vocab) to gain ~0.01 BPB, targeting ~1.088 BPB.

**Architecture:** Identical to Experiment 4 (PR #1435) but with vocab_size=4096 instead of 1024. SP4096 data is not in the public HF manifest, so we train the tokenizer and retokenize the dataset on-pod using `data/download_hf_docs_and_tokenize.py`.

**Tech Stack:** PyTorch, FlashAttention 3, SentencePiece (tokenizer training), RunPod (8xH100 SXM)

---

## File Structure

| File | Purpose |
|------|---------|
| `experiments/exp4b_train_gpt_sp4096.py` | Training script (BigramHash variant with vocab_size=4096) |
| `experiments/exp4b_tokenizer_specs.json` | Tokenizer spec for SP4096 (used on-pod to retokenize data) |
| `experiments/exp4b_results.md` | Run results summary |
| `records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/train_gpt.py` | Submission script |
| `records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/README.md` | Submission writeup |
| `records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/submission.json` | Submission metadata |
| `records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/train_seed*.log` | Training logs (3 seeds) |

---

### Task 1: Create Worktree and Branch

- [ ] **Step 1: Create a new branch from main in a git worktree**

```bash
git worktree add /Users/abhayanand/GithubFolders/pgolf-exp4b-sp4096 -b exp4b-recurrence-sp4096 main
```

- [ ] **Step 2: Verify worktree**

```bash
cd /Users/abhayanand/GithubFolders/pgolf-exp4b-sp4096
git branch --show-current
```

Expected: `exp4b-recurrence-sp4096`

All subsequent tasks work from `/Users/abhayanand/GithubFolders/pgolf-exp4b-sp4096`.

---

### Task 2: Prepare SP4096 Training Script

**Files:**
- Create: `experiments/exp4b_train_gpt_sp4096.py`

- [ ] **Step 1: Copy the BigramHash script from exp4-recurrence branch**

```bash
cp /Users/abhayanand/GithubFolders/pgolf-exp4-recurrence/experiments/exp4_train_gpt_bigram.py \
   experiments/exp4b_train_gpt_sp4096.py
```

- [ ] **Step 2: Change vocab_size default from 1024 to 4096**

In `experiments/exp4b_train_gpt_sp4096.py`, find:

```python
    vocab_size = int(os.environ.get('VOCAB_SIZE', 1024))
```

Change to:

```python
    vocab_size = int(os.environ.get('VOCAB_SIZE', 4096))
```

- [ ] **Step 3: Verify the script parses and has correct vocab_size**

```bash
python3 -c "import ast; ast.parse(open('experiments/exp4b_train_gpt_sp4096.py').read()); print('OK')"
grep "VOCAB_SIZE.*4096" experiments/exp4b_train_gpt_sp4096.py
```

Expected: `OK` and one grep match.

- [ ] **Step 4: Commit**

```bash
git add experiments/exp4b_train_gpt_sp4096.py
git commit -m "Add SP4096 depth recurrence + BigramHash training script"
```

---

### Task 3: Prepare Tokenizer Spec for SP4096

**Files:**
- Create: `experiments/exp4b_tokenizer_specs.json`

- [ ] **Step 1: Create tokenizer spec file**

Create `experiments/exp4b_tokenizer_specs.json` with:

```json
{
  "tokenizers": [
    {
      "name": "sp_bpe_4096",
      "dataset_suffix": "sp4096",
      "vocab_size": 4096
    }
  ]
}
```

This tells `download_hf_docs_and_tokenize.py` to train a SentencePiece BPE tokenizer with 4096 vocab and export tokenized shards to `fineweb10B_sp4096/`.

- [ ] **Step 2: Commit**

```bash
git add experiments/exp4b_tokenizer_specs.json
git commit -m "Add SP4096 tokenizer spec for retokenization"
```

---

### Task 4: Create RunPod Pod and Set Up Environment

- [ ] **Step 1: Create the pod**

```bash
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name "pgolf-exp4b-sp4096" \
  --cloud-type SECURE
```

Save the pod ID.

If SECURE fails:
```bash
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name "pgolf-exp4b-sp4096" \
  --cloud-type COMMUNITY
```

- [ ] **Step 2: Wait for pod, get SSH info**

```bash
runpodctl ssh info <pod-id>
```

Note `<ip>` and `<port>`.

- [ ] **Step 3: Clone repo and install deps**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'SETUP'
rm -rf /workspace/parameter-golf
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install --break-system-packages zstandard brotli sentencepiece
echo "=== Setup done ==="
SETUP
```

Note: `sentencepiece` is needed for tokenizer training. It may already be installed in the template.

---

### Task 5: Retokenize Data with SP4096 on Pod

This is the key new step. We download the raw docs from HF, train a 4096-vocab SentencePiece tokenizer, and retokenize all 10B tokens into shards.

- [ ] **Step 1: SCP the tokenizer spec to the pod**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  experiments/exp4b_tokenizer_specs.json \
  root@<ip>:/workspace/parameter-golf/data/tokenizer_specs_sp4096.json
```

- [ ] **Step 2: Run retokenization**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'RETOK'
cd /workspace/parameter-golf
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data \
  --tokenizer-config data/tokenizer_specs_sp4096.json \
  --with-docs \
  --skip-byte \
  2>&1 | tee /workspace/retokenize_sp4096.log
echo "=== Retokenization done ==="
RETOK
```

This command:
1. Downloads `docs_selected.jsonl` (~5GB raw text) from HF
2. Trains a SentencePiece BPE tokenizer with vocab_size=4096
3. Tokenizes all docs into train/val shards at `data/datasets/fineweb10B_sp4096/`
4. Creates tokenizer at `data/tokenizers/fineweb_4096_bpe.model`

Expected runtime: 10-30 minutes depending on CPU throughput (pod has 224 vCPUs).

- [ ] **Step 3: Verify retokenized data**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'CHECK'
echo "Train shards:"
ls /workspace/parameter-golf/data/datasets/fineweb10B_sp4096/fineweb_train_*.bin | wc -l
echo "Val shards:"
ls /workspace/parameter-golf/data/datasets/fineweb10B_sp4096/fineweb_val_*.bin | wc -l
echo "Tokenizer:"
ls -la /workspace/parameter-golf/data/tokenizers/fineweb_4096_bpe.*
CHECK
```

Expected: 80+ training shards, 1+ val shard, tokenizer .model and .vocab files present.

If retokenization fails, fall back to SP1024:
```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> \
  'cd /workspace/parameter-golf && python3 data/cached_challenge_fineweb.py --variant sp1024'
```
And use vocab_size=1024 (identical to Experiment 4, abort this experiment).

---

### Task 6: Run Training — Seed 1337

- [ ] **Step 1: SCP the SP4096 training script to pod**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  experiments/exp4b_train_gpt_sp4096.py \
  root@<ip>:/workspace/parameter-golf/train_gpt.py
```

- [ ] **Step 2: Run training**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=1337 RUN_ID=exp4b_sp4096_s1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4b_sp4096_s1337.log
TRAIN
```

Expected: ~590s training, ~5200-5400 steps. Recurrence activates at step 3000. Final `final_int6_sliding_window val_bpb` should be ~1.088-1.092.

Monitor for:
- `vocab_size: 4096` in hyperparameters printout
- `train_shards: 80+` (not 1!)
- `recurrence:activated at step 3000`
- Artifact under 16,000,000 bytes

- [ ] **Step 3: Collect log**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4b_sp4096_s1337.log \
  experiments/exp4b_sp4096_s1337.log
```

- [ ] **Step 4: Verify result**

```bash
grep -i "val_bpb\|sliding.*bpb\|final.*bpb\|vocab_size\|train_shards" experiments/exp4b_sp4096_s1337.log | tail -10
```

If BPB > 1.10, troubleshoot:
- Check `vocab_size: 4096` (not 1024)
- Check `train_shards` count (must be 80+)
- Check recurrence activated

---

### Task 7: Run Remaining Seeds (42 and 2024)

- [ ] **Step 1: Run seed 42**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=42 RUN_ID=exp4b_sp4096_s42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4b_sp4096_s42.log
TRAIN
```

- [ ] **Step 2: Run seed 2024**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=2024 RUN_ID=exp4b_sp4096_s2024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4b_sp4096_s2024.log
TRAIN
```

- [ ] **Step 3: Collect all logs**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4b_sp4096_s42.log \
  experiments/exp4b_sp4096_s42.log

scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4b_sp4096_s2024.log \
  experiments/exp4b_sp4096_s2024.log
```

- [ ] **Step 4: Compute 3-seed mean/std**

```bash
echo "=== 3-Seed Results ==="
for f in experiments/exp4b_sp4096_s1337.log experiments/exp4b_sp4096_s42.log experiments/exp4b_sp4096_s2024.log; do
  echo "$f:"
  grep -i "final_int6_sliding_window" "$f" | tail -1
done
```

Record mean and std. For a record submission:
- Mean BPB < 1.1097 (at least 0.005 below SOTA 1.1147)
- Std < 0.002

---

### Task 8: Stop RunPod Pod

- [ ] **Step 1: Stop the pod**

```bash
runpodctl pod stop <pod-id>
```

- [ ] **Step 2: Verify stopped**

```bash
runpodctl pod list
```

---

### Task 9: Create Submission Folder

**Files:**
- Create: `records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/`

- [ ] **Step 1: Create directory and copy files**

```bash
mkdir -p records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096

cp experiments/exp4b_train_gpt_sp4096.py \
   records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/train_gpt.py

cp experiments/exp4b_sp4096_s1337.log \
   records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/train_seed1337.log
cp experiments/exp4b_sp4096_s42.log \
   records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/train_seed42.log
cp experiments/exp4b_sp4096_s2024.log \
   records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/train_seed2024.log
```

- [ ] **Step 2: Create submission.json**

Fill in actual values from logs:

```json
{
    "author": "Abhay Anand",
    "github_id": "AbhayAnandUCSD",
    "name": "11L Depth Recurrence + BigramHash + SP4096 + EMA 0.9965",
    "blurb": "Depth recurrence (layers 4,5 repeat once, 13 virtual layers) with BigramHash(1536,112), SP4096 tokenizer (trained and tokenized locally), EMA 0.9965, GPTQ int6 + Brotli. Built on PR #1334/#1421 architecture.",
    "date": "2026-04-07T00:00:00Z",
    "track": "10min-16mb",
    "val_loss": <FILL>,
    "val_bpb": <FILL>,
    "step_stop": <FILL>,
    "wallclock_seconds": 600,
    "bytes_total": <FILL>,
    "gpu": "8xH100 SXM"
}
```

- [ ] **Step 3: Create README.md**

```markdown
# 11L Depth Recurrence + BigramHash + SP4096 — val_bpb <FILL>

**val_bpb: <FILL>** (3-seed mean) | **~15 MB** | 8xH100 SXM, 600s

## Summary

Depth recurrence architecture (PR #1334/#1421) with BigramHash(1536, 112) and SP4096
tokenizer. SP4096 tokenizer trained locally using `data/download_hf_docs_and_tokenize.py`
since it is not in the public HF manifest.

Improvement over our SP1024 variant (PR #1435, 1.0980 BPB): <FILL> BPB.

## 3-Seed Results (8xH100 SXM)

| Seed | Pre-quant BPB | Sliding BPB (s64) | Artifact |
|------|---------------|-------------------|----------|
| 1337 | <FILL> | <FILL> | <FILL> |
| 42 | <FILL> | <FILL> | <FILL> |
| 2024 | <FILL> | <FILL> | <FILL> |
| **Mean** | | **<FILL>** | |

## Architecture

Same as PR #1435 (our SP1024 submission) but with SP4096 tokenizer:
- 11 layers, 512-dim, depth recurrence on layers 4,5 (13 virtual)
- BigramHash(1536, 112) + SmearGate
- Skip gates, parallel residuals (layers 7+), MuonEq-R
- Value Embedding (dim=128, layers 9,10), QK-Gain init=5.0
- EMA 0.9965, warmdown 66.7%, GPTQ int6 + Brotli

## Data Preparation

SP4096 tokenizer trained on-pod:
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data \
  --tokenizer-config data/tokenizer_specs_sp4096.json \
  --skip-byte

## Attribution

- Base architecture: PR #1334 by @aryanbhosale
- EMA tuning: PR #1421 by @X-Abhishek-X
- BigramHash: cumulative competition stack
```

- [ ] **Step 4: Commit**

```bash
git add records/track_10min_16mb/2026-04-07_DepthRecurrence_BigramHash_SP4096/
git commit -m "Record: 11L Depth Recurrence + BigramHash + SP4096 — val_bpb <FILL>"
```

---

### Task 10: Create Pull Request

- [ ] **Step 1: Push to fork**

```bash
git push -u fork exp4b-recurrence-sp4096
```

If fork remote doesn't exist:
```bash
git remote add fork https://github.com/AbhayAnandUCSD/parameter-golf.git
git push -u fork exp4b-recurrence-sp4096
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --repo openai/parameter-golf \
  --head AbhayAnandUCSD:exp4b-recurrence-sp4096 \
  --base main \
  --title "Record: 11L Depth Recurrence + BigramHash + SP4096 — val_bpb <FILL>" \
  --body "$(cat <<'EOF'
## Summary

- Depth recurrence (layers 4,5 repeat once, 13 virtual), activated step 3000
- BigramHash(1536, 112) + SmearGate
- SP4096 tokenizer (trained locally, not in public HF manifest)
- EMA 0.9965, GPTQ int6 + Brotli

Improvement over our SP1024 submission (PR #1435, 1.0980 BPB).

## Results (3 seeds, 8xH100 SXM)

| Seed | Sliding BPB (s64) | Artifact |
|------|-------------------|----------|
| 1337 | <FILL> | <FILL> |
| 42 | <FILL> | <FILL> |
| 2024 | <FILL> | <FILL> |
| **Mean** | **<FILL>** | |

## Attribution

- Base architecture: PR #1334 by @aryanbhosale
- EMA tuning: PR #1421 by @X-Abhishek-X
EOF
)"
```

- [ ] **Step 3: Record PR URL in experiments/exp4b_results.md**

---

### Task 11: Cleanup

- [ ] **Step 1: Delete RunPod pod**

```bash
runpodctl pod delete <pod-id>
```

- [ ] **Step 2: Update experiments/exp4b_results.md**

Record final BPB, 3-seed results, comparison to SP1024, PR URL, cost.
