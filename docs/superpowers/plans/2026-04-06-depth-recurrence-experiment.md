# Experiment 4: Depth Recurrence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat merged SOTA (1.1147 BPB) by running PR #1421's depth recurrence script (proven at 1.0925 BPB) on RunPod 8xH100, with optional BigramHash enhancement.

**Architecture:** Adopt PR #1421 verbatim — 11L/512d with layers 4,5 repeating once (13 virtual layers), SP4096 tokenizer, MuonEq-R, skip gates, parallel residuals, EMA 0.9965, GPTQ int6 + Brotli.

**Tech Stack:** PyTorch, FlashAttention 3, Muon optimizer, GPTQ quantization, Brotli compression, RunPod (8xH100 SXM)

---

## File Structure

| File | Purpose |
|------|---------|
| `experiments/exp4_train_gpt.py` | Training script extracted from PR #1421 (vanilla) |
| `experiments/exp4_train_gpt_bigram.py` | Training script with BigramHash added (optional enhancement) |
| `experiments/exp4_results.md` | Run logs and results summary |
| `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_gpt.py` | Final submission script |
| `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/README.md` | Submission writeup |
| `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/submission.json` | Submission metadata |
| `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed*.log` | Training logs (3 seeds) |

---

### Task 1: Extract Training Script From PR #1421

**Files:**
- Create: `experiments/exp4_train_gpt.py`

- [ ] **Step 1: Extract the train_gpt.py from PR #1421 diff**

The script is a new file (all additions) in the PR diff. Extract it by stripping the `+` prefix from each line:

```bash
gh pr diff 1421 --repo openai/parameter-golf | python3 -c "
import sys
in_file = False
for line in sys.stdin:
    if line.startswith('+++ b/') and 'train_gpt.py' in line:
        in_file = True
        continue
    if in_file and line.startswith('diff --git'):
        break
    if in_file and line.startswith('+'):
        print(line[1:], end='')
    elif in_file and line.startswith('@@'):
        continue
" > experiments/exp4_train_gpt.py
```

Verify the file is valid Python:

```bash
python3 -c "import ast; ast.parse(open('experiments/exp4_train_gpt.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Verify key hyperparameters match PR #1421**

Check critical values are present:

```bash
grep "ema_decay.*0.9965" experiments/exp4_train_gpt.py
grep "recur_layers.*4,5" experiments/exp4_train_gpt.py
grep "recur_start_step.*3000" experiments/exp4_train_gpt.py
grep "vocab_size.*4096" experiments/exp4_train_gpt.py
grep "muon_wd.*0.09" experiments/exp4_train_gpt.py
```

Expected: all 5 grep commands produce matches.

- [ ] **Step 3: Commit**

```bash
git add experiments/exp4_train_gpt.py
git commit -m "Extract PR #1421 depth recurrence script for experiment 4"
```

---

### Task 2: Create BigramHash Variant (Optional Enhancement)

**Files:**
- Create: `experiments/exp4_train_gpt_bigram.py`
- Reference: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py` (lines 671-712 for SmearGate + BigramHashEmbedding classes)

- [ ] **Step 1: Copy vanilla script as base**

```bash
cp experiments/exp4_train_gpt.py experiments/exp4_train_gpt_bigram.py
```

- [ ] **Step 2: Add BigramHash hyperparameters to Hyperparameters class**

In `experiments/exp4_train_gpt_bigram.py`, add these lines to the `Hyperparameters` class after the `qk_gain_init` line:

```python
    # BigramHash
    bigram_vocab_size = int(os.environ.get('BIGRAM_VOCAB_SIZE', 1536))
    bigram_dim = int(os.environ.get('BIGRAM_DIM', 112))
```

- [ ] **Step 3: Add SmearGate and BigramHashEmbedding classes**

Add these classes after the `RMSNorm` class definition and before the `Rotary` class:

```python
class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
    def bigram_hash(self, tokens: Tensor) -> Tensor:
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()
    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)
```

- [ ] **Step 4: Wire BigramHash and SmearGate into GPT model**

In `GPT.__init__`, after `self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)`, add:

```python
        self.bigram = BigramHashEmbedding(h.bigram_vocab_size, h.bigram_dim, h.model_dim) if h.bigram_vocab_size > 0 else None
        self.smear = SmearGate(h.model_dim)
```

In both `GPT.forward_logits` and `GPT.forward` (the training forward), after `x = self.tok_emb(input_ids)` and before `x = F.rms_norm(x, (x.size(-1),))`, add:

```python
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
```

After the `F.rms_norm` line, add:

```python
        x = self.smear(x)
```

- [ ] **Step 5: Add BigramHash params to optimizer groups**

In the `Optimizers.__init__` method, find where `scalar_params` are being assembled. After `scalar_params.append(base_model.skip_gates)`, add:

```python
        if hasattr(base_model, 'smear') and base_model.smear is not None:
            scalar_params.append(base_model.smear.gate)
        if hasattr(base_model, 'bigram') and base_model.bigram is not None:
            scalar_params.append(base_model.bigram.scale)
            if base_model.bigram.proj is not None:
                matrix_params.append(base_model.bigram.proj.weight)
```

Also add the bigram embedding to the token optimizer group. Find the `tok_params` list and append:

```python
        if hasattr(base_model, 'bigram') and base_model.bigram is not None:
            tok_params.append({"params": [base_model.bigram.embed.weight], "lr": token_lr, "base_lr": token_lr})
```

- [ ] **Step 6: Verify BigramHash variant parses**

```bash
python3 -c "import ast; ast.parse(open('experiments/exp4_train_gpt_bigram.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add experiments/exp4_train_gpt_bigram.py
git commit -m "Add BigramHash variant of depth recurrence script"
```

---

### Task 3: Create RunPod Pod and Set Up Environment

- [ ] **Step 1: Create the pod**

```bash
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name "pgolf-exp4-recurrence" \
  --cloud-type SECURE
```

Expected: pod ID returned (e.g., `pod_abc123`). Save this ID — needed for all subsequent commands.

If SECURE fails, try community:
```bash
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name "pgolf-exp4-recurrence" \
  --cloud-type COMMUNITY
```

- [ ] **Step 2: Wait for pod to be ready, get SSH info**

```bash
runpodctl pod list
```

Wait until status shows RUNNING, then:

```bash
runpodctl ssh info <pod-id>
```

Expected: SSH IP and port. Note these as `<ip>` and `<port>`.

- [ ] **Step 3: SSH in and set up environment**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'SETUP'
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install --break-system-packages zstandard brotli
python3 data/cached_challenge_fineweb.py --variant sp4096
echo "Setup complete. Checking data:"
ls -la data/datasets/fineweb10B_sp4096/ | head -5
ls data/datasets/fineweb10B_sp4096/fineweb_train_*.bin | wc -l
SETUP
```

Expected: Data download completes. Should see 80+ training shards. If shard count is less than 80, the data download was incomplete — re-run the download command.

---

### Task 4: Run Vanilla Training (Seed 1337)

- [ ] **Step 1: SCP the vanilla script to pod**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  experiments/exp4_train_gpt.py \
  root@<ip>:/workspace/parameter-golf/train_gpt.py
```

- [ ] **Step 2: Run training with seed 1337**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=1337 RUN_ID=exp4_vanilla_s1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4_vanilla_s1337.log
TRAIN
```

Expected: Training runs for ~590s (~5200-5400 steps), then GPTQ quantization runs for ~10s. Final output should include:
- `final_int6_sliding_window_exact val_bpb` close to **1.093**
- Artifact size under 16,000,000 bytes

Training takes ~10 minutes. Monitor the log for:
- Early steps: `train_loss` should drop below 3.0 by step 500
- Step 3000: "recurrence:activated" message should appear
- Final: `val_bpb` result

- [ ] **Step 3: Collect the log**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4_vanilla_s1337.log \
  experiments/exp4_vanilla_s1337.log
```

- [ ] **Step 4: Record the result**

Check the log for the final BPB:

```bash
grep -i "val_bpb\|sliding.*bpb\|final.*bpb" experiments/exp4_vanilla_s1337.log | tail -5
```

Record the result. If BPB > 1.10, troubleshoot:
- Check shard count: `grep "train_shards" experiments/exp4_vanilla_s1337.log` (should be 100+)
- Check recurrence activated: `grep "recurrence" experiments/exp4_vanilla_s1337.log`
- Check wallclock: `grep "wallclock\|time" experiments/exp4_vanilla_s1337.log | tail -3`

---

### Task 5: Decision Point — BigramHash or Vanilla

This task is a decision gate based on Task 4 results.

- [ ] **Step 1: Evaluate vanilla result**

**If vanilla BPB <= 1.095** (successful reproduction):
- Proceed to Step 2 to test BigramHash variant.

**If vanilla BPB > 1.10** (reproduction failed):
- Skip BigramHash. Debug the vanilla run (check Task 4 troubleshooting).
- If unfixable, use the vanilla result as-is and proceed to Task 6 with the vanilla script.

- [ ] **Step 2: SCP BigramHash variant and run (only if vanilla succeeded)**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  experiments/exp4_train_gpt_bigram.py \
  root@<ip>:/workspace/parameter-golf/train_gpt.py
```

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=1337 RUN_ID=exp4_bigram_s1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4_bigram_s1337.log
TRAIN
```

- [ ] **Step 3: Collect BigramHash log and compare**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4_bigram_s1337.log \
  experiments/exp4_bigram_s1337.log
```

```bash
grep -i "val_bpb\|sliding.*bpb\|final.*bpb\|bytes_total\|artifact" experiments/exp4_bigram_s1337.log | tail -5
```

- [ ] **Step 4: Pick the winner**

**Decision rule:**
- If BigramHash BPB < vanilla BPB AND artifact < 16,000,000 bytes → use BigramHash variant
- Otherwise → use vanilla

Record which script is the winner. This is the script used for the remaining seeds.

---

### Task 6: Run Remaining Seeds (42 and 2024)

**Files:**
- The winning script from Task 5 should already be on the pod as `train_gpt.py`

- [ ] **Step 1: If BigramHash lost, restore vanilla script on pod**

Only needed if BigramHash was tested and lost:

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  experiments/exp4_train_gpt.py \
  root@<ip>:/workspace/parameter-golf/train_gpt.py
```

- [ ] **Step 2: Run seed 42**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=42 RUN_ID=exp4_best_s42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4_best_s42.log
TRAIN
```

- [ ] **Step 3: Run seed 2024**

```bash
ssh -i ~/.runpod/ssh/RunPod-Key-Go root@<ip> -p <port> << 'TRAIN'
cd /workspace/parameter-golf
SEED=2024 RUN_ID=exp4_best_s2024 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee /workspace/exp4_best_s2024.log
TRAIN
```

- [ ] **Step 4: Collect all logs**

```bash
scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4_best_s42.log \
  experiments/exp4_best_s42.log

scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> \
  root@<ip>:/workspace/exp4_best_s2024.log \
  experiments/exp4_best_s2024.log
```

- [ ] **Step 5: Compute 3-seed mean and std**

Extract BPB from all 3 seed logs and compute mean/std:

```bash
echo "=== 3-Seed Results ==="
for f in experiments/exp4_*s1337.log experiments/exp4_best_s42.log experiments/exp4_best_s2024.log; do
  echo "$f:"
  grep -i "val_bpb\|sliding.*bpb\|final.*bpb" "$f" | tail -1
done
```

Record the 3-seed mean and std. For a valid record submission, need:
- Mean BPB < 1.1097 (at least 0.005 below current SOTA 1.1147)
- Std < 0.002 (tight variance)

---

### Task 7: Stop RunPod Pod

- [ ] **Step 1: Stop the pod to stop billing**

```bash
runpodctl pod stop <pod-id>
```

- [ ] **Step 2: Verify pod is stopped**

```bash
runpodctl pod list
```

Expected: pod status shows STOPPED or EXITED.

Note: Do NOT delete the pod yet — we may need to re-run if results need debugging. Delete after PR is submitted.

---

### Task 8: Create Submission Folder

**Files:**
- Create: `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_gpt.py`
- Create: `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/README.md`
- Create: `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/submission.json`
- Create: `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed1337.log`
- Create: `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed42.log`
- Create: `records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed2024.log`

- [ ] **Step 1: Create submission directory**

```bash
mkdir -p records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965
```

- [ ] **Step 2: Copy winning script and logs**

```bash
# Copy the winning script (vanilla or bigram variant)
cp experiments/exp4_train_gpt.py records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_gpt.py

# Copy logs (rename to submission format)
cp experiments/exp4_*s1337.log records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed1337.log
cp experiments/exp4_best_s42.log records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed42.log
cp experiments/exp4_best_s2024.log records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_seed2024.log
```

If the BigramHash variant won, copy that instead:
```bash
cp experiments/exp4_train_gpt_bigram.py records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/train_gpt.py
```

- [ ] **Step 3: Create submission.json**

Fill in actual values from the 3-seed results. Template:

```json
{
    "author": "Abhay Anand",
    "github_id": "AbhayAnandUCSD",
    "name": "11L Depth Recurrence + EMA 0.9965",
    "blurb": "Reproduction and validation of PR #1421 depth recurrence architecture (11L, layers 4,5 repeat once for 13 virtual layers, skip gates, parallel residuals, EMA 0.9965, GPTQ int6 + Brotli). Built on PR #1334 by @aryanbhosale and PR #1421 by @X-Abhishek-X.",
    "date": "2026-04-06T00:00:00Z",
    "track": "10min-16mb",
    "val_loss": <FILL: mean val_loss from logs>,
    "val_bpb": <FILL: mean val_bpb from 3 seeds>,
    "step_stop": <FILL: from logs>,
    "wallclock_seconds": 600,
    "bytes_total": <FILL: mean artifact bytes from logs>,
    "gpu": "8xH100 SXM"
}
```

- [ ] **Step 4: Create README.md**

```markdown
# 11L Depth Recurrence + EMA 0.9965 — val_bpb <FILL>

**val_bpb: <FILL>** (3-seed mean) | **~15.95 MB** | 8xH100 SXM, 600s

## Summary

Reproduction and validation of depth recurrence on the PR #1334/#1421 architecture.
11 physical transformer layers with layers 4,5 repeating once, yielding 13 virtual layers.
Recurrence activates at step 3000. EMA decay tuned to 0.9965.

## 3-Seed Results

| Seed | Sliding BPB (s64) | Artifact |
|------|-------------------|----------|
| 1337 | <FILL> | <FILL> |
| 42 | <FILL> | <FILL> |
| 2024 | <FILL> | <FILL> |
| **Mean** | **<FILL>** | |

## Architecture

- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- Depth recurrence: layers 4,5 repeat (virtual 13 layers), activated at step 3000
- Skip gates (learnable residual gating)
- Parallel residuals (layers 7+)
- Value Embedding (dim=128, layers 9,10)
- SP4096 tokenizer (SentencePiece BPE)
- Tied embeddings, logit softcap=30.0

## Training

- Muon optimizer: lr=0.02, momentum=0.99, WD=0.09, backend_steps=5
- EMA decay=0.9965, warmdown 66.7%
- Batch: 786,432 tokens/step, seq_len=2048
- GPTQ int6 + Brotli compression

## Attribution

- Base architecture: PR #1334 by @aryanbhosale
- EMA tuning: PR #1421 by @X-Abhishek-X
```

- [ ] **Step 5: Commit submission**

```bash
git add records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/
git commit -m "Record: 11L Depth Recurrence + EMA 0.9965 — val_bpb <FILL>"
```

---

### Task 9: Create Pull Request

- [ ] **Step 1: Push to fork**

```bash
git push -u fork exp4-recurrence
```

If the `fork` remote doesn't exist:
```bash
git remote add fork https://github.com/AbhayAnandUCSD/parameter-golf.git
git push -u fork exp4-recurrence
```

- [ ] **Step 2: Create PR**

```bash
gh pr create --repo openai/parameter-golf \
  --head AbhayAnandUCSD:exp4-recurrence \
  --base main \
  --title "Record: 11L Depth Recurrence + EMA 0.9965 — val_bpb <FILL>" \
  --body "$(cat <<'EOF'
## Summary

- Reproduction of depth recurrence architecture from PR #1334/#1421
- 11 physical layers, layers 4,5 repeat once (13 virtual), activated at step 3000
- EMA decay 0.9965, GPTQ int6 + Brotli compression
- SP4096 tokenizer, MuonEq-R, skip gates, parallel residuals

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

- [ ] **Step 3: Record PR URL**

Save the PR URL in `experiments/exp4_results.md`.

---

### Task 10: Cleanup

- [ ] **Step 1: Delete RunPod pod**

Only after PR is submitted and verified:

```bash
runpodctl pod delete <pod-id>
```

- [ ] **Step 2: Update experiments/exp4_results.md with final summary**

Record: final BPB, 3-seed results, which variant won (vanilla vs BigramHash), PR URL, total cost.
