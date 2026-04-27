# Parameter Golf NGramMix — Session Handoff

**Updated:** 2026-04-24 ~16:45 local time

If you're picking this up from a fresh CLI Claude session, paste the
"Briefing prompt" section below as your first message. Everything you need
is on disk or in cloud accounts.

---

## 1. State of the world

### Code (local)
- **Repo:** `/Users/yinchen/Git_Projects/parameter-golf/.claude/worktrees/jovial-bartik-557ac3`
- **Branch:** `submission/pr1797-ngram-mix` (committed and pushed to `origin = Fija/parameter-golf`)
- **Last commits:** `83c751a` (Phase B infra), `ae59992` (Day 4 TTT), `54bb76d` (mixer V1)
- **Submission folder:** `records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/`
- **What's in there:**
  - `train_gpt.py` (3811 lines) — PR #1797 base + `NGramMixer` (eval_val) + `BatchUnigramMixer` (TTT phased)
  - `test_ngram_legality.py` — 10/10 tests pass
  - `test_mixer_gpu.py` — H100 smoke
  - `sweep_eval_val.py` — multi-config sweep
  - `prepare_caseops_data_parallel.py` — 96-worker tokenizer
  - `upload_caseops_to_hf.py` — uses `upload_large_folder`
  - `submission.json` (stub), `README.md`, `lossless_caps.py`, `tokenizers/*.model`
- **RunPod scripts:** `runpod/{Dockerfile, bootstrap.sh, bootstrap_no_volume.sh, launch_experiment.sh, phase_b_launch.sh, phase_b_onpod.sh, upload_caseops_to_hf.py, README.md}`

### Cloud accounts
- **RunPod:** API key in `~/.runpod/config.toml`. Verified via `runpodctl get pod`. Account balance **$162.82**.
- **HF:** token in `~/.config/parameter-golf/hf_token` (mode 600). User handle `FijaEE`.
- **GitHub:** `gh auth status` ✓ as `Fija`.
- **SSH key:** `~/.ssh/id_runpod` (public key already uploaded to RunPod).

### Cloud resources currently allocated (costing $)
| ID | What | Where | Cost/hr | Status |
|----|------|-------|--------:|--------|
| `bp56l63b70` | 150 GB Network Volume | EU-NL-1 | ~$0.02/hr | KEPT — has 29 GB CaseOps shards + 45 GB raw docs |
| `96bjk3hvbs` | 50 GB Network Volume | US-KS-2 | ~$0.007/hr | OLD — pre-existing, can delete if you want |
| `f8cbdzhy1icxte` | 1×H100 EU-NL-1 pod | (stopped) | $0/hr | Stopped after Phase A |
| `mhriki2p72m45q` | 8×H100 US-MO-1 pod | (stopped) | $0/hr | Stopped after Phase B sweep |

To delete the old volume: `curl -sS -X POST "https://api.runpod.io/graphql?api_key=$(grep apikey ~/.runpod/config.toml | cut -d\" -f2)" -H "Content-Type: application/json" -d '{"query":"mutation { deleteNetworkVolume(input:{id:\"96bjk3hvbs\"}) }"}'`

### Pre-tokenized data on HF (the expensive output of Phase A)
- **Dataset:** `hf://datasets/FijaEE/parameter-golf-sp8192-caseops` (private, 1499 train + 5 val + 5 val_bytes shards, ~29 GB on-disk, ~5 GB after Xet dedup)
- Pull from any new pod with: `python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='FijaEE/parameter-golf-sp8192-caseops', repo_type='dataset', local_dir='/workspace/data/datasets/fineweb10B_sp8192_caseops', token=open('/proc/1/environ').read().split('HF_TOKEN=')[1].split('\\x00')[0], max_workers=16)"`

---

## 2. What ran, what we learned, what blocks us

### Phase A (EU-NL-1, $4.5)
- ✅ Built local code (NGramMixer + BatchUnigramMixer, 10/10 legality tests).
- ✅ 1×H100 pod up, deps (`brotli zstandard python-minifier`) installed, GPU smoke ✓.
- ✅ Downloaded `docs_selected.jsonl` (44.9 GB) from `willdepueoai/parameter-golf` HF dataset.
- ✅ Ran `prepare_caseops_data_parallel.py` (96 workers, ~76 min, 3500 docs/s).
- ✅ Uploaded 29 GB to `FijaEE/parameter-golf-sp8192-caseops` via `upload_large_folder` (~3 min, Xet dedup → 5 GB wire).

### Phase B (US-MO-1, ~$15)
- ❌ First train CRASHED at serialize step — `pyminify` CLI missing. Fix: install **`python-minifier`** (NOT `pyminify`, that's a different placeholder package).
- ✅ Re-ran. Baseline trained 4948 steps in 10 min, then GPTQ + diagnostic eval + TTT phased eval.
- ✅ Ran 11-point `sweep_eval_val.py` over `(alpha, beta)` grid in one torchrun.

### Numbers
- **Baseline diagnostic eval_val val_bpb: 1.07570** (mixer OFF)
- **Baseline TTT phased val_bpb: 1.06232** (mixer OFF; PR #1797 reported 1.06157 — we're +0.00075 worse, likely missing PR #1797 env vars below)
- **Best mixer config**: 1.13393 (+0.058 vs baseline) — every mixer config HURT.
- **Sanity anchors**: `full_nn (λ=1)` matched baseline exactly; `full_bi (λ=0)` was 2.34 (uniform-prior n-gram is awful). Math is right; the design is wrong.

### Two blockers found
1. **Artifact 16.93 MB > 16 MB cap.** PR #1797 was 15.95 MB. Likely cause: PR #1797's README lists env vars I didn't set:
   ```
   SPARSE_ATTN_GATE_ENABLED=1  MIN_LR=0.1  FUSED_CE_ENABLED=1  TTT_WARM_START_A=1
   SMEAR_GATE_ENABLED=1  LQER_ENABLED=1  LQER_ASYM_ENABLED=1
   ```
   Not setting them means we ran a *different* model than PR #1797's intended config. Need to verify which env var actually controls the artifact size delta.
2. **Mixer fundamentally fails with cold-start uniform `q_bi`.** At λ=0.88, mixing the neural distribution with a near-uniform bigram on a fresh stream injects ~0.13 nats per token of pure noise. To make it work we'd need (a) a baked unigram prior in the artifact (~16 KB extra code bytes — but we have no headroom) OR (b) λ near 1 always (which is just baseline).

Kill-switch I committed to (Day 3 plan): "if no mixer config beats base by ≥ 0.003 BPB, stop and pivot." → **Triggered.**

---

## 3. Decision pending (this is where you ask the user)

Three pivots discussed, no decision yet:
- **A. Fix both issues, retry mixer.** Set the env vars to bring artifact ≤ 16 MB; bake a tiny unigram prior in code. Cost ~1-2 days. Same family of knob.
- **B. New orthogonal knob.** Recommended sub-option: per-doc temperature scaling gated by prefix-only running entropy. Zero artifact cost. Easy to prove legal. Estimated +0.005-0.015 BPB. ~1 day.
- **C. Non-record submission.** Ship the bigram explorer as a documented negative result against the Apr 30 deadline.

---

## 4. Briefing prompt (paste this into a fresh CLI Claude session)

> I'm continuing a Parameter Golf challenge submission for the OpenAI 16MB / 10-min / 8×H100 leaderboard (deadline 2026-04-30).
>
> Read `SESSION_HANDOFF.md` at the repo root first — it has the full state. Then read these files for context:
> - `records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/README.md`
> - `records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/train_gpt.py` (search for `class NGramMixer` and `class BatchUnigramMixer`)
> - `runpod/phase_b_onpod.sh`
>
> RunPod is configured (`runpodctl get pod` works). HF token at `~/.config/parameter-golf/hf_token`. SSH key at `~/.ssh/id_runpod`. Working dir is the current `pwd`.
>
> The previous session triggered the Day 3 kill-switch: every bigram-mixer config UNDERPERFORMED the baseline (best at +0.058 BPB; full table in handoff). Two issues need addressing: artifact > 16 MB cap, and bigram cold-start failure.
>
> I'm choosing pivot path **[A / B / C]** described in `SESSION_HANDOFF.md` section 3. Please confirm understanding and propose the next 3 concrete actions.

---

## 5. Practical re-entry commands

```bash
# Check pod inventory + balance
runpodctl get pod
curl -sS -X POST "https://api.runpod.io/graphql?api_key=$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { myself { currentSpendPerHr clientBalance networkVolumes { id name size dataCenterId } } }"}' | python3 -m json.tool

# Re-pull our pre-tokenized data anywhere (HF token from local)
HF_TOKEN=$(cat ~/.config/parameter-golf/hf_token) python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='FijaEE/parameter-golf-sp8192-caseops', repo_type='dataset',
                  local_dir='./data/sp8192_caseops',
                  token=os.environ['HF_TOKEN'], max_workers=16)
"

# Spin a Phase B style 8xH100 pod (NB: only AP-IN-1 / US-MO-1 have stock for 8xH100 Secure)
HF_TOKEN_VAL=$(cat ~/.config/parameter-golf/hf_token)
runpodctl create pod --name "pg-resume-$(date +%H%M)" \
  --gpuType "NVIDIA H100 80GB HBM3" --gpuCount 8 --secureCloud \
  --imageName "runpod/parameter-golf:latest" \
  --containerDiskSize 80 --startSSH --ports "22/tcp" \
  --env "HF_TOKEN=$HF_TOKEN_VAL" --mem 64 --vcpu 32

# IMMEDIATELY check pod cost rate after creation (to catch surprises)
curl -sS -X POST "https://api.runpod.io/graphql?api_key=$(grep apikey ~/.runpod/config.toml | cut -d'"' -f2)" \
  -H "Content-Type: application/json" \
  -d '{"query":"query { myself { currentSpendPerHr } }"}' | python3 -m json.tool

# Stop a pod (kill-switch)
runpodctl stop pod <POD_ID>

# Re-run the unit-test self-check anywhere with Python + torch installed
python3 records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/test_ngram_legality.py

# Re-run the GPU smoke (needs a CUDA box)
python3 records/track_10min_16mb/2026-04-25_PR1797Base_NGramMix/test_mixer_gpu.py
```

---

## 6. Cost ledger so far

| Item | $ |
|------|--:|
| 1×H100 EU-NL-1 (Phase A, 91 min) | 4.54 |
| 8×H100 US-MO-1 first train (crash, 8 min) | 3.20 |
| 8×H100 US-MO-1 second train + sweep (~30 min) | 12.50 |
| Volumes (200 GB across 2 vols, all day) | ~0.5 |
| **Total** | **~20.7** |
| **Remaining budget** | **$162** out of stated $1000 cap |

Plenty of headroom. If you continue with pivot B you're still well under budget.
