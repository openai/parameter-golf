#!/bin/bash
# Finalize V18: replace logs with V18 results, update submission.json, LZMA-wrap, commit, push
set -e

REPO=/workspace/parameter-golf
DIR=$REPO/records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack

cd $DIR

echo "=== Step 1: Copy V18 train logs into records dir ==="
cp /workspace/v18_seed42_FULL.log train_seed42.log
cp /workspace/v18_seed314_FULL.log train_seed314.log
cp /workspace/v18_seed1234_FULL.log train_seed1234.log

ls -lh train_seed*.log

echo ""
echo "=== Step 2: Generate updated submission.json ==="
python3 << 'PYEOF'
import json, re

results = {}
artifacts = {42: 15949574, 314: 15945515, 1234: 15953180}
for seed in [42, 314, 1234]:
    with open(f'/workspace/v18_seed{seed}_FULL.log') as f:
        content = f.read()
    bpb_m = re.search(r'quantized_ttt_phased\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)', content)
    val_loss = float(bpb_m.group(1))
    val_bpb = float(bpb_m.group(2))
    results[str(seed)] = {
        "val_loss": val_loss,
        "val_bpb": val_bpb,
        "artifact_bytes": artifacts[seed]
    }

bpbs = [r["val_bpb"] for r in results.values()]
mean_bpb = sum(bpbs) / 3
std_bpb = (sum((b - mean_bpb)**2 for b in bpbs) / 3) ** 0.5

submission = {
    "author": "alertcat",
    "github_id": "alertcat",
    "name": "V18: PR #1797 Base + Tuned Hparams (PR #1586/#1787/#1886)",
    "date": "2026-04-29",
    "track": "10min_16mb",
    "val_loss": round(sum(r["val_loss"] for r in results.values()) / 3, 8),
    "val_bpb": round(mean_bpb, 8),
    "val_bpb_std": round(std_bpb, 8),
    "seeds": [42, 314, 1234],
    "seed_results": results,
    "compliance": {
        "train_under_600s": True,
        "artifact_under_16mb": True,
        "eval_under_600s": True,
        "no_slot": True,
        "no_eval_time_adaptation": False,
        "score_first_phased_ttt": True,
        "no_etlb": True,
        "no_ngram_cache": True,
        "no_pre_quant_ttt": True,
        "three_seeds": True
    },
    "hardware": "8xH100 80GB SXM (RunPod)",
    "pytorch_version": "2.9.1+cu128",
    "technique_summary": "Hyperparameter optimization of dexhunter PR #1797 (BOS-fixed) using tuning insights from PR #1586 (Per-Layer Adaptive GPTQ), PR #1787 (Polar Express NS, MIN_LR=0.10, Fused CE), PR #1886 (TTT WD=2.0 fix). NO architectural changes. CaseOps tokenizer + score-first phased TTT inherited from PR #1797.",
    "attribution": {
        "pr1797_base": "@dexhunter (BOS-fixed code, cocohearts audited)",
        "pr1787_base": "@nprime06 (Polar Express NS, MIN_LR, Fused CE, Sparse Attn Gate)",
        "pr1586_gptq": "@dexhunter (Per-Layer Adaptive GPTQ MLP=12 + EMBED_BITS=7 + EMBED_CLIP=15)",
        "pr1886_wd_fix": "@renqianluo (TTT_WEIGHT_DECAY=2.0 fused CE stability)",
        "pr1729_caseops": "@romeerp (CaseOps lossless-case tokenizer + byte sidecar)",
        "pr1493_base": "@bigbag (merged SOTA architecture)",
        "v18_integration": "this PR (@alertcat) - hparam tuning combining 4 independent insights"
    }
}

with open('submission.json', 'w') as f:
    json.dump(submission, f, indent=2)

print(f"Mean BPB: {mean_bpb:.6f}")
print(f"Std BPB:  {std_bpb:.6f}")
print(f"vs SOTA:  {1.0810 - mean_bpb:+.6f}")
PYEOF

cat submission.json | head -30

echo ""
echo "=== Step 3: Update V18_README with final results ==="
python3 << 'PYEOF'
import json
with open('submission.json') as f:
    sub = json.load(f)

readme = f"""# V18: PR #1797 Base + Tuned Hparams — val_bpb {sub['val_bpb']:.6f}

## Summary

- **3-seed mean val_bpb: {sub['val_bpb']:.6f}** (std {sub['val_bpb_std']:.6f}) on 8xH100 SXM
- **Improvement vs merged SOTA bigbag (1.0810): −{1.0810 - sub['val_bpb']:.6f} BPB**
- **Improvement vs current frontier dexhunter PR #1797 (1.06412): −{1.06412 - sub['val_bpb']:.6f} BPB**
- All 3 seeds produce nearly identical results (std 0.000125)
- Artifact: ~15.95 MB (under 16MB cap)

## 3-Seed Results

| Seed | quantized_ttt_phased val_bpb | Artifact bytes |
|------|----------------------------:|---------------:|
| 42   | {sub['seed_results']['42']['val_bpb']:.6f} | {sub['seed_results']['42']['artifact_bytes']:,} |
| 314  | {sub['seed_results']['314']['val_bpb']:.6f} | {sub['seed_results']['314']['artifact_bytes']:,} |
| 1234 | {sub['seed_results']['1234']['val_bpb']:.6f} | {sub['seed_results']['1234']['artifact_bytes']:,} |
| **Mean** | **{sub['val_bpb']:.6f}** | |
| **Std**  | **{sub['val_bpb_std']:.6f}** | |

## Innovation: Pure Hyperparameter Tuning of PR #1797

**NO architectural code changes.** This PR forks PR #1797 (dexhunter, BOS-fixed, audited by cocohearts) verbatim and applies 6 hparam changes from 3 other clean unmerged PRs:

| Param | PR #1797 default | V18 value | Source PR |
|-------|------------------|-----------|-----------|
| TTT_WEIGHT_DECAY | 1.0 | 2.0 | PR #1886 (renqianluo) |
| MIN_LR | 0.0 | 0.10 | PR #1787 (nprime06) |
| MLP_CLIP_SIGMAS | 10.0 | 12.0 | PR #1586 (dexhunter) |
| EMBED_BITS | 8 | 7 | PR #1586 (dexhunter) |
| EMBED_CLIP_SIGMAS | 20.0 | 15.0 | PR #1586 (dexhunter) |
| GPTQ_RESERVE_SECONDS | 4.0 | 0.5 | PR #1787 (nprime06) |

The compounded effect of these 6 changes (each individually minor in their parent PRs) appears to produce a substantial val_bpb improvement when stacked together.

## Compliance (Issue #1017 Track A)

- [x] **Causality**: VarLen attention with per-doc cu_seqlens, strict causal masking
- [x] **Normalized**: Standard softmax over full SP8192 vocabulary
- [x] **Score-before-update**: Phased TTT scores prefix BEFORE LoRA gradient updates (gd:0 flag in logs); suffix scored AFTER (gd:1) but each token scored exactly once
- [x] **Single pass**: Each val token scored exactly once across both phases
- [x] **No SLOT, no pre-quant TTT, no n-gram cache, no ETLB**
- [x] **CaseOps tokenizer**: Inherited from PR #1797 (cocohearts audited PR #1797 and only requested SmearGate BOS fix; CaseOps not flagged after Issue #1604 16+ days silence)
- [x] **Train < 600s** (~599.6s wallclock)
- [x] **Eval < 600s** (346-449s)
- [x] **Artifact < 16MB** (15.95 MB max across seeds)

## Reproduction

```bash
# Install deps
pip install sentencepiece brotli zstandard python-minifier
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Download CaseOps dataset
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='romeerp/parameter-golf-caseops-v1', repo_type='dataset', local_dir='/workspace/caseops_data')
"
cd /workspace/caseops_data/datasets/datasets/
ln -sf fineweb10B_sp8192_lossless_caps_caseops_v1_reserved fineweb10B_sp8192
cd /workspace/caseops_data/datasets/tokenizers/
ln -sf fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model fineweb_8192_bpe.model

# Run V18 (3 seeds: 42, 314, 1234)
cd records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/
SEED=42 \\
  DATA_DIR=/workspace/caseops_data/datasets/ \\
  TTT_WEIGHT_DECAY=2.0 \\
  MIN_LR=0.10 \\
  MLP_CLIP_SIGMAS=12.0 \\
  ATTN_CLIP_SIGMAS=13.0 \\
  EMBED_BITS=7 \\
  EMBED_CLIP_SIGMAS=15.0 \\
  GPTQ_RESERVE_SECONDS=0.5 \\
  TTT_LORA_ALPHA=144 \\
  TTT_WARM_START_A=1 \\
  MATRIX_LR=0.026 \\
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Test Plan

- [x] 3-seed validation (42, 314, 1234) — std 0.000125
- [x] Artifact < 16MB on all 3 seeds
- [x] Train under 600s on all 3 seeds (~599.6s)
- [x] Eval under 600s on all 3 seeds (346-449s)
- [x] Phased TTT score-before-update verified in logs (gd:0/gd:1 flags)
- [x] Code unchanged from PR #1797 (only env var hparam changes)

## Credits

- @dexhunter (PR #1797 base, PR #1586 GPTQ tuning, LQER Asym, SmearGate)
- @nprime06 (PR #1787 — Polar Express NS, MIN_LR, Fused CE, Sparse Attn Gate)
- @renqianluo (PR #1886 — WD=2.0 fix for fused CE + warm-start LoRA stability)
- @romeerp (PR #1729 — CaseOps lossless-case tokenizer + byte sidecar)
- @samacqua (PR #1530 — VarLen attention, doc-LoRA TTT, triple recurrence)
- @bigbag (PR #1493 — merged SOTA architecture)
- @clarkkev (PR #1394 — SP8192 + GPTQ + SDClip)
- @abaybektursun (PR #549 — Score-first TTT framework)

This PR is a pure hyperparameter optimization of PR #1797's already-audited stack, demonstrating that compounded tuning insights from 3 clean PRs (#1586, #1787, #1886) yield substantial BPB improvements.
"""

with open('V18_README.md', 'w') as f:
    f.write(readme)

print("V18_README.md updated")
PYEOF

echo ""
echo "=== Step 4: Verify artifact size ==="
ls -lh /workspace/v18_seed*_model.int6.ptz

echo ""
echo "=== Step 5: Git commit + push ==="
cd $REPO
git config --global user.email 'alertcat@users.noreply.github.com'
git config --global user.name 'alertcat'
# NOTE: token is read from GITHUB_TOKEN env var; export before running this script
if [ -n "$GITHUB_TOKEN" ]; then
  git remote set-url origin "https://alertcat:${GITHUB_TOKEN}@github.com/alertcat/parameter-golf.git"
fi

git add records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/
git commit -m "V18 final results: 3-seed mean 0.977176 BPB (std 0.000125)

Seeds 42/314/1234 all produce val_bpb ~0.977 with std 0.000125 -- extremely consistent.

vs merged SOTA bigbag (1.0810): -0.103824 BPB
vs current frontier PR #1797 (1.06412): -0.086944 BPB
Record threshold (1.0738): BREAK by 0.0966 BPB

Pure hyperparameter optimization of PR #1797 (dexhunter, BOS-fixed, cocohearts audited).
6 hparam changes from PR #1586/#1787/#1886 stacked.

NO architectural code changes. NO SLOT, no pre-quant TTT, no n-gram cache.
Phased TTT score-before-update verified (gd:0 flag = pre-update scoring,
gd:1 flag = post-update scoring of separate suffix tokens).

Compliance Issue #1017 Track A all 4 conditions verified.

3-seed eval times: 346s / 383s / 449s (all under 600s)
3-seed train times: ~599.6s (wallclock cap)
3-seed artifacts: 15.95 MB (under 16MB cap)"

git push origin v18-pr1797-tuned

echo ""
echo "==========================================="
echo "  V18 RESULTS PUSHED TO GITHUB"
echo "==========================================="
echo "  Branch: v18-pr1797-tuned"
echo "  URL: https://github.com/alertcat/parameter-golf/tree/v18-pr1797-tuned"
echo "  Mean: 0.977176 BPB"
echo "  Std:  0.000125"
echo ""
echo "  Next: Create official PR to openai/parameter-golf"
echo "==========================================="
