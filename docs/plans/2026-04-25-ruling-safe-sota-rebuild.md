# Ruling-Safe SOTA Rebuild Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild `train_gpt_v3.py` into a ruling-safe Parameter Golf contender by reconstructing the strongest public SP8192/CaseOps/TTT frontier stack and adding only defensible stackable improvements.

**Architecture:** Treat PR #1797 as the reconstruction base, because it already includes PR #1787 plus SmearGate and LQER. Preserve the current local script before replacement, import the required CaseOps/tokenizer prep support files, then add our improvements behind env flags and verify every promoted path with tests and artifact checks.

**Tech Stack:** Python 3.12, PyTorch CUDA, torchrun/DDP, Flash Attention 3, Triton custom op, SentencePiece, Brotli, int6 GPTQ, LoRA TTT.

---

## Guardrails

- Do not revert unrelated dirty worktree files.
- Keep PPM and byte-level online predictors disabled in the primary path.
- Preserve current `train_gpt_v3.py` before replacing it.
- Use the upstream PR source URLs in commit messages for provenance.
- Every feature that changes model forward must be mirrored in the TTT helper path.
- Final submission candidate must be validated by three 8xH100 seeds before promotion.

## Task 1: Snapshot Local v3 Before Reconstruction

**Files:**
- Read: `train_gpt_v3.py`
- Create: `backups/20260425_pre_sota_rebuild/train_gpt_v3.py`
- Create: `backups/20260425_pre_sota_rebuild/git_status.txt`

**Step 1: Create backup directory**

Run:

```bash
mkdir -p backups/20260425_pre_sota_rebuild
```

Expected: directory exists.

**Step 2: Copy current script and worktree status**

Run:

```bash
cp train_gpt_v3.py backups/20260425_pre_sota_rebuild/train_gpt_v3.py
git status --short > backups/20260425_pre_sota_rebuild/git_status.txt
```

Expected: backup file exists and `git_status.txt` records current dirty state.

**Step 3: Commit backup metadata only**

Run:

```bash
git add backups/20260425_pre_sota_rebuild/train_gpt_v3.py backups/20260425_pre_sota_rebuild/git_status.txt
git commit -m "chore: snapshot pre-rebuild train_gpt_v3"
```

Expected: commit succeeds without adding unrelated files.

## Task 2: Reconstruct PR #1797 Frontier Script

**Files:**
- Modify: `train_gpt_v3.py`
- Create: `frontier_sources/2026-04-24_pr1797_train_gpt.py`

**Step 1: Fetch PR #1797 script to provenance cache**

Run:

```bash
mkdir -p frontier_sources
curl -L "https://github.com/openai/parameter-golf/raw/04d35edaad74fc88b5ef08a814c94596b616ec1b/records%2Ftrack_10min_16mb%2F2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157%2Ftrain_gpt.py" -o frontier_sources/2026-04-24_pr1797_train_gpt.py
```

Expected: cached file exists and starts with imports for `flash_attn_interface`.

**Step 2: Replace `train_gpt_v3.py` mechanically**

Run:

```bash
cp frontier_sources/2026-04-24_pr1797_train_gpt.py train_gpt_v3.py
```

Expected: `train_gpt_v3.py` now contains `LQER`, `BatchedTTTLoRA`, `PHASED_TTT`, and `SMEAR_GATE_ENABLED`.

**Step 3: Add local provenance header**

Modify the module header to include:

```python
# Local reconstruction base:
# openai/parameter-golf PR #1797, commit 04d35edaad74fc88b5ef08a814c94596b616ec1b
# Ruling-safe primary path: PPM_MIX_ENABLED must remain off.
```

**Step 4: Compile**

Run:

```bash
python3 -m py_compile train_gpt_v3.py
```

Expected: no syntax errors.

**Step 5: Commit**

Run:

```bash
git add train_gpt_v3.py frontier_sources/2026-04-24_pr1797_train_gpt.py
git commit -m "feat: reconstruct ruling-safe pr1797 frontier script"
```

Expected: commit contains only the reconstructed script and cached source.

## Task 3: Import CaseOps Support Files

**Files:**
- Create: `lossless_caps.py`
- Create: `prepare_caseops_data.py`
- Create: `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- Test: `tests/test_caseops_roundtrip.py`

**Step 1: Fetch support files from PR #1797**

Run:

```bash
mkdir -p tokenizers
curl -L "https://github.com/openai/parameter-golf/raw/04d35edaad74fc88b5ef08a814c94596b616ec1b/records%2Ftrack_10min_16mb%2F2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157%2Flossless_caps.py" -o lossless_caps.py
curl -L "https://github.com/openai/parameter-golf/raw/04d35edaad74fc88b5ef08a814c94596b616ec1b/records%2Ftrack_10min_16mb%2F2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157%2Fprepare_caseops_data.py" -o prepare_caseops_data.py
curl -L "https://github.com/openai/parameter-golf/raw/04d35edaad74fc88b5ef08a814c94596b616ec1b/records%2Ftrack_10min_16mb%2F2026-04-24_PR1787Base_Smear_LQERAsym_PhasedTTT_1.06157%2Ftokenizers%2Ffineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model" -o tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Expected: all three files exist.

**Step 2: Write roundtrip test**

Create `tests/test_caseops_roundtrip.py`:

```python
from lossless_caps import decode_lossless_caps_v2, encode_lossless_caps_v2


def test_caseops_roundtrip_ascii_caps_and_controls():
    samples = [
        "The NASA Launch",
        "camelCase HTTPServer XYZ",
        "already lower",
        "JSON.parse('URL')",
    ]
    for sample in samples:
        assert decode_lossless_caps_v2(encode_lossless_caps_v2(sample)) == sample
```

**Step 3: Run test**

Run:

```bash
python3 -m pytest tests/test_caseops_roundtrip.py -q
```

Expected: test passes.

**Step 4: Commit**

Run:

```bash
git add lossless_caps.py prepare_caseops_data.py tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model tests/test_caseops_roundtrip.py
git commit -m "feat: add caseops support files"
```

Expected: commit contains only CaseOps support and test.

## Task 4: Add Submission-Safety Tests

**Files:**
- Create: `tests/test_submission_safety.py`
- Modify: `train_gpt_v3.py` only if functions need import guards

**Step 1: Write invariant tests**

Create `tests/test_submission_safety.py`:

```python
from pathlib import Path


SOURCE = Path("train_gpt_v3.py").read_text()


def test_primary_path_does_not_enable_ppm_by_default():
    assert 'PPM_MIX_ENABLED", "0"' in SOURCE or "PPM_MIX_ENABLED', '0'" in SOURCE


def test_frontier_features_are_present():
    required = [
        "BatchedTTTLoRA",
        "LQER",
        "SMEAR_GATE_ENABLED",
        "SPARSE_ATTN_GATE_ENABLED",
        "FUSED_CE_ENABLED",
        "MIN_LR",
        "TTT_LORA_ALPHA",
        "brotli.compress",
    ]
    missing = [item for item in required if item not in SOURCE]
    assert not missing


def test_score_first_markers_are_present():
    required = [
        "score",
        "BEFORE",
        "reset",
        "PHASED_TTT_PREFIX_DOCS",
    ]
    missing = [item for item in required if item not in SOURCE]
    assert not missing
```

**Step 2: Run tests**

Run:

```bash
python3 -m pytest tests/test_submission_safety.py -q
```

Expected: tests pass. If the PPM default string is absent because this script has no PPM path, update the first assertion to pass when `PPM_MIX_ENABLED` is absent.

**Step 3: Commit**

Run:

```bash
git add tests/test_submission_safety.py train_gpt_v3.py
git commit -m "test: add ruling-safe submission invariants"
```

Expected: commit includes tests and any import-guard fix only.

## Task 5: Repair And Verify Value Embedding Path

**Files:**
- Modify: `train_gpt_v3.py`
- Test: `tests/test_submission_safety.py`

**Step 1: Locate VE implementation**

Run:

```bash
rg -n "ve_enabled|VE_ENABLED|ve_layers|v_embed|value" train_gpt_v3.py
```

Expected: VE appears in hyperparameters, model init, normal forward, and TTT helper forward.

**Step 2: Add parity test markers**

Append to `tests/test_submission_safety.py`:

```python
def test_value_embedding_is_mirrored_in_ttt_path():
    assert "ve_enabled" in SOURCE
    assert "v_embed" in SOURCE
    assert "_block_with_lora" in SOURCE
    lora_section = SOURCE[SOURCE.index("_block_with_lora") :]
    assert "v_embed" in lora_section
```

**Step 3: If missing, patch TTT path**

In the TTT helper where Q/K/V are constructed, add the same value embedding argument used by normal attention:

```python
v_embed = self._value_embedding_for_layer(input_ids, block_idx) if self.ve_enabled else None
y = self.blocks[block_idx].attn.forward_with_weights(n, q_w, k_w, v_w, out_w, v_embed=v_embed)
```

Use the exact local helper names found by `rg`.

**Step 4: Run tests and compile**

Run:

```bash
python3 -m pytest tests/test_submission_safety.py -q
python3 -m py_compile train_gpt_v3.py
```

Expected: tests and compile pass.

**Step 5: Commit**

Run:

```bash
git add train_gpt_v3.py tests/test_submission_safety.py
git commit -m "fix: mirror value embeddings in ttt path"
```

Expected: commit only if a patch was needed. If PR #1797 already has parity, commit the test only.

## Task 6: Add TTT-Robust Regularizer Against Real LoRA Path

**Files:**
- Modify: `train_gpt_v3.py`
- Test: `tests/test_submission_safety.py`

**Step 1: Add hyperparameters**

Add near other TTT settings:

```python
ttt_robust_lambda = float(os.environ.get("TTT_ROBUST_LAMBDA", "0.0"))
ttt_robust_every = int(os.environ.get("TTT_ROBUST_EVERY", "250"))
ttt_robust_start_frac = float(os.environ.get("TTT_ROBUST_START_FRAC", "0.70"))
ttt_robust_batch_seqs = int(os.environ.get("TTT_ROBUST_BATCH_SEQS", "4"))
```

**Step 2: Add helper using existing BatchedTTTLoRA**

Add a helper near TTT utilities:

```python
def ttt_robust_lora_drift_loss(model, h, x, y):
    if x.size(0) > h.ttt_robust_batch_seqs:
        x = x[: h.ttt_robust_batch_seqs]
        y = y[: h.ttt_robust_batch_seqs]
    with torch.no_grad():
        base_logits = model.forward_logits(x).detach()
    lora = BatchedTTTLoRA(
        x.size(0),
        model,
        h.ttt_lora_rank,
        k_lora=h.ttt_k_lora,
        mlp_lora=h.ttt_mlp_lora,
        o_lora=h.ttt_o_lora,
    ).to(x.device)
    logits = model.forward_logits_with_lora(x, lora)
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    grads = torch.autograd.grad(ce, lora.parameters(), create_graph=True)
    with torch.no_grad():
        for p, g in zip(lora.parameters(), grads, strict=True):
            p.add_(g, alpha=-h.ttt_lora_lr)
    drift_logits = model.forward_logits_with_lora(x, lora)
    return torch.nn.functional.mse_loss(drift_logits.float(), base_logits.float())
```

Adjust helper names to match the reconstructed script. If there is no
`forward_logits_with_lora`, factor the existing phased TTT scoring call into a
small reusable helper first.

**Step 3: Add training-loop hook**

Inside the training loop after CE loss is computed:

```python
if (
    h.ttt_robust_lambda > 0
    and step >= int(h.iterations * h.ttt_robust_start_frac)
    and step % h.ttt_robust_every == 0
):
    loss = loss + h.ttt_robust_lambda * ttt_robust_lora_drift_loss(base_model, h, x, y)
```

**Step 4: Add marker test**

Append:

```python
def test_ttt_robust_regularizer_uses_batched_lora():
    assert "TTT_ROBUST_LAMBDA" in SOURCE
    assert "BatchedTTTLoRA" in SOURCE[SOURCE.find("ttt_robust") :]
```

**Step 5: Run tests and compile**

Run:

```bash
python3 -m pytest tests/test_submission_safety.py -q
python3 -m py_compile train_gpt_v3.py
```

Expected: tests and compile pass.

**Step 6: Commit**

Run:

```bash
git add train_gpt_v3.py tests/test_submission_safety.py
git commit -m "feat: add lora-aware ttt robust regularizer"
```

Expected: commit succeeds.

## Task 7: Add Ruling-Safe Launch Script

**Files:**
- Create: `run_ruling_safe_sota.sh`

**Step 1: Create launcher**

Create:

```bash
#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"
NPROC="${NPROC:-8}"

NCCL_NET="${NCCL_NET:-Socket}" \
DATA_DIR="${DATA_DIR:-./data}" \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_PREFIX_DOCS="${PHASED_TTT_PREFIX_DOCS:-2000}" \
PHASED_TTT_NUM_PHASES="${PHASED_TTT_NUM_PHASES:-3}" \
MLP_CLIP_SIGMAS="${MLP_CLIP_SIGMAS:-12.0}" \
ATTN_CLIP_SIGMAS="${ATTN_CLIP_SIGMAS:-13.0}" \
EMBED_BITS="${EMBED_BITS:-7}" \
EMBED_CLIP_SIGMAS="${EMBED_CLIP_SIGMAS:-15.0}" \
MATRIX_LR="${MATRIX_LR:-0.026}" \
GPTQ_RESERVE_SECONDS="${GPTQ_RESERVE_SECONDS:-0.5}" \
GPTQ_CALIBRATION_BATCHES="${GPTQ_CALIBRATION_BATCHES:-16}" \
VAL_LOSS_EVERY=0 \
MIN_LR="${MIN_LR:-0.10}" \
FUSED_CE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_INIT_STD=0.0 \
SPARSE_ATTN_GATE_SCALE=1.0 \
GATED_ATTN_ENABLED=0 \
ATTN_OUT_GATE_ENABLED=0 \
GATED_ATTN_QUANT_GATE=1 \
SMEAR_GATE_ENABLED="${SMEAR_GATE_ENABLED:-1}" \
LQER_ENABLED="${LQER_ENABLED:-1}" \
LQER_RANK="${LQER_RANK:-4}" \
LQER_TOP_K="${LQER_TOP_K:-3}" \
TTT_LORA_ALPHA="${TTT_LORA_ALPHA:-144}" \
TTT_WARM_START_A="${TTT_WARM_START_A:-1}" \
TTT_WEIGHT_DECAY="${TTT_WEIGHT_DECAY:-1.0}" \
TTT_ROBUST_LAMBDA="${TTT_ROBUST_LAMBDA:-0.0}" \
SEED="$SEED" \
torchrun --standalone --nproc_per_node="$NPROC" train_gpt_v3.py
```

**Step 2: Make executable and shell-check syntax**

Run:

```bash
chmod +x run_ruling_safe_sota.sh
bash -n run_ruling_safe_sota.sh
```

Expected: no syntax errors.

**Step 3: Commit**

Run:

```bash
git add run_ruling_safe_sota.sh
git commit -m "chore: add ruling-safe sota launcher"
```

Expected: commit succeeds.

## Task 8: Add Ablation Launchers

**Files:**
- Create: `run_ablation_qk525.sh`
- Create: `run_ablation_ttt_robust.sh`

**Step 1: QK5.25 launcher**

Create `run_ablation_qk525.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
QK_GAIN_INIT=5.25 ./run_ruling_safe_sota.sh
```

**Step 2: TTT robust launcher**

Create `run_ablation_ttt_robust.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
TTT_ROBUST_LAMBDA="${TTT_ROBUST_LAMBDA:-0.001}" \
TTT_ROBUST_EVERY="${TTT_ROBUST_EVERY:-250}" \
TTT_ROBUST_START_FRAC="${TTT_ROBUST_START_FRAC:-0.70}" \
./run_ruling_safe_sota.sh
```

**Step 3: Validate and commit**

Run:

```bash
chmod +x run_ablation_qk525.sh run_ablation_ttt_robust.sh
bash -n run_ablation_qk525.sh
bash -n run_ablation_ttt_robust.sh
git add run_ablation_qk525.sh run_ablation_ttt_robust.sh
git commit -m "chore: add sota ablation launchers"
```

Expected: commit succeeds.

## Task 9: Add Artifact Packaging Check

**Files:**
- Create: `check_submission_artifact.py`

**Step 1: Create checker**

Create:

```python
from pathlib import Path
import sys


LIMIT = 16_000_000


def main() -> int:
    code = Path("train_gpt_v3.py")
    artifacts = sorted(Path(".").glob("final_model.int*.ptz"))
    if not artifacts:
        print("missing artifact: final_model.int*.ptz")
        return 2
    artifact = artifacts[-1]
    total = code.stat().st_size + artifact.stat().st_size
    print(f"code_bytes={code.stat().st_size}")
    print(f"artifact={artifact}")
    print(f"artifact_bytes={artifact.stat().st_size}")
    print(f"total_submission_bytes={total}")
    if total > LIMIT:
        print(f"FAIL: exceeds {LIMIT}")
        return 1
    print(f"PASS: margin={LIMIT - total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Step 2: Run no-artifact failure check**

Run:

```bash
python3 check_submission_artifact.py
```

Expected: exits non-zero with `missing artifact` unless a previous artifact exists.

**Step 3: Commit**

Run:

```bash
git add check_submission_artifact.py
git commit -m "chore: add submission artifact checker"
```

Expected: commit succeeds.

## Task 10: Local Static Verification

**Files:**
- Read: `train_gpt_v3.py`
- Read: tests

**Step 1: Run compile and tests**

Run:

```bash
python3 -m py_compile train_gpt_v3.py lossless_caps.py prepare_caseops_data.py
python3 -m pytest tests/test_caseops_roundtrip.py tests/test_submission_safety.py -q
bash -n run_ruling_safe_sota.sh run_ablation_qk525.sh run_ablation_ttt_robust.sh
```

Expected: all pass.

**Step 2: Commit any verification fixes**

If fixes were required:

```bash
git add train_gpt_v3.py lossless_caps.py prepare_caseops_data.py tests run_*.sh check_submission_artifact.py
git commit -m "fix: pass local sota rebuild verification"
```

Expected: clean verification.

## Task 11: H100 Smoke Run

**Files:**
- Read/execute: `run_ruling_safe_sota.sh`
- Output: `logs/*`, `final_model.int6.ptz`

**Step 1: Prepare data**

Run on the H100 machine:

```bash
python3 prepare_caseops_data.py --help
```

Expected: script prints usage. Then run the documented CaseOps data prep using
the canonical FineWeb docs path available on the machine.

**Step 2: Smoke train**

Run:

```bash
SEED=42 NPROC=8 ITERATIONS=100 VAL_LOSS_EVERY=0 PREQUANT_ONLY=1 ./run_ruling_safe_sota.sh
```

Expected: training starts, reaches 100 iterations, no compile/runtime errors.

**Step 3: Fix smoke-only failures**

Patch only the failing path. Re-run:

```bash
python3 -m py_compile train_gpt_v3.py
SEED=42 NPROC=8 ITERATIONS=100 VAL_LOSS_EVERY=0 PREQUANT_ONLY=1 ./run_ruling_safe_sota.sh
```

Expected: smoke completes.

## Task 12: Full Candidate Runs

**Files:**
- Read/execute: `run_ruling_safe_sota.sh`
- Output: logs and artifacts

**Step 1: Baseline three seeds**

Run:

```bash
for SEED in 42 0 1234; do
  SEED=$SEED ./run_ruling_safe_sota.sh > "train_seed${SEED}.log" 2>&1
  python3 check_submission_artifact.py | tee "artifact_seed${SEED}.txt"
done
```

Expected:

- Mean post-TTT BPB below `1.06157` at minimum.
- Promotion threshold below `1.0595`.
- Every artifact total below `16,000,000`.
- Train and eval both below 600 seconds.

**Step 2: TTT robust ablation**

Run:

```bash
for SEED in 42 0 1234; do
  SEED=$SEED ./run_ablation_ttt_robust.sh > "train_tttrobust_seed${SEED}.log" 2>&1
done
```

Expected: promote only if mean BPB improves and artifact/time remain valid.

**Step 3: QK5.25 ablation**

Run:

```bash
for SEED in 42 0 1234; do
  SEED=$SEED ./run_ablation_qk525.sh > "train_qk525_seed${SEED}.log" 2>&1
done
```

Expected: promote only if mean BPB improves and no instability appears.

## Task 13: Package Record Folder

**Files:**
- Create: `records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/README.md`
- Create: `records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/submission.json`
- Create: `records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/train_gpt.py`
- Add: final run logs
- Add: tokenizer/support files required by script

**Step 1: Create folder and copy final script**

Run:

```bash
mkdir -p records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild
cp train_gpt_v3.py records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/train_gpt.py
cp lossless_caps.py prepare_caseops_data.py records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/
mkdir -p records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/tokenizers
cp tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild/tokenizers/
```

Expected: folder contains self-contained script and support files.

**Step 2: Write README**

Include:

- 3-seed table.
- Artifact bytes table.
- Architecture summary.
- Exact run command.
- Compliance notes: no PPM, score-first TTT, no validation during training,
  full distribution scoring, CaseOps reversibility.
- Lineage: PR #1797/#1787 plus local additions.

**Step 3: Write `submission.json`**

Use:

```json
{
  "author": "KaiLean",
  "github_id": "kailean",
  "name": "Ruling-safe SP8192 CaseOps + Smear + LQER + TTT robust rebuild",
  "date": "2026-04-25",
  "track": "10min_16mb",
  "val_bpb": 0.0,
  "val_bpb_std": 0.0,
  "seeds": [42, 0, 1234],
  "hardware": "8xH100 80GB SXM",
  "technique_summary": "SP8192 CaseOps, sparse attention gate, Loop4-5, parallel residuals, full GPTQ int6, Brotli, SmearGate, LQER asym rank-4, phased score-first LoRA-TTT, optional TTT-robust training regularizer",
  "compliance": {
    "train_under_600s": true,
    "artifact_under_16mb": true,
    "eval_under_600s": true,
    "no_ppm_mix": true,
    "validation_not_used_for_training": true,
    "score_first_ttt": true,
    "three_seeds": true
  }
}
```

Replace the zero scores with measured values.

**Step 4: Compile in record folder**

Run:

```bash
cd records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild
python3 -m py_compile train_gpt.py lossless_caps.py prepare_caseops_data.py
```

Expected: compiles.

**Step 5: Commit package**

Run:

```bash
git add records/track_10min_16mb/2026-04-25_KaiLean_RulingSafe_SOTA_Rebuild
git commit -m "records: add ruling-safe sota rebuild submission"
```

Expected: record folder committed.

## Task 14: Final Review

**Files:**
- Review: all modified files

**Step 1: Inspect diff**

Run:

```bash
git status --short
git log --oneline -8
git diff --stat HEAD~8..HEAD
```

Expected: only planned files are included.

**Step 2: Final verification**

Run:

```bash
python3 -m pytest tests/test_caseops_roundtrip.py tests/test_submission_safety.py -q
python3 -m py_compile train_gpt_v3.py
```

Expected: pass.

**Step 3: Prepare PR notes**

Summarize:

- Baseline PR lineage.
- What changed.
- Three-seed mean and std.
- Artifact margin.
- Train/eval timing.
- Compliance guarantees.
