# Spec 027 — LoRA warm-start-A + Depth curriculum + MLPClip12

**Slug:** `lora-warmstart-depth-curriculum`
**Created:** 2026-04-22
**Status:** READY
**Branch:** `exp/027-three-stack`
**Commit:** `e3f47a1`
**Links to:** `research/ideas/ttt-lora-warmstart-a.md`, `research/ideas/recurrence-depth-curriculum.md`, `research/ideas/mlp-clip-sigmas-12.md`

## Hypothesis

Three additive levers — all independently validated by the community on the #1736 base —
stacked on our #1736 baseline:

1. **MLPClip12** (env-var only): GPTQ MLP σ-clip 10→12 improves quantization fidelity on
   the 4×MLP stack. Validated by dexhunter (#1769, 7-seed 1.06477). Already in spec 026's
   run command; included here for completeness. Δ ~−0.001.

2. **LoRA warm-start-A**: Only zero B in TTT batch reset; A accumulates feature directions
   across batches. Add alpha/rank output scaling (alpha=144, rank=96, scale=1.5). Raise LoRA
   WD 0.5→1.0. Validated by renqianluo (#1767) and composited by bigbag (#1771). Δ ~−0.001.

3. **Depth curriculum**: Training phases NUM_LOOPS at depth 1→3→4 (looping activates at 35%
   as usual; depth upgrades from 3→4 at 67%). Eval and TTT always at depth=4. Validated by
   romeerp (#1756) and bigbag (#1771). Δ ~−0.0004 to −0.001.

Combined expected result: ~1.062–1.063 from #1736 baseline of 1.06549.

## Baselines

| run | post-TTT bpb | note |
|---|---|---|
| #1736 (#021e equivalent) | 1.06549 | our baseline |
| #1769 (MLPClip12 only) | 1.06453 | community validation, 7-seed |
| #1771 (curriculum + LoRA-TTT) | 1.06513 | community validation, 3-seed |

## Expected Δ

~−0.002 to −0.003 bpb vs #1736 baseline. Moderate confidence — all three are independently
validated on the #1736 CaseOps stack; combined delta could be sub-additive if there's
TTT headroom compression.

## Accept criteria

| post-TTT bpb | bucket | action |
|---|---|---|
| < 1.062 | Clear beat | 3-seed (42/43/44) same pod → submission |
| [1.062, 1.065] | Beats baseline, matches community | 3-seed to confirm |
| (1.065, 1.068] | Within noise of baseline | Compare seeds; iterate |
| > 1.068 | Regression | Kill; debug which lever hurts |

Mini pass/fail: `loop_depth:upgraded` log line must appear at ~67% of training. No NaN.
Throughput within 10% of spec 021e's 8×H (~700 tok/s on 2×H mini reference).

## Config diff vs #1736 baseline run command

| env var | #1736 | spec 027 |
|---|---|---|
| `NUM_LOOPS` | `2` | **`3`** (depth=4 final; curriculum handles phases) |
| `LOOP_DEPTH_UPGRADE_AT` | (absent) | **`0.67`** |
| `TTT_LORA_ALPHA` | (absent, default 96) | **`144`** |
| `TTT_WEIGHT_DECAY` | `0.5` | **`1.0`** |
| `MLP_CLIP_SIGMAS` | (absent, default 10.0) | **`12.0`** |

All other env vars identical to spec 026's run command.

## Code changes

**Branch:** `exp/027-three-stack` · **Commit:** `e3f47a1`

```diff
 class Hyperparameters:
+    loop_depth_upgrade_at = float(os.environ.get("LOOP_DEPTH_UPGRADE_AT", 0.0))
+    ttt_lora_alpha = int(os.environ.get("TTT_LORA_ALPHA", 96))

 class BatchedLinearLoRA(nn.Module):
-    def __init__(self, bsz, in_features, out_features, rank):
+    def __init__(self, bsz, in_features, out_features, rank, alpha=96):
+        self._scale = alpha / rank
     def reset(self):
-        self.A.uniform_(-self._bound, self._bound)  # re-randomize A
+        # warm-start A: keep; only zero B (LoRA output = 0 at batch start)
         self.B.zero_()
     def forward(self, x):
-        return (x @ self.A.T) @ self.B.T
+        return (x @ self.A.T) @ self.B.T * self._scale

 class BatchedTTTLoRA(nn.Module):
-    def __init__(self, bsz, model, rank, ...):
+    def __init__(self, bsz, model, rank, alpha=96, ...):
     # passes alpha to all BatchedLinearLoRA instantiations

 class GPT(nn.Module):  # in __init__
+    self._num_loops = h.num_loops
     if h.loop_depth_upgrade_at > 0 and h.num_loops >= 2:
+        # precompute intermediate (depth=3) index lists for phase 2
+        self._enc_idx_intermediate, self._dec_idx_intermediate = ...
+        self.looping_depth = h.num_loops - 1  # start at depth=3 after loop activates

 # forward_logits: use intermediate indices when looping_depth < num_loops
+    if self.looping_depth < self._num_loops and hasattr(self, '_enc_idx_intermediate'):
+        enc_iter, dec_iter = self._enc_idx_intermediate, self._dec_idx_intermediate

 # training loop: depth upgrade trigger
+    if h.loop_depth_upgrade_at > 0 and base_model.looping_active
+            and base_model.looping_depth < h.num_loops and frac >= h.loop_depth_upgrade_at:
+        base_model.looping_depth = h.num_loops
+        log(f"loop_depth:upgraded ...")

 # eval/TTT model setup
+    eval_model.looping_depth = h.num_loops  # always full depth at eval/TTT
+    ttt_model.looping_depth = h.num_loops
```

TTT LoRA slot count increases from 17 (depth=3) to 20 (depth=4) — more LoRA
parameters per phase, beneficial for TTT expressivity.

## Hardware ladder

**2×H100 JP mini required** — code changes in training and eval paths.

**Throughput note:** depth=4 adds one extra loop pass (layers 3-5 ~3/11 of model). Expect
~+9% compute per step. Verify actual tok/s in mini run — must not trigger step-time stop-early.

## Seed plan

- **Mini:** seed 42, `MAX_WALLCLOCK_SECONDS=1200`, override phase triggers for fast verification
  (see mini run command below)
- **Official:** 3 seeds (42, 43, 44), 8×H100 JP, same pod sequential

## Run commands

### Mini (2×H100 JP) — phase trigger verification

```bash
pip install --break-system-packages brotli sentencepiece
python -c "import brotli, sentencepiece"

cd /runpod/parameter-golf/records/track_10min_16mb/2026-04-19_SP8192_CaseOps_GatedAttn_QuantGate_Loop45_PhasedTTT
git fetch fork
git checkout e3f47a1

# Sanity verify patches
grep "loop_depth_upgrade_at" train_gpt.py          # must be present
grep "warm-start A" train_gpt.py                   # must be present
grep "_scale = alpha / rank" train_gpt.py           # must be present
grep "loop_depth:upgraded" train_gpt.py             # must be present

mkdir -p /runpod/runs/027-lora-warmstart-depth-curriculum/mini_seed_42
mkdir -p /tmp/torch_inductor_cache_027_mini

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/027-lora-warmstart-depth-curriculum/mini_seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_027_mini \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=0 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=3 LOOP_DEPTH_UPGRADE_AT=0.20 ENABLE_LOOPING_AT=0.10 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=2 train_gpt.py \
  > /runpod/runs/027-lora-warmstart-depth-curriculum/mini_seed_42/train.log 2>&1
```

**What to verify in mini log:**
1. `layer_loop:enabled` fires at ~10% (early-trigger test)
2. `loop_depth:upgraded` fires at ~20% — **required; fail spec if absent**
3. No NaN in train_loss
4. No unexpected recompile warnings beyond compile + first-loop-activation + depth-upgrade
5. Tok/s ≥ 600 (2×H100 baseline)

### Official (8×H100 JP) — seed 42 first

```bash
mkdir -p /runpod/runs/027-lora-warmstart-depth-curriculum/seed_42
mkdir -p /tmp/torch_inductor_cache_027_8h_jp

nvidia-smi --query-gpu=timestamp,index,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,memory.used \
  --format=csv -l 1 \
  > /runpod/runs/027-lora-warmstart-depth-curriculum/seed_42/diag_nvsmi.csv &
NVSMI_PID=$!

NCCL_NET=Socket DATA_DIR=/runpod/data \
ARTIFACT_DIR=/runpod/runs/027-lora-warmstart-depth-curriculum/seed_42 \
TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache_027_8h_jp \
CASEOPS_ENABLED=1 \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
MATRIX_LR=0.026 \
GATED_ATTN_ENABLED=1 GATED_ATTN_INIT_STD=0.005 GATED_ATTN_QUANT_GATE=1 \
NUM_LOOPS=3 LOOP_DEPTH_UPGRADE_AT=0.67 \
TTT_LORA_ALPHA=144 TTT_WEIGHT_DECAY=1.0 \
GPTQ_RESERVE_SECONDS=4 GPTQ_CALIBRATION_BATCHES=16 \
TRAIN_LOG_EVERY=100 \
SEED=42 \
TORCH_LOGS=recompiles \
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  > /runpod/runs/027-lora-warmstart-depth-curriculum/seed_42/train.log 2>&1

kill $NVSMI_PID
```

## Inputs

- Data: `/runpod/data/` (same volume as prior runs)
- Tokenizer: SP8192 (baked into script, path from CASEOPS_ENABLED)
- No hotstart — trains from scratch same as all prior specs on this base

## Checkpoints / artifacts

- `final_model.pt` — post-EMA FP state dict
- `final_model.int6.ptz` — quantized submission artifact
- `train.log` — every-100-step tok/s, val_bpb, train_loss, TTT trace
- `diag_nvsmi.csv` — per-GPU telemetry
- `final.json` — `val_bpb_pre_gptq_post_ema`, `val_bpb_post_gptq`, `val_bpb_post_ttt`

## Stop-early criteria

- NaN/inf in train_loss → halt immediately
- `loop_depth:upgraded` not found by step 2500 (75% of training) on 8×H → halt, bug in curriculum
- Step time > 2.0 s/step at 8×H (depth-4 adds compute; normal is ~1.3 s/step + ~10% buffer)
- `loop_warmup` or `layer_loop:enabled` fires without subsequent `loop_depth:upgraded` by
  step 2500 → halt

## Cost estimate

| item | cost |
|---|---|
| 2×H JP mini × ~25 min | ~$3 |
| 8×H JP × ~28 min (train + GPTQ + TTT) seed 42 | ~$11 |
| Conditional 8×H × 2 more seeds | ~$22 |
| **Max total** | **~$36** |

## Open questions for executor interview

1. **Spec 026 result available?** If spec 026 (cross-layer carry) has a post-TTT result before
   this pod is provisioned, report it so research can update the accept criteria. Spec 027 and
   026 are parallel arcs; if 026 delivers < 1.062, spec 027's bar may need to move.

2. **Recompile count:** After training, check `TORCH_LOGS=recompiles` output. Expect at most
   3 compile events: (a) initial, (b) loop activation at 35%, (c) depth upgrade at 67%.
   If more than 5 recompiles total, halt and report — likely means compile-time specialization
   is re-triggering unexpectedly.

3. **JP stock?** Provision with `--template-id y5cejece4j` (parameter-golf image). Do not
   use other regions or other templates.

4. **Monitoring cadence?** Ask user: poll every 30s or leave to notification?

5. **Failure mode:** If mini shows `loop_depth:upgraded` absent, the `loop_depth_upgrade_at`
   env var may not be reaching the code — halt and report back to research before 8×H launch.
