#!/bin/bash
# V21 OPTIMIZED 2-seed: seed 0 + seed 1234 with GPTQ_RESERVE_SECONDS=4.0 (strict <600s wallclock)
#
# V21 3-seed seed 42 (FSS=4920, GPTQ_RESERVE=0.5): wallclock 602.048s, val_bpb 1.05834
# Issue: GPTQ_RESERVE=0.5 -> effective training = 599.5s, last step overshoots ~2s -> 602s
#
# Fix: GPTQ_RESERVE_SECONDS=4.0 -> effective training = 596s -> wallclock ~596-598s ✅
# No FORCE_STOP_STEP (let wallclock cap trigger naturally)
#
# Cost: ~5-7 fewer steps of training -> pre-quant +0.0001-0.0002 BPB worse -> final ~1.0585-1.0590
# Still breaks frontier 1.06081 by 0.001-0.002 BPB
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V21 OPT seeds 0 + 1234 (GPTQ_RESERVE=4.0, no FSS)"
echo "  Start: $(date)"
echo "===================================================="

ENV_VARS_OPTIMIZED="DATA_DIR=/workspace/caseops_data/datasets/ \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  CASEOPS_ENABLED=1 VOCAB_SIZE=8192 \
  ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
  WARMUP_STEPS=20 WARMDOWN_FRAC=0.85 BETA2=0.99 \
  GRAD_CLIP_NORM=0.3 MIN_LR=0.1 MATRIX_LR=0.026 \
  GLOBAL_TTT_MOMENTUM=0.9 \
  SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_SCALE=0.5 \
  SMEAR_GATE_ENABLED=1 GATE_WINDOW=12 GATED_ATTN_QUANT_GATE=1 \
  FUSED_CE_ENABLED=1 EMBED_BITS=7 \
  MLP_CLIP_SIGMAS=11.5 ATTN_CLIP_SIGMAS=13.0 EMBED_CLIP_SIGMAS=14.0 \
  GPTQ_RESERVE_SECONDS=4.0 GPTQ_CALIBRATION_BATCHES=16 COMPRESSOR=pergroup \
  LQER_ENABLED=1 LQER_ASYM_ENABLED=1 LQER_RANK=4 LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 LQER_TOP_K=3 \
  AWQ_LITE_ENABLED=1 AWQ_LITE_BITS=8 AWQ_LITE_GROUP_TOP_K=1 AWQ_LITE_GROUP_SIZE=64 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2500 PHASED_TTT_NUM_PHASES=3 \
  TTT_CHUNK_SIZE=48 TTT_BETA2=0.99 TTT_WEIGHT_DECAY=0.5 TTT_LORA_RANK=80 \
  MUON_BACKEND_STEPS=5 NCCL_NET=Socket VAL_LOSS_EVERY=0 \
  ASYM_LOGIT_RESCALE=1"

for SEED in 0 1234; do
  echo ""
  echo "========================================"
  echo "  SEED $SEED  Start: $(date)"
  echo "========================================"

  env SEED=$SEED $ENV_VARS_OPTIMIZED \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v21opt_seed${SEED}.log 2>&1

  cp final_model.int6.ptz /workspace/v21opt_seed${SEED}_model.int6.ptz 2>/dev/null || true

  echo "--- Seed $SEED done at $(date) ---"
  grep -E "stopping_early|train_time|quantized_ttt_phased|Total submission|total_eval_time" /workspace/scout_v21opt_seed${SEED}.log | tail -8
done

echo ""
echo "===================================================="
echo "  V21 3-SEED FINAL (seed42 from earlier + opt 0/1234)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_bpb_from(path):
    try:
        with open(path) as f:
            content = f.read()
        m = re.search(r'quantized_ttt_phased\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)', content)
        sm = re.search(r'Total submission size quantized\+pergroup:\s+(\d+)', content)
        tm = re.search(r'stopping_early:\s+wallclock_cap\s+train_time:\s+(\d+)ms', content)
        if m:
            return float(m.group(1)), int(sm.group(1)) if sm else 0, int(tm.group(1))/1000.0 if tm else 0
    except FileNotFoundError:
        pass
    return None, None, None

results = {
    42: get_bpb_from('/workspace/scout_v21_seed42.log'),
    0: get_bpb_from('/workspace/scout_v21opt_seed0.log'),
    1234: get_bpb_from('/workspace/scout_v21opt_seed1234.log'),
}

print(f"{'seed':>6} {'val_bpb':>12} {'artifact':>12} {'wallclock':>10}")
for s in [42, 0, 1234]:
    bpb, size, wt = results[s]
    if bpb:
        print(f"{s:>6} {bpb:>12.6f} {size:>12,} {wt:>9.2f}s")
    else:
        print(f"{s:>6} MISSING")

vals = [r[0] for r in results.values() if r[0]]
if len(vals) == 3:
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print()
    print(f"  3-seed MEAN: {mean:.6f}")
    print(f"  3-seed STD:  {std:.6f}")
    print()
    print(f"  vs PR #1908 frontier (1.06081): delta {1.06081 - mean:+.6f}")
    print(f"  vs PR #1855 official#1(1.06108): delta {1.06108 - mean:+.6f}")
    print(f"  vs win threshold     (1.06021): delta {1.06021 - mean:+.6f}")
    print(f"  vs MERGED SOTA bigbag(1.0810):  delta {1.0810  - mean:+.6f}")
    if mean < 1.06021:
        print(f"  RECORD! Mean below community 0.0006 floor by {1.06021 - mean:.6f} BPB")
PYEOF
