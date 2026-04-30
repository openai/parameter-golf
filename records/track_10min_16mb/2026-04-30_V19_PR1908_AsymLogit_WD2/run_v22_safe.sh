#!/bin/bash
# V22 SAFE: V21 base + PR #1953's 7 levers + EVAL_SEQ_LEN=2816 (intermediate safe value)
#
# vs PR #1953 (1.05855):
#   - EVAL_SEQ_LEN: 2816 (vs 2560) -- longer context, ~10% eval time penalty
#   - All other 6 levers identical
#
# Predicted: ~1.0578-1.0586 (3-seed mean), ~5% chance eval > 600s
# Win threshold (vs PR #1967 N-gram Tilt 1.05851): need < 1.05851 = 50% prob if eval works
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V22 SAFE: V21 + PR #1953 7 levers + EVAL=2816"
echo "  3-seed: 42, 0, 1234   Start: $(date)"
echo "===================================================="

# Common env vars: V21 base + PR #1953 lever stack + EVAL=2816
ENV_VARS_V22="DATA_DIR=/workspace/caseops_data/datasets/ \
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
  ASYM_LOGIT_RESCALE=1 \
  EVAL_SEQ_LEN=2816 \
  TTT_EVAL_SEQ_LEN=2816 \
  TTT_MASK=no_qv \
  TTT_Q_LORA=0 \
  TTT_V_LORA=0 \
  TTT_LOCAL_LR_MULT=0.75 \
  QK_GAIN_INIT=5.25"

for SEED in 42 0 1234; do
  echo ""
  echo "========================================"
  echo "  V22 SEED $SEED  Start: $(date)"
  echo "========================================"

  env SEED=$SEED $ENV_VARS_V22 \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v22_seed${SEED}.log 2>&1

  cp final_model.int6.ptz /workspace/v22_seed${SEED}_model.int6.ptz 2>/dev/null || true

  echo "--- V22 Seed $SEED done at $(date) ---"
  grep -E "stopping_early|train_time|quantized_ttt_phased|Total submission|total_eval_time" /workspace/scout_v22_seed${SEED}.log | tail -8
done

echo ""
echo "===================================================="
echo "  V22 3-SEED FINAL RESULTS  $(date)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_data(seed):
    with open(f'/workspace/scout_v22_seed{seed}.log') as f:
        c = f.read()
    bpb_m = re.search(r'quantized_ttt_phased\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)', c)
    sz_m  = re.search(r'Total submission size quantized\+pergroup:\s+(\d+)', c)
    wt_m  = re.search(r'stopping_early:\s+wallclock_cap\s+train_time:\s+(\d+)ms', c)
    et_m  = re.search(r'total_eval_time:([\d.]+)s', c)
    return {
        'val_bpb': float(bpb_m.group(1)) if bpb_m else None,
        'artifact': int(sz_m.group(1)) if sz_m else None,
        'train_ms': int(wt_m.group(1)) if wt_m else None,
        'eval_s': float(et_m.group(1)) if et_m else None,
    }

results = {s: get_data(s) for s in [42, 0, 1234]}
print(f"{'seed':>6} {'val_bpb':>11} {'artifact':>12} {'train':>10} {'eval':>10}")
for s in [42, 0, 1234]:
    r = results[s]
    if r['val_bpb']:
        print(f"{s:>6} {r['val_bpb']:>11.6f} {r['artifact']:>12,} {r['train_ms']/1000:>9.2f}s {r['eval_s']:>9.2f}s")
    else:
        print(f"{s:>6} MISSING")

vals = [r['val_bpb'] for r in results.values() if r['val_bpb']]
if len(vals) == 3:
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print(f"\n  V22 3-SEED MEAN: {mean:.6f}")
    print(f"  V22 3-SEED STD:  {std:.6f}")
    print()
    print(f"  vs V21        (1.059434): delta {1.059434 - mean:+.6f}")
    print(f"  vs PR #1965   (1.058749): delta {1.058749 - mean:+.6f}")
    print(f"  vs PR #1953   (1.058554): delta {1.058554 - mean:+.6f}")
    print(f"  vs PR #1967   (1.058510): delta {1.058510 - mean:+.6f}")
    print(f"  vs MERGED SOTA (1.0810):  delta {1.0810   - mean:+.6f}")
    if mean < 1.05851:
        print(f"\n  *** V22 BEATS PR #1967 1.05851! Likely #1 legal ***")
    elif mean < 1.05855:
        print(f"\n  *** V22 BEATS PR #1953 1.05855! Likely #1-2 ***")
    elif mean < 1.05875:
        print(f"\n  *** V22 BEATS PR #1965, between 1953 and 1965 (#3) ***")
    elif mean < 1.05943:
        print(f"\n  *** V22 BEATS V21, between 1965 and V21 (#4-5) ***")
    else:
        print(f"\n  V22 doesn't improve V21 - regression")
PYEOF
