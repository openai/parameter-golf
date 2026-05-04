#!/bin/bash
# V19 3-seed validation: 42, 314, 1234 (matches PR #1908 / dexhunter convention).
# Expected runtime: ~80 min total. Cost ~$2.5.
# RUN ONLY AFTER scout shows V19 < 0.9755 on seed 42.
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-30_V19_PR1908_AsymLogit_WD2/

echo "===================================================="
echo "  V19 3-seed: PR #1908 + AsymLogit + TTT_WD=2.0"
echo "  Seeds 42 + 314 + 1234   Start: $(date)"
echo "===================================================="

# 3-seed includes the V19c stacked recipe: AsymLogit + simon-marcus hparams.
# CRITICAL CASEOPS_ENABLED=1 (matches PR #1908 actual training run).
# Without this BPB is computed with SP LUT byte counting -> ~0.97 instead of ~1.06
ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ \
  CASEOPS_ENABLED=1 \
  DATA_PATH=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  ASYM_LOGIT_RESCALE=1 \
  TTT_WEIGHT_DECAY=2.0 \
  MATRIX_LR=0.028 \
  PHASED_TTT_PREFIX_DOCS=3500 \
  AWQ_LITE_ENABLED=1 \
  AWQ_LITE_BITS=8 \
  AWQ_LITE_GROUP_TOP_K=1 \
  AWQ_LITE_GROUP_SIZE=64 \
  LQER_ENABLED=1 \
  LQER_ASYM_ENABLED=1 \
  LQER_RANK=4 \
  LQER_FACTOR_BITS=4 \
  LQER_ASYM_GROUP=64 \
  LQER_TOP_K=3"

for SEED in 42 314 1234; do
  echo ""
  echo "========== SEED $SEED [$(date)] =========="
  env SEED=$SEED $ENV_VARS \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v19_seed${SEED}.log 2>&1

  cp final_model.int6.ptz /workspace/v19_seed${SEED}_model.int6.ptz 2>/dev/null || true
  cp /workspace/scout_v19_seed${SEED}.log /workspace/v19_seed${SEED}_FULL.log 2>/dev/null || true

  echo "--- Seed $SEED done [$(date)] ---"
  grep -E "stopping_early|quantized_ttt_phased" /workspace/scout_v19_seed${SEED}.log | tail -5
done

echo ""
echo "===================================================="
echo "  V19 3-SEED FINAL RESULTS  $(date)"
echo "===================================================="
python3 << 'PYEOF'
import re

def get_bpb(seed):
    paths = [f'/workspace/v19_seed{seed}_FULL.log', f'/workspace/scout_v19_seed{seed}.log']
    for p in paths:
        try:
            with open(p) as f:
                content = f.read()
            m = re.search(r'quantized_ttt_phased\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)', content)
            if m:
                return float(m.group(1))
        except FileNotFoundError:
            continue
    return None

results = {s: get_bpb(s) for s in [42, 314, 1234]}
print("=== V19 3-SEED RESULTS ===")
for s, bpb in results.items():
    print(f"  Seed {s}: {bpb}")

vals = [v for v in results.values() if v]
if len(vals) == 3:
    mean = sum(vals) / 3
    std = (sum((v - mean) ** 2 for v in vals) / 3) ** 0.5
    print()
    print(f"  3-seed MEAN: {mean:.6f}")
    print(f"  3-seed STD:  {std:.6f}")
    print()
    print(f"  vs PR #1908 reported (1.06081):    delta {1.06081 - mean:+.6f}")
    print(f"  vs PR #1855 reported (1.06108):    delta {1.06108 - mean:+.6f}")
    print(f"  vs PR #1925 reported (1.06049):    delta {1.06049 - mean:+.6f}")
    print()
    print(f"  Community merge floor: 0.0006 BPB delta")
    print(f"  Win threshold (< 1.06021): {'WIN' if mean < 1.06021 else 'tied/loss'}")
PYEOF

echo ""
echo "Files backed up:"
ls -lh /workspace/v19_seed*_model.int6.ptz 2>/dev/null
ls -lh /workspace/v19_seed*_FULL.log 2>/dev/null
