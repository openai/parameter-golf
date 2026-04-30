#!/bin/bash
# Run 2 more V18 seeds (314 + 1234) after seed 42 already done
# Backs up models + logs to /workspace/, prints final 3-seed summary
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/

ENV_VARS="DATA_DIR=/workspace/caseops_data/datasets/ TTT_WEIGHT_DECAY=2.0 MIN_LR=0.10 MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_RESERVE_SECONDS=0.5 TTT_LORA_ALPHA=144 TTT_WARM_START_A=1 MATRIX_LR=0.026"

for SEED in 314 1234; do
  echo "========== SEED $SEED [$(date)] =========="
  env SEED=$SEED $ENV_VARS \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v18_seed${SEED}.log 2>&1

  cp final_model.int6.ptz /workspace/v18_seed${SEED}_model.int6.ptz 2>/dev/null || true
  cp /workspace/scout_v18_seed${SEED}.log /workspace/v18_seed${SEED}_FULL.log 2>/dev/null || true

  echo "--- Seed $SEED done [$(date)] ---"
  grep -E "quantized_ttt_phased|val_bpb:" /workspace/scout_v18_seed${SEED}.log | tail -5
done

echo ""
echo "========== ALL DONE [$(date)] =========="
echo ""

python3 << 'PYEOF'
import re

def get_bpb(seed):
    paths = [f'/workspace/v18_seed{seed}_FULL.log', f'/workspace/scout_v18_seed{seed}.log']
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
print("=== 3-SEED V18 RESULTS ===")
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
    print(f"  vs merged SOTA bigbag (1.0810): delta {1.0810 - mean:+.6f}")
    print(f"  vs PR #1797 dexhunter (1.06412): delta {1.06412 - mean:+.6f}")
    print(f"  Record threshold (1.0738): {'BREAK' if mean <= 1.0738 else 'MISS'}")
PYEOF

echo ""
echo "Files backed up:"
ls -lh /workspace/v18_seed*_model.int6.ptz 2>/dev/null
ls -lh /workspace/v18_seed*_FULL.log 2>/dev/null
