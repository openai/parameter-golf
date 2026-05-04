#!/bin/bash
# V18 3-seed validation: 42, 314, 1234 (matching dexhunter PR #1797 seeds for direct comparison)
set -e

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-29_V18_PR1797Tuned_FullStack/

echo "===================================================="
echo "  V18 3-seed validation: 42 + 314 + 1234"
echo "  Start: $(date)"
echo "===================================================="

ENV_VARS="TTT_WEIGHT_DECAY=2.0 MIN_LR=0.10 MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 GPTQ_RESERVE_SECONDS=0.5 TTT_LORA_ALPHA=144 TTT_WARM_START_A=1 MATRIX_LR=0.026"

for SEED in 42 314 1234; do
  echo ""
  echo "========== SEED $SEED [$(date)] =========="
  env SEED=$SEED $ENV_VARS \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
    > /workspace/scout_v18_seed${SEED}.log 2>&1

  # Backup
  cp final_model.int6.ptz /workspace/v18_seed${SEED}_model.int6.ptz 2>/dev/null || true
  cp /workspace/scout_v18_seed${SEED}.log /workspace/v18_seed${SEED}_FULL.log 2>/dev/null || true

  echo "--- Seed $SEED done ---"
  grep -E "sliding_val_bpb|val_bpb:|Total submission|stopping_early" /workspace/scout_v18_seed${SEED}.log | tail -8
done

echo ""
echo "===================================================="
echo "  V18 3-SEED FINAL RESULTS [$(date)]"
echo "===================================================="
python3 -c "
import re
seeds_data = {}
for s in [42, 314, 1234]:
    try:
        with open(f'/workspace/scout_v18_seed{s}.log') as f:
            content = f.read()
        m = re.search(r'(post_ttt_val_bpb|sliding_val_bpb)[\s:=]+([\d.]+)', content)
        if m:
            seeds_data[s] = float(m.group(2))
            print(f'Seed {s}: {m.group(2)}')
    except Exception as e:
        print(f'Seed {s}: error {e}')

if len(seeds_data) == 3:
    vals = list(seeds_data.values())
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print(f'\\nMEAN: {mean:.6f}')
    print(f'STD:  {std:.6f}')
    print(f'\\nvs dexhunter PR #1797 BOS-fixed: 1.06412')
    print(f'vs record threshold (1.0810 - 0.0072 = 1.0738): {\"BREAK\" if mean <= 1.0738 else \"miss\"}')
    if mean < 1.06412:
        print(f'BEATS dexhunter by {1.06412 - mean:.6f} BPB')
    else:
        print(f'MISSED dexhunter by {mean - 1.06412:.6f} BPB')
"
