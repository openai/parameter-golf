#!/bin/bash
# V15 3-seed validation: runs seed 42 then seed 999 sequentially
# Total time: ~50 min on 8x H100 SXM
# Outputs: /workspace/seeds_42_999_master.log

set -e
echo "===================================================="
echo "  V15 3-seed validation: 42 + 999"
echo "  Start: $(date)"
echo "===================================================="

cd /workspace/parameter-golf/records/track_10min_16mb/2026-04-18_SP8192_ParallelPreQuantTTT/

# ============ SEED 42 ============
echo ""
echo "========== SEED 42 START [$(date)] =========="
SEED=42 \
  DATA_DIR=/workspace/caseops_data/datasets/ \
  DATASETS_DIR=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  TTT_EMA_ENABLED=0 \
  PREQUANT_TTT_ENABLED=1 \
  PREQUANT_TTT_EPOCHS=21 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > /workspace/scout_v15_seed42.log 2>&1

echo "========== SEED 42 DONE [$(date)] =========="

# Backup seed 42 outputs
cp final_model.int6.ptz /workspace/v15_seed42_model.int6.ptz
cp final_model.pt /workspace/v15_seed42_model.pt 2>/dev/null || true
cp /workspace/scout_v15_seed42.log /workspace/v15_seed42_FULL.log

SEED42_BPB=$(grep "quantized_sliding_window val_bpb" /workspace/scout_v15_seed42.log | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
echo "Seed 42 final_int6_sliding val_bpb: $SEED42_BPB"

# ============ SEED 999 ============
echo ""
echo "========== SEED 999 START [$(date)] =========="
SEED=999 \
  DATA_DIR=/workspace/caseops_data/datasets/ \
  DATASETS_DIR=/workspace/caseops_data/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved \
  TOKENIZER_PATH=/workspace/caseops_data/datasets/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
  TTT_EMA_ENABLED=0 \
  PREQUANT_TTT_ENABLED=1 \
  PREQUANT_TTT_EPOCHS=21 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py > /workspace/scout_v15_seed999.log 2>&1

echo "========== SEED 999 DONE [$(date)] =========="

# Backup seed 999 outputs
cp final_model.int6.ptz /workspace/v15_seed999_model.int6.ptz
cp final_model.pt /workspace/v15_seed999_model.pt 2>/dev/null || true
cp /workspace/scout_v15_seed999.log /workspace/v15_seed999_FULL.log

SEED999_BPB=$(grep "quantized_sliding_window val_bpb" /workspace/scout_v15_seed999.log | grep -oP "val_bpb:\K[0-9.]+" | tail -1)
echo "Seed 999 final_int6_sliding val_bpb: $SEED999_BPB"

# ============ FINAL SUMMARY ============
SEED1337_BPB=$(grep "quantized_sliding_window val_bpb" /workspace/v15_seed1337_FULL.log | grep -oP "val_bpb:\K[0-9.]+" | tail -1)

echo ""
echo "===================================================="
echo "  V15 3-SEED FINAL RESULTS"
echo "  End: $(date)"
echo "===================================================="
echo ""
printf "  Seed 1337: %s\n" "$SEED1337_BPB"
printf "  Seed 42:   %s\n" "$SEED42_BPB"
printf "  Seed 999:  %s\n" "$SEED999_BPB"
echo ""

python3 -c "
seeds = ['$SEED1337_BPB', '$SEED42_BPB', '$SEED999_BPB']
vals = [float(s) for s in seeds if s and s != '']
if len(vals) == 3:
    mean = sum(vals)/3
    std = (sum((v-mean)**2 for v in vals)/3)**0.5
    print(f'  3-seed MEAN: {mean:.6f}')
    print(f'  3-seed STD:  {std:.6f}')
    print(f'')
    print(f'  AjAnubolu PR #1735 mean: 1.0429')
    print(f'  Record threshold (-0.0072): 1.0357')
    print(f'')
    if mean <= 1.0357:
        print(f'  RESULT: BREAK RECORD by {1.0357 - mean:.6f} BPB')
        print(f'  Submit as RECORD PR immediately')
    elif mean <= 1.0429:
        print(f'  RESULT: Beats AjAnubolu by {1.0429 - mean:.6f} but not record threshold')
        print(f'  Submit as non-record (Top frontier)')
    else:
        print(f'  RESULT: Worse than AjAnubolu by {mean - 1.0429:.6f}')
else:
    print(f'  ERROR: Could not parse all BPBs: {seeds}')
"

echo ""
echo "  Files backed up:"
ls -lh /workspace/v15_seed*_model.int6.ptz 2>/dev/null
echo ""
echo "  All logs:"
ls -lh /workspace/v15_seed*_FULL.log 2>/dev/null
