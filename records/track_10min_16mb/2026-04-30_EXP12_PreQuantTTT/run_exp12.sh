#!/bin/bash

# run_exp12.sh
# -------------------------------------------------------------
# EXP12: Depth Recurrence + PreQuantTTT + LZMA Compression
# -------------------------------------------------------------

# 1. Setup Depth Recurrence natively supported by the code
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5

# 2. Add hyperparams for the PreQuantTTT
export PRE_QUANT_EPOCHS=21
export PRE_QUANT_LR=5e-4
export VOCAB_SIZE=8192

# Point DATA_DIR to the workspace root so relative paths resolve correctly
export DATA_DIR=/workspace/parameter-golf/data/

# 3. Execute the custom EXP12 Python script from workspace root
echo "Running EXP12 Training Pipeline with Depth Recurrence..."
cd /workspace/parameter-golf
python records/track_10min_16mb/2026-04-30_EXP12_PreQuantTTT/train_gpt_exp12.py

# 4. Check if successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    
    # 5. Pack the script using LZMA
    echo "Compressing python script for final artifact to save space..."
    python3 -c "
import lzma, base64
with open('train_gpt_exp12.py', 'rb') as f:
    code = f.read()
compressed = base64.b85encode(lzma.compress(code)).decode()
with open('submission.json', 'r') as f:
    import json
    data = json.load(f)
data['code_compressed'] = compressed
with open('submission.json', 'w') as f:
    json.dump(data, f)
"
    echo "Done! The artifact is now ready to be pushed to GitHub."
else
    echo "Training failed. Please check logs."
    exit 1
fi
