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
export TOKENIZER_VOCAB_SIZE=10240

# 3. Execute the custom EXP12 Python script
echo "Running EXP12 Training Pipeline with Depth Recurrence..."
python train_gpt_exp12.py

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
