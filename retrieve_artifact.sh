#!/bin/bash
while true; do
  ssh -o StrictHostKeyChecking=no -i .env root@103.207.149.77 -p 18369 'grep -q "Total submission size" /workspace/parameter-golf/nohup_final.log'
  if [ $? -eq 0 ]; then
    echo "Training completed! Downloading the SOTA artifact and train.log..."
    scp -o StrictHostKeyChecking=no -i .env -P 18369 root@103.207.149.77:/workspace/parameter-golf/final_model.int6.ptz ./final_model_SOTA.int6.ptz
    scp -o StrictHostKeyChecking=no -i .env -P 18369 root@103.207.149.77:/workspace/parameter-golf/logs/competition_run_compliant.txt ./train.log
    echo "Download completed. Both final_model_SOTA.int6.ptz and train.log are secured!"
    break
  fi
  sleep 45
done
