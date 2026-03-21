#!/bin/bash
# Run locally AFTER training finishes to copy artifacts from the pod.
# Update HOST, PORT with your pod's TCP connection info from RunPod dashboard.
HOST=103.207.149.62
PORT=14873

scp -P $PORT -i ~/.ssh/id_ed25519 root@$HOST:/workspace/parameter-golf/submission.log ./submission.log
scp -P $PORT -i ~/.ssh/id_ed25519 root@$HOST:/workspace/parameter-golf/final_model.int8.ptz ./final_model.int8.ptz
scp -P $PORT -i ~/.ssh/id_ed25519 root@$HOST:/workspace/parameter-golf/final_model.int6.ptz ./final_model.int6.ptz
scp -P $PORT -i ~/.ssh/id_ed25519 root@$HOST:/workspace/parameter-golf/final_model.pt ./final_model.pt
