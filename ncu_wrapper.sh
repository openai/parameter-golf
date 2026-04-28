#!/bin/bash
cd /workspace/parameter-golf
if [ "$LOCAL_RANK" = "0" ]; then
    exec /usr/local/cuda/bin/ncu --set full \
        --kernel-name-base demangled \
        -o /workspace/parameter-golf/pr1493_profile \
        -f \
        /usr/bin/python3 /workspace/parameter-golf/profile_pr1493.py "$@"
else
    exec /usr/bin/python3 /workspace/parameter-golf/profile_pr1493.py "$@"
fi
