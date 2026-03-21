#!/bin/bash
cd /home/ubuntu/parameter-golf
# Uses modified train_gpt.py with mode="max-autotune"
export ITERATIONS=2000 RUN_ID=exp022_max_autotune
export LOGIT_SOFTCAP=15 ADAM_EPS=1e-10
python3 train_gpt.py
