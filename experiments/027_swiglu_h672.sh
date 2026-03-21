#!/bin/bash
cd /home/ubuntu/parameter-golf
export USE_SWIGLU=1 SWIGLU_HIDDEN=672
export LOGIT_SOFTCAP=15 ADAM_EPS=1e-10
export RUN_ID=exp027_swiglu_h672
python3 train_gpt_swiglu.py
