#!/bin/bash
# Reproduce: Mixed-Temperature Self-Generated GPTQ Calibration on V6
#
# Hardware: 1×H100 80GB SXM (RunPod). Total ~3h end-to-end per variant.
# Requirements: PyTorch 2.6+ (for `enable_gqa` in scaled_dot_product_attention),
#               sentencepiece, numpy. No FA3, no lrzip required.
#
# This reproduces both variant A (single temp=0.8 baseline, leader's recipe ported)
# and variant B (mixed-temperature, this submission's contribution).
# Variant B is the submission; A is the same-pipeline baseline for the 0.0054 BPB delta.

set -e

# Setup
pip install --upgrade "torch>=2.6,<2.7" sentencepiece numpy

# Variant A — leader's recipe (single temp=0.8 sampling, BOS-only seed)
python -u gptq_v6_dreamcal.py \
    --self-gen --calib-temp 0.8 --bos-seed \
    --calib-seqs 64 --seq-len 2048 --emb6 \
    --suffix-tag dreamcal_A_t08 \
    2>&1 | tee dreamcal_A_t08_run.log

python -u sliding_window_eval_v6.py \
    best_model_v6_ema_gptq_4bit_emb6_dreamcal_A_t08_hessian_roundtrip.pt \
    --gpu 0 \
    2>&1 | tee eval_dreamcal_A_t08.log

# Variant B — mixed-temperature (this submission)
python -u gptq_v6_dreamcal.py \
    --self-gen --mixed-temp --temp-low 0.5 --temp-high 1.5 --bos-seed \
    --calib-seqs 64 --seq-len 2048 --emb6 \
    --suffix-tag dreamcal_B_mix0515 \
    2>&1 | tee dreamcal_B_mix0515_run.log

python -u sliding_window_eval_v6.py \
    best_model_v6_ema_gptq_4bit_emb6_dreamcal_B_mix0515_hessian_roundtrip.pt \
    --gpu 0 \
    2>&1 | tee eval_dreamcal_B_mix0515.log

echo "Variant A val_bpb:" ; grep "val_bpb:" eval_dreamcal_A_t08.log
echo "Variant B val_bpb:" ; grep "val_bpb:" eval_dreamcal_B_mix0515.log
